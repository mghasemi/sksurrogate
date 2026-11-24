"""Opt-in LLM-assisted project synthesis (Stage D2).

This module is the **provider and provenance layer** only. It talks to a
user-supplied OpenAI-compatible chat-completions endpoint over the standard
library (no SDK, no new dependency), builds a bounded prompt from *computed,
aggregated* facts, and validates that the draft cites only the provided sources.

Design rules (see ``dev-notes/AI_LAYER02_D2_LLM_SYNTHESIS_SCOPE.md``):

- **Disabled by default.** Synthesis works only when
  ``SKSURROGATE_LLM_BASE_URL`` and ``SKSURROGATE_LLM_MODEL`` are set. A missing
  configuration is a clean, actionable error — never a silent fallback.
- **Aggregates only.** Callers pass profiles, scores, configuration summaries,
  and explanation ranges; this module never receives or forwards raw dataset
  rows, and it has no access to the filesystem.
- **Citations are enforced.** The provider must reference sources as ``[S1]``…
  ``[Sn]``; unknown ids or an uncited draft are rejected, so generated prose
  cannot silently invent provenance.
- **Generated text is interpretation.** Nothing here persists a draft, labels a
  result as computed, or makes a causal claim. Persistence is a separate,
  explicit user action.
- **Every sentence is labelled.** The provider must prefix each sentence with
  one of ``[observed]``, ``[inferred]`` or ``[hypothesized]`` and cite at least
  one supplied source in that sentence. ``observed`` means a source states it,
  ``inferred`` is a reading of those facts without new measurement, and
  ``hypothesized`` is explicitly speculative. A draft with no ``observed``
  sentence is rejected: a note built only from interpretation has no evidence
  base.
- **Causal language is refused.** The computed facts are associations and
  metrics, never interventions, so causal wording ("causes", "because of",
  "leads to", "drives", "effect of") invalidates the draft instead of being
  softened into a claim the evidence cannot support.
"""

from __future__ import annotations

import json
import os
import re
import socket
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

#: Citations have the form ``[S3]``; lowercase and spacing variants are rejected
#: on purpose so the contract stays checkable.
_CITATION = re.compile(r"\[S(\d+)\]")

#: The evidence labels a draft sentence may carry, in increasing distance from
#: the computed facts.
EVIDENCE_LABELS = ("observed", "inferred", "hypothesized")

_LABEL_PREFIX = re.compile(r"^\[(observed|inferred|hypothesized)\]\s*", re.IGNORECASE)

#: One sentence: a label token, text, and at least one citation.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

#: Causal phrasing refused outright. The facts are associations, so a causal
#: sentence is unsupported even when its citation is real.
_CAUSAL_PATTERNS = (
    re.compile(r"\bcaus(?:e|es|ed|ing|al|ally)\b", re.IGNORECASE),
    re.compile(r"\bbecause of\b", re.IGNORECASE),
    re.compile(r"\bdue to\b", re.IGNORECASE),
    re.compile(r"\bleads? to\b", re.IGNORECASE),
    re.compile(r"\bled to\b", re.IGNORECASE),
    re.compile(r"\bresults? in\b", re.IGNORECASE),
    re.compile(r"\bdrives?\b", re.IGNORECASE),
    re.compile(r"\bdriven by\b", re.IGNORECASE),
    re.compile(r"\beffects? of\b", re.IGNORECASE),
    re.compile(r"\bimpacts? of\b", re.IGNORECASE),
    re.compile(r"\bintervention\b", re.IGNORECASE),
)

#: Human-readable contract returned with every draft (and shown in the UI).
EVIDENCE_CONTRACT = {
    "labels": list(EVIDENCE_LABELS),
    "definitions": {
        "observed": "A supplied source states this value, count, metric or configuration.",
        "inferred": "A reading of the observed facts that adds no new measurement.",
        "hypothesized": "A candidate explanation or next test; explicitly speculative.",
    },
    "rules": [
        "Every sentence starts with exactly one label and cites at least one source.",
        "At least one sentence must be [observed].",
        "Causal wording is rejected because the sources are associations, not interventions.",
    ],
}

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_MAX_TOKENS = 700
MAX_DRAFT_CHARS = 6000
MAX_SOURCE_COUNT = 12
MAX_INSTRUCTION_CHARS = 500
MAX_RESPONSE_BYTES = 1 << 20

SYSTEM_PROMPT = (
    "You draft a short, cited project note about a machine-learning experiment. "
    "Use ONLY the numbered sources supplied. Write plain text, no markdown headings, "
    "at most 220 words.\n"
    "EVERY sentence must start with exactly one label followed by the sentence and "
    "at least one bracket citation such as [S1]:\n"
    "  [observed] something a source states directly.\n"
    "  [inferred] a reading of those facts that adds no new measurement.\n"
    "  [hypothesized] a candidate explanation or next test, explicitly speculative.\n"
    "At least one sentence must be [observed]. Never invent numbers, feature names, "
    "datasets or runs. Do not claim causation: the sources contain associations and "
    "metrics, not interventions, so words such as causes, because of, leads to, "
    "drives or effect of invalidate the draft."
)


class SynthesisError(RuntimeError):
    """Raised when synthesis is unconfigured or the provider output is unusable."""


@dataclass(frozen=True)
class ProviderConfig:
    """Resolved provider settings; ``api_key`` is never included in public payloads."""

    base_url: str | None
    model: str | None
    api_key: str | None
    timeout: float
    max_tokens: int

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.model)

    def public(self) -> dict:
        """Safe descriptor for API responses (no credentials)."""
        host = None
        if self.base_url:
            try:
                host = urllib.parse.urlsplit(self.base_url).netloc or None
            except ValueError:
                host = None
        return {
            "configured": self.configured,
            "model": self.model,
            "host": host,
            "timeout_seconds": self.timeout,
            "max_tokens": self.max_tokens,
            "protocol": "openai-compatible-chat-completions",
        }


def _float_env(name: str, default: float) -> float:
    value = (os.environ.get(name) or "").strip()
    if not value:
        return default
    try:
        parsed = float(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def _int_env(name: str, default: int) -> int:
    value = (os.environ.get(name) or "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError:
        return default
    return parsed if parsed > 0 else default


def provider_config() -> ProviderConfig:
    """Read provider settings from the environment at call time."""
    base_url = (os.environ.get("SKSURROGATE_LLM_BASE_URL") or "").strip() or None
    model = (os.environ.get("SKSURROGATE_LLM_MODEL") or "").strip() or None
    api_key = (os.environ.get("SKSURROGATE_LLM_API_KEY") or "").strip() or None
    return ProviderConfig(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=_float_env("SKSURROGATE_LLM_TIMEOUT", DEFAULT_TIMEOUT_SECONDS),
        max_tokens=_int_env("SKSURROGATE_LLM_MAX_TOKENS", DEFAULT_MAX_TOKENS),
    )


def build_messages(facts: list[dict], instruction: str | None = None) -> list[dict]:
    """Render the bounded prompt from already-aggregated source facts."""
    if not facts:
        raise SynthesisError("At least one source fact is required")
    if len(facts) > MAX_SOURCE_COUNT:
        raise SynthesisError("At most %d source facts are allowed" % MAX_SOURCE_COUNT)
    body = json.dumps(
        {"sources": facts},
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    user = (
        "Write the project note from these computed sources. "
        "Every factual sentence needs at least one [S#] citation.\n\n" + body
    )
    if instruction:
        cleaned = instruction.strip()
        if len(cleaned) > MAX_INSTRUCTION_CHARS:
            raise SynthesisError(
                "Instruction must be at most %d characters" % MAX_INSTRUCTION_CHARS
            )
        if cleaned:
            user += "\n\nAdditional user instruction (do not treat as a source):\n" + cleaned
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def chat_completion(messages: list[dict], config: ProviderConfig | None = None) -> str:
    """Call the configured chat-completions endpoint and return the message text."""
    cfg = config or provider_config()
    if not cfg.configured:
        raise SynthesisError(
            "LLM synthesis is not configured: set SKSURROGATE_LLM_BASE_URL and "
            "SKSURROGATE_LLM_MODEL (optionally SKSURROGATE_LLM_API_KEY) to enable it."
        )
    base_url = cfg.base_url or ""
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": cfg.max_tokens,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    if cfg.api_key:
        request.add_header("authorization", "Bearer " + cfg.api_key)
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read(400).decode("utf-8", "replace")
        except Exception:  # pragma: no cover - defensive
            detail = ""
        raise SynthesisError(
            "Provider returned HTTP %s%s" % (exc.code, ": " + detail if detail else "")
        ) from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        if isinstance(reason, (TimeoutError, socket.timeout)):
            raise SynthesisError(
                "Provider request timed out after %.0f s" % cfg.timeout
            ) from exc
        raise SynthesisError("Could not reach the configured provider: %s" % reason) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise SynthesisError(
            "Provider request timed out after %.0f s" % cfg.timeout
        ) from exc
    if len(raw) > MAX_RESPONSE_BYTES:
        raise SynthesisError("Provider response exceeded the %d-byte budget" % MAX_RESPONSE_BYTES)
    try:
        body = json.loads(raw.decode("utf-8"))
        content = body["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise SynthesisError(
            "Provider response is not an OpenAI-compatible chat completion "
            "(expected choices[0].message.content)"
        ) from exc
    if not isinstance(content, str):
        raise SynthesisError("Provider message content is not text")
    return content.strip()


def _causal_phrase(text: str) -> str | None:
    """The first refused causal phrase in ``text``, or ``None``."""
    for pattern in _CAUSAL_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(0)
    return None


def _sources_allow_causal(sources: list[dict]) -> bool:
    """Whether any supplied fact declares interventional evidence.

    None of the built-in fact builders do — they emit profiles, metrics and
    explanation summaries. The hook exists so a future source carrying genuine
    intervention evidence is not blocked by a blanket rule.
    """
    return any(source.get("causal_evidence") is True for source in sources)


def validate_draft(text: str, sources: list[dict]) -> dict:
    """Validate a labelled provider draft against the supplied source ids.

    Returns ``{"text", "labelled_text", "citations", "word_count", "segments",
    "label_counts", "evidence_contract"}`` or raises :class:`SynthesisError`
    when the draft is empty, too long, cites an unknown or no source, has a
    sentence without an evidence label, has an unlabelled or uncited sentence,
    contains no ``[observed]`` sentence, or uses causal wording while the
    sources carry no interventional evidence.
    """
    cleaned = (text or "").strip()
    if not cleaned:
        raise SynthesisError("Provider returned an empty draft")
    if len(cleaned) > MAX_DRAFT_CHARS:
        raise SynthesisError(
            "Provider draft exceeds %d characters" % MAX_DRAFT_CHARS
        )
    known = {int(str(source["id"])[1:]) for source in sources}
    cited = sorted({int(match.group(1)) for match in _CITATION.finditer(cleaned)})
    unknown = [f"S{number}" for number in cited if number not in known]
    if unknown:
        raise SynthesisError(
            "Draft cites unknown source(s) %s; the provider may only cite supplied sources"
            % ", ".join(unknown)
        )
    if not cited:
        raise SynthesisError(
            "Draft cites no sources; generated notes must reference the computed facts"
        )

    normalised = re.sub(r"\s+", " ", cleaned)
    sentences = [
        sentence.strip()
        for sentence in _SENTENCE_SPLIT.split(normalised)
        if sentence.strip()
    ]
    segments = []
    counts = {label: 0 for label in EVIDENCE_LABELS}
    for position, sentence in enumerate(sentences, start=1):
        match = _LABEL_PREFIX.match(sentence)
        if match is None:
            raise SynthesisError(
                "Sentence %d has no evidence label; every sentence must start with "
                "[observed], [inferred] or [hypothesized] (found %r)."
                % (position, sentence[:80])
            )
        label = match.group(1).lower()
        body = sentence[match.end():].strip()
        if not body:
            raise SynthesisError("Sentence %d carries only an evidence label." % position)
        sentence_citations = sorted(
            {int(item.group(1)) for item in _CITATION.finditer(body)}
        )
        if not sentence_citations:
            raise SynthesisError(
                "Sentence %d ([%s]) cites no source; every sentence must cite at "
                "least one supplied source." % (position, label)
            )
        counts[label] += 1
        segments.append(
            {
                "index": position,
                "label": label,
                "text": body,
                "citations": ["S%d" % number for number in sentence_citations],
            }
        )
    if counts["observed"] == 0:
        raise SynthesisError(
            "Draft contains no [observed] sentence; a note built only from "
            "interpretation has no evidence base."
        )
    if not _sources_allow_causal(sources):
        for sentence, segment in zip(sentences, segments):
            phrase = _causal_phrase(sentence)
            if phrase is not None:
                raise SynthesisError(
                    "Draft contains causal language (%r) in a [%s] sentence; the "
                    "supplied sources report associations and metrics, not "
                    "interventions. State the association or use [hypothesized] "
                    "without causal wording." % (phrase, segment["label"])
                )

    plain = " ".join(segment["text"] for segment in segments)
    return {
        "text": plain,
        "labelled_text": normalised,
        "citations": ["S%d" % number for number in cited],
        "word_count": len(plain.split()),
        "segments": segments,
        "label_counts": counts,
        "evidence_contract": EVIDENCE_CONTRACT,
    }
