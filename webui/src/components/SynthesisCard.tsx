import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import {
  addGeneratedProjectNote,
  getJob,
  getSynthesisStatus,
  submitSynthesis,
} from "../api/client";
import type { Project, SynthesisResult } from "../api/client";
import { Card, ErrorNote } from "./ui";

/**
 * Stage D2 — opt-in cited-draft synthesis for one research project.
 *
 * The draft is generated prose, not a computed result: it is rendered with its
 * citation chips and the exact computed facts separately, and is persisted only
 * when the user explicitly adds it as a note.
 */
export function SynthesisCard({ project }: { project: Project }) {
  const qc = useQueryClient();
  const bundles = useMemo(
    () => project.artifacts.filter((artifact) => artifact.kind === "model_bundle"),
    [project.artifacts],
  );
  const jobs = useMemo(
    () => project.artifacts.filter((artifact) => artifact.kind === "job"),
    [project.artifacts],
  );
  const tasks = useMemo(
    () => [...new Set(project.artifacts.map((artifact) => artifact.task_name))],
    [project.artifacts],
  );

  const [task, setTask] = useState(tasks[0] ?? "");
  const [version, setVersion] = useState(bundles[0]?.artifact_id ?? "");
  const [jobId, setJobId] = useState("");
  const [includeExplanations, setIncludeExplanations] = useState(true);
  const [instruction, setInstruction] = useState("");
  const [synthesisJob, setSynthesisJob] = useState("");

  useEffect(() => {
    if (!task && tasks.length) setTask(tasks[0]);
  }, [task, tasks]);
  useEffect(() => {
    if (!version && bundles.length) setVersion(bundles[0].artifact_id);
  }, [version, bundles]);

  const statusQ = useQuery({ queryKey: ["synthesis-status"], queryFn: getSynthesisStatus });
  const jobQ = useQuery({
    queryKey: ["synthesis-job", synthesisJob],
    queryFn: () => getJob(synthesisJob),
    enabled: !!synthesisJob,
    refetchIntervalInBackground: true,
    refetchInterval: (query) => {
      const state = query.state.data?.status;
      return state === "completed" || state === "failed" ? false : 1000;
    },
  });

  const submitMut = useMutation({
    mutationFn: () =>
      submitSynthesis(project.project_id, {
        task_name: task,
        model_version: version || null,
        job_id: jobId || null,
        include_explanations: includeExplanations,
        instruction: instruction.trim() || null,
      }),
    onSuccess: (payload) => setSynthesisJob(payload.job_id),
  });

  const record = jobQ.data;
  const result =
    record?.status === "completed" ? (record.result as unknown as SynthesisResult) : null;
  const failed = record?.status === "failed";

  const noteMut = useMutation({
    mutationFn: () => {
      if (!result) throw new Error("No generated draft is available to save.");
      return addGeneratedProjectNote(project.project_id, {
        text: result.draft.text,
        model: result.provider.model ?? "unknown",
        citations: result.draft.citations,
        sources: result.sources,
        computed_facts: result.computed_facts,
        synthesis_job_id: synthesisJob,
      });
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["projects"] }),
  });

  const status = statusQ.data?.provider;
  const enabled = statusQ.data?.enabled ?? false;

  return (
    <Card
      title="Draft synthesis"
      sub="Ask a user-supplied model to draft a cited note from this project's computed artifacts. The draft is generated interpretation, not a computed result; nothing is saved until you add it as a note."
    >
      {statusQ.isError && <ErrorNote error={statusQ.error} />}
      {statusQ.data && !enabled && (
        <>
          <p className="muted">
            LLM synthesis is disabled because no provider is configured. Set these environment
            variables on the API process and restart it:
          </p>
          <ul className="muted">
            {Object.entries(statusQ.data.instructions).map(([name, hint]) => (
              <li key={name}>
                <code>{name}</code> — {hint}
              </li>
            ))}
          </ul>
          <p className="hint">
            No request has been sent anywhere. Aggregated profiles, scores, configuration, and
            explanation summaries would be sent to that endpoint; raw dataset rows never are.
          </p>
        </>
      )}
      {statusQ.data && enabled && (
        <>
          <p className="muted">
            Provider <code>{status?.model}</code>
            {status?.host ? ` at ${status.host}` : ""}; timeout {status?.timeout_seconds}s. Only
            aggregated computed facts are sent.
          </p>
          <div className="row">
            <div className="field">
              <label htmlFor="synthesis-task">Task</label>
              <select id="synthesis-task" value={task} onChange={(event) => setTask(event.target.value)}>
                {tasks.length === 0 && <option value="">No task artifacts in this project</option>}
                {tasks.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="synthesis-bundle">Model bundle</label>
              <select
                id="synthesis-bundle"
                value={version}
                onChange={(event) => setVersion(event.target.value)}
              >
                <option value="">None (dataset profile only)</option>
                {bundles.map((artifact) => (
                  <option key={artifact.artifact_id} value={artifact.artifact_id}>
                    {artifact.artifact_id.slice(0, 12)} · {artifact.task_name}
                  </option>
                ))}
              </select>
            </div>
            <div className="field">
              <label htmlFor="synthesis-job">Run job (optional)</label>
              <select id="synthesis-job" value={jobId} onChange={(event) => setJobId(event.target.value)}>
                <option value="">None</option>
                {jobs.map((artifact) => (
                  <option key={artifact.artifact_id} value={artifact.artifact_id}>
                    {artifact.artifact_id.slice(0, 12)} · {artifact.task_name}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className="row">
            <label className="field" style={{ flexDirection: "row", alignItems: "center", gap: 8 }}>
              <input
                type="checkbox"
                checked={includeExplanations}
                onChange={(event) => setIncludeExplanations(event.target.checked)}
              />
              Include explanation summaries
            </label>
          </div>
          <div className="field">
            <label htmlFor="synthesis-instruction">Extra instruction (optional)</label>
            <textarea
              id="synthesis-instruction"
              rows={2}
              value={instruction}
              onChange={(event) => setInstruction(event.target.value)}
              placeholder="What should the note emphasize?"
            />
          </div>
          <div className="row">
            <button
              className="btn primary"
              disabled={!task || submitMut.isPending}
              onClick={() => submitMut.mutate()}
            >
              {submitMut.isPending ? "Queueing…" : "Generate draft"}
            </button>
            {synthesisJob && record && record.status !== "completed" && record.status !== "failed" && (
              <span className="muted">Synthesizing… ({record.status})</span>
            )}
          </div>
          {submitMut.isError && <ErrorNote error={submitMut.error} />}
          {failed && (
            <ErrorNote
              error={new Error((record?.error ?? "Synthesis failed").split("\n").slice(-2).join(" "))}
            />
          )}
        </>
      )}

      {result && (
        <>
          <h3>Generated interpretation</h3>
          <p className="muted">
            Drafted by <code>{result.provider.model}</code> ({result.draft.word_count} words,{" "}
            {result.draft.citations.length} citation
            {result.draft.citations.length === 1 ? "" : "s"}). This text is model-generated and
            can be wrong; verify it against the computed facts below.
          </p>
          <blockquote style={{ margin: "8px 0", paddingLeft: 12, borderLeft: "3px solid var(--border)" }}>
            {result.draft.text}
          </blockquote>
          <p>
            {result.draft.citations.map((citation) => (
              <span key={citation} className="badge info" style={{ marginRight: 6 }}>
                {citation}
              </span>
            ))}
          </p>

          <h3>Computed facts sent</h3>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Source</th>
                  <th>Kind</th>
                  <th>Label</th>
                </tr>
              </thead>
              <tbody>
                {result.sources.map((source) => (
                  <tr key={source.id}>
                    <td>{source.id}</td>
                    <td>{source.kind}</td>
                    <td>{source.label}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <details>
            <summary className="muted">Full computed-fact payload</summary>
            <pre style={{ whiteSpace: "pre-wrap", fontSize: 12 }}>
              {JSON.stringify(result.computed_facts, null, 2)}
            </pre>
          </details>
          <ul className="muted">
            {result.interpretation.limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
          <div className="row">
            <button
              className="btn"
              disabled={noteMut.isPending || noteMut.isSuccess}
              onClick={() => noteMut.mutate()}
            >
              {noteMut.isSuccess ? "Added to notes" : "Add as note"}
            </button>
          </div>
          {noteMut.isError && <ErrorNote error={noteMut.error} />}
        </>
      )}
    </Card>
  );
}
