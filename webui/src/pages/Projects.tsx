import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { CheckCircle2, Circle } from "lucide-react";

import {
  addProjectNote,
  addProjectRecord,
  createProject,
  getDatasetMetadata,
  getProjectSummary,
  listBundles,
  listJobs,
  listProjects,
  updateProject,
} from "../api/client";
import type { Project, ProjectArtifact } from "../api/client";
import { LineageRail } from "../components/LineageRail";
import { Badge, Card, EmptyState, ErrorNote, Loading, StatusBadge } from "../components/ui";
import { useTask } from "../lib/task-context";

const STAGE_LINKS: Record<string, { to: string; cta: string }> = {
  data: { to: "/datasets", cta: "Register data" },
  model: { to: "/experiments", cta: "Run an experiment" },
  release: { to: "/bundles", cta: "Review models" },
};

const keyOf = (a: ProjectArtifact) => `${a.kind}:${a.task_name}:${a.artifact_id}`;

function ProjectDetail({ project }: { project: Project }) {
  const { task } = useTask();
  const qc = useQueryClient();
  const [note, setNote] = useState("");
  const [recordKind, setRecordKind] = useState<"hypothesis" | "analysis" | "finding" | "conclusion">("hypothesis");
  const [recordTitle, setRecordTitle] = useState("");
  const [recordText, setRecordText] = useState("");
  const [evidenceKey, setEvidenceKey] = useState("");
  const [evidenceRelation, setEvidenceRelation] = useState<"supports" | "contradicts">("supports");
  const [recordEvidence, setRecordEvidence] = useState<
    Array<{ key: string; relation: "supports" | "contradicts" }>
  >([]);

  const summaryQ = useQuery({
    queryKey: ["project-summary", project.project_id, project.updated_at],
    queryFn: () => getProjectSummary(project.project_id),
  });

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["projects"] });
    qc.invalidateQueries({ queryKey: ["project-summary", project.project_id] });
  };

  const noteMut = useMutation({
    mutationFn: () => addProjectNote(project.project_id, note),
    onSuccess: () => {
      setNote("");
      refresh();
    },
  });

  const recordMut = useMutation({
    mutationFn: () => {
      return addProjectRecord(project.project_id, {
        kind: recordKind,
        title: recordTitle,
        text: recordText,
        evidence: recordEvidence.flatMap(({ key, relation }) => {
          const artifact = project.artifacts.find((candidate) => keyOf(candidate) === key);
          return artifact ? [{ relation, artifact }] : [];
        }),
      });
    },
    onSuccess: () => {
      setRecordTitle("");
      setRecordText("");
      setEvidenceKey("");
      setRecordEvidence([]);
      refresh();
    },
  });

  const addEvidence = () => {
    if (!evidenceKey || recordEvidence.some((item) => item.key === evidenceKey)) return;
    setRecordEvidence((current) => [...current, { key: evidenceKey, relation: evidenceRelation }]);
    setEvidenceKey("");
  };

  // Offer every dataset partition, bundle and completed job of the current task that is not linked yet.
  const candidatesQ = useQuery({
    queryKey: ["project-candidates", task],
    enabled: !!task,
    queryFn: async () => {
      const found: ProjectArtifact[] = [];
      const [meta, bundles, jobs] = await Promise.allSettled([
        getDatasetMetadata(task),
        listBundles(task),
        listJobs(task),
      ]);
      if (meta.status === "fulfilled") {
        for (const p of Object.keys(meta.value.partitions ?? {})) {
          found.push({ kind: "dataset_partition", task_name: task, artifact_id: p });
        }
      }
      if (bundles.status === "fulfilled") {
        for (const b of bundles.value.bundles) found.push({ kind: "model_bundle", task_name: task, artifact_id: b });
      }
      if (jobs.status === "fulfilled") {
        for (const j of jobs.value.jobs) {
          if (j.status === "completed") found.push({ kind: "job", task_name: task, artifact_id: j.job_id, label: j.kind });
        }
      }
      return found;
    },
  });
  const linked = new Set(project.artifacts.map(keyOf));
  const available = (candidatesQ.data ?? []).filter((a) => !linked.has(keyOf(a)));

  const linkMut = useMutation({
    mutationFn: (items: ProjectArtifact[]) =>
      updateProject(project.project_id, {
        name: project.name,
        description: project.description,
        research_questions: project.research_questions,
        artifacts: [...project.artifacts, ...items],
      }),
    onSuccess: refresh,
  });

  const stages = summaryQ.data?.stages ?? [];
  const next = stages.find((s) => !s.done);

  return (
    <div className="stack">
      <Card title="Progress" sub={next ? `Next step: ${next.label.toLowerCase()}` : "All stages covered"}>
        {summaryQ.isLoading && <Loading />}
        {summaryQ.isError && <ErrorNote error={summaryQ.error} />}
        <div className="stack">
          {stages.map((s) => (
            <div key={s.stage} className="row fixed" style={{ alignItems: "center" }}>
              {s.done ? <CheckCircle2 size={16} color="#10b981" /> : <Circle size={16} />}
              <span>{s.label}</span>
              {!s.done && STAGE_LINKS[s.stage] && (
                <Link className="btn" to={STAGE_LINKS[s.stage].to}>{STAGE_LINKS[s.stage].cta}</Link>
              )}
            </div>
          ))}
        </div>
      </Card>

      <Card
        title="Linked artifacts"
        sub="Artifact payloads stay in their stores; project links retain metadata snapshots."
        actions={
          available.length > 0 ? (
            <button className="btn" disabled={linkMut.isPending} onClick={() => linkMut.mutate(available)}>
              Link all from “{task}” ({available.length})
            </button>
          ) : undefined
        }
      >
        {linkMut.isError && <ErrorNote error={linkMut.error} />}
        {!task && <p className="muted">Pick a task in the top bar to link its datasets, models and jobs.</p>}
        {(summaryQ.data?.artifacts ?? []).length === 0 ? (
          <EmptyState title="Nothing linked yet" />
        ) : (
          <ul>
            {summaryQ.data?.artifacts.map((a) => (
              <li key={keyOf(a)}>
                <Badge tone="info">{a.kind.replace("_", " ")}</Badge> <span className="mono">{a.task_name}/{a.artifact_id}</span>{" "}
                {a.status && <StatusBadge status={a.status} />}
                {!a.exists && <Badge tone="err">missing</Badge>}
                {a.snapshot_status && (
                  <Badge
                    tone={
                      a.snapshot_status === "current"
                        ? "ok"
                        : a.snapshot_status === "unverified"
                          ? "warn"
                          : "err"
                    }
                  >
                    {a.snapshot_status === "current" ? "snapshot current" : `snapshot ${a.snapshot_status}`}
                  </Badge>
                )}
                {a.kind === "model_bundle" && a.exists && <LineageRail task={a.task_name} modelVersion={a.artifact_id} />}
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card
        title="Research records"
        sub="Append hypotheses, analysis notes, findings, and conclusions linked to up to 20 project artifacts."
      >
        <div className="row">
          <div className="field">
            <label>Record type</label>
            <select value={recordKind} onChange={(e) => setRecordKind(e.target.value as typeof recordKind)}>
              <option value="hypothesis">Hypothesis</option>
              <option value="analysis">Analysis</option>
              <option value="finding">Finding</option>
              <option value="conclusion">Conclusion</option>
            </select>
          </div>
          <div className="field">
            <label>Title</label>
            <input value={recordTitle} onChange={(e) => setRecordTitle(e.target.value)} />
          </div>
        </div>
        <div className="field">
          <label>Record</label>
          <textarea rows={2} value={recordText} onChange={(e) => setRecordText(e.target.value)} />
        </div>
        <div className="row">
          <div className="field">
            <label>Evidence artifact</label>
            <select value={evidenceKey} onChange={(e) => setEvidenceKey(e.target.value)}>
              <option value="">Select an artifact</option>
              {project.artifacts.map((artifact) => (
                <option
                  key={keyOf(artifact)}
                  value={keyOf(artifact)}
                  disabled={recordEvidence.some((item) => item.key === keyOf(artifact))}
                >
                  {artifact.kind.replace("_", " ")} — {artifact.task_name}/{artifact.artifact_id}
                </option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>Evidence relation</label>
            <select
              value={evidenceRelation}
              onChange={(e) => setEvidenceRelation(e.target.value as typeof evidenceRelation)}
            >
              <option value="supports">Supports</option>
              <option value="contradicts">Contradicts</option>
            </select>
          </div>
          <button className="btn" type="button" disabled={!evidenceKey} onClick={addEvidence}>
            Add evidence
          </button>
        </div>
        {recordEvidence.length > 0 && (
          <ul>
            {recordEvidence.map(({ key, relation }) => {
              const artifact = project.artifacts.find((candidate) => keyOf(candidate) === key);
              if (!artifact) return null;
              return (
                <li key={key}>
                  <Badge tone={relation === "supports" ? "ok" : "warn"}>{relation}</Badge>{" "}
                  {artifact.kind.replace("_", " ")}{" "}
                  <span className="mono">{artifact.task_name}/{artifact.artifact_id}</span>{" "}
                  <button
                    className="btn"
                    type="button"
                    aria-label={`Remove evidence ${artifact.task_name}/${artifact.artifact_id}`}
                    onClick={() => setRecordEvidence((current) => current.filter((item) => item.key !== key))}
                  >
                    Remove
                  </button>
                </li>
              );
            })}
          </ul>
        )}
        <div className="row">
          <button
            className="btn primary"
            disabled={
              !recordTitle.trim()
              || !recordText.trim()
              || recordMut.isPending
              || ((recordKind === "finding" || recordKind === "conclusion") && recordEvidence.length === 0)
            }
            onClick={() => recordMut.mutate()}
          >
            Add record
          </button>
        </div>
        {recordMut.isError && <ErrorNote error={recordMut.error} />}
        {project.records.length === 0 ? (
          <EmptyState title="No research records yet" />
        ) : (
          <ul>
            {[...project.records].reverse().map((record) => (
              <li key={record.record_id}>
                <Badge tone="info">{record.kind}</Badge> <strong>{record.title}</strong>
                <p>{record.text}</p>
                {record.evidence.map((evidence) => (
                  <div key={`${record.record_id}:${keyOf(evidence.artifact)}:${evidence.relation}`} className="muted">
                    {evidence.relation} — {evidence.artifact.kind.replace("_", " ")}{" "}
                    <span className="mono">
                      {evidence.artifact.task_name}/{evidence.artifact.artifact_id}
                    </span>
                  </div>
                ))}
              </li>
            ))}
          </ul>
        )}
      </Card>

      <Card title="Notes">
        <div className="row">
          <div className="field">
            <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)} placeholder="What did you learn?" />
          </div>
          <button className="btn primary" disabled={!note.trim() || noteMut.isPending} onClick={() => noteMut.mutate()}>
            Add note
          </button>
        </div>
        {noteMut.isError && <ErrorNote error={noteMut.error} />}
        <ul>
          {[...project.notes].reverse().map((n) => (
            <li key={n.note_id}>
              <span className="muted">{new Date(n.created_at).toLocaleString()}</span> — {n.text}
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}

export default function ProjectsPage() {
  const qc = useQueryClient();
  const [selected, setSelected] = useState("");
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [questions, setQuestions] = useState("");

  const projectsQ = useQuery({ queryKey: ["projects"], queryFn: listProjects });
  const projects = projectsQ.data?.projects ?? [];
  const current = projects.find((p) => p.project_id === selected) ?? projects[0];

  const createMut = useMutation({
    mutationFn: () =>
      createProject({
        name,
        description,
        research_questions: questions.split("\n").map((q) => q.trim()).filter(Boolean),
      }),
    onSuccess: (p) => {
      setName("");
      setDescription("");
      setQuestions("");
      setSelected(p.project_id);
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Projects</h1>
        <span className="desc">Group the datasets, models and runs of one investigation and track how far along it is</span>
      </div>

      <Card title="New project">
        <div className="row">
          <div className="field">
            <label>Name</label>
            <input value={name} onChange={(e) => setName(e.target.value)} />
          </div>
          <div className="field">
            <label>Description</label>
            <input value={description} onChange={(e) => setDescription(e.target.value)} />
          </div>
        </div>
        <div className="field">
          <label>Research questions (one per line)</label>
          <textarea rows={2} value={questions} onChange={(e) => setQuestions(e.target.value)} />
        </div>
        {createMut.isError && <ErrorNote error={createMut.error} />}
        <button className="btn primary" disabled={!name.trim() || createMut.isPending} onClick={() => createMut.mutate()}>
          Create project
        </button>
      </Card>

      {projectsQ.isLoading && <Loading />}
      {projectsQ.isError && <ErrorNote error={projectsQ.error} />}
      {projectsQ.data && projects.length === 0 && (
        <EmptyState title="No projects yet" hint="Create one above to start tracking an investigation." />
      )}

      {current && (
        <>
          <Card title="Open project">
            <div className="row">
              <div className="field">
                <select value={current.project_id} onChange={(e) => setSelected(e.target.value)}>
                  {projects.map((p) => (
                    <option key={p.project_id} value={p.project_id}>{p.name}</option>
                  ))}
                </select>
              </div>
            </div>
            {current.description && <p>{current.description}</p>}
            {current.research_questions.length > 0 && (
              <ul>
                {current.research_questions.map((q) => (
                  <li key={q}>{q}</li>
                ))}
              </ul>
            )}
          </Card>
          <ProjectDetail key={current.project_id} project={current} />
        </>
      )}
    </div>
  );
}
