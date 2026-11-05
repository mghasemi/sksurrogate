import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { CheckCircle2, Circle } from "lucide-react";

import {
  addProjectNote,
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
        sub="References only — nothing is copied."
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
                {a.kind === "model_bundle" && a.exists && <LineageRail task={a.task_name} modelVersion={a.artifact_id} />}
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
