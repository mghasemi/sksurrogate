import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { getMonitoringAlerts, listJobs, getEffectiveApiBase } from "../api/client";
import type { JobRecord, MonitoringAlert } from "../api/client";
import { Badge, Card, Stat, StatusBadge, Table, ErrorNote, Loading, fmtNum } from "../components/ui";

const STAGES: Array<{ to: string; title: string; desc: string }> = [
  { to: "/projects", title: "Projects", desc: "Group an investigation and track its progress" },
  { to: "/datasets", title: "1. Datasets", desc: "Register data, inspect schema & fingerprint" },
  { to: "/features", title: "Feature Analysis", desc: "Sobol / Morris sensitivity, correlation pruning" },
  { to: "/design", title: "Experiment Design", desc: "Plan the next experiments, suggest the next point" },
  { to: "/experiments", title: "2. Experiments", desc: "Quick start presets or full AML/EOA search" },
  { to: "/explain", title: "Explain", desc: "Feature importance, partial dependence, prediction intervals" },
  { to: "/bundles", title: "Model Bundles", desc: "Train baselines, inspect schema & metrics" },
  { to: "/quality-gates", title: "3. Quality Gates", desc: "CI checks before promotion" },
  { to: "/registry", title: "Registry & Deployment", desc: "Lifecycle, approvals, rollback" },
  { to: "/inference", title: "4. Inference", desc: "Predict console & batch runs" },
  { to: "/monitoring", title: "Monitoring", desc: "Drift, fairness, latency" },
  { to: "/retraining", title: "Retraining", desc: "Triggered refresh with promotion checks" },
];

const FIRST_STEPS: Array<{ to: string; text: string }> = [
  { to: "/datasets", text: "Pick a task name in the top bar, then upload a CSV on Datasets." },
  { to: "/experiments", text: "Open Experiments and press “Find best model” (Quick start)." },
  { to: "/explain", text: "Check what the model learned on Explain." },
  { to: "/quality-gates", text: "Run a quality gate, then promote the model from Registry." },
];

export default function DashboardPage() {
  const jobsQ = useQuery({ queryKey: ["jobs"], queryFn: () => listJobs(), refetchInterval: 5000 });
  const alertsQ = useQuery({
    queryKey: ["monitoring-alerts", "all"],
    queryFn: () => getMonitoringAlerts(undefined, 20),
    refetchInterval: 15000,
  });

  return (
    <div className="stack">
      <div className="page-head">
        <h1>Dashboard</h1>
        <span className="desc">Cross-task overview of the control plane at {getEffectiveApiBase()}</span>
      </div>

      <div className="grid cols-4">
        <Stat label="API" value={jobsQ.isError ? "offline" : jobsQ.isLoading ? "…" : "online"} sub={getEffectiveApiBase()} />
        <Stat label="Jobs (all tasks)" value={fmtNum(jobsQ.data?.jobs.length)} sub={`${jobsQ.data?.jobs.filter((j) => j.status === "running").length ?? 0} running`} />
        <Stat label="Completed" value={fmtNum(jobsQ.data?.jobs.filter((j) => j.status === "completed").length)} />
        <Stat label="Failed" value={fmtNum(jobsQ.data?.jobs.filter((j) => j.status === "failed").length)} />
      </div>

      {jobsQ.data && jobsQ.data.jobs.length === 0 && (
        <Card title="Getting started" sub="Four steps from a CSV to a served model.">
          <ol>
            {FIRST_STEPS.map((step) => (
              <li key={step.to}>
                <Link to={step.to}>{step.text}</Link>
              </li>
            ))}
          </ol>
        </Card>
      )}

      <Card title="Recent monitoring alerts" sub="Latest drift findings and inference failures across tasks.">
        {alertsQ.isLoading && <Loading />}
        {alertsQ.isError && <ErrorNote error={alertsQ.error} />}
        {alertsQ.data && alertsQ.data.alerts.length === 0 && (
          <p className="muted">No monitoring alerts have been recorded.</p>
        )}
        {alertsQ.data && alertsQ.data.alerts.length > 0 && (
          <Table<MonitoringAlert>
            columns={["Severity", "Task", "Version", "Type", "Detail", "When", ""]}
            rows={alertsQ.data.alerts}
            keyOf={(alert, index) => `${alert.timestamp}-${alert.task}-${alert.version ?? "none"}-${index}`}
            render={(alert) => [
              <td key="severity">
                <Badge tone={alert.severity === "error" ? "err" : "warn"}>{alert.severity}</Badge>
              </td>,
              <td key="task">{alert.task}</td>,
              <td key="version" className="mono">{alert.version ?? "—"}</td>,
              <td key="type"><span className="badge info">{alert.type}</span></td>,
              <td key="detail">{alert.detail}</td>,
              <td key="time" className="mono">{new Date(alert.timestamp).toLocaleString()}</td>,
              <td key="link">
                <Link
                  className="btn"
                  to={`/monitoring?task=${encodeURIComponent(alert.task)}${alert.version ? `&version=${encodeURIComponent(alert.version)}` : ""}`}
                >
                  View monitoring
                </Link>
              </td>,
            ]}
          />
        )}
      </Card>

      <Card title="Pipeline stages">
        <div className="grid cols-4">
          {STAGES.map((s) => (
            <Link key={s.to} to={s.to} style={{ textDecoration: "none", color: "inherit" }}>
              <div className="card" style={{ height: "100%" }}>
                <h3>{s.title}</h3>
                <div className="muted">{s.desc}</div>
              </div>
            </Link>
          ))}
        </div>
      </Card>

      <Card title="Recent jobs" sub="All tasks, refreshed every 5 s">
        {jobsQ.isLoading && <Loading />}
        {jobsQ.isError && <ErrorNote error={jobsQ.error} />}
        {jobsQ.data && (
          <Table<JobRecord>
            columns={["Task", "Kind", "Status", "Created", "Updated"]}
            rows={[...jobsQ.data.jobs].sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, 15)}
            keyOf={(j) => j.job_id}
            render={(j) => [
              <td key="t">{j.task_name}</td>,
              <td key="k"><span className="badge info">{j.kind}</span></td>,
              <td key="s"><StatusBadge status={j.status} /></td>,
              <td key="c" className="mono">{j.created_at}</td>,
              <td key="u" className="mono">{j.updated_at}</td>,
            ]}
          />
        )}
      </Card>
    </div>
  );
}
