import { lazy, Suspense, useEffect, useState } from "react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, NavLink, Route, Routes, useLocation } from "react-router-dom";
import {
  Activity,
  Boxes,
  ClipboardCheck,
  Cpu,
  Database,
  FlaskConical,
  Gauge,
  LayoutDashboard,
  LineChart,
  ListChecks,
  Menu,
  RefreshCw,
  Settings2,
  Sigma,
} from "lucide-react";

import { Loading } from "./components/ui";
import { TaskProvider, useTask } from "./lib/task-context";
import { useHealth } from "./lib/hooks";

// Route-level code splitting: each page is its own chunk so the initial load
// only pays for the shell + the first route (see docs/HANDOFF.md next steps).
const DashboardPage = lazy(() => import("./pages/Dashboard"));
const DatasetsPage = lazy(() => import("./pages/Datasets"));
const FeatureAnalysisPage = lazy(() => import("./pages/FeatureAnalysis"));
const ExperimentsPage = lazy(() => import("./pages/Experiments"));
const EvaluationPage = lazy(() => import("./pages/Evaluation"));
const BundlesPage = lazy(() => import("./pages/Bundles"));
const QualityGatesPage = lazy(() => import("./pages/QualityGates"));
const RegistryPage = lazy(() => import("./pages/Registry"));
const InferencePage = lazy(() => import("./pages/Inference"));
const MonitoringPage = lazy(() => import("./pages/Monitoring"));
const RetrainingPage = lazy(() => import("./pages/Retraining"));
const JobsPage = lazy(() => import("./pages/Jobs"));
const SettingsPage = lazy(() => import("./pages/Settings"));

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
});

interface NavEntry {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
}

const NAV_GROUPS: Array<{ label: string; items: NavEntry[] }> = [
  {
    label: "Overview",
    items: [{ to: "/", label: "Dashboard", icon: LayoutDashboard }],
  },
  {
    label: "Pipeline",
    items: [
      { to: "/datasets", label: "Datasets", icon: Database },
      { to: "/features", label: "Feature Analysis", icon: Sigma },
      { to: "/experiments", label: "Experiments", icon: FlaskConical },
      { to: "/evaluation", label: "Evaluation", icon: LineChart },
      { to: "/bundles", label: "Model Bundles", icon: Boxes },
    ],
  },
  {
    label: "Release",
    items: [
      { to: "/quality-gates", label: "Quality Gates", icon: ClipboardCheck },
      { to: "/registry", label: "Registry & Deployment", icon: Gauge },
    ],
  },
  {
    label: "Serve",
    items: [
      { to: "/inference", label: "Inference", icon: Cpu },
      { to: "/monitoring", label: "Monitoring", icon: Activity },
      { to: "/retraining", label: "Retraining", icon: RefreshCw },
    ],
  },
  {
    label: "System",
    items: [
      { to: "/jobs", label: "Jobs", icon: ListChecks },
      { to: "/settings", label: "Settings / Audit", icon: Settings2 },
    ],
  },
];

const TITLES: Record<string, string> = {
  "/": "Dashboard",
  "/datasets": "Datasets",
  "/features": "Feature Analysis",
  "/experiments": "Experiments",
  "/evaluation": "Evaluation",
  "/bundles": "Model Bundles",
  "/quality-gates": "Quality Gates",
  "/registry": "Registry & Deployment",
  "/inference": "Inference",
  "/monitoring": "Monitoring",
  "/retraining": "Retraining",
  "/jobs": "Jobs",
  "/settings": "Settings / Audit",
};

function Sidebar({ onNavigate }: { onNavigate: () => void }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <div className="logo">SK</div>
        <div>
          <div className="name">SKSurrogate</div>
          <div className="sub">Control Plane</div>
        </div>
      </div>
      {NAV_GROUPS.map((group) => (
        <nav key={group.label}>
          <div className="nav-group-label">{group.label}</div>
          {group.items.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              onClick={onNavigate}
              className={({ isActive }) => `nav-item${isActive ? " active" : ""}`}
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>
      ))}
    </aside>
  );
}

function Topbar({ onMenu }: { onMenu: () => void }) {
  const { task, setTask } = useTask();
  const healthOk = useHealth();
  const location = useLocation();
  const title = TITLES[location.pathname] ?? "SKSurrogate";

  return (
    <header className="topbar">
      <button type="button" className="btn menu-btn" onClick={onMenu} aria-label="Toggle navigation">
        <Menu size={16} />
      </button>
      <span className="title">{title}</span>
      <span className="spacer" />
      <label className="task-picker">
        Task
        <input
          value={task}
          onChange={(e) => setTask(e.target.value.trim())}
          placeholder="task_name (e.g. galaxy3)"
          spellCheck={false}
        />
      </label>
      <span className={`health-dot${healthOk ? " ok" : ""}`}>
        <span className="dot" />
        {healthOk === null ? "checking…" : healthOk ? "API online" : "API offline"}
      </span>
    </header>
  );
}

function Shell() {
  const [navOpen, setNavOpen] = useState(false);
  const location = useLocation();

  // Close the off-canvas sidebar whenever the route changes.
  useEffect(() => {
    setNavOpen(false);
  }, [location.pathname]);

  return (
    <div className={`shell${navOpen ? " nav-open" : ""}`}>
      {navOpen && <div className="scrim" onClick={() => setNavOpen(false)} />}
      <Sidebar onNavigate={() => setNavOpen(false)} />
      <div className="main">
        <Topbar onMenu={() => setNavOpen((open) => !open)} />
        <main className="content">
          <Suspense fallback={<Loading label="Loading page…" />}>
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/datasets" element={<DatasetsPage />} />
            <Route path="/features" element={<FeatureAnalysisPage />} />
            <Route path="/experiments" element={<ExperimentsPage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
            <Route path="/bundles" element={<BundlesPage />} />
            <Route path="/quality-gates" element={<QualityGatesPage />} />
            <Route path="/registry" element={<RegistryPage />} />
            <Route path="/inference" element={<InferencePage />} />
            <Route path="/monitoring" element={<MonitoringPage />} />
            <Route path="/retraining" element={<RetrainingPage />} />
            <Route path="/jobs" element={<JobsPage />} />
            <Route path="/settings" element={<SettingsPage />} />
          </Routes>
          </Suspense>
        </main>
      </div>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <TaskProvider>
        <BrowserRouter>
          <Shell />
        </BrowserRouter>
      </TaskProvider>
    </QueryClientProvider>
  );
}
