import { useCallback, useEffect, useState } from "react";
import { API_BASE_URL, healthApi } from "./api/client.js";

const initialChecks = {
  backend: { state: "loading", data: null, error: null },
  models: { state: "loading", data: null, error: null },
  recognition: { state: "loading", data: null, error: null },
};

function StatusPill({ tone, children }) {
  return <span className={`status-pill status-pill--${tone}`}>{children}</span>;
}

function DashboardCard({ title, description, state, children }) {
  return (
    <section className="dashboard-card" aria-live="polite">
      <div className="card-heading">
        <div>
          <h2>{title}</h2>
          <p>{description}</p>
        </div>
        {state === "loading" ? <span className="loading-mark" aria-label="Loading" /> : null}
      </div>
      <div className="card-content">{children}</div>
    </section>
  );
}

function CheckError({ message }) {
  return (
    <div className="check-message check-message--error" role="alert">
      <span className="status-dot" />
      <span>{message || "The status could not be loaded."}</span>
    </div>
  );
}

export default function App() {
  const [checks, setChecks] = useState(initialChecks);
  const [refreshing, setRefreshing] = useState(false);

  const refresh = useCallback(async () => {
    setRefreshing(true);
    setChecks(initialChecks);
    const [backend, models, recognition] = await Promise.allSettled([
      healthApi.liveness(),
      healthApi.models(),
      healthApi.recognition(),
    ]);
    const result = {};
    for (const [key, item] of Object.entries({ backend, models, recognition })) {
      result[key] = item.status === "fulfilled"
        ? { state: "loaded", data: item.value.data, httpStatus: item.value.status, error: null }
        : { state: "error", data: null, error: item.reason?.message || "Request failed" };
    }
    setChecks(result);
    setRefreshing(false);
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const backend = checks.backend;
  const models = checks.models;
  const recognition = checks.recognition;

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#home" aria-label="Offline Face Recognition home">
          <span className="brand-mark" aria-hidden="true">OF</span>
          <span>Offline Face Recognition</span>
        </a>
        <span className="local-tag"><span className="local-dot" /> Local runtime</span>
      </header>

      <section className="intro" id="home">
        <div className="eyebrow">SYSTEM STATUS</div>
        <h1>Local service dashboard</h1>
        <p>Check backend connectivity, local model availability, and recognition readiness.</p>
        <div className="intro-actions">
          <button className="refresh-button" onClick={refresh} disabled={refreshing}>
            <span aria-hidden="true">↻</span> {refreshing ? "Checking…" : "Refresh status"}
          </button>
          <span className="api-address">API <code>{API_BASE_URL}</code></span>
        </div>
      </section>

      <section className="status-grid" aria-label="Backend health checks">
        <DashboardCard title="Backend connection" description="Liveness check" state={backend.state}>
          {backend.state === "loading" ? <p className="muted">Checking local backend…</p> : null}
          {backend.state === "error" ? <CheckError message={backend.error} /> : null}
          {backend.state === "loaded" && backend.data.status === "alive" ? (
            <div className="check-message"><span className="status-dot status-dot--green" /><strong>Connected</strong><StatusPill tone="green">Alive</StatusPill></div>
          ) : null}
          {backend.state === "loaded" && backend.data.status !== "alive" ? <CheckError message="Backend did not report an alive status." /> : null}
        </DashboardCard>

        <DashboardCard title="Local models" description="YuNet detection and SFace embeddings" state={models.state}>
          {models.state === "loading" ? <p className="muted">Checking model status…</p> : null}
          {models.state === "error" ? <CheckError message={models.error} /> : null}
          {models.state === "loaded" && models.data.models_ready ? (
            <div className="check-message"><span className="status-dot status-dot--green" /><strong>Available</strong><StatusPill tone="green">Ready</StatusPill></div>
          ) : null}
          {models.state === "loaded" && !models.data.models_ready ? (
            <div className="stacked-status">
              <div className="check-message"><span className="status-dot status-dot--amber" /><strong>Unavailable</strong><StatusPill tone="amber">Not ready</StatusPill></div>
              <p className="detail-text">YuNet: {models.data.yunet_ready ? "ready" : "unavailable"} · SFace: {models.data.sface_ready ? "ready" : "unavailable"}</p>
              {models.data.error_code ? <p className="detail-text">Backend code: <code>{models.data.error_code}</code></p> : null}
            </div>
          ) : null}
        </DashboardCard>

        <DashboardCard title="Recognition" description="Matching policy readiness" state={recognition.state}>
          {recognition.state === "loading" ? <p className="muted">Checking recognition readiness…</p> : null}
          {recognition.state === "error" ? <CheckError message={recognition.error} /> : null}
          {recognition.state === "loaded" && recognition.data.recognition_ready ? (
            <div className="check-message"><span className="status-dot status-dot--green" /><strong>Ready</strong><StatusPill tone="green">Available</StatusPill></div>
          ) : null}
          {recognition.state === "loaded" && !recognition.data.recognition_ready ? (
            <div className="stacked-status">
              <div className="check-message"><span className="status-dot status-dot--amber" /><strong>Unavailable</strong><StatusPill tone="amber">Not ready</StatusPill></div>
              <p className="detail-text">Recognition remains disabled until matching thresholds are validated and configured.</p>
              {recognition.data.reason ? <p className="detail-text">Reason: <code>{recognition.data.reason}</code></p> : null}
            </div>
          ) : null}
        </DashboardCard>
      </section>

      <footer className="page-footer">
        <span>Runs locally on this device</span>
        <span>Recognition is gated until its policy is validated.</span>
      </footer>
    </main>
  );
}
