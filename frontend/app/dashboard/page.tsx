"use client";

import Link from "next/link";

export default function DashboardPage() {
  return (
    <main className="dashboard-page">
      <header className="dashboard-hero">
        <div>
          <span className="dashboard-kicker">Aegis Command Center</span>
          <h1>Welcome to your identity defense hub</h1>
          <p>
            Monitor fraud alerts, understand anomaly trends, and activate response playbooks in
            one AWS-native console. Connect this view to your FastAPI backend to surface live data.
          </p>
        </div>
        <Link href="/" className="hero-cta hero-cta-ghost">
          Back to landing
        </Link>
      </header>

      <section className="dashboard-panels">
        <article className="dashboard-panel">
          <header>
            <h2>Alerts Overview</h2>
            <span className="badge badge-high">3 critical</span>
          </header>
          <p>
            Integrate your real-time alert feed here to visualize incident severity, affected
            accounts, and recommended mitigations.
          </p>
          <div className="panel-placeholder" />
        </article>

        <article className="dashboard-panel">
          <header>
            <h2>Identity Risk Graph</h2>
            <span className="badge badge-medium">Live</span>
          </header>
          <p>
            Plot login velocity spikes, session anomalies, and GPS deviations once your analytics
            pipeline is connected.
          </p>
          <div className="panel-placeholder" />
        </article>

        <article className="dashboard-panel">
          <header>
            <h2>Privacy Health</h2>
            <span className="badge badge-low">Stable</span>
          </header>
          <p>
            Surface hashed breach lookups, dark web hits, and remediation guidance sourced from
            your privacy-preserving scans.
          </p>
          <div className="panel-placeholder" />
        </article>
      </section>
    </main>
  );
}
