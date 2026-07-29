import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import { ApprovalModal } from "./components/ApprovalModal";
import { HowItWorks } from "./components/HowItWorks";
import { Launcher } from "./components/Launcher";
import { RunHistory } from "./components/RunHistory";
import { RunView } from "./components/RunView";
import type { RunRow, Scenario, Topology } from "./types";
import { useRunStream } from "./useRunStream";

export function App() {
  const [scenarios, setScenarios] = useState<Scenario[]>([]);
  const [runs, setRuns] = useState<RunRow[]>([]);
  const [activeRun, setActiveRun] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const state = useRunStream(activeRun);

  const refreshRuns = useCallback(() => {
    api.runs().then(setRuns).catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    api.scenarios().then(setScenarios).catch((e) => setError(String(e)));
    refreshRuns();
  }, [refreshRuns]);

  // When a run finishes, refresh the history so it appears with its final status.
  useEffect(() => {
    if (state && ["succeeded", "failed", "budget_exceeded"].includes(state.status)) {
      refreshRuns();
    }
  }, [state?.status, refreshRuns]);

  const launch = useCallback(
    async (incidentId: string, topology: Topology, mock: boolean) => {
      setError(null);
      try {
        const { run_id } = await api.startRun(incidentId, topology, mock);
        setActiveRun(run_id);
        refreshRuns();
      } catch (e) {
        setError(String(e));
      }
    },
    [refreshRuns],
  );

  const resolve = useCallback(
    async (approvalId: string, decision: "approved" | "denied", reason: string) => {
      try {
        await api.resolveApproval(approvalId, decision, reason);
      } catch (e) {
        setError(String(e));
      }
    },
    [],
  );

  const pendingApproval = state?.approvals[0] ?? null;

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <h1>
            <span className="shield">◈</span> Aegis
          </h1>
          <p>Agentic incident triage &amp; Root Cause Analysis (RCA)</p>
          <button className="brand-link" onClick={() => setActiveRun(null)}>
            How it works
          </button>
        </div>
        <Launcher scenarios={scenarios} onLaunch={launch} />
        <RunHistory runs={runs} activeRun={activeRun} onSelect={setActiveRun} />
      </aside>

      <main className="main">
        {error && <div className="err-banner">{error}</div>}
        {state ? <RunView state={state} /> : <HowItWorks />}
      </main>

      {pendingApproval && (
        <ApprovalModal approval={pendingApproval} onResolve={resolve} />
      )}
    </div>
  );
}
