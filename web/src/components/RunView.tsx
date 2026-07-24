import type { AgentLane, RunState, ToolEntry } from "../useRunStream";

const ROLE_ORDER = { coordinator: 0, investigator: 1, verifier: 2 } as const;

export function RunView({ state }: { state: RunState }) {
  const running = state.status === "running" || state.status === "connecting";

  // Sort lanes so the coordinator sits first, then investigators, then verifiers, roughly in
  // spawn order — the shape a reader expects to scan.
  const lanes = [...state.agents].sort(
    (a, b) => ROLE_ORDER[a.role] - ROLE_ORDER[b.role],
  );

  return (
    <>
      <div className="run-header">
        <div className="alert">{state.alert || "…"}</div>
        <div className="meters">
          {state.budgetWarning && (
            <div className="meter" title="Budget warning">
              <div className="v" style={{ color: "var(--warn)" }}>
                {Math.round(state.budgetWarning.fraction * 100)}%
              </div>
              <div className="k">budget</div>
            </div>
          )}
          <div className="meter">
            <div className="v">${state.costUsd.toFixed(4)}</div>
            <div className="k">cost</div>
          </div>
          <div className="meter">
            <div className="v">
              <span className={`dot ${state.status}`} /> {state.status}
            </div>
            <div className="k">
              {state.durationS != null ? `${state.durationS.toFixed(1)}s` : `${lanes.length} agents`}
            </div>
          </div>
        </div>
      </div>

      <div className="lanes">
        {lanes.map((lane) => (
          <Lane key={lane.agentId} lane={lane} runActive={running} />
        ))}
      </div>

      {state.rca && <RcaCard rca={state.rca} />}
    </>
  );
}

function Lane({ lane, runActive }: { lane: AgentLane; runActive: boolean }) {
  const active = runActive && !lane.finished;
  const sub = lane.role !== "coordinator";
  return (
    <div className={`lane ${sub ? "sub" : ""}`}>
      <div className="lane-head">
        <div className={`lane-role role-${lane.role}`} />
        <span className="lane-label">{lane.label}</span>
        <div className="lane-meta">
          {lane.turns != null && <span>{lane.turns} turns</span>}
          {lane.costUsd != null && <span>${lane.costUsd.toFixed(4)}</span>}
          {lane.error && <span style={{ color: "var(--err)" }}>error</span>}
        </div>
      </div>
      <div className="lane-body">
        {lane.thinking && <div className="thinking">{lane.thinking}</div>}
        <div className={`agent-text ${active && !lane.tools.length ? "cursor" : ""}`}>
          {lane.text}
        </div>
        {lane.tools.length > 0 && (
          <div className="tools">
            {lane.tools.map((t) => (
              <ToolRow key={t.toolUseId} tool={t} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ToolRow({ tool }: { tool: ToolEntry }) {
  const args = Object.entries(tool.input)
    .filter(([k]) => k !== "incident_id")
    .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
    .join(" ");
  const cls = tool.isError ? "err" : tool.done ? "" : "pending";
  return (
    <div className={`tool ${cls}`} title={tool.preview}>
      <span className="tname">{tool.name}</span>
      <span className="targs">{args}</span>
      <span className="tdur">{tool.done ? `${tool.durationMs}ms` : "…"}</span>
    </div>
  );
}

function RcaCard({ rca }: { rca: NonNullable<RunState["rca"]> }) {
  const none = rca.root_cause_class === "none";
  return (
    <div className={`rca ${none ? "none" : ""}`}>
      <h2>Root-cause analysis</h2>
      <div className="cause">
        {rca.root_cause_class}
        {!none && (
          <>
            {" @ "}
            <span className="svc">{rca.culprit_service}</span>
          </>
        )}
        <span className="conf">confidence {rca.confidence.toFixed(2)}</span>
      </div>
      <div className="summary">{rca.summary}</div>
      {rca.evidence_event_ids.length > 0 && (
        <div className="evidence">evidence: {rca.evidence_event_ids.join(", ")}</div>
      )}
      {rca.remediation && (
        <div className="remediation">
          <strong>Proposed remediation:</strong> {rca.remediation.action} →{" "}
          <span className="mono">{rca.remediation.target}</span>
          <br />
          <span style={{ color: "var(--text-dim)" }}>
            blast radius: {rca.remediation.blast_radius}
            {rca.remediation.reversible ? " (reversible)" : ""}
          </span>
        </div>
      )}
    </div>
  );
}
