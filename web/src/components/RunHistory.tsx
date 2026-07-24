import type { RunRow } from "../types";

export function RunHistory({
  runs,
  activeRun,
  onSelect,
}: {
  runs: RunRow[];
  activeRun: string | null;
  onSelect: (id: string) => void;
}) {
  return (
    <div className="history">
      <p className="section-label">Recent runs</p>
      {runs.map((r) => (
        <div
          key={r.id}
          className={`run-item ${activeRun === r.id ? "selected" : ""}`}
          onClick={() => onSelect(r.id)}
        >
          <div style={{ display: "flex", alignItems: "center", gap: 8, minWidth: 0 }}>
            <span className={`dot ${r.status}`} />
            <span className="rid">{r.scenario_id}</span>
          </div>
          <span className="rid">${r.cost_usd.toFixed(3)}</span>
        </div>
      ))}
      {runs.length === 0 && <div className="rid">No runs yet.</div>}
    </div>
  );
}
