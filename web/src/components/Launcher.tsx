import { useState } from "react";
import type { Scenario, Topology } from "../types";

const TOPOLOGIES: { key: Topology; label: string }[] = [
  { key: "full", label: "Full" },
  { key: "no_verifier", label: "No verifier" },
  { key: "single_agent", label: "Single agent" },
];

export function Launcher({
  scenarios,
  onLaunch,
}: {
  scenarios: Scenario[];
  onLaunch: (incidentId: string, topology: Topology, mock: boolean) => void;
}) {
  const [selected, setSelected] = useState<string | null>(null);
  const [topology, setTopology] = useState<Topology>("full");
  const [mock, setMock] = useState(true);

  return (
    <div className="launcher">
      <p className="section-label">Incidents</p>
      <div className="scenario-list">
        {scenarios.map((s) => (
          <div
            key={s.id}
            className={`scenario ${selected === s.id ? "selected" : ""}`}
            onClick={() => setSelected(s.id)}
          >
            <div>
              <span className="sid">{s.id}</span>
              {s.with_injection && <span className="badge inj">injection</span>}
            </div>
            <div className="salert">{s.alert}</div>
          </div>
        ))}
        {scenarios.length === 0 && (
          <div className="salert" style={{ padding: "8px 10px" }}>
            No scenarios loaded. Run <code>make scenarios</code>.
          </div>
        )}
      </div>

      <div className="controls">
        <div className="seg">
          {TOPOLOGIES.map((t) => (
            <button
              key={t.key}
              className={topology === t.key ? "on" : ""}
              onClick={() => setTopology(t.key)}
            >
              {t.label}
            </button>
          ))}
        </div>
        <label className="toggle">
          <input type="checkbox" checked={mock} onChange={(e) => setMock(e.target.checked)} />
          Mock model (no spend - Uncheck to use real LLM API)
        </label>
        <button
          className="launch-btn"
          disabled={!selected}
          onClick={() => selected && onLaunch(selected, topology, mock)}
        >
          Start triage
        </button>
      </div>
    </div>
  );
}
