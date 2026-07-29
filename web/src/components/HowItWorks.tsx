// A static, in-app "understand Aegis in 30 seconds" panel. Shown as the empty
// state before a run is selected; doubles as a legend for the run-view lane colors.
// The full version with diagrams lives in docs/HOW_IT_WORKS.md — link below.
const REPO_DOC_URL = "https://github.com/siliconshells/agentic_rca/blob/main/docs/HOW_IT_WORKS.md";

const STEPS = [
  { label: "Alert fires", role: "warn" },
  { label: "Coordinator plans", role: "coordinator" },
  { label: "Investigators dig ×N", role: "investigator" },
  { label: "Verifier attacks", role: "verifier" },
  { label: "Cited RCA + your approval", role: "warn" },
];

export function HowItWorks() {
  return (
    <div className="howto">
      <h2>How Aegis works</h2>
      <p className="howto-hook">
        A team of AI agents figures out <em>why</em> a web system broke — and shows its evidence.
      </p>

      <div className="howto-flow">
        {STEPS.map((s, i) => (
          <span key={s.label} style={{ display: "contents" }}>
            <span className={`howto-step hs-${s.role}`}>{s.label}</span>
            {i < STEPS.length - 1 && <span className="howto-arrow">→</span>}
          </span>
        ))}
      </div>

      <dl className="howto-defs">
        <div className="howto-def">
          <dt>Incident</dt>
          <dd>
            A live web system is dozens of services calling each other. When one misbehaves, an
            alert fires — but it names the symptom, not the cause.
          </dd>
        </div>
        <div className="howto-def">
          <dt>RCA (root-cause analysis)</dt>
          <dd>
            The written answer: what broke, in which service, with a confidence score and{" "}
            <em>cited</em> log, metric, and deploy events as proof.
          </dd>
        </div>
        <div className="howto-def">
          <dt>Where the data comes from</dt>
          <dd>
            A deterministic fake world built from a seed, with exactly one injected fault. The true
            answer (ground truth) is hidden from the agents and used only to grade them.
          </dd>
        </div>
      </dl>

      <p className="howto-see">
        <strong>See it yourself:</strong> pick an incident on the left — mock mode needs no API key —
        or replay a finished run with <code>aegis show &lt;run-id&gt;</code>.
      </p>

      <a className="howto-link" href={REPO_DOC_URL} target="_blank" rel="noreferrer">
        Full 2-minute explainer with diagrams ↗
      </a>
    </div>
  );
}
