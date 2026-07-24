import { useState } from "react";
import type { PendingApproval } from "../useRunStream";

// The human-in-the-loop gate, rendered when the harness raises a write action. Until the operator
// decides, the run is paused server-side — this modal is the decision point.
export function ApprovalModal({
  approval,
  onResolve,
}: {
  approval: PendingApproval;
  onResolve: (id: string, decision: "approved" | "denied", reason: string) => void;
}) {
  const [reason, setReason] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const resolve = (decision: "approved" | "denied") => {
    setSubmitting(true);
    onResolve(approval.approvalId, decision, reason);
  };

  return (
    <div className="modal-scrim">
      <div className="modal">
        <h2>Approval required</h2>
        <div className="sub">The agent wants to take an action that changes the world.</div>

        <div className="field">
          <div className="k">action</div>
          <div className="v">{approval.name}</div>
        </div>
        {Object.entries(approval.input)
          .filter(([k]) => !["incident_id", "blast_radius"].includes(k))
          .map(([k, v]) => (
            <div className="field" key={k}>
              <div className="k">{k}</div>
              <div className="v">{typeof v === "string" ? v : JSON.stringify(v)}</div>
            </div>
          ))}

        <div className="blast">⚠ blast radius: {approval.blastRadius}</div>

        <textarea
          rows={2}
          placeholder="Reason (optional; shown to the agent on denial)"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
        />

        <div className="modal-actions">
          <button className="btn deny" disabled={submitting} onClick={() => resolve("denied")}>
            Deny
          </button>
          <button className="btn approve" disabled={submitting} onClick={() => resolve("approved")}>
            Approve
          </button>
        </div>
      </div>
    </div>
  );
}
