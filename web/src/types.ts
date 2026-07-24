// Mirrors the event and resource shapes the API emits. Kept deliberately close to the Python
// `aegis.types` definitions so the two ends stay legible together.

export type Role = "coordinator" | "investigator" | "verifier";

export interface Scenario {
  id: string;
  alert: string;
  fault_class: string;
  with_injection: boolean;
}

export interface PromptArg {
  name: string;
  description: string;
  required: boolean;
}

export interface Prompt {
  name: string;
  description: string;
  arguments: PromptArg[];
}

export interface Remediation {
  action: string;
  target: string;
  rationale: string;
  blast_radius: string;
  reversible: boolean;
}

export interface RCA {
  root_cause_class: string;
  culprit_service: string;
  summary: string;
  evidence_event_ids: string[];
  confidence: number;
  rejected_hypotheses: string[];
  remediation: Remediation | null;
}

export interface RunRow {
  id: string;
  scenario_id: string;
  alert: string;
  status: string;
  created_at: number;
  cost_usd: number;
  duration_s: number | null;
  rca: RCA | null;
}

// The discriminated event union delivered over SSE. `seq` is attached by the server.
export type AegisEvent =
  | { type: "run.started"; seq: number; run_id: string; scenario_id: string; alert: string }
  | {
      type: "run.finished";
      seq: number;
      run_id: string;
      status: string;
      cost_usd: number;
      duration_s: number;
      rca: RCA | null;
      error: string | null;
    }
  | {
      type: "agent.started";
      seq: number;
      agent_id: string;
      role: Role;
      label: string;
      parent_agent_id: string | null;
    }
  | {
      type: "agent.finished";
      seq: number;
      agent_id: string;
      role: Role;
      label: string;
      cost_usd: number;
      turns: number;
      error: string | null;
    }
  | { type: "agent.text_delta"; seq: number; agent_id: string; text: string }
  | { type: "agent.thinking_delta"; seq: number; agent_id: string; text: string }
  | {
      type: "tool.called";
      seq: number;
      agent_id: string;
      tool_use_id: string;
      name: string;
      input: Record<string, unknown>;
    }
  | {
      type: "tool.resulted";
      seq: number;
      agent_id: string;
      tool_use_id: string;
      name: string;
      duration_ms: number;
      is_error: boolean;
      preview: string;
    }
  | {
      type: "approval.requested";
      seq: number;
      agent_id: string;
      approval_id: string;
      tool_use_id: string;
      name: string;
      input: Record<string, unknown>;
      blast_radius: string;
    }
  | {
      type: "approval.resolved";
      seq: number;
      agent_id: string;
      approval_id: string;
      decision: string;
      reason: string;
    }
  | {
      type: "budget.warning";
      seq: number;
      spent_usd: number;
      limit_usd: number;
      fraction: number;
    };

export type Topology = "full" | "single_agent" | "no_verifier";
