// The run reducer: turns the SSE event stream into the view model the dashboard renders.
//
// One agent "lane" per agent, each accumulating streamed text/thinking and a tool timeline. The
// EventSource replays history from seq 0 on connect (the server backs this with its event log),
// so opening the stream late still reconstructs the whole run — no missed events.

import { useEffect, useReducer, useRef } from "react";
import type { AegisEvent, RCA, Role } from "./types";

export interface ToolEntry {
  toolUseId: string;
  name: string;
  input: Record<string, unknown>;
  durationMs?: number;
  isError?: boolean;
  preview?: string;
  done: boolean;
}

export interface AgentLane {
  agentId: string;
  role: Role;
  label: string;
  parentAgentId: string | null;
  text: string;
  thinking: string;
  tools: ToolEntry[];
  costUsd?: number;
  turns?: number;
  finished: boolean;
  error?: string | null;
}

export interface PendingApproval {
  approvalId: string;
  agentId: string;
  name: string;
  input: Record<string, unknown>;
  blastRadius: string;
}

export interface RunState {
  runId: string;
  status: "connecting" | "running" | "succeeded" | "failed" | "budget_exceeded" | "cancelled";
  alert: string;
  agents: AgentLane[];
  approvals: PendingApproval[];
  rca: RCA | null;
  costUsd: number;
  durationS: number | null;
  budgetWarning?: { spent: number; limit: number; fraction: number };
  lastSeq: number;
}

function initial(runId: string): RunState {
  return {
    runId,
    status: "connecting",
    alert: "",
    agents: [],
    approvals: [],
    rca: null,
    costUsd: 0,
    durationS: null,
    lastSeq: 0,
  };
}

function laneById(agents: AgentLane[], id: string | undefined): AgentLane | undefined {
  return id ? agents.find((a) => a.agentId === id) : undefined;
}

function reduce(state: RunState, event: AegisEvent): RunState {
  // Ignore anything we've already folded in — the history replay overlaps the live tail.
  if (event.seq <= state.lastSeq) return state;
  const agents = state.agents.map((a) => ({ ...a, tools: [...a.tools] }));
  const next: RunState = { ...state, agents, lastSeq: event.seq };

  switch (event.type) {
    case "run.started":
      next.status = "running";
      next.alert = event.alert;
      return next;

    case "run.finished":
      next.status = event.status as RunState["status"];
      next.rca = event.rca;
      next.costUsd = event.cost_usd;
      next.durationS = event.duration_s;
      return next;

    case "agent.started":
      if (!laneById(agents, event.agent_id)) {
        agents.push({
          agentId: event.agent_id!,
          role: event.role,
          label: event.label,
          parentAgentId: event.parent_agent_id,
          text: "",
          thinking: "",
          tools: [],
          finished: false,
        });
      }
      return next;

    case "agent.finished": {
      const lane = laneById(agents, event.agent_id);
      if (lane) {
        lane.finished = true;
        lane.costUsd = event.cost_usd;
        lane.turns = event.turns;
        lane.error = event.error;
      }
      return next;
    }

    case "agent.text_delta": {
      const lane = laneById(agents, event.agent_id);
      if (lane) lane.text += event.text;
      return next;
    }

    case "agent.thinking_delta": {
      const lane = laneById(agents, event.agent_id);
      if (lane) lane.thinking += event.text;
      return next;
    }

    case "tool.called": {
      const lane = laneById(agents, event.agent_id);
      if (lane) {
        lane.tools.push({
          toolUseId: event.tool_use_id,
          name: event.name,
          input: event.input,
          done: false,
        });
      }
      return next;
    }

    case "tool.resulted": {
      const lane = laneById(agents, event.agent_id);
      const entry = lane?.tools.find((t) => t.toolUseId === event.tool_use_id);
      if (entry) {
        entry.done = true;
        entry.durationMs = event.duration_ms;
        entry.isError = event.is_error;
        entry.preview = event.preview;
      }
      return next;
    }

    case "approval.requested":
      next.status = "running";
      next.approvals = [
        ...state.approvals,
        {
          approvalId: event.approval_id,
          agentId: event.agent_id!,
          name: event.name,
          input: event.input,
          blastRadius: event.blast_radius,
        },
      ];
      return next;

    case "approval.resolved":
      next.approvals = state.approvals.filter((a) => a.approvalId !== event.approval_id);
      return next;

    case "budget.warning":
      next.budgetWarning = {
        spent: event.spent_usd,
        limit: event.limit_usd,
        fraction: event.fraction,
      };
      return next;

    default:
      return next;
  }
}

type Action =
  | { kind: "reset"; runId: string }
  | { kind: "clear" }
  | { kind: "event"; event: AegisEvent };

function rootReducer(state: RunState | null, action: Action): RunState | null {
  if (action.kind === "clear") return null;
  if (action.kind === "reset") return initial(action.runId);
  if (!state) return state;
  return reduce(state, action.event);
}

const EVENT_CHANNELS = [
  "run.started",
  "run.finished",
  "agent.started",
  "agent.finished",
  "agent.text_delta",
  "agent.thinking_delta",
  "tool.called",
  "tool.resulted",
  "approval.requested",
  "approval.resolved",
  "budget.warning",
];

/** Subscribe to a run's SSE stream and reduce it into render state. */
export function useRunStream(runId: string | null): RunState | null {
  const [state, dispatch] = useReducer(rootReducer, null);
  const openFor = useRef<string | null>(null);

  useEffect(() => {
    if (!runId) {
      // Deselecting a run (e.g. clicking "How it works") must drop the prior run's
      // state, otherwise the stale object keeps <RunView> mounted over <HowItWorks>.
      if (openFor.current !== null) {
        openFor.current = null;
        dispatch({ kind: "clear" });
      }
      return;
    }
    // Fresh state whenever the selected run changes, before the stream opens.
    if (openFor.current !== runId) {
      openFor.current = runId;
      dispatch({ kind: "reset", runId });
    }

    const source = new EventSource(`/api/runs/${runId}/stream?after=0`);
    const onMessage = (raw: MessageEvent) => {
      try {
        dispatch({ kind: "event", event: JSON.parse(raw.data) as AegisEvent });
      } catch {
        /* ignore malformed frames */
      }
    };
    source.onmessage = onMessage;
    // The server names each SSE frame's `event:` after its type, so listen on those channels too.
    for (const channel of EVENT_CHANNELS) {
      source.addEventListener(channel, onMessage as EventListener);
    }
    return () => source.close();
  }, [runId]);

  // No run selected → no run state, so <HowItWorks> shows.
  if (!runId) return null;
  // Between selecting a run and the reset landing, hand back a seeded object so the view renders.
  if (!state || state.runId !== runId) return initial(runId);
  return state;
}
