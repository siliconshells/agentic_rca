// Thin API client. Everything is same-origin via the Vite proxy (dev) or the reverse path
// (container), so no base URL is needed in the browser.

import type { Prompt, RunRow, Scenario, Topology } from "./types";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path}: ${res.status}`);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const detail = await res.text();
    throw new Error(`${path}: ${res.status} ${detail}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  scenarios: () => get<Scenario[]>("/api/scenarios"),
  prompts: () => get<Prompt[]>("/api/prompts"),
  runs: () => get<RunRow[]>("/api/runs"),
  run: (id: string) =>
    get<{ run: RunRow; spans: unknown[]; tool_calls: unknown[]; approvals: unknown[] }>(
      `/api/runs/${id}`,
    ),
  startRun: (incident_id: string, topology: Topology, mock: boolean) =>
    post<{ run_id: string }>("/api/runs", { incident_id, topology, mock }),
  resolveApproval: (approval_id: string, decision: "approved" | "denied", reason: string) =>
    post<{ resolved: boolean }>(`/api/approvals/${approval_id}`, { decision, reason }),
};
