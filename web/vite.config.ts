import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The dashboard talks to the FastAPI backend. Vite proxies /api to it so the browser makes
// same-origin requests — no CORS dance, and SSE streams cleanly. VITE_API_BASE points the proxy
// at the backend: http://localhost:8000 in dev, http://api:8000 inside compose.
const API = process.env.VITE_API_BASE ?? "http://localhost:8000";
const proxy = { "/api": { target: API, changeOrigin: true } };

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, host: true, proxy },
  preview: { port: 5173, host: true, proxy },
});
