# The dashboard, built to static assets and served by Vite preview. In compose it proxies /api to
# the api service so the browser makes same-origin calls (SSE included).
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm install
COPY . .
RUN npm run build

FROM node:20-slim AS runtime
WORKDIR /app
COPY --from=build /app/package.json ./
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/dist ./dist
COPY --from=build /app/vite.config.ts ./
EXPOSE 5173
# `vite preview` serves the built assets and honours the proxy config for /api.
CMD ["npx", "vite", "preview", "--host", "--port", "5173"]
