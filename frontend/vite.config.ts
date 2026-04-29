import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { resolve } from "node:path";

export default defineConfig(({ mode }) => ({
  plugins: [svelte()],
  build: {
    // Build directly into the FastAPI static-files dir.
    outDir: resolve(__dirname, "../src/neme_extractor/server/static"),
    emptyOutDir: true,
    sourcemap: mode === "development",
  },
  server: {
    port: 5173,
    strictPort: false,
    proxy: {
      // In dev mode, forward /api/* + /api/ws to a running FastAPI server.
      // The user is expected to start uvicorn separately on this port.
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        ws: true,
      },
    },
  },
  test: {
    environment: "happy-dom",
    globals: true,
  },
}));
