import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { closeSync, openSync } from "node:fs";
import { resolve } from "node:path";

const STATIC_DIR = resolve(__dirname, "../src/neme_anima/server/static");

// emptyOutDir wipes the FastAPI static dir on every build, including the
// tracked .gitkeep that keeps the directory present on fresh CI checkouts.
// Recreate it after the bundle closes so commits made post-build don't
// silently stage its deletion.
const keepStaticGitkeep = {
  name: "keep-static-gitkeep",
  closeBundle() {
    closeSync(openSync(resolve(STATIC_DIR, ".gitkeep"), "a"));
  },
};

export default defineConfig(({ mode }) => ({
  plugins: [svelte(), keepStaticGitkeep],
  resolve: {
    alias: {
      $lib: resolve(__dirname, "src/lib"),
    },
  },
  build: {
    // Build directly into the FastAPI static-files dir.
    outDir: STATIC_DIR,
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
