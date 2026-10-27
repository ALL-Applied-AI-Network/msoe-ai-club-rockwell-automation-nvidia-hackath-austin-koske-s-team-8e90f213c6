import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backend = process.env.API_PROXY_TARGET || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/health": { target: backend, changeOrigin: true },
      "/v1": { target: backend, changeOrigin: true },
      "/ws": { target: backend, changeOrigin: true, ws: true },
    },
  },
});

