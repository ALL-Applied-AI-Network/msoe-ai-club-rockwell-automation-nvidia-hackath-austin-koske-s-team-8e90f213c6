// vite.config.ts
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      // API_PROXY_TARGET is optional; the local API defaults to8443.
      '/api': {
        target: process.env.API_PROXY_TARGET || 'http://127.0.0.1:8443',
        changeOrigin: true,
        secure: false,
        ws: true,
      },
      '/health': {
        target: process.env.API_PROXY_TARGET || 'http://127.0.0.1:8443',
        changeOrigin: true,
        secure: false,
      },
    },
  },
})
