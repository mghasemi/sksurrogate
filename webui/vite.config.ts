import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      // Forward API calls — and the job progress WebSocket streams — through the
      // dev server so the browser only ever talks to its own origin. This is what
      // lets the UI work from localhost, a LAN IP, or a forwarded/tunnel URL
      // without CORS or mixed-content errors (see src/api/client.ts).
      '/api': {
        target: 'http://localhost:8013',
        changeOrigin: true,
        ws: true,
      },
    },
  },
})
