import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: '0.0.0.0',
    hmr: {
      port: 5173,
      // Allow HMR to work through proxy
      clientPort: process.env.VITE_HMR_PORT ? parseInt(process.env.VITE_HMR_PORT) : undefined,
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: true,
  },
})