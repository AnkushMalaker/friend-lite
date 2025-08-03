import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: process.env.REACT_UI_HOST || '0.0.0.0',
    port: parseInt(process.env.REACT_UI_PORT || '5173'),
    https: process.env.REACT_UI_HTTPS === 'true' ? true : false,
    proxy: {
      '/api': {
        target: `http://${process.env.SPEAKER_SERVICE_HOST || 'speaker-service'}:${process.env.SPEAKER_SERVICE_PORT || '8085'}`,
        changeOrigin: true,
        secure: false, // Allow self-signed certificates for internal use
        rewrite: (path) => path.replace(/^\/api/, '')
      }
    }
  },
  define: {
    global: 'globalThis',
  },
  resolve: {
    alias: {
      buffer: 'buffer',
      stream: 'stream-browserify',
      util: 'util'
    }
  },
  optimizeDeps: {
    include: ['buffer', 'stream-browserify', 'util']
  }
})