import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react(), basicSsl()],
  server: {
    host: '0.0.0.0',
    port: parseInt(process.env.REACT_UI_PORT || '5173'),
    https: process.env.REACT_UI_HTTPS === 'true' ? true : false,
    allowedHosts: [
      ...(process.env.REACT_UI_HOST ? [process.env.REACT_UI_HOST] : []),
      'speaker-recognition.local',
      'localhost', 
      '127.0.0.1'
    ],
    proxy: {
      '/api': {
        target: `http://${process.env.SPEAKER_SERVICE_HOST || 'localhost'}:${process.env.SPEAKER_SERVICE_PORT || '8085'}`,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, '')
      },
      '/health': {
        target: `http://${process.env.SPEAKER_SERVICE_HOST || 'localhost'}:${process.env.SPEAKER_SERVICE_PORT || '8085'}`,
        changeOrigin: true
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