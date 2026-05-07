import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig({
  base: '/app/',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['icons/icon-192.png', 'icons/icon-512.png', 'favicon.svg'],
      manifest: {
        name: 'Code Names AI',
        short_name: 'CodeNamesAI',
        description: 'Play Codenames with AI spymasters and guessers',
        theme_color: '#0f1117',
        background_color: '#0f1117',
        display: 'standalone',
        start_url: '/app/',
        scope: '/app/',
        icons: [
          {
            src: 'icons/icon-192.png',
            sizes: '192x192',
            type: 'image/png',
            purpose: 'any maskable',
          },
          {
            src: 'icons/icon-512.png',
            sizes: '512x512',
            type: 'image/png',
            purpose: 'any maskable',
          },
        ],
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg,woff2}'],
      },
    }),
  ],
  build: {
    outDir: '../src/codenames_ai/web/static/pwa',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': { target: 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
