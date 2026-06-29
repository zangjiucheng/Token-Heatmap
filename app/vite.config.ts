import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const port = Number(env.VITE_DEV_PORT ?? 5173);

  return {
    // Relative asset paths so the built SPA loads from the Tauri app's
    // `tauri://` filesystem origin (and any static host) without a server.
    base: './',
    plugins: [react()],
    resolve: {
      alias: {
        '@': path.resolve(__dirname, 'src'),
      },
    },
    server: {
      port,
      strictPort: false,
    },
    preview: {
      port,
    },
  };
});
