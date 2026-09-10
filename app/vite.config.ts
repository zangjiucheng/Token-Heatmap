import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// `import.meta.dirname`, not `__dirname`: this package is `type: module`, so the
// config is ESM and `__dirname` only happens to resolve today because Vite's
// default loader bundles the config to CJS first. Under `configLoader: 'native'`
// — slated to become the default — it would be undefined at runtime.
// Requires Node >= 20.11, which is below the floor CI already builds on.
const rootDir = import.meta.dirname;

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
        '@': path.resolve(rootDir, 'src'),
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
