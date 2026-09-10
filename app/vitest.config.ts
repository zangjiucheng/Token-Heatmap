import { defineConfig, mergeConfig } from 'vitest/config';
// Explicit extension: the native config loader resolves specifiers the way
// Node does, and Node does not guess extensions.
import viteConfig from './vite.config.ts';

export default mergeConfig(
  viteConfig({ mode: 'test', command: 'serve' }),
  defineConfig({
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./src/setupTests.ts'],
      css: false,
      include: ['src/**/*.test.{ts,tsx}'],
    },
  }),
);
