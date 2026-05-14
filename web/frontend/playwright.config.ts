import { defineConfig, devices } from '@playwright/test';

const PORT = Number(process.env.E2E_PORT ?? 4173);
const BASE_URL = `http://127.0.0.1:${PORT}`;

/**
 * Playwright config for the LLM Token Heatmap frontend.
 *
 * Specs live under `e2e/` and run against a production-mode preview server
 * (`vite preview`) so the bundle behaves identically to deployments. A fresh
 * `npm run build` is part of the webServer command so CI never tests stale
 * output. Three projects (Chromium, Firefox, WebKit) cover the supported
 * browser matrix; sharding is delegated to CI via the `--shard` flag.
 */
export default defineConfig({
  testDir: 'e2e',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 5 : 1,
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI
    ? [
        ['list'],
        ['html', { open: 'never', outputFolder: 'playwright-report' }],
        ['junit', { outputFile: 'test-results/junit.xml' }],
      ]
    : [['list'], ['html', { open: 'never', outputFolder: 'playwright-report' }]],
  outputDir: 'test-results',
  expect: {
    toHaveScreenshot: {
      // Allow a small amount of anti-aliasing / font rasterisation drift.
      // Canvas-rendered heatmap cells vary subtly between GPU drivers; this
      // threshold is wide enough to absorb that without hiding real layout
      // regressions.
      maxDiffPixelRatio: 0.03,
      threshold: 0.2,
      animations: 'disabled',
    },
  },
  use: {
    baseURL: BASE_URL,
    trace: 'retain-on-failure',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
    {
      name: 'firefox',
      use: { ...devices['Desktop Firefox'] },
    },
    {
      name: 'webkit',
      use: { ...devices['Desktop Safari'] },
    },
  ],
  webServer: {
    command: `npm run build && npm run preview -- --port ${PORT} --strictPort`,
    url: BASE_URL,
    reuseExistingServer: !process.env.CI,
    timeout: 180_000,
    stdout: 'pipe',
    stderr: 'pipe',
  },
  // Visual snapshot baselines are only authoritative on Linux (CI). Local
  // runs on other OSes will write their own copy under a platform-suffixed
  // directory so they don't fight the committed Linux baselines.
  snapshotPathTemplate:
    '{testDir}/__screenshots__/{testFilePath}/{arg}-{projectName}{ext}',
});
