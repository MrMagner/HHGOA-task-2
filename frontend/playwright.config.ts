import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: 'tests/e2e',
  fullyParallel: true,
  retries: 0,
  workers: 1,
  reporter: 'list',
  use: {
    browserName: 'firefox',
    launchOptions: {
      executablePath: '/usr/bin/firefox'
    },
    trace: 'on-first-retry',
  },
});
