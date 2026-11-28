const { defineConfig, devices } = require('@playwright/test');

module.exports = defineConfig({
  testDir: './e2e',
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: 'list',
  use: { baseURL: 'http://127.0.0.1:8766', trace: 'retain-on-failure' },
  projects: [
    { name: 'desktop-chromium', use: { ...devices['Desktop Chrome'] } },
    { name: 'mobile-chromium', use: { ...devices['iPhone SE'], browserName: 'chromium' } },
  ],
  webServer: {
    command: 'uv run python -m uvicorn clipforge.main:app --host 127.0.0.1 --port 8766',
    url: 'http://127.0.0.1:8766/api/v1/health',
    reuseExistingServer: false,
  },
});
