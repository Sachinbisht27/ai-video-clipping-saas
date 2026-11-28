const { test, expect } = require('@playwright/test');

test('public navigation opens the intended pages and returns home', async ({ page, isMobile }) => {
  await page.goto('/');
  if (isMobile) await page.getByRole('button', { name: /menu/i }).click();
  await page.getByRole('link', { name: 'How it works', exact: true }).click();
  await expect(page.getByRole('heading', { name: /a better moment starts with context/i })).toBeVisible();
  if (isMobile) await page.getByRole('button', { name: /menu/i }).click();
  await page.getByRole('link', { name: 'About', exact: true }).click();
  await expect(page.getByRole('heading', { name: /stories deserve to be heard in full/i })).toBeVisible();
  await page.getByRole('link', { name: /clipforge/i }).first().click();
  await expect(page.getByRole('heading', { name: /big conversations/i })).toBeVisible();
});

test('suggestion selection updates the example', async ({ page }) => {
  await page.goto('/');
  await page.getByRole('button', { name: /the spark for the idea.*0:28/i }).click();
  await expect(page.locator('#demo-title')).toHaveText('The spark for the idea');
  await expect(page.locator('#demo-duration')).toHaveText('0:28');
  await expect(page.getByRole('status')).toContainText('no video is playing');
});

test('story follows scrolling in both directions', async ({ page }) => {
  await page.goto('/');
  const first = page.locator('[data-story="0"]');
  const last = page.locator('[data-story="3"]');
  await last.scrollIntoViewIfNeeded();
  await expect(last).toHaveAttribute('aria-pressed', 'true');
  await first.scrollIntoViewIfNeeded();
  await expect(first).toHaveAttribute('aria-pressed', 'true');
});

test('story is usable with reduced motion and keyboard', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/');
  const chapter = page.locator('[data-story="2"]');
  await chapter.focus();
  await page.keyboard.press('Enter');
  await expect(chapter).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('#story-visual-title')).toHaveText('Shape a complete thought');
});

test('suggestions fit their container on a narrow screen', async ({ page }) => {
  await page.setViewportSize({ width: 320, height: 640 });
  await page.goto('/');
  const sizes = await page.evaluate(() => {
    const card = document.querySelector('.choice').getBoundingClientRect();
    const container = document.querySelector('.demo').getBoundingClientRect();
    return { cardLeft: card.left, cardRight: card.right, left: container.left, right: container.right, pageWidth: document.documentElement.scrollWidth, viewport: innerWidth };
  });
  expect(sizes.cardLeft).toBeGreaterThanOrEqual(sizes.left);
  expect(sizes.cardRight).toBeLessThanOrEqual(sizes.right);
  expect(sizes.pageWidth).toBeLessThanOrEqual(sizes.viewport);
});
