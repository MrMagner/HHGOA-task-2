import { test, expect } from '@playwright/test';

test.describe('Voice-Enabled RAG Smoke Test', () => {
  test('Text Query E2E flow', async ({ page }) => {
    // 1. Open the application
    await page.goto('http://localhost:3000');

    // 2. Verify page loads correctly
    await expect(page).toHaveTitle(/Voice-Enabled RAG/i);
    await expect(page.locator('h1')).toContainText('Voice-Enabled RAG');

    // 3. Verify no "Failed to fetch"
    const content = await page.content();
    expect(content).not.toContain('Failed to fetch');

    // 4. Test text query
    const searchInput = page.locator('input[type="text"]');
    await searchInput.fill('Manhattan Project');
    
    // We listen to console to check for TypeErrors
    const consoleErrors: string[] = [];
    page.on('console', msg => {
      if (msg.type() === 'error') {
        consoleErrors.push(msg.text());
      }
    });
    
    page.on('pageerror', err => {
      consoleErrors.push(err.message);
    });

    // 5. Submit query
    await page.locator('button', { hasText: 'Ask' }).click();

    // 6. Verify Answer renders (give it up to 20 seconds for LLM generation)
    const answerBlock = page.locator('div').filter({ hasText: 'Answer' }).last();
    await expect(answerBlock).toBeVisible({ timeout: 20000 });

    // 7. Verify Sources render
    const sourcesBlock = page.locator('div').filter({ hasText: 'Sources' }).last();
    await expect(sourcesBlock).toBeVisible();
    await expect(page.locator('.text-sm.text-gray-600').first()).toBeVisible();

    // 8. Verify Latency renders
    const latencyBlock = page.locator('text=/Total:.*ms/');
    await expect(latencyBlock).toBeVisible();

    // 9. Check that no TypeErrors were thrown in console
    expect(consoleErrors.length).toBe(0);
  });
});
