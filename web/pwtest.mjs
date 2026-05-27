import { chromium } from 'playwright';

const OUT = 'C:/Users/ElmoShim/Projects/diffusion-renderer/output';
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1320, height: 1040 } });

const errors = [];
page.on('console', (m) => {
  const t = m.text();
  if (m.type() === 'error') errors.push(t);
  if (t.includes('[OPACITY]')) console.log(t);
});
page.on('pageerror', (e) => errors.push('PAGEERROR: ' + e.message));

await page.goto('http://localhost:8080', { waitUntil: 'domcontentloaded' });
// Default scene is outfit.zprj (the 'top' sweater has the opacity map).
// Wait for the WASM module + scene to load and the panels to render.
await page.waitForTimeout(9000);

await page.screenshot({ path: `${OUT}/web_full.png` });
for (const id of ['panel-basecolor', 'panel-normal', 'panel-depth', 'panel-roughness']) {
  const el = await page.$('#' + id);
  if (el) await el.screenshot({ path: `${OUT}/web_${id}.png` });
}

console.log('CONSOLE ERRORS:', errors.length ? '\n' + errors.join('\n') : 'none');
await browser.close();
