// Playwright (Node, global package via NODE_PATH): launch the bundled Chromium and screenshot Chinese text.
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 600, height: 120 } });
  await page.setContent('<p style="font: 40px sans-serif; margin: 30px">Node Playwright 中文 ✓</p>');
  await page.screenshot({ path: 'node.png' });
  console.log(`node.png, chromium ${browser.version()}`);
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
