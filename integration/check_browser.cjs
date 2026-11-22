const assert = require('node:assert/strict');
const fs = require('node:fs');
const { chromium } = require('/tmp/manimations-qa/node_modules/playwright');

(async () => {
  const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox'] });
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
  const requests = [], errors = [];
  page.on('request', request => requests.push(request.url()));
  page.on('pageerror', error => errors.push(String(error)));
  const response = await page.goto('http://127.0.0.1:8767/Manimations/#huffman');
  assert.equal(response.status(), 200);
  await page.waitForFunction(() => document.querySelector('video')?.currentTime > 0);
  assert.equal(await page.locator('.card').count(), 1);
  assert.equal(await page.locator('#huffman h3').textContent(), 'Huffman encoding');
  assert(await page.locator('video').evaluate(video => video.currentSrc.endsWith('/Manimations/media/huffman.mp4') && video.muted));
  assert(requests.every(url => url.startsWith('http://127.0.0.1:8767/')));
  assert(!requests.some(url => url.includes('playback-test') || url.includes('dummy-slug')));
  await page.screenshot({ path: 'build/integration/github-pages-gallery.png', fullPage: true });
  assert.deepEqual(errors, []);
  await browser.close();
  fs.writeFileSync('build/integration/browser-report.json', JSON.stringify({ passed: true, route: '/Manimations/#huffman', selectedClips: ['huffman'], sameOriginAssetsOnly: true, mutedAutoplay: true, javascriptErrors: errors }, null, 2));
  console.log('PASS: staged /Manimations/#huffman, selected media only, muted autoplay, all assets from the Pages origin, no JS errors.');
})().catch(error => { console.error(error); process.exit(1); });
