/* Test the next-day data-only addition in a real browser, without touching docs/. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const { spawn } = require('node:child_process');
const { chromium } = require('/tmp/manimations-qa/node_modules/playwright');

(async () => {
  const report = JSON.parse(fs.readFileSync('build/add-video-qa/report.json'));
  const server = spawn('python3', ['-m', 'http.server', '8768', '--bind', '127.0.0.1', '--directory', report.host_directory], { stdio: 'ignore' });
  let browser;
  try {
    const base = 'http://127.0.0.1:8768/Manimations/';
    let ready = false;
    for (let n = 0; n < 30; n++) {
      try { ready = (await fetch(base)).ok; } catch {}
      if (ready) break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert(ready, 'Local data-only fixture server started');
    browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox'] });
    const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
    const errors = [];
    page.on('pageerror', error => errors.push(String(error)));
    await page.goto(`${base}#second-sample`);
    await page.waitForFunction(() => document.querySelector('#second-sample video')?.currentTime > 0);
    assert.equal(await page.locator('.card').count(), 2);
    assert.equal(await page.locator('#second-sample h3').textContent(), 'Updated second title');
    await page.locator('#topic').selectOption('New topic');
    assert.equal(await page.locator('.card:visible').count(), 1);
    assert(await page.locator('#second-sample').isVisible());
    assert(await page.locator('#first-sample').isHidden());
    assert.deepEqual(errors, []);
    await page.screenshot({ path: 'build/add-video-qa/second-sample-browser.png', fullPage: true });
    report.browser = { passed: true, twoCards: true, newTagAutomatic: true, deepLinkPlayback: true, metadataTitleUpdated: true, javascriptErrors: errors };
    fs.writeFileSync('build/add-video-qa/report.json', JSON.stringify(report, null, 2));
    console.log('PASS: second JSON/media-only sample renders, plays via deep link, adds its tag automatically, and shows its updated title.');
  } finally {
    if (browser) await browser.close();
    server.kill('SIGTERM');
  }
})().catch(error => { console.error(error); process.exit(1); });
