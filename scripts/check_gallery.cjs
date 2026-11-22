/* Browser QA only. Install Playwright outside the repo; see README.md. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { chromium } = require(path.join(process.env.GALLERY_QA_TOOLS || '/tmp/manimations-qa', 'node_modules/playwright'));

(async () => {
  const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args: ['--no-sandbox'] });
  const base = 'http://127.0.0.1:8766/100/Manimations/';
  const single = 'http://127.0.0.1:8766/1/Manimations/';
  const context = await browser.newContext({ viewport: { width: 1280, height: 900 } });
  const page = await context.newPage();
  const errors = [];
  const requests = [];
  page.on('pageerror', error => errors.push(String(error)));
  page.on('request', request => requests.push(request.url()));
  await page.goto(base);
  await page.waitForFunction(() => document.querySelectorAll('.card').length === 12);
  assert(!requests.some(url => url.includes('page-2.json')), 'Must not eagerly load next manifest');
  await page.waitForFunction(() => [...document.querySelectorAll('video')].some(v => !v.paused));
  await page.locator('.card').first().scrollIntoViewIfNeeded();
  await page.waitForFunction(() => [...document.querySelectorAll('video')].some(v => !v.paused));
  assert.equal(await page.locator('video').count(), 1, 'Only the selected visible card creates a player');
  assert(await page.locator('video').evaluate(v => v.muted && v.playsInline && v.controls));
  await page.waitForTimeout(250);
  assert(await page.locator('video').evaluate(v => v.currentTime > 0), 'Autoplay advances without clicking Play');
  await page.locator('video').evaluate(v => v.pause());
  await page.waitForTimeout(150);
  assert(await page.locator('video').evaluate(v => v.paused), 'Manual pause is respected');
  await page.locator('#autoplay').click();
  assert.equal(await page.locator('#autoplay').getAttribute('aria-pressed'), 'false');
  await page.locator('#load-more').click();
  await page.waitForFunction(() => document.querySelectorAll('.card').length === 24);
  await page.locator('#topic').selectOption('Compression');
  assert.equal(await page.locator('.card:visible').count(), 12);
  await page.goto('about:blank');
  await page.goto(`${base}#sample-0099`);
  await page.waitForFunction(() => Boolean(document.querySelector('#sample-0099')));
  assert.equal(await page.locator('.card').count(), 13, 'Deep link fetches one record, not remaining catalog pages');
  assert(await page.locator('#sample-0099').evaluate(element => {
    const bounds = element.getBoundingClientRect();
    return bounds.top < innerHeight && bounds.bottom > 0;
  }));

  const reduced = await browser.newContext({ viewport: { width: 390, height: 844 }, reducedMotion: 'reduce', colorScheme: 'light' });
  const reducedPage = await reduced.newPage();
  const reducedRequests = [];
  reducedPage.on('request', request => reducedRequests.push(request.url()));
  await reducedPage.goto(single);
  await reducedPage.waitForSelector('.card');
  await reducedPage.locator('.card').scrollIntoViewIfNeeded();
  await reducedPage.waitForTimeout(200);
  assert.equal(await reducedPage.locator('video').count(), 0);
  assert(!reducedRequests.some(url => url.includes('.mp4')), 'Reduced motion must not request video before interaction');
  assert.equal(await reducedPage.locator('#autoplay').getAttribute('aria-pressed'), 'false');
  await reducedPage.keyboard.press('Tab');
  await reducedPage.locator('.start').focus();
  await reducedPage.keyboard.press('Enter');
  await reducedPage.waitForFunction(() => document.querySelector('video')?.currentTime > 0);
  assert(await reducedPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), 'No mobile horizontal overflow');
  await reducedPage.screenshot({ path: 'build/gallery-qa/mobile-light.png', fullPage: true });

  const mobile = await browser.newContext({ viewport: { width: 390, height: 844 }, colorScheme: 'dark' });
  const mobilePage = await mobile.newPage();
  await mobilePage.goto(single);
  await mobilePage.waitForSelector('.card');
  await mobilePage.waitForFunction(() => document.querySelector('video')?.currentTime > 0);
  await mobilePage.screenshot({ path: 'build/gallery-qa/mobile-dark.png', fullPage: true });
  await page.goto(single);
  await page.waitForSelector('.card');
  await page.screenshot({ path: 'build/gallery-qa/desktop.png', fullPage: true });

  // Autoplay rejection must leave a working, labelled manual play affordance.
  const blocked = await browser.newContext();
  await blocked.addInitScript(() => { HTMLMediaElement.prototype.play = () => Promise.reject(new DOMException('Blocked', 'NotAllowedError')); });
  const blockedPage = await blocked.newPage();
  await blockedPage.goto(base);
  await blockedPage.waitForSelector('.card');
  await blockedPage.locator('.card').first().scrollIntoViewIfNeeded();
  await blockedPage.waitForFunction(() => document.querySelector('.media-status')?.textContent.includes('Select Play'));
  assert(await blockedPage.locator('.start').first().isVisible());
  await blockedPage.locator('.start').first().click();

  // A missing manifest can be retried, rather than leaving a blank collection.
  const failure = await context.newPage();
  await failure.route('**/animations.json', route => route.fulfill({ status: 503, body: 'unavailable' }));
  await failure.goto(base);
  await failure.waitForFunction(() => document.querySelector('#load-more').textContent === 'Retry loading');
  await failure.unroute('**/animations.json');
  await failure.locator('#load-more').click();
  await failure.waitForFunction(() => document.querySelectorAll('.card').length === 12);
  assert.deepEqual(errors, []);
  fs.writeFileSync('build/gallery-qa/browser-report.json', JSON.stringify({ passed: true, checks: ['muted inline autoplay', 'one player initially', 'manual pause', 'autoplay toggle', '12-entry pagination', 'loaded-topic filter', 'deep link beyond first page', 'GitHub Pages subpath', 'reduced-motion no-video-fetch', 'keyboard play', 'mobile overflow', 'blocked-autoplay fallback', 'manifest retry', 'no JS errors'] }, null, 2));
  await browser.close();
  console.log('Gallery browser checks passed. Screenshots and report: build/gallery-qa/');
})().catch(error => { console.error(error); process.exit(1); });
