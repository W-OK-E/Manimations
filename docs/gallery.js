"use strict";

const gallery = document.querySelector("#gallery");
const status = document.querySelector("#status");
const more = document.querySelector("#load-more");
const topic = document.querySelector("#topic");
const autoplayButton = document.querySelector("#autoplay");
const reducedMotion = matchMedia("(prefers-reduced-motion: reduce)");
const cards = new Map();
const visible = new Map();
let autoplay = !reducedMotion.matches && !navigator.connection?.saveData;
let nextPage = "animations.json";
let loading = false;
let current = null;
let preferred = null;

// Only allow relative files from this static gallery, including under /Manimations/.
function localURL(path) {
  const base = new URL("./", location.href);
  const url = new URL(path, base);
  if (url.origin !== base.origin || !url.pathname.startsWith(base.pathname)) {
    throw new Error("Invalid gallery asset path");
  }
  return url.href;
}

async function readJSON(path) {
  const response = await fetch(localURL(path));
  if (!response.ok) throw new Error(`Could not load ${path}`);
  return response.json();
}

function updateAutoplay() {
  autoplayButton.setAttribute("aria-pressed", String(autoplay));
  autoplayButton.textContent = `Autoplay: ${autoplay ? "on" : "off"}`;
  document.querySelector("#playback-note").textContent = autoplay
    ? "Visible clips play automatically, muted. Use the player controls to pause or enable sound."
    : "Autoplay is off. Select Play to watch a clip. Reduced-motion and data-saving preferences are respected.";
}

function pauseOthers(except) {
  for (const card of cards.values()) {
    if (card !== except && card.video && !card.video.paused) card.video.pause();
  }
}

async function play(card, automatic = false) {
  if (automatic && (!autoplay || document.hidden || card.suppressed || card.ended)) return;
  if (!automatic) preferred = card;
  if (!card.video) {
    const video = document.createElement("video");
    video.width = 1280;
    video.height = 720;
    video.controls = true;
    video.muted = true;
    video.defaultMuted = true;
    video.playsInline = true;
    video.preload = "none";
    video.poster = localURL(card.item.poster);
    video.setAttribute("aria-label", card.item.title);
    video.setAttribute("aria-describedby", `${card.item.slug}-description`);
    video.addEventListener("play", () => {
      if (current !== card) preferred = card;
      pauseOthers(card);
      current = card;
      card.start.hidden = true;
      card.message.textContent = "";
    });
    video.addEventListener("ended", () => { card.ended = true; });
    video.addEventListener("pause", () => {
      // A native-control pause while still in view must stay paused.
      if (current === card && (visible.get(card) || 0) >= 0.25 && !document.hidden) {
        card.suppressed = true;
      }
    });
    video.addEventListener("error", () => {
      card.start.hidden = false;
      card.start.textContent = "Retry video";
      card.message.textContent = "This video could not be loaded. Try again.";
    });
    video.src = localURL(card.item.video);
    card.video = video;
    card.media.append(video);
  }
  pauseOthers(card);
  current = card;
  if (card.video.error) card.video.load();
  try {
    await card.video.play();
  } catch {
    card.start.hidden = false;
    card.message.textContent = "Select Play to start this video.";
  }
}

function chooseVisible() {
  if (document.hidden || !autoplay) return;
  const candidates = [...visible.entries()]
    .filter(([card, ratio]) => ratio >= 0.25 && !card.element.hidden)
    .sort((a, b) => b[1] - a[1]);
  const selected = candidates.find(([card]) => card === preferred)?.[0] || candidates[0]?.[0];
  if (current && current !== selected) {
    const previous = current;
    current = null;
    previous.video?.pause();
  }
  if (selected && (!selected.video || selected.video.paused)) play(selected, true);
}

const observer = "IntersectionObserver" in window ? new IntersectionObserver(entries => {
  for (const entry of entries) {
    const card = cards.get(entry.target.closest(".card").id);
    visible.set(card, entry.intersectionRatio);
    if (!entry.isIntersecting) card.suppressed = false;
  }
  if (current && (visible.get(current) || 0) < 0.25) {
    const previous = current;
    current = null;
    previous.video?.pause();
  }
  chooseVisible();
}, { threshold: [0, 0.25, 0.6, 1] }) : null;

function render(item) {
  if (cards.has(item.slug)) return cards.get(item.slug);
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(item.slug)) throw new Error("Invalid animation slug");
  const element = document.createElement("article");
  element.className = "card";
  element.id = item.slug;
  // This template contains only fixed markup. Manifest text uses textContent.
  element.innerHTML = '<div class="media"><img width="1280" height="720" loading="lazy" alt=""><button class="start" type="button">Play</button></div><div class="card-meta"><span class="tags"></span><span class="duration"></span></div><h3><a></a></h3><p class="description"></p><p class="media-status" aria-live="polite"></p>';
  const media = element.querySelector(".media");
  const image = element.querySelector("img");
  image.src = localURL(item.poster);
  const title = element.querySelector("h3 a");
  title.textContent = item.title;
  title.href = `#${item.slug}`;
  element.querySelector(".tags").textContent = item.tags.join(" · ");
  const duration = Math.round(item.duration);
  element.querySelector(".duration").textContent = `${Math.floor(duration / 60)}:${String(duration % 60).padStart(2, "0")}`;
  const description = element.querySelector(".description");
  description.id = `${item.slug}-description`;
  description.textContent = item.description;
  const card = { item, element, media, start: element.querySelector(".start"), message: element.querySelector(".media-status"), video: null, suppressed: false, ended: false };
  card.start.setAttribute("aria-label", `Play ${item.title}`);
  card.start.addEventListener("click", () => {
    card.suppressed = false;
    card.ended = false;
    play(card);
  });
  cards.set(item.slug, card);
  gallery.append(element);
  for (const tag of item.tags) {
    if (![...topic.options].some(option => option.value === tag)) {
      topic.add(new Option(tag, tag));
    }
  }
  observer?.observe(media);
  return card;
}

function filter() {
  let count = 0;
  for (const card of cards.values()) {
    card.element.hidden = Boolean(topic.value && !card.item.tags.includes(topic.value));
    if (card.element.hidden) {
      if (current === card) current = null;
      card.video?.pause();
    } else count++;
  }
  status.textContent = cards.size
    ? `${count} ${count === 1 ? "animation" : "animations"} shown${nextPage ? " · more available below" : ""}.`
    : "The collection is just getting started. New animations will appear here.";
  chooseVisible();
}

async function loadPage() {
  if (loading || !nextPage) return;
  loading = true;
  more.disabled = true;
  try {
    const page = await readJSON(nextPage);
    for (const item of page.items) render(item);
    nextPage = page.next;
    more.hidden = !nextPage;
    more.textContent = "Load more animations";
    filter();
  } catch {
    status.textContent = "The collection could not be loaded. Please try again.";
    more.hidden = false;
    more.textContent = "Retry loading";
  } finally {
    loading = false;
    more.disabled = false;
  }
}

async function deepLink() {
  const slug = location.hash.slice(1);
  if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) || slug === "collection") return;
  try {
    const card = cards.get(slug) || render(await readJSON(`catalog/${slug}.json`));
    preferred = card;
    topic.value = "";
    filter();
    card.element.scrollIntoView({ behavior: "instant", block: "center" });
  } catch {
    status.textContent = "That animation is not available. Browse the collection below.";
  }
}

autoplayButton.addEventListener("click", () => {
  autoplay = !autoplay;
  updateAutoplay();
  if (!autoplay) { current = null; pauseOthers(null); }
  else { for (const card of cards.values()) card.suppressed = false; chooseVisible(); }
});
reducedMotion.addEventListener("change", () => {
  if (reducedMotion.matches) {
    autoplay = false;
    current = null;
    pauseOthers(null);
    updateAutoplay();
  }
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { current = null; pauseOthers(null); }
  else chooseVisible();
});
topic.addEventListener("change", filter);
more.addEventListener("click", loadPage);
window.addEventListener("hashchange", deepLink);
updateAutoplay();
loadPage().then(deepLink);
