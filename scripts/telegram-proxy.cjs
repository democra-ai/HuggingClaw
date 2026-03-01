/**
 * Telegram API proxy preload script for HF Spaces.
 *
 * HF Spaces blocks DNS for api.telegram.org. This script intercepts
 * globalThis.fetch() calls and redirects api.telegram.org requests
 * to a working mirror (set via TELEGRAM_API_ROOT env var).
 *
 * This works because grammY (OpenClaw's Telegram library) uses Node 22's
 * built-in fetch (undici), which bypasses dns.lookup monkey-patching.
 * Intercepting at the fetch level is the only reliable approach.
 *
 * Loaded via: NODE_OPTIONS="--require /path/to/telegram-proxy.cjs"
 */
"use strict";

const TELEGRAM_API_ROOT = process.env.TELEGRAM_API_ROOT;
const OFFICIAL = "https://api.telegram.org/";

if (TELEGRAM_API_ROOT && TELEGRAM_API_ROOT.replace(/\/+$/, "") !== "https://api.telegram.org") {
  const mirror = TELEGRAM_API_ROOT.replace(/\/+$/, "") + "/";
  const mirrorHost = (() => {
    try { return new URL(mirror).hostname; } catch { return mirror; }
  })();

  const originalFetch = globalThis.fetch;
  let logged = false;

  globalThis.fetch = function patchedFetch(input, init) {
    let url;

    if (typeof input === "string") {
      url = input;
    } else if (input instanceof URL) {
      url = input.toString();
    } else if (input && typeof input === "object" && input.url) {
      url = input.url;
    }

    if (url && url.startsWith(OFFICIAL)) {
      const newUrl = mirror + url.slice(OFFICIAL.length);
      if (!logged) {
        console.log(`[telegram-proxy] Redirecting api.telegram.org → ${mirrorHost}`);
        logged = true;
      }

      if (typeof input === "string") {
        return originalFetch.call(this, newUrl, init);
      }
      // For Request objects, create a new one with the redirected URL
      if (input instanceof Request) {
        const newReq = new Request(newUrl, input);
        return originalFetch.call(this, newReq, init);
      }
      return originalFetch.call(this, newUrl, init);
    }

    return originalFetch.call(this, input, init);
  };

  console.log(`[telegram-proxy] Loaded: api.telegram.org → ${mirrorHost}`);
}
