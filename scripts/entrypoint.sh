#!/bin/sh
set -e

BOOT_START=$(date +%s)

echo "[entrypoint] OpenClaw HuggingFace Spaces Entrypoint"
echo "[entrypoint] ======================================="

# ── DNS pre-resolution (background — non-blocking) ───────────────────────
# Resolves WhatsApp domains via DoH for dns-fix.cjs to consume.
# Telegram connectivity is handled by API base auto-probe in sync_hf.py.
echo "[entrypoint] Starting DNS resolution in background..."
python3 /home/node/scripts/dns-resolve.py /tmp/dns-resolved.json 2>&1 &
DNS_PID=$!
echo "[entrypoint] DNS resolver PID: $DNS_PID"

# ── Node.js memory limit (only if explicitly set) ─────────────────────────
if [ -n "$NODE_MEMORY_LIMIT" ]; then
  export NODE_OPTIONS="${NODE_OPTIONS:+$NODE_OPTIONS }--max-old-space-size=$NODE_MEMORY_LIMIT"
  echo "[entrypoint] Node.js memory limit: ${NODE_MEMORY_LIMIT}MB"
fi

# Enable Node.js DNS fix (will use resolved file when ready)
export NODE_OPTIONS="${NODE_OPTIONS:+$NODE_OPTIONS }--require /home/node/scripts/dns-fix.cjs"

# Enable Telegram API proxy (redirects fetch() to working mirror if needed)
export NODE_OPTIONS="${NODE_OPTIONS:+$NODE_OPTIONS }--require /home/node/scripts/telegram-proxy.cjs"

# token-redirect.cjs is no longer needed — the Express proxy (proxy.cjs)
# handles token injection for /admin and serves the animation at /
# export NODE_OPTIONS="${NODE_OPTIONS:+$NODE_OPTIONS }--require /home/node/scripts/token-redirect.cjs"

# ── Extensions symlink ──────────────────────────────────────────────────────
SYMLINK_START=$(date +%s)
if [ ! -L /home/node/.openclaw/extensions ]; then
  rm -rf /home/node/.openclaw/extensions 2>/dev/null || true
  ln -s /app/openclaw/extensions /home/node/.openclaw/extensions
  echo "[entrypoint] Created extensions symlink -> /app/openclaw/extensions"
fi
echo "[TIMER] Extensions symlink: $(($(date +%s) - SYMLINK_START))s"

# ── WhatsApp credentials check ──────────────────────────────────────────────
if [ -d /home/node/.openclaw/credentials/whatsapp ]; then
  echo "[entrypoint] Found existing WhatsApp credentials - will use for auto-connect"
fi

# ── Build artifacts check ───────────────────────────────────────────────────
cd /app/openclaw
echo "[entrypoint] Build artifacts check:"
test -f dist/entry.js && echo "  OK dist/entry.js" || echo "  WARNING: dist/entry.js missing!"
test -f dist/plugin-sdk/index.js && echo "  OK dist/plugin-sdk/index.js" || echo "  WARNING: dist/plugin-sdk/index.js missing!"
echo "  Extensions: $(ls extensions/ 2>/dev/null | wc -l | tr -d ' ') found"
echo "  Global extensions link: $(readlink /home/node/.openclaw/extensions 2>/dev/null || echo 'NOT SET')"

# Create logs directory
mkdir -p /home/node/logs
touch /home/node/logs/app.log

ENTRYPOINT_END=$(date +%s)
echo "[TIMER] Entrypoint (before sync_hf.py): $((ENTRYPOINT_END - BOOT_START))s"

# ── Set version from build artifact ────────────────────────────────────────
if [ -f /app/openclaw/.version ]; then
  export OPENCLAW_VERSION=$(cat /app/openclaw/.version)
  echo "[entrypoint] OpenClaw version: $OPENCLAW_VERSION"
fi

# ── Start OpenClaw via sync_hf.py ─────────────────────────────────────────
echo "[entrypoint] Starting OpenClaw via sync_hf.py..."
exec python3 -u /home/node/scripts/sync_hf.py
