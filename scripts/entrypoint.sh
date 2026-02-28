#!/bin/sh
set -e

BOOT_START=$(date +%s)

echo "[entrypoint] OpenClaw HuggingFace Spaces Entrypoint"
echo "[entrypoint] ======================================="

# ── DNS pre-resolution (run in BACKGROUND — was 121s blocking) ──────────────
echo "[entrypoint] Resolving WhatsApp domains via DNS-over-HTTPS (background)..."
DNS_START=$(date +%s)
(
  python3 /home/node/scripts/dns-resolve.py /tmp/dns-resolved.json 2>&1
  DNS_END=$(date +%s)
  echo "[TIMER] DNS pre-resolve (background): $((DNS_END - DNS_START))s"
) &
DNS_PID=$!

# Enable Node.js DNS fix (will use resolved file when ready)
export NODE_OPTIONS="${NODE_OPTIONS:+$NODE_OPTIONS }--require /home/node/scripts/dns-fix.cjs"

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

# ── Start OpenClaw via sync_hf.py (don't wait for DNS — it runs in bg) ─────
echo "[entrypoint] Starting OpenClaw via sync_hf.py..."
echo "[entrypoint] DNS resolution running in background (PID $DNS_PID), app will use it when ready"
exec python3 -u /home/node/scripts/sync_hf.py
