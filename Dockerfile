# OpenClaw on Hugging Face Spaces — Pre-built image (v4.1)
# Uses official pre-built image to avoid 30+ minute builds on cpu-basic

# ── Stage 1: Pull pre-built OpenClaw ─────────────────────────────────────────
FROM ghcr.io/openclaw/openclaw:latest AS openclaw-prebuilt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM node:22-bookworm
SHELL ["/bin/bash", "-c"]

# ── System dependencies (root) ───────────────────────────────────────────────
RUN echo "[build] Installing system deps..." && START=$(date +%s) \
  && apt-get update \
  && apt-get install -y --no-install-recommends git ca-certificates curl python3 python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && pip3 install --no-cache-dir --break-system-packages huggingface_hub \
  && corepack enable \
  && mkdir -p /app/openclaw \
  && chown -R node:node /app \
  && mkdir -p /home/node/.openclaw/workspace /home/node/.openclaw/credentials \
  && chown -R node:node /home/node \
  && echo "[build] System deps: $(($(date +%s) - START))s"

# ── Copy pre-built OpenClaw (skips clone + install + build entirely) ─────────
COPY --from=openclaw-prebuilt --chown=node:node /app /app/openclaw

USER node
ENV HOME=/home/node
WORKDIR /app

# ── A2A Gateway Extension ───────────────────────────────────────────────────
RUN echo "[build] Installing A2A gateway..." && START=$(date +%s) \
  && git clone --depth 1 https://github.com/win4r/openclaw-a2a-gateway.git /app/openclaw/extensions/a2a-gateway \
  && cd /app/openclaw/extensions/a2a-gateway \
  && npm install --production \
  && echo "[build] A2A gateway: $(($(date +%s) - START))s"

# ── Prepare runtime dirs ────────────────────────────────────────────────────
RUN mkdir -p /app/openclaw/empty-bundled-plugins \
  && node -e "try{console.log(require('/app/openclaw/package.json').version)}catch(e){console.log('unknown')}" > /app/openclaw/.version \
  && echo "[build] OpenClaw version: $(cat /app/openclaw/.version)"

# ── Scripts + Config + Frontend ──────────────────────────────────────────────
COPY --chown=node:node scripts /home/node/scripts
COPY --chown=node:node frontend /home/node/frontend
COPY --chown=node:node openclaw.json /home/node/scripts/openclaw.json.default
RUN chmod +x /home/node/scripts/entrypoint.sh /home/node/scripts/sync_hf.py \
  && VERSION_TS=$(date +%s) \
  && sed "s/{{VERSION_TIMESTAMP}}/${VERSION_TS}/g" /home/node/frontend/electron-standalone.html > /home/node/frontend/index.html \
  && echo "[build] Frontend index.html generated (timestamp=${VERSION_TS})"

ENV NODE_ENV=production
ENV OPENCLAW_BUNDLED_PLUGINS_DIR=/app/openclaw/empty-bundled-plugins
ENV OPENCLAW_PREFER_PNPM=1
ENV PATH="/home/node/.local/bin:$PATH"
WORKDIR /home/node

CMD ["/home/node/scripts/entrypoint.sh"]
