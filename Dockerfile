# OpenClaw on Hugging Face Spaces — 从源码构建（带计时）
# 文档: https://huggingface.co/docs/hub/spaces-sdks-docker

FROM node:22-bookworm
SHELL ["/bin/bash", "-c"]

# ── Step 1: System dependencies ──────────────────────────────────────────────
RUN echo "[build][step1] Installing system deps..." && START=$(date +%s) \
  && apt-get update && apt-get install -y --no-install-recommends git ca-certificates curl python3 python3-pip \
  && rm -rf /var/lib/apt/lists/* \
  && echo "[build][step1] System deps: $(($(date +%s) - START))s"

# ── Step 2: Python dependencies ──────────────────────────────────────────────
RUN echo "[build][step2] Installing huggingface_hub..." && START=$(date +%s) \
  && pip3 install --no-cache-dir --break-system-packages huggingface_hub \
  && echo "[build][step2] huggingface_hub: $(($(date +%s) - START))s"

# ── Step 3: Node tooling ────────────────────────────────────────────────────
RUN echo "[build][step3] Enabling corepack + bun..." && START=$(date +%s) \
  && corepack enable \
  && curl -fsSL https://bun.sh/install | bash \
  && echo "[build][step3] Corepack + Bun: $(($(date +%s) - START))s"
ENV PATH="/root/.bun/bin:${PATH}"

# ── Step 4: Clone OpenClaw ───────────────────────────────────────────────────
WORKDIR /app
RUN echo "[build][step4] Cloning OpenClaw..." && START=$(date +%s) \
  && git clone --depth 1 https://github.com/openclaw/openclaw.git openclaw \
  && echo "[build][step4] Git clone: $(($(date +%s) - START))s"
WORKDIR /app/openclaw

# ── Step 5: Apply patches ───────────────────────────────────────────────────
COPY patches /app/patches
RUN echo "[build][step5] Applying patches..." && START=$(date +%s) \
  && if [ -f /app/patches/web-inbound-record-activity-after-body.patch ]; then \
       patch -p1 < /app/patches/web-inbound-record-activity-after-body.patch; \
     fi \
  && echo "[build][step5] Patches: $(($(date +%s) - START))s"

# ── Step 6: pnpm install ────────────────────────────────────────────────────
RUN echo "[build][step6] pnpm install..." && START=$(date +%s) \
  && pnpm install --frozen-lockfile \
  && echo "[build][step6] pnpm install: $(($(date +%s) - START))s"

# ── Step 7: pnpm build ──────────────────────────────────────────────────────
RUN echo "[build][step7] pnpm build..." && START=$(date +%s) \
  && pnpm build \
  && echo "[build][step7] pnpm build: $(($(date +%s) - START))s"

# ── Step 8: pnpm ui:build ───────────────────────────────────────────────────
ENV OPENCLAW_PREFER_PNPM=1
RUN echo "[build][step8] pnpm ui:build..." && START=$(date +%s) \
  && pnpm ui:build \
  && echo "[build][step8] pnpm ui:build: $(($(date +%s) - START))s"

# ── Step 9: Verify build artifacts ──────────────────────────────────────────
RUN echo "[build][step9] Verifying build artifacts..." \
  && test -f dist/entry.js && echo "  OK dist/entry.js" \
  && test -f dist/plugin-sdk/index.js && echo "  OK dist/plugin-sdk/index.js" \
  && test -d extensions/telegram && echo "  OK extensions/telegram" \
  && test -d extensions/whatsapp && echo "  OK extensions/whatsapp" \
  && test -d dist/control-ui && echo "  OK dist/control-ui"

# ── Step 10: Inject auto-token into Control UI ──────────────────────────────
COPY --chown=node:node scripts /home/node/scripts
RUN chmod +x /home/node/scripts/inject-token.sh && bash /home/node/scripts/inject-token.sh

# ── Step 11: Final setup ────────────────────────────────────────────────────
ENV NODE_ENV=production
RUN mkdir -p /app/openclaw/empty-bundled-plugins
ENV OPENCLAW_BUNDLED_PLUGINS_DIR=/app/openclaw/empty-bundled-plugins
RUN chown -R node:node /app

RUN mkdir -p /home/node/.openclaw/workspace /home/node/.openclaw/credentials

COPY --chown=node:node openclaw.json /home/node/scripts/openclaw.json.default
RUN chmod +x /home/node/scripts/entrypoint.sh \
  && chmod +x /home/node/scripts/sync_hf.py \
  && chown -R node:node /home/node

USER node
ENV HOME=/home/node
ENV PATH="/home/node/.local/bin:$PATH"
WORKDIR /home/node

CMD ["/home/node/scripts/entrypoint.sh"]
