# OpenClaw on Hugging Face Spaces — 优化构建（v2）
# 优化点：node 用户构建（消除 chown）、合并 RUN 层（减少层开销）
FROM node:22-bookworm
SHELL ["/bin/bash", "-c"]

# ── Layer 1 (root): 系统依赖 + 工具（全部合并为一层）─────────────────────────
RUN echo "[build][layer1] System deps + tools..." && START=$(date +%s) \
  && apt-get update \
  && apt-get install -y --no-install-recommends git ca-certificates curl python3 python3-pip patch \
  && rm -rf /var/lib/apt/lists/* \
  && pip3 install --no-cache-dir --break-system-packages huggingface_hub \
  && corepack enable \
  && mkdir -p /app \
  && chown node:node /app \
  && mkdir -p /home/node/.openclaw/workspace /home/node/.openclaw/credentials \
  && chown -R node:node /home/node \
  && echo "[build][layer1] System deps + tools: $(($(date +%s) - START))s"

# ── 切换到 node 用户（后续所有操作都以 node 身份，无需 chown）───────────────
USER node
ENV HOME=/home/node
WORKDIR /app

# ── Layer 2 (node): Clone + Patch + Install + Build（合并为一层）─────────────
COPY --chown=node:node patches /app/patches
RUN echo "[build][layer2] Clone + install + build..." && START=$(date +%s) \
  && git clone --depth 1 https://github.com/openclaw/openclaw.git openclaw \
  && echo "[build] git clone: $(($(date +%s) - START))s" \
  && cd openclaw \
  && for p in /app/patches/*.patch; do \
       if [ -f "$p" ]; then \
         patch -p1 < "$p" \
         && echo "[build] patch applied: $(basename $p)"; \
       fi; \
     done \
  && T1=$(date +%s) \
  && pnpm install --frozen-lockfile \
  && echo "[build] pnpm install: $(($(date +%s) - T1))s" \
  && T2=$(date +%s) \
  && pnpm build \
  && echo "[build] pnpm build: $(($(date +%s) - T2))s" \
  && T3=$(date +%s) \
  && OPENCLAW_PREFER_PNPM=1 pnpm ui:build \
  && echo "[build] pnpm ui:build: $(($(date +%s) - T3))s" \
  && test -f dist/entry.js && echo "[build] OK dist/entry.js" \
  && test -f dist/plugin-sdk/index.js && echo "[build] OK dist/plugin-sdk/index.js" \
  && test -d extensions/telegram && echo "[build] OK extensions/telegram" \
  && test -d extensions/whatsapp && echo "[build] OK extensions/whatsapp" \
  && test -d dist/control-ui && echo "[build] OK dist/control-ui" \
  && mkdir -p /app/openclaw/empty-bundled-plugins \
  && node -e "console.log(require('./package.json').version)" > /app/openclaw/.version \
  && echo "[build] version: $(cat /app/openclaw/.version)" \
  && echo "[build][layer2] Total clone+install+build: $(($(date +%s) - START))s"

# ── Layer 3 (node): Scripts + Config ──────────────────────────────────────────
COPY --chown=node:node scripts /home/node/scripts
COPY --chown=node:node openclaw.json /home/node/scripts/openclaw.json.default
RUN chmod +x /home/node/scripts/entrypoint.sh /home/node/scripts/sync_hf.py

ENV NODE_ENV=production
ENV OPENCLAW_BUNDLED_PLUGINS_DIR=/app/openclaw/empty-bundled-plugins
ENV OPENCLAW_PREFER_PNPM=1
ENV PATH="/home/node/.local/bin:$PATH"
WORKDIR /home/node

CMD ["/home/node/scripts/entrypoint.sh"]
