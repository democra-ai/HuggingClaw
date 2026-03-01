---
title: HuggingClaw
emoji: 🔥
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
license: mit
short_description: Deploy OpenClaw on HuggingFace Spaces
app_port: 7860
tags:
  - huggingface
  - huggingface-dataset
  - openrouter
  - chatbot
  - llm
  - openclaw
  - ai-assistant
  - whatsapp
  - telegram
  - text-generation
  - openai-api
  - huggingface-spaces
  - docker
  - deployment
  - persistent-storage
  - agents
  - multi-channel
  - openai-compatible
  - free-tier
  - one-click-deploy
  - self-hosted
  - messaging-bot
---

<div align="center">
  <img src="HuggingClaw.png" alt="HuggingClaw" width="720"/>
  <br/><br/>
  <strong>The best way to deploy <a href="https://github.com/openclaw/openclaw">OpenClaw</a> on the cloud</strong>
  <br/>
  <sub>Zero hardware · Always online · Auto-persistent · One-click deploy</sub>
  <br/><br/>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Hugging Face](https://img.shields.io/badge/🤗-Hugging%20Face-yellow)](https://huggingface.co)
  [![HF Spaces](https://img.shields.io/badge/Spaces-HuggingFace-blue)](https://huggingface.co/spaces/tao-shen/HuggingClaw)
  [![OpenClaw](https://img.shields.io/badge/OpenClaw-Powered-orange)](https://github.com/openclaw/openclaw)
  [![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
  [![OpenAI Compatible](https://img.shields.io/badge/OpenAI--compatible-API-green)](https://openclawdoc.com/docs/reference/environment-variables)
  [![WhatsApp](https://img.shields.io/badge/WhatsApp-Enabled-25D366?logo=whatsapp)](https://www.whatsapp.com/)
  [![Telegram](https://img.shields.io/badge/Telegram-Enabled-26A5E4?logo=telegram)](https://telegram.org/)
  [![Free Tier](https://img.shields.io/badge/Free%20Tier-16GB%20RAM-brightgreen)](https://huggingface.co/spaces)
</div>

---

## Why HuggingClaw?

[OpenClaw](https://github.com/openclaw/openclaw) is a powerful, popular AI assistant (Telegram, WhatsApp, 40+ channels), but it’s meant to run on your own machine (e.g. a Mac Mini). Not everyone has that. You can deploy on the cloud, but most providers either charge by the hour or offer only very limited resources. **HuggingFace Spaces** gives you 2 vCPU and **16 GB RAM** for free — a good fit for OpenClaw, but Spaces have two problems we fix.

**HuggingClaw** is this repo. It fixes two Hugging Face Space issues: **(1) Data is not persistent** — we use a private **HuggingFace Dataset** to sync and restore your conversations, settings, and credentials so they survive restarts; **(2) DNS resolution fails** for some domains (e.g. WhatsApp) — we fix it with DNS-over-HTTPS and a Node.js DNS patch so OpenClaw can connect reliably.

## Architecture

<div align="center">
  <img src="assets/architecture.svg" alt="Architecture" width="720"/>
</div>

## Quick Start

### 1. Duplicate this Space

Click **Duplicate this Space** on the [HuggingClaw Space page](https://huggingface.co/spaces/tao-shen/HuggingClaw).

### 2. Set Secrets

Go to **Settings → Repository secrets** and configure:

| Secret | Status | Description | Example |
|--------|:------:|-------------|---------|
| `OPENCLAW_PASSWORD` | Recommended | Password for the Control UI (default: `huggingclaw`) | `my-secret-password` |
| `HF_TOKEN` | **Required** | HF Access Token with write permission ([create one](https://huggingface.co/settings/tokens)) | `hf_AbCdEfGhIjKlMnOpQrStUvWxYz` |
| `OPENCLAW_DATASET_REPO` | **Required** | Dataset repo for backup — format: `username/repo-name` | `tao-shen/HuggingClaw-data` |
| `OPENAI_API_KEY` | Recommended | OpenAI (or any [OpenAI-compatible](https://openclawdoc.com/docs/reference/environment-variables)) API key | `sk-proj-xxxxxxxxxxxx` |
| `OPENROUTER_API_KEY` | Optional | [OpenRouter](https://openrouter.ai) API key (200+ models, free tier available) | `sk-or-v1-xxxxxxxxxxxx` |
| `ANTHROPIC_API_KEY` | Optional | Anthropic Claude API key | `sk-ant-xxxxxxxxxxxx` |
| `GOOGLE_API_KEY` | Optional | Google / Gemini API key | `AIzaSyXxXxXxXxXx` |
| `OPENCLAW_DEFAULT_MODEL` | Optional | Default model for new conversations | `openrouter/openai/gpt-oss-20b:free` |

### Environment Variables

In addition to the secrets above, HuggingClaw provides environment variables to fine-tune persistence and performance. Set these the same way — as **Repository Secrets** in HF Spaces, or in your `.env` file for local Docker.

| Variable | Default | Description |
|----------|---------|-------------|
| `AUTO_CREATE_DATASET` | `true` | **Auto-create the Dataset repo** if it doesn't exist. When set to `true`, HuggingClaw automatically creates a **private** HuggingFace Dataset repo (using the name from `OPENCLAW_DATASET_REPO`) on first startup. Set to `false` if you prefer to [create the repo manually](https://huggingface.co/new-dataset) before deploying. Accepted values: `true`, `1`, `yes` (enabled) / `false`, `0`, `no` (disabled). |
| `SYNC_INTERVAL` | `60` | **Backup interval in seconds.** How often HuggingClaw syncs the `~/.openclaw` directory (conversations, settings, credentials) to the HuggingFace Dataset repo. Lower values mean less data loss on restart but more API calls. Recommended: `60`–`300`. |
| `NODE_MEMORY_LIMIT` | `512` | **Node.js heap memory limit in MB.** HF free tier provides 16 GB RAM; the default 512 MB is enough for most cases. Increase if you run complex agent workflows or handle very large conversations. |
| `TZ` | `UTC` | **Timezone** for log timestamps and scheduled tasks. Example: `Asia/Shanghai`, `America/New_York`. |

> For the full list of environment variables (including `OPENAI_BASE_URL`, `OLLAMA_HOST`, proxy settings, and more), see [`.env.example`](.env.example).

### 3. Open the Control UI

Visit your Space URL. Click the settings icon, enter your password, and connect.

Messaging integrations (Telegram, WhatsApp) can be configured directly inside the Control UI after connecting.

## Configuration

HuggingClaw supports **all OpenClaw environment variables** — it passes the entire environment to the OpenClaw process (`env=os.environ.copy()`), so any variable from the [OpenClaw docs](https://openclawdoc.com/docs/reference/environment-variables) works out of the box in HF Spaces. This includes:

- **API Keys** — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `OPENROUTER_API_KEY`
- **Server** — `OPENCLAW_API_PORT`, `OPENCLAW_WS_PORT`, `OPENCLAW_HOST`
- **Memory** — `OPENCLAW_MEMORY_BACKEND`, `OPENCLAW_REDIS_URL`, `OPENCLAW_SQLITE_PATH`
- **Network** — `OPENCLAW_HTTP_PROXY`, `OPENCLAW_HTTPS_PROXY`, `OPENCLAW_NO_PROXY`
- **Ollama** — `OLLAMA_HOST`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_KEEP_ALIVE`
- **Secrets** — `OPENCLAW_SECRETS_BACKEND`, `VAULT_ADDR`, `VAULT_TOKEN`

HuggingClaw adds its own variables for persistence and deployment: `HF_TOKEN`, `OPENCLAW_DATASET_REPO`, `AUTO_CREATE_DATASET`, `SYNC_INTERVAL`, `OPENCLAW_PASSWORD`, `OPENCLAW_DEFAULT_MODEL`, etc. See [`.env.example`](.env.example) for the complete reference.

## Security

- **Password-protected** — the Control UI requires a password to connect and manage the instance
- **Secrets stay server-side** — API keys and tokens are never exposed to the browser
- **Private backups** — the Dataset repo is created as private by default

> **Tip:** Change the default password from `huggingclaw` to something unique by setting the `OPENCLAW_PASSWORD` secret.

## License

MIT
