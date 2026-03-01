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
| `HF_TOKEN` | **Required** | HF Access Token with write permission ([create one](https://huggingface.co/settings/tokens)) | `hf_AbCdEfGhIjKlMnOpQrStUvWxYz` |
| `OPENCLAW_DATASET_REPO` | See below | Dataset repo for backup — format: `username/repo-name`. Required in manual mode; optional in auto mode (see [Data Persistence](#data-persistence)) | `tao-shen/HuggingClaw-data` |
| `OPENAI_API_KEY` | Recommended | OpenAI (or any [OpenAI-compatible](https://openclawdoc.com/docs/reference/environment-variables)) API key | `sk-proj-xxxxxxxxxxxx` |
| `OPENROUTER_API_KEY` | Optional | [OpenRouter](https://openrouter.ai) API key (200+ models, free tier available) | `sk-or-v1-xxxxxxxxxxxx` |
| `ANTHROPIC_API_KEY` | Optional | Anthropic Claude API key | `sk-ant-xxxxxxxxxxxx` |
| `GOOGLE_API_KEY` | Optional | Google / Gemini API key | `AIzaSyXxXxXxXxXx` |
| `OPENCLAW_DEFAULT_MODEL` | Optional | Default model for new conversations | `openai/gpt-oss-20b:free` |

### Data Persistence

HuggingClaw syncs `~/.openclaw` (conversations, settings, credentials) to a private HuggingFace Dataset repo so data survives restarts. There are two ways to set this up:

**Option A — Manual mode (default, recommended)**

1. Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset) and create a **private** Dataset repo (e.g. `your-name/HuggingClaw-data`)
2. Set `OPENCLAW_DATASET_REPO` = `your-name/HuggingClaw-data` in your Space secrets
3. Set `HF_TOKEN` with write permission
4. Done — HuggingClaw will sync to this repo every 60 seconds

**Option B — Auto mode**

1. Set `AUTO_CREATE_DATASET` = `true` in your Space secrets
2. Set `HF_TOKEN` with write permission
3. (Optional) Set `OPENCLAW_DATASET_REPO` if you want a custom repo name
4. On first startup, HuggingClaw automatically creates a **private** Dataset repo. If `OPENCLAW_DATASET_REPO` is not set, it derives the name from your HF username + Space name: `your-username/SpaceName-data` (e.g. `tao-shen/HuggingClaw-data`). Each Space gets its own dataset, so duplicating a Space won't cause conflicts

> **Security note:** `AUTO_CREATE_DATASET` defaults to `false` — the system will not create repos on your behalf unless you explicitly opt in.

### Environment Variables

Fine-tune persistence and performance. Set these as **Repository Secrets** in HF Spaces, or in `.env` for local Docker.

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_TOKEN` | _(none)_ | **Gateway token.** If set, Control UI requires this token to connect. If not set, anyone with the URL can access. |
| `AUTO_CREATE_DATASET` | `false` | **Auto-create the Dataset repo.** Set to `true` to auto-create a private Dataset repo on first startup. |
| `SYNC_INTERVAL` | `60` | **Backup interval in seconds.** How often data syncs to the Dataset repo. |

> For the full list (including `OPENAI_BASE_URL`, `OLLAMA_HOST`, proxy settings, etc.), see [`.env.example`](.env.example).

### 3. Open the Control UI

Visit your Space URL — the Control UI connects automatically, no token needed. To add access control, set the `GATEWAY_TOKEN` secret.

Messaging integrations (Telegram, WhatsApp) can be configured directly inside the Control UI after connecting.

> **Telegram note:** HF Spaces blocks `api.telegram.org` DNS. HuggingClaw automatically probes alternative API endpoints at startup and selects one that works — no manual configuration needed.

## Configuration

HuggingClaw supports **all OpenClaw environment variables** — it passes the entire environment to the OpenClaw process (`env=os.environ.copy()`), so any variable from the [OpenClaw docs](https://openclawdoc.com/docs/reference/environment-variables) works out of the box in HF Spaces. This includes:

- **API Keys** — `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `OPENROUTER_API_KEY`
- **Server** — `OPENCLAW_API_PORT`, `OPENCLAW_WS_PORT`, `OPENCLAW_HOST`
- **Memory** — `OPENCLAW_MEMORY_BACKEND`, `OPENCLAW_REDIS_URL`, `OPENCLAW_SQLITE_PATH`
- **Network** — `OPENCLAW_HTTP_PROXY`, `OPENCLAW_HTTPS_PROXY`, `OPENCLAW_NO_PROXY`
- **Ollama** — `OLLAMA_HOST`, `OLLAMA_NUM_PARALLEL`, `OLLAMA_KEEP_ALIVE`
- **Secrets** — `OPENCLAW_SECRETS_BACKEND`, `VAULT_ADDR`, `VAULT_TOKEN`

HuggingClaw adds its own variables for persistence and deployment: `HF_TOKEN`, `OPENCLAW_DATASET_REPO`, `AUTO_CREATE_DATASET`, `SYNC_INTERVAL`, `OPENCLAW_DEFAULT_MODEL`, etc. See [`.env.example`](.env.example) for the complete reference.

## Security

- **Optional token auth** — set `GATEWAY_TOKEN` to restrict Control UI access; without it, the UI is open for easy setup
- **Secrets stay server-side** — API keys and tokens are never exposed to the browser
- **Private backups** — the Dataset repo is created as private by default

## License

MIT
