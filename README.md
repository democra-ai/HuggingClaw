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
  [![HF Spaces](https://img.shields.io/badge/%F0%9F%A4%97-HuggingFace%20Spaces-blue)](https://huggingface.co/spaces/tao-shen/HuggingClaw)
</div>

---

## Why HuggingClaw?

[OpenClaw](https://github.com/openclaw/openclaw) is a powerful self-hosted AI assistant platform supporting Telegram, WhatsApp, and 40+ messaging channels. But running it requires a server — and not everyone has a Mac Mini or spare hardware sitting around.

Cloud hosting is an option, but most providers charge by the hour, and OpenClaw's Node.js runtime needs real resources (it runs comfortably at ~16 GB RAM). That rules out most free tiers.

**HuggingFace Spaces** offers the perfect fit: free CPU instances with 2 vCPU and 16 GB RAM, always-on HTTPS, and zero maintenance. There's just one catch — Space containers are **ephemeral**. Every restart wipes your data.

**HuggingClaw** solves this by using a private **HuggingFace Dataset** repo as persistent storage. Your conversations, credentials, and settings are automatically synced on a schedule, and restored on every cold start. You get all the benefits of free cloud hosting with none of the data loss.

| Feature | Detail |
|---------|--------|
| **Free compute** | Runs on HF's free CPU tier (2 vCPU, 16 GB RAM) |
| **Always online** | No need to keep your computer running |
| **Auto-persistent** | Syncs data to a private HF Dataset repo on schedule |
| **HTTPS built-in** | Secure WebSocket connections out of the box |
| **One-click deploy** | Fork → set secrets → done |

## Architecture

<div align="center">
  <img src="assets/architecture.svg" alt="Architecture" width="720"/>
</div>

## Quick Start

### 1. Duplicate this Space

Click **Duplicate this Space** on the [HuggingClaw Space page](https://huggingface.co/spaces/tao-shen/HuggingClaw).

### 2. Set Secrets

Go to **Settings → Repository secrets** and configure:

| Secret | Status | Description |
|--------|:------:|-------------|
| `OPENCLAW_PASSWORD` | Recommended | Password for the Control UI (default: `huggingclaw`) |
| `HF_TOKEN` | **Required** | HF Access Token with write permission ([create one](https://huggingface.co/settings/tokens)) |
| `OPENCLAW_DATASET_REPO` | **Required** | Dataset repo for backup, e.g. `your-name/openclaw-data` |
| `OPENAI_API_KEY` | Recommended | OpenAI (or any [OpenAI-compatible](https://openclawdoc.com/docs/reference/environment-variables)) API key for LLM |
| `OPENROUTER_API_KEY` | Optional | [OpenRouter](https://openrouter.ai) API key (200+ models, free tier) |
| `ANTHROPIC_API_KEY` | Optional | Anthropic Claude API key |
| `GOOGLE_API_KEY` | Optional | Google / Gemini API key |
| `OPENCLAW_DEFAULT_MODEL` | Optional | Default model, e.g. `openai/gpt-4o-mini` or `openrouter/deepseek/deepseek-chat:free` |

> For the full list of environment variables, see [`.env.example`](.env.example).

### 3. Open the Control UI

Visit your Space URL. Click the settings icon, enter your password, and connect.

Messaging integrations (Telegram, WhatsApp) can be configured directly inside the Control UI after connecting.

## Configuration

HuggingClaw is configured entirely through **environment variables**. A fully documented template is provided in [`.env.example`](.env.example).

| Category | Variables | Purpose |
|----------|-----------|---------|
| **Security** | `OPENCLAW_PASSWORD` | Protect the Control UI with a password |
| **Persistence** | `HF_TOKEN`, `OPENCLAW_DATASET_REPO`, `AUTO_CREATE_DATASET`, `SYNC_INTERVAL` | Auto-backup to HF Dataset |
| **LLM (OpenAI-compatible)** | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENROUTER_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `MISTRAL_API_KEY`, `COHERE_API_KEY`, `OPENCLAW_DEFAULT_MODEL` | Power AI conversations ([OpenClaw env reference](https://openclawdoc.com/docs/reference/environment-variables)) |
| **Performance** | `NODE_MEMORY_LIMIT` | Tune Node.js memory usage |
| **Locale** | `TZ` | Set timezone for logs |

For local development, copy the template and fill in your values:

```bash
cp .env.example .env
# Edit .env with your values
```

## Local Development

```bash
git clone https://huggingface.co/spaces/tao-shen/HuggingClaw
cd HuggingClaw

# Configure
cp .env.example .env
# Edit .env — at minimum set HF_TOKEN and OPENCLAW_DATASET_REPO

# Build and run
docker build -t huggingclaw .
docker run --rm -p 7860:7860 --env-file .env huggingclaw
```

Open `http://localhost:7860` in your browser.

## Security

- **Password-protected** — the Control UI requires a password to connect and manage the instance
- **Secrets stay server-side** — API keys and tokens are never exposed to the browser
- **CSP headers** — Content Security Policy restricts script and resource loading
- **Private backups** — the Dataset repo is created as private by default

> **Tip:** Change the default password from `huggingclaw` to something unique by setting the `OPENCLAW_PASSWORD` secret.

## License

MIT
