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

| Secret | Status | Description |
|--------|:------:|-------------|
| `OPENCLAW_PASSWORD` | Recommended | Password for the Control UI (default: `huggingclaw`) |
| `HF_TOKEN` | **Required** | HF Access Token with write permission ([create one](https://huggingface.co/settings/tokens)) |
| `OPENCLAW_DATASET_REPO` | **Required** | Dataset repo for backup, e.g. `your-name/HuggingClaw-data` |
| `OPENAI_API_KEY` | Recommended | OpenAI (or any [OpenAI-compatible](https://openclawdoc.com/docs/reference/environment-variables)) API key for LLM |
| `OPENROUTER_API_KEY` | Optional | [OpenRouter](https://openrouter.ai) API key (200+ models, free tier) |
| `ANTHROPIC_API_KEY` | Optional | Anthropic Claude API key |
| `GOOGLE_API_KEY` | Optional | Google / Gemini API key |
| `OPENCLAW_DEFAULT_MODEL` | Optional | Default model, e.g. `openrouter/openai/gpt-oss-20b:free` or `openai/gpt-4o-mini` |

> For the full list of environment variables, see [`.env.example`](.env.example).

### 3. Open the Control UI

Visit your Space URL. Click the settings icon, enter your password, and connect.

Messaging integrations (Telegram, WhatsApp) can be configured directly inside the Control UI after connecting.

## Configuration

HuggingClaw supports **all OpenClaw environment variables** — it passes the entire environment to the OpenClaw process (`env=os.environ.copy()`), so any variable from the [OpenClaw docs](https://openclawdoc.com/docs/reference/environment-variables) (API Keys, Server, Memory, Network, Ollama, Secrets Manager, etc.) works out of the box in HF Spaces.

HuggingClaw adds a few variables of its own for persistence and deployment: `HF_TOKEN`, `OPENCLAW_DATASET_REPO`, `AUTO_CREATE_DATASET`, `SYNC_INTERVAL`, `OPENCLAW_PASSWORD`, `OPENCLAW_DEFAULT_MODEL`, etc. For the full list, see [`.env.example`](.env.example).

## Security

- **Password-protected** — the Control UI requires a password to connect and manage the instance
- **Secrets stay server-side** — API keys and tokens are never exposed to the browser
- **Private backups** — the Dataset repo is created as private by default

> **Tip:** Change the default password from `huggingclaw` to something unique by setting the `OPENCLAW_PASSWORD` secret.

## License

MIT
