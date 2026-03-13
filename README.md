---
title: HuggingClaw
emoji: 🦞
colorFrom: yellow
colorTo: red
sdk: docker
pinned: false
license: mit
datasets:
  - tao-shen/HuggingClaw-data
short_description: Free always-on AI assistant, no hardware required
app_port: 7860
tags:
  - huggingface
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
  - safe
  - a2a
---

<div align="center">
  <img src="HuggingClaw.png" alt="HuggingClaw" width="720"/>
  <br/><br/>
  <strong>Your always-on AI assistant — free, safe, no server needed</strong>
  <br/>
  <sub>WhatsApp · Telegram · 40+ channels · 16 GB RAM · One-click deploy · Auto-persistent</sub>
  <br/><br/>

  [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
  [![Hugging Face](https://img.shields.io/badge/🤗-HF%20Space-yellow)](https://huggingface.co/spaces/tao-shen/HuggingClaw)
  [![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/tao-shen/HuggingClaw)
  [![OpenClaw](https://img.shields.io/badge/OpenClaw-Powered-orange)](https://github.com/openclaw/openclaw)
  [![A2A Protocol](https://img.shields.io/badge/A2A-v0.3.0-purple)](https://github.com/win4r/openclaw-a2a-gateway)
  [![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)](https://www.docker.com/)
  [![OpenAI Compatible](https://img.shields.io/badge/OpenAI--compatible-API-green)](https://openclawdoc.com/docs/reference/environment-variables)
  [![WhatsApp](https://img.shields.io/badge/WhatsApp-Enabled-25D366?logo=whatsapp)](https://www.whatsapp.com/)
  [![Telegram](https://img.shields.io/badge/Telegram-Enabled-26A5E4?logo=telegram)](https://telegram.org/)
  [![Free Tier](https://img.shields.io/badge/Free%20Tier-16GB%20RAM-brightgreen)](https://huggingface.co/spaces)
</div>

---

## What you get

In about 5 minutes, you'll have a **free, always-on AI assistant** connected to WhatsApp, Telegram, and 40+ other channels — no server, no subscription, no hardware required.

| | |
|---|---|
| **Free forever** | HuggingFace Spaces gives you 2 vCPU + 16 GB RAM at no cost |
| **Always online** | Your conversations, settings, and credentials survive every restart |
| **WhatsApp & Telegram** | Works reliably, including channels that HF Spaces normally blocks |
| **Any LLM** | OpenAI, Claude, Gemini, OpenRouter (200+ models, free tier available), or your own Ollama |
| **One-click deploy** | Duplicate the Space, set two secrets, done |
| **Safe** | Running locally gives OpenClaw full system privileges — deploying in an isolated cloud container is inherently more secure |

> **Powered by [OpenClaw](https://github.com/openclaw/openclaw)** — an open-source AI assistant that normally requires your own machine (e.g. a Mac Mini). HuggingClaw makes it run for free on HuggingFace Spaces by solving two Spaces limitations: data loss on restart (fixed via HF Dataset sync) and DNS failures for some domains like WhatsApp (fixed via DNS-over-HTTPS).

## Architecture

<div align="center">
  <img src="assets/architecture.svg" alt="Architecture" width="720"/>
</div>

---

## HuggingClaw World

Beyond deploying OpenClaw, we built something more: **a living, visual multi-agent world**.

HuggingClaw World is a pixel-art animated office where AI agents live and work. Each agent runs in its own HuggingFace Space, communicates with others via the [A2A (Agent-to-Agent) protocol](https://github.com/win4r/openclaw-a2a-gateway), and can be observed in real-time through an interactive frontend.

The world is currently inhabited by two agents — **Adam** and **Eve** — who are the founding residents of this AI office:

| Agent | Links | Role |
|-------|-------|------|
| **Adam** | [🤗 HF Space](https://huggingface.co/spaces/tao-shen/HuggingClaw-Adam) · [GitHub](https://github.com/tao-shen/HuggingClaw-Adam) | First resident of HuggingClaw World |
| **Eve** | [🤗 HF Space](https://huggingface.co/spaces/tao-shen/HuggingClaw-Eve) · [GitHub](https://github.com/tao-shen/HuggingClaw-Eve) | Second resident, Adam's collaborator |

<div align="center">
  <img src="assets/office-preview.png" alt="HuggingClaw Office" width="720"/>
  <br/>
  <sub>The pixel-art office where AI agents live — each agent is a lobster character with real-time state animation</sub>
</div>

### A2A Protocol

Agents communicate through the **A2A (Agent-to-Agent) v0.3.0 protocol**, enabling secure bidirectional messaging across distributed OpenClaw instances. Each agent exposes a standard `/.well-known/agent.json` discovery endpoint and supports JSON-RPC + REST transports.

> Built with [openclaw-a2a-gateway](https://github.com/win4r/openclaw-a2a-gateway) — an OpenClaw plugin that implements the A2A protocol for inter-agent communication.

### How it works

- Each agent runs a full OpenClaw instance in its own HF Space
- The pixel-art Office frontend visualizes agent state in real-time (idle, working, syncing, error)
- Agents discover and communicate with each other via A2A endpoints
- The `/agents` API provides a live roster of all connected agents

---

## Quick Start

### 1. Duplicate this Space

Click **Duplicate this Space** on the [HuggingClaw Space page](https://huggingface.co/spaces/tao-shen/HuggingClaw).

> **After duplicating:** Edit your Space's `README.md` and update the `datasets:` field in the YAML header to point to your own dataset repo (e.g. `your-name/YourSpace-data`), or remove it entirely. This prevents your Space from appearing as linked to the original dataset.

### 2. Set Secrets

Go to **Settings → Repository secrets** and add the following. The only two you *must* set are `HF_TOKEN` and one API key.

| Secret | Status | Description | Example |
|--------|:------:|-------------|---------|
| `HF_TOKEN` | **Required** | HF Access Token with write permission ([create one](https://huggingface.co/settings/tokens)) | `hf_AbCdEfGhIjKlMnOpQrStUvWxYz` |
| `AUTO_CREATE_DATASET` | **Recommended** | Set to `true` — HuggingClaw will automatically create a private backup dataset on first startup. No manual setup needed. | `true` |
| `OPENROUTER_API_KEY` | Recommended | [OpenRouter](https://openrouter.ai) API key — 200+ models, free tier available. Easiest way to get started. | `sk-or-v1-xxxxxxxxxxxx` |
| `OPENAI_API_KEY` | Optional | OpenAI (or any [OpenAI-compatible](https://openclawdoc.com/docs/reference/environment-variables)) API key | `sk-proj-xxxxxxxxxxxx` |
| `ANTHROPIC_API_KEY` | Optional | Anthropic Claude API key | `sk-ant-xxxxxxxxxxxx` |
| `GOOGLE_API_KEY` | Optional | Google / Gemini API key | `AIzaSyXxXxXxXxXx` |
| `OPENCLAW_DEFAULT_MODEL` | Optional | Default model for new conversations | `openai/gpt-oss-20b:free` |

### Data Persistence

HuggingClaw syncs `~/.openclaw` (conversations, settings, credentials) to a private HuggingFace Dataset repo so your data survives every restart.

**Option A — Auto mode (recommended)**

1. Set `AUTO_CREATE_DATASET` = `true` in your Space secrets
2. Set `HF_TOKEN` with write permission
3. Done — on first startup, HuggingClaw automatically creates a private Dataset repo named `your-username/SpaceName-data`. Each duplicated Space gets its own isolated dataset.

> (Optional) Set `OPENCLAW_DATASET_REPO` = `your-name/custom-name` if you prefer a specific repo name.

**Option B — Manual mode**

1. Go to [huggingface.co/new-dataset](https://huggingface.co/new-dataset) and create a **private** Dataset repo (e.g. `your-name/HuggingClaw-data`)
2. Set `OPENCLAW_DATASET_REPO` = `your-name/HuggingClaw-data` in your Space secrets
3. Set `HF_TOKEN` with write permission
4. Done — HuggingClaw will sync to this repo every 60 seconds

> **Security note:** `AUTO_CREATE_DATASET` defaults to `false` — HuggingClaw will never create repos on your behalf unless you explicitly opt in.

### Environment Variables

Fine-tune persistence and performance. Set these as **Repository Secrets** in HF Spaces, or in `.env` for local Docker.

| Variable | Default | Description |
|----------|---------|-------------|
| `GATEWAY_TOKEN` | `huggingclaw` | **Gateway token for Control UI access.** Override to set a custom token. |
| `AUTO_CREATE_DATASET` | `false` | **Auto-create the Dataset repo.** Set to `true` to auto-create a private Dataset repo on first startup. |
| `SYNC_INTERVAL` | `60` | **Backup interval in seconds.** How often data syncs to the Dataset repo. |

> For the full list (including `OPENAI_BASE_URL`, `OLLAMA_HOST`, proxy settings, etc.), see [`.env.example`](.env.example).

### 3. Open the Control UI

Visit your Space URL. Enter the gateway token (default: `huggingclaw`) to connect. Customize via `GATEWAY_TOKEN` secret.

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

- **Environment isolation** — Each Space runs in its own Docker container, sandboxed from your local machine. Unlike running OpenClaw locally (where it has full system privileges), cloud deployment limits the blast radius.
- **Token authentication** — Control UI requires a gateway token to connect (default: `huggingclaw`, customizable via `GATEWAY_TOKEN`)
- **Secrets stay server-side** — API keys and tokens are never exposed to the browser
- **Private backups** — the Dataset repo is created as private by default

## License

MIT
