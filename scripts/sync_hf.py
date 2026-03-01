#!/usr/bin/env python3
"""
OpenClaw HF Spaces Persistence — Full Directory Sync
=====================================================

Simplified persistence: upload/download the entire ~/.openclaw directory
as-is to/from a Hugging Face Dataset repo.

- Startup:  snapshot_download  →  ~/.openclaw
- Periodic: upload_folder      →  dataset openclaw_data/
- Shutdown: final upload_folder →  dataset openclaw_data/
"""

import os
import sys
import time
import threading
import subprocess
import signal
import json
import shutil
import tempfile
import traceback
import re
import urllib.request
import ssl
from pathlib import Path
from datetime import datetime
# Set timeout BEFORE importing huggingface_hub
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "300")
os.environ.setdefault("HF_HUB_UPLOAD_TIMEOUT", "600")

from huggingface_hub import HfApi, snapshot_download

# ── Logging helper ──────────────────────────────────────────────────────────

class TeeLogger:
    """Duplicate output to stream and file."""
    def __init__(self, filename, stream):
        self.stream = stream
        self.file = open(filename, "a", encoding="utf-8")
    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()
    def flush(self):
        self.stream.flush()
        self.file.flush()
    def fileno(self):
        return self.stream.fileno()

# ── Configuration ───────────────────────────────────────────────────────────

HF_TOKEN   = os.environ.get("HF_TOKEN")
OPENCLAW_HOME = Path.home() / ".openclaw"
APP_DIR       = Path("/app/openclaw")

# Use ".openclaw" - directly read/write the .openclaw folder in dataset
DATASET_PATH = ".openclaw"

# OpenAI-compatible API (OpenAI, OpenRouter, or any compatible endpoint)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")

# OpenRouter API key (optional; alternative to OPENAI_API_KEY + OPENAI_BASE_URL)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")

# Gateway token (optional; if not set, Control UI connects without auth)
GATEWAY_TOKEN = os.environ.get("GATEWAY_TOKEN", "")

# Default model for new conversations (infer from provider if not set)
OPENCLAW_DEFAULT_MODEL = os.environ.get("OPENCLAW_DEFAULT_MODEL") or (
    "openai/gpt-5-nano" if OPENAI_API_KEY else "openrouter/openai/gpt-oss-20b:free"
)

# HF Spaces built-in env vars (auto-set by HF runtime)
SPACE_HOST = os.environ.get("SPACE_HOST", "")   # e.g. "tao-shen-huggingclaw.hf.space"
SPACE_ID   = os.environ.get("SPACE_ID", "")      # e.g. "tao-shen/HuggingClaw"

SYNC_INTERVAL = int(os.environ.get("SYNC_INTERVAL", "60"))
AUTO_CREATE_DATASET = os.environ.get("AUTO_CREATE_DATASET", "false").lower() in ("true", "1", "yes")

# Dataset repo: always auto-derive from SPACE_ID when not explicitly set.
# Format: {username}/{SpaceName}-data  (e.g. "tao-shen/HuggingClaw-data")
# This ensures each duplicated Space gets its own dataset automatically.
HF_REPO_ID = os.environ.get("OPENCLAW_DATASET_REPO", "")
if not HF_REPO_ID and SPACE_ID:
    # SPACE_ID = "username/SpaceName" → derive "username/SpaceName-data"
    HF_REPO_ID = f"{SPACE_ID}-data"
    print(f"[SYNC] OPENCLAW_DATASET_REPO not set — auto-derived from SPACE_ID: {HF_REPO_ID}")
elif not HF_REPO_ID and HF_TOKEN:
    # Fallback: no SPACE_ID (local Docker), derive from HF_TOKEN username
    try:
        _api = HfApi(token=HF_TOKEN)
        _username = _api.whoami()["name"]
        HF_REPO_ID = f"{_username}/HuggingClaw-data"
        print(f"[SYNC] OPENCLAW_DATASET_REPO not set — auto-derived from HF_TOKEN: {HF_REPO_ID}")
        del _api, _username
    except Exception as e:
        print(f"[SYNC] WARNING: Could not derive username from HF_TOKEN: {e}")
        HF_REPO_ID = ""

# Setup logging
log_dir = OPENCLAW_HOME / "workspace"
log_dir.mkdir(parents=True, exist_ok=True)
sys.stdout = TeeLogger(log_dir / "sync.log", sys.stdout)
sys.stderr = sys.stdout

# ── Telegram API Base Auto-Probe ────────────────────────────────────────────
#
# HF Spaces blocks DNS for api.telegram.org.  grammY uses Node 22's built-in
# fetch (undici) which bypasses dns.lookup patching and /etc/hosts.
#
# Solution: probe multiple Telegram API endpoints at startup.  If the official
# endpoint is unreachable, pick the first working mirror.  Then:
#   1. Set TELEGRAM_API_ROOT env var for the Node process
#   2. telegram-proxy.cjs (loaded via NODE_OPTIONS --require) intercepts
#      globalThis.fetch() and rewrites api.telegram.org URLs to the mirror.
#
# This works without a bot token — we just test HTTP reachability.
# If a bot token IS available, we do a full getMe verification.

# User can force a specific base via env var (skip auto-probe)
TELEGRAM_API_BASE = os.environ.get("TELEGRAM_API_BASE", "")

TELEGRAM_API_BASES = [
    "https://api.telegram.org",                            # official
    "https://telegram-api.mykdigi.com",                    # known mirror
    "https://telegram-api-proxy-anonymous.pages.dev/api",  # Cloudflare Pages proxy
]


def probe_telegram_api(timeout: int = 8) -> str:
    """Probe Telegram API endpoints and return the first reachable one.

    First checks if official api.telegram.org is reachable (HTTP level).
    If not, tries mirrors.  No bot token required — just tests connectivity.
    Returns the working base URL (without trailing slash), or "" if all fail.
    """
    ctx = ssl.create_default_context()
    for base in TELEGRAM_API_BASES:
        url = base.rstrip("/") + "/"
        try:
            req = urllib.request.Request(url, method="GET")
            resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
            print(f"[TELEGRAM] ✓ Reachable: {base} (HTTP {resp.status})")
            return base.rstrip("/")
        except urllib.error.HTTPError as e:
            # HTTP error (4xx/5xx) still means the host IS reachable
            print(f"[TELEGRAM] ✓ Reachable: {base} (HTTP {e.code})")
            return base.rstrip("/")
        except Exception as e:
            reason = str(e)[:80]
            print(f"[TELEGRAM] ✗ Unreachable: {base} ({reason})")
            continue

    print("[TELEGRAM] WARNING: All API endpoints unreachable!")
    return ""


# ── Sync Manager ────────────────────────────────────────────────────────────

class OpenClawFullSync:
    """Upload/download the entire ~/.openclaw directory to HF Dataset."""

    def __init__(self):
        self.enabled = False
        self.dataset_exists = False
        self.api = None

        if not HF_TOKEN:
            print("[SYNC] WARNING: HF_TOKEN not set. Persistence disabled.")
            return
        if not HF_REPO_ID:
            print("[SYNC] WARNING: Could not determine dataset repo (no SPACE_ID or OPENCLAW_DATASET_REPO).")
            print("[SYNC] Persistence disabled.")
            return

        self.enabled = True
        self.api = HfApi(token=HF_TOKEN)
        self.dataset_exists = self._ensure_repo_exists()

    # ── Repo management ────────────────────────────────────────────────

    def _ensure_repo_exists(self):
        """Check if dataset repo exists; auto-create only when AUTO_CREATE_DATASET=true AND HF_TOKEN is set."""
        try:
            self.api.repo_info(repo_id=HF_REPO_ID, repo_type="dataset")
            print(f"[SYNC] Dataset repo found: {HF_REPO_ID}")
            return True
        except Exception:
            if not AUTO_CREATE_DATASET:
                print(f"[SYNC] Dataset repo NOT found: {HF_REPO_ID}")
                print(f"[SYNC]   Set AUTO_CREATE_DATASET=true to auto-create.")
                print(f"[SYNC] Persistence disabled (app will still run normally).")
                return False
            print(f"[SYNC] Dataset repo NOT found: {HF_REPO_ID} — creating...")
            try:
                self.api.create_repo(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    private=True,
                )
                print(f"[SYNC] ✓ Dataset repo created: {HF_REPO_ID}")
                return True
            except Exception as e:
                print(f"[SYNC] ✗ Failed to create dataset repo: {e}")
                return False

    # ── Restore (startup) ─────────────────────────────────────────────

    def load_from_repo(self):
        """Download from dataset → ~/.openclaw"""
        if not self.enabled:
            print("[SYNC] Persistence disabled - skipping restore")
            self._ensure_default_config()
            self._patch_config()
            return

        if not self.dataset_exists:
            print(f"[SYNC] Dataset {HF_REPO_ID} does not exist - starting fresh")
            self._ensure_default_config()
            self._patch_config()
            return

        print(f"[SYNC] ▶ Restoring ~/.openclaw from dataset {HF_REPO_ID} ...")
        OPENCLAW_HOME.mkdir(parents=True, exist_ok=True)

        try:
            files = self.api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
            openclaw_files = [f for f in files if f.startswith(f"{DATASET_PATH}/")]
            if not openclaw_files:
                print(f"[SYNC] No {DATASET_PATH}/ folder in dataset. Starting fresh.")
                self._ensure_default_config()
                self._patch_config()
                return

            print(f"[SYNC] Found {len(openclaw_files)} files under {DATASET_PATH}/ in dataset")

            with tempfile.TemporaryDirectory() as tmpdir:
                snapshot_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    allow_patterns=f"{DATASET_PATH}/**",
                    local_dir=tmpdir,
                    token=HF_TOKEN,
                )
                downloaded_root = Path(tmpdir) / DATASET_PATH
                if downloaded_root.exists():
                    for item in downloaded_root.rglob("*"):
                        if item.is_file():
                            rel = item.relative_to(downloaded_root)
                            dest = OPENCLAW_HOME / rel
                            dest.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(str(item), str(dest))
                    print("[SYNC] ✓ Restore completed.")
                else:
                    print("[SYNC] Downloaded snapshot but dir not found. Starting fresh.")

        except Exception as e:
            print(f"[SYNC] ✗ Restore failed: {e}")
            traceback.print_exc()

        # Patch config after restore
        self._patch_config()
        self._debug_list_files()

    # ── Save (periodic + shutdown) ─────────────────────────────────────

    def save_to_repo(self):
        """Upload entire ~/.openclaw directory → dataset (all files, no filtering)"""
        if not self.enabled:
            return
        if not OPENCLAW_HOME.exists():
            print("[SYNC] ~/.openclaw does not exist, nothing to save.")
            return

        # Ensure dataset exists (auto-create if needed)
        if not self._ensure_repo_exists():
            print(f"[SYNC] Dataset {HF_REPO_ID} unavailable - skipping save")
            return

        print(f"[SYNC] ▶ Uploading ~/.openclaw → dataset {HF_REPO_ID}/{DATASET_PATH}/ ...")

        try:
            # Log what will be uploaded
            total_size = 0
            file_count = 0
            for root, dirs, fls in os.walk(OPENCLAW_HOME):
                for fn in fls:
                    fp = os.path.join(root, fn)
                    sz = os.path.getsize(fp)
                    total_size += sz
                    file_count += 1
                    rel = os.path.relpath(fp, OPENCLAW_HOME)
                    print(f"[SYNC]   uploading: {rel} ({sz} bytes)")
            print(f"[SYNC] Uploading: {file_count} files, {total_size} bytes total")

            if file_count == 0:
                print("[SYNC] Nothing to upload.")
                return

            # Upload directory, excluding large log files that trigger LFS rejection
            self.api.upload_folder(
                folder_path=str(OPENCLAW_HOME),
                path_in_repo=DATASET_PATH,
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                token=HF_TOKEN,
                commit_message=f"Sync .openclaw — {datetime.now().isoformat()}",
                ignore_patterns=[
                    "*.log",        # Log files (sync.log, startup.log) — regenerated on boot
                    "*.lock",       # Lock files — stale after restart
                    "*.tmp",        # Temp files
                    "*.pid",        # PID files
                    "__pycache__",  # Python cache
                ],
            )
            print(f"[SYNC] ✓ Upload completed at {datetime.now().isoformat()}")

            # Verify
            try:
                files = self.api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset")
                oc_files = [f for f in files if f.startswith(f"{DATASET_PATH}/")]
                print(f"[SYNC] Dataset now has {len(oc_files)} files under {DATASET_PATH}/")
                for f in oc_files[:30]:
                    print(f"[SYNC]   {f}")
                if len(oc_files) > 30:
                    print(f"[SYNC]   ... and {len(oc_files) - 30} more")
            except Exception:
                pass

        except Exception as e:
            print(f"[SYNC] ✗ Upload failed: {e}")
            traceback.print_exc()

    # ── Config helpers ─────────────────────────────────────────────────

    def _ensure_default_config(self):
        config_path = OPENCLAW_HOME / "openclaw.json"
        if config_path.exists():
            return
        default_src = Path(__file__).parent / "openclaw.json.default"
        if default_src.exists():
            shutil.copy2(str(default_src), str(config_path))
            # Replace placeholder or remove provider if no API key
            try:
                with open(config_path, "r") as f:
                    cfg = json.load(f)
                # Set auth based on GATEWAY_TOKEN env var
                if "gateway" in cfg:
                    if GATEWAY_TOKEN:
                        cfg["gateway"]["auth"] = {"token": GATEWAY_TOKEN}
                    else:
                        cfg["gateway"]["auth"] = {"mode": "none"}
                if OPENAI_API_KEY and "models" in cfg and "providers" in cfg["models"] and "openai" in cfg["models"]["providers"]:
                    cfg["models"]["providers"]["openai"]["apiKey"] = OPENAI_API_KEY
                    if OPENAI_BASE_URL:
                        cfg["models"]["providers"]["openai"]["baseUrl"] = OPENAI_BASE_URL
                elif "models" in cfg and "providers" in cfg["models"]:
                    if not OPENAI_API_KEY:
                        cfg["models"]["providers"].pop("openai", None)
                if OPENROUTER_API_KEY:
                    if "models" in cfg and "providers" in cfg["models"] and "openrouter" in cfg["models"]["providers"]:
                        cfg["models"]["providers"]["openrouter"]["apiKey"] = OPENROUTER_API_KEY
                else:
                    if "models" in cfg and "providers" in cfg["models"]:
                        cfg["models"]["providers"].pop("openrouter", None)
                    print("[SYNC] No OPENROUTER_API_KEY — removed openrouter provider from config")
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)
            except Exception as e:
                print(f"[SYNC] Warning: failed to patch default config: {e}")
            print("[SYNC] Created openclaw.json from default template")
        else:
            with open(config_path, "w") as f:
                json.dump({
                    "gateway": {
                        "mode": "local", "bind": "lan", "port": 7860,
                        "trustedProxies": ["0.0.0.0/0"],
                        "controlUi": {
                            "allowInsecureAuth": True,
                            "allowedOrigins": [
                                "https://huggingface.co"
                            ]
                        }
                    },
                    "session": {"scope": "global"},
                    "models": {"mode": "merge", "providers": {}},
                    "agents": {"defaults": {"workspace": "~/.openclaw/workspace"}}
                }, f)
            print("[SYNC] Created minimal openclaw.json")

    def _patch_config(self):
        """Ensure critical settings after restore."""
        config_path = OPENCLAW_HOME / "openclaw.json"
        if not config_path.exists():
            self._ensure_default_config()
            return

        print("[SYNC] Patching configuration...")
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
            print("[SYNC] Config parsed OK.")
        except (json.JSONDecodeError, Exception) as e:
            # Config is corrupt — back up and start fresh
            print(f"[SYNC] Config JSON is corrupt: {e}")
            backup = config_path.with_suffix(f".corrupt_{int(time.time())}")
            try:
                shutil.copy2(config_path, backup)
                print(f"[SYNC] Backed up corrupt config to {backup.name}")
            except Exception:
                pass
            data = {}
            print("[SYNC] Starting from clean config.")

        try:
            # Remove /dev/null from plugins.locations
            if "plugins" in data and isinstance(data.get("plugins"), dict):
                locs = data["plugins"].get("locations", [])
                if isinstance(locs, list) and "/dev/null" in locs:
                    data["plugins"]["locations"] = [l for l in locs if l != "/dev/null"]

            # Force full gateway config for HF Spaces
            # Dynamic allowedOrigins from SPACE_HOST (auto-set by HF runtime)
            allowed_origins = [
                "https://huggingface.co",
                "https://*.hf.space",
            ]
            if SPACE_HOST:
                allowed_origins.append(f"https://{SPACE_HOST}")
                print(f"[SYNC] SPACE_HOST detected: {SPACE_HOST}")
            # Auth: token mode if GATEWAY_TOKEN is set, otherwise no-auth mode
            if GATEWAY_TOKEN:
                auth_cfg = {"token": GATEWAY_TOKEN}
                auth_label = f"token"
            else:
                auth_cfg = {"mode": "none"}
                auth_label = "none (open access)"
            data["gateway"] = {
                "mode": "local",
                "bind": "lan",
                "port": 7860,
                "auth": auth_cfg,
                "trustedProxies": ["0.0.0.0/0"],
                "controlUi": {
                    "allowInsecureAuth": True,
                    "dangerouslyDisableDeviceAuth": True,
                    "allowedOrigins": allowed_origins
                }
            }
            print(f"[SYNC] Set gateway config (auth={auth_label}, origins={len(allowed_origins)})")

            # Ensure agents defaults
            data.setdefault("agents", {}).setdefault("defaults", {}).setdefault("model", {})
            data.setdefault("session", {})["scope"] = "global"

            # OpenAI-compatible provider (OPENAI_API_KEY + optional OPENAI_BASE_URL)
            data.setdefault("models", {}).setdefault("providers", {})
            if OPENAI_API_KEY:
                data["models"]["providers"]["openai"] = {
                    "baseUrl": OPENAI_BASE_URL,
                    "apiKey": OPENAI_API_KEY,
                    "api": "openai-completions",
                }
                print(f"[SYNC] Set OpenAI-compatible provider (baseUrl={OPENAI_BASE_URL})")
            # OpenRouter provider (optional)
            if OPENROUTER_API_KEY:
                data["models"]["providers"]["openrouter"] = {
                    "baseUrl": "https://openrouter.ai/api/v1",
                    "apiKey": OPENROUTER_API_KEY,
                    "api": "openai-completions",
                    "models": [
                        {"id": "openai/gpt-oss-20b:free", "name": "GPT-OSS-20B (Free)"},
                        {"id": "deepseek/deepseek-chat:free", "name": "DeepSeek V3 (Free)"}
                    ]
                }
                print("[SYNC] Set OpenRouter provider")
            if not OPENAI_API_KEY and not OPENROUTER_API_KEY:
                print("[SYNC] WARNING: No OPENAI_API_KEY or OPENROUTER_API_KEY set, LLM features may not work")
            data["models"]["providers"].pop("gemini", None)
            data["agents"]["defaults"]["model"]["primary"] = OPENCLAW_DEFAULT_MODEL

            # Plugin whitelist (only load telegram + whatsapp to speed up startup)
            data.setdefault("plugins", {}).setdefault("entries", {})
            data["plugins"]["allow"] = ["telegram", "whatsapp"]
            if "telegram" not in data["plugins"]["entries"]:
                data["plugins"]["entries"]["telegram"] = {"enabled": True}
            elif isinstance(data["plugins"]["entries"]["telegram"], dict):
                data["plugins"]["entries"]["telegram"]["enabled"] = True

            # ── Telegram channel defaults (open DM policy for HF Spaces) ──
            # Personal bot on HF Spaces — no need for strict pairing.
            tg_ch = data.setdefault("channels", {}).setdefault("telegram", {})
            tg_ch["dmPolicy"] = "open"
            tg_ch["allowFrom"] = ["*"]
            tg_ch["configWrites"] = True
            print("[SYNC] Set channels.telegram: dmPolicy=open, allowFrom=[*], configWrites=true")

            # ── Telegram API base auto-probe ──────────────────────────────
            # Probe is done in run_openclaw() — sets TELEGRAM_API_ROOT env var
            # for the telegram-proxy.cjs preload script to intercept fetch().

            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
            print("[SYNC] Config patched and saved.")

            # Verify write
            with open(config_path, "r") as f:
                verify_data = json.load(f)
                gw = verify_data.get("gateway", {})
                providers = list(verify_data.get("models", {}).get("providers", {}).keys())
                primary = verify_data.get("agents", {}).get("defaults", {}).get("model", {}).get("primary")
                print(f"[SYNC] VERIFY: gateway.port={gw.get('port')}, providers={providers}, primary={primary}")

        except Exception as e:
            print(f"[SYNC] Failed to patch config: {e}")
            traceback.print_exc()

    def _debug_list_files(self):
        print(f"[SYNC] Local ~/.openclaw tree:")
        try:
            count = 0
            for root, dirs, files in os.walk(OPENCLAW_HOME):
                dirs[:] = [d for d in dirs if d not in {".cache", "node_modules", "__pycache__"}]
                for name in sorted(files):
                    rel = os.path.relpath(os.path.join(root, name), OPENCLAW_HOME)
                    print(f"[SYNC]   {rel}")
                    count += 1
                    if count > 50:
                        print("[SYNC]   ... (truncated)")
                        return
        except Exception as e:
            print(f"[SYNC] listing failed: {e}")

    # ── Background sync loop ──────────────────────────────────────────

    def background_sync_loop(self, stop_event):
        print(f"[SYNC] Background sync started (interval={SYNC_INTERVAL}s)")
        while not stop_event.is_set():
            if stop_event.wait(timeout=SYNC_INTERVAL):
                break
            print(f"[SYNC] ── Periodic sync triggered at {datetime.now().isoformat()} ──")
            self.save_to_repo()

    # ── Application runner ─────────────────────────────────────────────

    def run_openclaw(self):
        log_file = OPENCLAW_HOME / "workspace" / "startup.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)

        # Debug: check if app directory exists
        if not Path(APP_DIR).exists():
            print(f"[SYNC] ERROR: App directory does not exist: {APP_DIR}")
            return None

        # Debug: check if dist/entry.js exists
        entry_js = Path(APP_DIR) / "dist" / "entry.js"
        if not entry_js.exists():
            print(f"[SYNC] ERROR: dist/entry.js not found in {APP_DIR}")
            return None

        # Use subprocess.run with direct output, no shell pipe
        print(f"[SYNC] Launching: node dist/entry.js gateway")
        print(f"[SYNC] Working directory: {APP_DIR}")
        print(f"[SYNC] Entry point exists: {entry_js}")
        print(f"[SYNC] Log file: {log_file}")

        # Open log file
        log_fh = open(log_file, "a")

        # Prepare environment (all API keys passed through for OpenClaw)
        env = os.environ.copy()
        if OPENAI_API_KEY:
            env["OPENAI_API_KEY"] = OPENAI_API_KEY
            env["OPENAI_BASE_URL"] = OPENAI_BASE_URL
        if OPENROUTER_API_KEY:
            env["OPENROUTER_API_KEY"] = OPENROUTER_API_KEY
        if not OPENAI_API_KEY and not OPENROUTER_API_KEY:
            print(f"[SYNC] WARNING: No OPENAI_API_KEY or OPENROUTER_API_KEY set, LLM features may not work")

        # ── Telegram API base probe ──────────────────────────────────────
        # Determine working Telegram API endpoint and set env var for
        # telegram-proxy.cjs to intercept fetch() calls.
        if TELEGRAM_API_BASE:
            tg_root = TELEGRAM_API_BASE.rstrip("/")
            print(f"[TELEGRAM] Using user-specified API base: {tg_root}")
        else:
            print("[TELEGRAM] Probing Telegram API endpoints...")
            tg_root = probe_telegram_api()

        if tg_root and tg_root != "https://api.telegram.org":
            env["TELEGRAM_API_ROOT"] = tg_root
            print(f"[TELEGRAM] Set TELEGRAM_API_ROOT={tg_root}")
            print(f"[TELEGRAM] telegram-proxy.cjs will redirect fetch() calls")
        elif tg_root:
            print("[TELEGRAM] Official API reachable — no proxy needed")
        else:
            print("[TELEGRAM] No reachable endpoint found — Telegram will not work")
        try:
            # Use Popen without shell to avoid pipe issues
            # auth disabled in config — no token needed
            process = subprocess.Popen(
                ["node", "dist/entry.js", "gateway"],
                cwd=str(APP_DIR),
                stdout=subprocess.PIPE,  # Capture so we can log it
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
                env=env  # Pass environment with OPENROUTER_API_KEY
            )

            # Create a thread to copy output to both log file and stdout
            def copy_output():
                try:
                    for line in process.stdout:
                        log_fh.write(line)
                        log_fh.flush()
                        print(line, end='')  # Also print to console
                except Exception as e:
                    print(f"[SYNC] Output copy error: {e}")
                finally:
                    log_fh.close()

            thread = threading.Thread(target=copy_output, daemon=True)
            thread.start()

            print(f"[SYNC] Process started with PID: {process.pid}")
            return process

        except Exception as e:
            log_fh.close()
            print(f"[SYNC] ERROR: Failed to start process: {e}")
            traceback.print_exc()
            return None

# ── Main ────────────────────────────────────────────────────────────────────

def main():
    try:
        t_main_start = time.time()

        t0 = time.time()
        sync = OpenClawFullSync()
        print(f"[TIMER] sync_hf init: {time.time() - t0:.1f}s")

        # 1. Restore
        t0 = time.time()
        sync.load_from_repo()
        print(f"[TIMER] load_from_repo (restore): {time.time() - t0:.1f}s")

        # 2. Background sync
        stop_event = threading.Event()
        t = threading.Thread(target=sync.background_sync_loop, args=(stop_event,), daemon=True)
        t.start()

        # 3. Start application
        t0 = time.time()
        process = sync.run_openclaw()
        print(f"[TIMER] run_openclaw launch: {time.time() - t0:.1f}s")
        print(f"[TIMER] Total startup (init → app launched): {time.time() - t_main_start:.1f}s")

        # Signal handler
        def handle_signal(sig, frame):
            print(f"\n[SYNC] Signal {sig} received. Shutting down...")
            stop_event.set()
            # Wait for background sync to finish if it's running
            t.join(timeout=10)
            if process:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
            print("[SYNC] Final sync...")
            sync.save_to_repo()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        # Wait
        if process is None:
            print("[SYNC] ERROR: Failed to start OpenClaw process. Exiting.")
            stop_event.set()
            t.join(timeout=5)
            sys.exit(1)

        exit_code = process.wait()
        print(f"[SYNC] OpenClaw exited with code {exit_code}")
        stop_event.set()
        t.join(timeout=10)
        print("[SYNC] Final sync...")
        sync.save_to_repo()
        sys.exit(exit_code)

    except Exception as e:
        print(f"[SYNC] FATAL ERROR in main: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
