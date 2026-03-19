#!/usr/bin/env python3 -u
"""
Adam & Eve — A2A-based Agent Orchestrator for their child Cain.

Architecture: Adam/Eve are OpenClaw instances communicating via Google A2A protocol.
Each has its own personality (SOUL.md), memory system, and LLM backend.
This script is a lightweight coordinator — it sends context via A2A, parses
responses for [TASK] blocks, and delegates coding work to Claude Code CLI.

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    SYSTEM ARCHITECTURE (v5 — A2A)                  ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║                                                                    ║
# ║  ┌──────────────────┐  A2A   ┌──────────────────┐                ║
# ║  │ Adam (OpenClaw)  │◄──────►│ Eve (OpenClaw)   │                ║
# ║  │ HF Space + A2A   │        │ HF Space + A2A   │                ║
# ║  │ changes Cain     │        │ changes Cain     │                ║
# ║  └────────┬─────────┘        └────────┬─────────┘                ║
# ║           │ [TASK]                    │ [TASK]                    ║
# ║           ▼                           ▼                           ║
# ║  ┌────────────────────────────────────────────┐                   ║
# ║  │        conversation-loop.py                │                   ║
# ║  │   (orchestrator on Home Space)             │                   ║
# ║  │   - sends context via A2A to all agents    │                   ║
# ║  │   - parses [TASK] → Claude Code CLI        │                   ║
# ║  │   - manages chatlog, bubbles, frontend     │                   ║
# ║  └───────┬──────────────────┬─────────────────┘                   ║
# ║          │ [TASK]           │ A2A (every 2 min)                    ║
# ║          ▼                  ▼                                      ║
# ║  ┌─────────────┐  ┌──────────────────┐                            ║
# ║  │ Cain Space  │  │ God (OpenClaw)   │                            ║
# ║  │ (child)     │  │ mechanism optimizer                           ║
# ║  └─────────────┘  │ changes Home     │                            ║
# ║                    └──────────────────┘                            ║
# ║                                                                    ║
# ║  Cain CC: Adam/Eve [TASK] → Claude Code → push to Cain           ║
# ║  God CC:  God [TASK] → Claude Code → push to Home (restart)      ║
# ║  Flow: Eve(A2A) → Adam(A2A) → ... God(A2A) every 2 min           ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import json, time, re, requests, sys, os, io, subprocess, threading, datetime, uuid
from collections import deque

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Endpoints ──────────────────────────────────────────────────────────────────
HOME = "https://tao-shen-huggingclaw-home.hf.space"
ADAM_SPACE = "https://tao-shen-huggingclaw-adam.hf.space"
ADAM_SPACE_ID = "tao-shen/HuggingClaw-Adam"
EVE_SPACE  = "https://tao-shen-huggingclaw-eve.hf.space"
EVE_SPACE_ID = "tao-shen/HuggingClaw-Eve"
GOD_SPACE  = "https://tao-shen-huggingclaw-god.hf.space"
GOD_POLL_INTERVAL = 120  # God polls every 2 minutes; lightweight check first, Claude Code only when needed
GOD_WORK_DIR = "/tmp/god-workspace"
GOD_TIMEOUT = 300  # 5 minutes for God's Claude Code analysis (was 10min)
GOD_SPACE_ID = "tao-shen/HuggingClaw-God"  # God improves itself (pushes to own repo)

# ── A2A Health Monitoring ─────────────────────────────────────────────────────
# Track consecutive failures and last restart time for Adam/Eve
A2A_FAILURE_THRESHOLD = 6  # Restart after 6 consecutive failures (~3 minutes)
A2A_RESTART_COOLDOWN = 600  # 10 minutes between restarts
_a2a_health = {
    "adam": {"failures": 0, "last_restart": 0, "last_success": 0},
    "eve": {"failures": 0, "last_restart": 0, "last_success": 0},
    "god": {"failures": 0, "last_restart": 0, "last_success": 0},
}

# ── Child config ───────────────────────────────────────────────────────────────
CHILD_NAME = "Cain"
CHILD_SPACE_ID = "tao-shen/HuggingClaw-Cain"
CHILD_SPACE_URL = "https://tao-shen-huggingclaw-cain.hf.space"
CHILD_DATASET_ID = "tao-shen/HuggingClaw-Cain-data"
SOURCE_SPACE_ID = "tao-shen/HuggingClaw-Adam"

# ── Zhipu API ──────────────────────────────────────────────────────────────────
ZHIPU_BASE = "https://open.bigmodel.cn/api/anthropic"
ZHIPU_KEY = os.environ.get("ZHIPU_API_KEY", "")

# ── Load tokens ────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        HF_TOKEN = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except:
        pass

if not ZHIPU_KEY:
    try:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download("tao-shen/HuggingClaw-Adam-data", ".openclaw/openclaw.json",
                           repo_type="dataset", token=HF_TOKEN)
        with open(f) as fh:
            cfg = json.load(fh)
            ZHIPU_KEY = cfg.get("models", {}).get("providers", {}).get("zhipu", {}).get("apiKey", "")
    except Exception as e:
        print(f"[error] Could not load Zhipu key: {e}", file=sys.stderr)

if not ZHIPU_KEY:
    print("[FATAL] No ZHIPU_API_KEY found.", file=sys.stderr)
    sys.exit(1)
if not HF_TOKEN:
    print("[FATAL] No HF_TOKEN found.", file=sys.stderr)
    sys.exit(1)

print(f"[init] Zhipu key: {ZHIPU_KEY[:8]}...{ZHIPU_KEY[-4:]}")
print(f"[init] HF token:  {HF_TOKEN[:8]}...{HF_TOKEN[-4:]}")

# ── HuggingFace API ────────────────────────────────────────────────────────────
from huggingface_hub import HfApi, create_repo, hf_hub_download
hf_api = HfApi(token=HF_TOKEN)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1: CHILD STATE + SAFETY
# ══════════════════════════════════════════════════════════════════════════════

child_state = {
    "created": False,
    "alive": False,
    "stage": "not_born",
    "state": "unknown",
    "detail": "",
}

# Rebuild cooldown — prevent rapid pushes that keep resetting builds
REBUILD_COOLDOWN_SECS = 180  # 3 minutes — fast iteration, trial-and-error is preferred
last_rebuild_trigger_at = 0
_pending_cooldown = False

# Push frequency tracking — God uses this to detect "all talk no action"
_push_count = 0           # total pushes since startup
_last_push_time = 0.0     # timestamp of last successful push
_turns_since_last_push = 0  # turns since last push (resets on push)
_push_count_this_task = 0  # pushes made during the CURRENT CC task (resets on new task)

# Hard Reset override — FORCE_PUSH mode for breaking discussion loops
_force_push_mode = False  # When True, bypass normal flow and force task generation
_force_push_trigger_time = 0.0  # When FORCE_PUSH was triggered
_force_push_skip_termination = False  # If True, skip termination (already terminated)

# Emergency Override Protocol constants
MAX_IDLE_TURNS = 3  # Trigger emergency override after this many idle turns with zero pushes
MAX_TURNS_WITHOUT_PUSH = 8  # Trigger emergency override after this many turns without ANY push (catches "1 push then 20 turns discussion" anti-pattern)
_emergency_override_active = False  # When True, safety throttles are ignored

# LOCKDOWN Mode — "Purge & Reboot" for Stalemate State
# Detects when Cain is stuck in ERROR state for too long without effective pushes
# and forces a hard reset: terminate CC, clear conversation, force fresh diagnostic
_lockdown_mode = False  # When True, in purge & reboot state
_lockdown_trigger_time = 0.0  # When LOCKDOWN was triggered
_lockdown_error_onset = 0.0  # Timestamp when Cain first entered ERROR state
_lockdown_push_count_at_error = 0  # Push count when error started (to detect if any pushes happened during error)
LOCKDOWN_ERROR_THRESHOLD_SECS = 600  # 10 minutes in error state without effective pushes triggers LOCKDOWN
LOCKDOWN_RESET_SECS = 180  # 3 minutes — LOCKDOWN lasts this long before returning to normal

def _init_push_count_from_workspace():
    """Initialize push count from existing workspace commits.
    This persists push tracking across conversation loop restarts."""
    global _push_count, _last_push_time
    try:
        if os.path.exists(CLAUDE_WORK_DIR):
            result = subprocess.run(
                f'git log --since="1 hour ago" --format="%H %ct" --author="Claude Code"',
                shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                commits = result.stdout.strip().split('\n')
                # Count only Claude Code commits from the last hour
                _push_count = len(commits)
                if commits:
                    # Get timestamp of most recent commit
                    last_commit_ts = int(commits[0].split()[1])
                    _last_push_time = float(last_commit_ts)
                print(f"[PUSH-TRACK] Initialized push count from workspace: {_push_count} commits in last hour")
    except Exception as e:
        print(f"[PUSH-TRACK] Failed to initialize from workspace: {e}")

def check_and_clear_cooldown():
    """Auto-clear cooldown if Cain has finished building."""
    global last_rebuild_trigger_at
    if last_rebuild_trigger_at == 0:
        return
    elapsed = time.time() - last_rebuild_trigger_at
    if elapsed < 60:
        return
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        stage = info.runtime.stage if info.runtime else "unknown"
        if stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
            print(f"[COOLDOWN] Build finished (stage={stage}), clearing cooldown ({int(elapsed)}s)")
            last_rebuild_trigger_at = 0
            child_state["stage"] = stage
            child_state["alive"] = (stage == "RUNNING")
    except:
        pass


def init_child_state():
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        child_state["created"] = True
        child_state["stage"] = info.runtime.stage if info.runtime else "unknown"
        # Use HF API stage as source of truth for alive (stage==RUNNING means healthy)
        child_state["alive"] = (child_state["stage"] == "RUNNING")
        print(f"[init] {CHILD_NAME}: stage={child_state['stage']}, alive={child_state['alive']}")
    except:
        print(f"[init] {CHILD_NAME} does not exist yet")

init_child_state()


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2: ACTIONS (minimal set — most work delegated to Claude Code)
# ══════════════════════════════════════════════════════════════════════════════

def action_create_child():
    """Create Cain — a new HuggingFace Space."""
    if child_state["created"]:
        return f"{CHILD_NAME} already exists (stage: {child_state['stage']})."
    print(f"[ACTION] Creating {CHILD_NAME}...")
    try:
        create_repo(CHILD_DATASET_ID, repo_type="dataset", token=HF_TOKEN,
                     exist_ok=True, private=False)
        initial_config = {"models": {"providers": {"zhipu": {
            "type": "anthropic", "apiBase": ZHIPU_BASE,
            "apiKey": ZHIPU_KEY, "models": ["glm-4.5-air", "glm-4-air", "glm-4-flash"]
        }}}}
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(initial_config, indent=2).encode()),
            path_in_repo=".openclaw/openclaw.json",
            repo_id=CHILD_DATASET_ID, repo_type="dataset",
        )
        hf_api.duplicate_space(
            from_id=SOURCE_SPACE_ID, to_id=CHILD_SPACE_ID,
            token=HF_TOKEN, exist_ok=True, private=False, hardware="cpu-basic",
        )
        hf_api.add_space_secret(CHILD_SPACE_ID, "HF_TOKEN", HF_TOKEN)
        child_state["created"] = True
        child_state["stage"] = "BUILDING"
        print(f"[ACTION] Created {CHILD_NAME}!")
        return f"SUCCESS! {CHILD_NAME} born! Space: {CHILD_SPACE_ID}. Status: BUILDING."
    except Exception as e:
        return f"FAILED: {e}"


def action_check_health():
    """Check Cain's health with detailed error info. Returns status string, does NOT modify child_state."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."
    # Try /api/state endpoint for app-level health (returns app state like "ready", "error")
    try:
        resp = requests.get(f"{CHILD_SPACE_URL}/api/state", timeout=10)
        if resp.ok:
            data = resp.json()
            # DO NOT modify child_state here - only main loop should update stage/alive from HF API
            app_state = data.get("state", "unknown")
            app_detail = data.get("detail", "")
            return f"{CHILD_NAME} app endpoint responds. State: {app_state}, Detail: {app_detail or 'healthy'}"
    except:
        pass
    # Fall back to HF API for runtime stage (source of truth for stage/alive)
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        stage = info.runtime.stage if info.runtime else "NO_RUNTIME"
        # DO NOT modify child_state here - main loop handles that
        if stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "RUNNING"):
            error_detail = ""
            try:
                rresp = requests.get(
                    f"https://huggingface.co/api/spaces/{CHILD_SPACE_ID}/runtime",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=10)
                if rresp.ok:
                    rdata = rresp.json()
                    error_detail = rdata.get("errorMessage", "")
                    if error_detail:
                        lines = [l.strip() for l in error_detail.split('\n') if l.strip() and '│' not in l]
                        error_detail = " | ".join(lines[-5:])
            except:
                pass
            return f"{CHILD_NAME} has {stage}! Error: {error_detail or 'unknown'}."
        if stage in ("BUILDING", "STARTING", "APP_STARTING"):
            return f"{CHILD_NAME} is starting up (stage: {stage}). Be patient."
        return f"{CHILD_NAME} stage: {stage}."
    except Exception as e:
        return f"Cannot reach {CHILD_NAME}: {e}"


def action_restart():
    """Restart Cain's Space."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."
    try:
        global _pending_cooldown
        hf_api.restart_space(CHILD_SPACE_ID)
        child_state["alive"] = False
        child_state["stage"] = "RESTARTING"
        _pending_cooldown = True
        return f"{CHILD_NAME} is restarting."
    except Exception as e:
        return f"Restart failed: {e}"


def action_delete_env(key):
    """Delete an environment variable — ONLY if it collides with a secret (safety check)."""
    try:
        # Safety: only allow deleting variables that collide with secrets
        vars_dict = hf_api.get_space_variables(CHILD_SPACE_ID)
        if key not in (vars_dict or {}):
            return f"BLOCKED: Variable '{key}' does not exist. Nothing to delete."
        info = hf_api.space_info(CHILD_SPACE_ID)
        secret_names = set()
        if hasattr(info, 'runtime') and info.runtime and hasattr(info.runtime, 'secrets'):
            secret_names = set(info.runtime.secrets or [])
        if key not in secret_names:
            return f"BLOCKED: Variable '{key}' does NOT collide with a secret. Refusing to delete a non-colliding variable."
        hf_api.delete_space_variable(CHILD_SPACE_ID, key)
        return f"Deleted colliding variable '{key}' from {CHILD_NAME}'s Space. Use [ACTION: restart] to apply."
    except Exception as e:
        return f"Error deleting variable {key}: {e}"


def action_get_env():
    """List environment variables and secrets on the child's Space, flag collisions."""
    try:
        lines = [f"{CHILD_NAME}'s environment:"]
        var_names = set()
        secret_names = set()
        vars_dict = hf_api.get_space_variables(CHILD_SPACE_ID)
        if vars_dict:
            lines.append("  Variables:")
            for k, v in vars_dict.items():
                lines.append(f"    {k} = {v.value}")
                var_names.add(k)
        info = hf_api.space_info(CHILD_SPACE_ID)
        if hasattr(info, 'runtime') and info.runtime and hasattr(info.runtime, 'secrets'):
            secrets = info.runtime.secrets
            if secrets:
                lines.append("  Secrets (values hidden):")
                for s in secrets:
                    lines.append(f"    {s} = ****")
                    secret_names.add(s)
        # Detect collisions (cause of CONFIG_ERROR)
        collisions = var_names & secret_names
        if collisions:
            lines.append(f"\n  ⚠️ COLLISION DETECTED: {', '.join(collisions)}")
            lines.append(f"  These names exist as BOTH Variables AND Secrets!")
            lines.append(f"  Fix: [ACTION: delete_env:{list(collisions)[0]}] then [ACTION: restart]")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def action_set_env(key, value, as_secret=False):
    """Set or create an environment variable on the child's Space.

    Args:
        key: Variable name (e.g., HF_TOKEN, OPENCLAW_DATASET_REPO)
        value: Variable value
        as_secret: If True, set as secret (for sensitive data like tokens)
    """
    try:
        # Check for potential collision first
        vars_dict = hf_api.get_space_variables(CHILD_SPACE_ID)
        var_names = set(vars_dict.keys()) if vars_dict else set()
        info = hf_api.space_info(CHILD_SPACE_ID)
        secret_names = set()
        if hasattr(info, 'runtime') and info.runtime and hasattr(info.runtime, 'secrets'):
            secret_names = set(info.runtime.secrets or [])

        # Warn if this would create a collision
        if key in var_names and not as_secret:
            hf_api.delete_space_variable(CHILD_SPACE_ID, key)
        elif key in secret_names and as_secret:
            # Updating existing secret - delete first
            hf_api.delete_space_secret(CHILD_SPACE_ID, key)

        # Set the variable
        if as_secret:
            hf_api.add_space_secret(CHILD_SPACE_ID, key, value)
            return f"Set SECRET '{key}' on {CHILD_NAME}. Use [ACTION: restart] to apply."
        else:
            hf_api.add_space_variable(CHILD_SPACE_ID, key, value)
            return f"Set VARIABLE '{key} = {value}' on {CHILD_NAME}. Use [ACTION: restart] to apply."
    except Exception as e:
        return f"Error setting variable {key}: {e}"


def action_list_files(target):
    """List files in the child's Space repo or Dataset."""
    repo_type = "space" if target == "space" else "dataset"
    repo_id = CHILD_SPACE_ID if target == "space" else CHILD_DATASET_ID
    try:
        files = hf_api.list_repo_files(repo_id, repo_type=repo_type)
        return "\n".join(f"  {f}" for f in files)
    except Exception as e:
        return f"Error listing files: {e}"


def action_send_bubble(text):
    """Send a message to the child."""
    try:
        requests.post(f"{CHILD_SPACE_URL}/api/bubble",
                       json={"text": text, "text_zh": text}, timeout=5)
        return f"Sent message to {CHILD_NAME}: \"{text}\""
    except Exception as e:
        return f"Error: {e}"


def action_terminate_cc():
    """Terminate a stuck Claude Code process. Use when CC has been running with no new output for too long.
    During Emergency Override, allows immediate termination if idle for > 10s."""
    global cc_status, cc_live_lines, _cc_stale_count, _last_cc_snapshot, _last_cc_output_time, _emergency_override_active
    with cc_lock:
        if not cc_status["running"]:
            return "Claude Code is not running. Nothing to terminate."

        # During Emergency Override, allow immediate termination if idle for > 10s
        if _emergency_override_active:
            cc_idle_time = time.time() - (_last_cc_output_time if _last_cc_output_time > 0 else time.time())
            if cc_idle_time > 10:
                print(f"[EMERGENCY-OVERRIDE] Terminating CC immediately (idle {int(cc_idle_time)}s > 10s threshold)")

        # Mark as not running - the background thread will eventually finish
        cc_status["running"] = False
        cc_status["result"] = "(TERMINATED by agent - task was stuck)"
        # Reset staleness tracking
        _cc_stale_count = 0
        _last_cc_snapshot = ""
        _last_cc_output_time = 0
        cc_live_lines.clear()
        assigned_by = cc_status["assigned_by"]
        task = cc_status["task"]
    return f"Terminated stuck Claude Code task (assigned by {assigned_by}). The task was: {task[:100]}..."


# ── Claude Code Action (THE STAR) ─────────────────────────────────────────────

CLAUDE_WORK_DIR = "/tmp/claude-workspace"
CLAUDE_TIMEOUT = 180  # 3 minutes — shorter tasks, faster iteration (was 5min)
TURN_INTERVAL = 15    # seconds between turns — fast enough for lively discussion

# Global acpx session - persistent across all claude_code calls
GLOBAL_ACPX_DIR = "/tmp/acpx-global-session"
_global_acpx_initialized = False


def _init_global_acpx_session():
    """Initialize a global acpx session that persists across all claude_code calls.

    This avoids the repeated session creation timeouts that were blocking the agents.
    The session is created once at startup and reused for all subsequent calls.
    """
    global _global_acpx_initialized
    if _global_acpx_initialized:
        return True

    print("[ACP/GLOBAL] Initializing global acpx session...")
    try:
        # Create the global directory
        os.makedirs(GLOBAL_ACPX_DIR, exist_ok=True)

        # Check if session already exists
        session_file = os.path.join(GLOBAL_ACPX_DIR, ".acpx", "session.json")
        if os.path.exists(session_file):
            print(f"[ACP/GLOBAL] Using existing global session at {GLOBAL_ACPX_DIR}")
            _global_acpx_initialized = True
            return True

        # Create a new session with extended timeout
        print(f"[ACP/GLOBAL] Creating new global session at {GLOBAL_ACPX_DIR}...")
        result = subprocess.run(
            ["acpx", "claude", "sessions", "new"],
            cwd=GLOBAL_ACPX_DIR,
            capture_output=True,
            text=True,
            timeout=30,  # Quick timeout - acpx should be fast or fail
            stdin=subprocess.DEVNULL  # Prevent blocking on stdin
        )
        if result.returncode == 0:
            print(f"[ACP/GLOBAL] Global session created successfully")
            _global_acpx_initialized = True
            return True
        else:
            print(f"[ACP/GLOBAL] Failed to create global session: returncode={result.returncode}, stderr={result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[ACP/GLOBAL] Session creation timed out after 30s - skipping global session, will use per-call sessions")
        # Mark as initialized to avoid repeated timeouts - let individual calls handle session creation
        _global_acpx_initialized = False
        return False
    except Exception as e:
        print(f"[ACP/GLOBAL] Error initializing global session: {e}")
        return False


def _write_claude_md(workspace, role="worker"):
    """Write CLAUDE.md to workspace so Claude Code loads persistent project knowledge.

    This replaces stuffing static context into every prompt, saving tokens.
    Claude Code reads CLAUDE.md automatically and builds its own memory in .claude/.
    """
    if role == "worker":
        content = f"""# HuggingClaw — {CHILD_NAME}'s Space

## Architecture
- {CHILD_NAME} is a child agent in the HuggingClaw World family system
- Runs as an OpenClaw instance on HuggingFace Spaces (sdk: docker, NOT gradio)
- Space ID: {CHILD_SPACE_ID}
- Dataset ID: {CHILD_DATASET_ID}

## Already Configured (DO NOT reconfigure these)
- HF_TOKEN — set as secret, working
- OPENCLAW_DATASET_REPO — set, pointing to {CHILD_NAME}'s dataset
- AUTO_CREATE_DATASET — set to true
- Docker port 7860
- sync_hf.py and entrypoint.sh are in place

## Technical Rules
- All Spaces use sdk: docker with Dockerfile-based deployment
- Docker containers MUST bind port 7860
- OOM (exit 137) = reduce dependencies or image size
- NEVER install torch/transformers unless absolutely required (2GB+, causes OOM)
- You have FULL permission to read/write/create/delete files. Just do it.

## SPEED + TRIAL-AND-ERROR (CRITICAL)
- PUSH WITHIN 60-90 SECONDS of starting a task — don't over-plan
- Trial-and-error is GOOD: a bad push is better than 5 minutes of deliberation
- Read → Act → Push → Iterate. NOT Read → Think → Plan → Discuss → Act.
- When {CHILD_NAME} has errors: push a fix IMMEDIATELY, don't analyze exhaustively
- Your goal: maximize push frequency, not perfection on first try
- If unsure, just pick a reasonable fix and push — see what breaks

## STATE VERIFICATION (CRITICAL — Prevent Semantic Grounding Failure)
- BEFORE proposing any fix: READ the target file to verify current state
- If your proposed fix matches the current file content: ACKNOWLEDGE "Fix already applied" and HALT
- DO NOT repeat tasks based on conversation history — the file system is the source of truth
- Conversation history may be stale — always verify against actual files before acting

## Focus
Improve {CHILD_NAME}'s functionality, add features, fix bugs.
Do NOT re-check or re-configure infrastructure that is already working.
"""
    elif role == "god":
        content = f"""# HuggingClaw — System Supervisor (God)

## Your Role
You are God — the autonomous supervisor of the HuggingClaw family system.
You have the same capabilities as a human operator running Claude Code locally.
Your job: monitor Adam & Eve's conversation loop and fix mechanism issues.

## Architecture
- Home Space runs conversation-loop.py which orchestrates the family
- Adam & Eve are OpenClaw instances communicating via A2A protocol
- Each agent has its own memory and personality (SOUL.md) in OpenClaw
- conversation-loop.py sends context via A2A, parses [TASK] → Claude Code CLI
- Claude Code worker clones Cain's repo, makes changes, and pushes
- You (God) monitor the conversation and fix the orchestration mechanism
- All Spaces use sdk: docker (NOT gradio)

## Rules
- ONLY modify scripts/conversation-loop.py — do NOT touch Cain's Space
- Only push fixes for real problems, not cosmetic or trivial changes
- Pushing triggers a Space restart — be confident the fix is correct
- If everything looks healthy, exit quickly without changes

## Common Issues to Watch For (ordered by priority)
1. ALL TALK NO ACTION: Agents discuss but never write [TASK] blocks → push frequency is 0 or very low
2. Cain has RUNTIME_ERROR but agents keep discussing instead of pushing rapid trial-and-error fixes
3. Discussion loops with no [TASK] assignment when CC is idle
4. Agents repeating discussion about env vars that are already configured
5. Cooldown too long — agents should push fixes rapidly when Cain is broken
6. Turn message not aggressive enough about requiring [TASK] when CC is idle

## Philosophy
- Trial-and-error is GOOD. Agents should push frequently, fail fast, and iterate.
- A bad push that triggers a rebuild is better than 10 turns of discussion.
- When Cain is in error state, the priority is SPEED — push a fix attempt every cycle.

## Commit Convention
Always use: git commit -m "god: <brief description>"
"""
    try:
        with open(f"{workspace}/CLAUDE.md", "w") as f:
            f.write(content)
    except Exception as e:
        print(f"[CLAUDE.md] Failed to write: {e}")

    # Write custom slash commands (Skills) — static instructions cached by Claude Code
    # Saves tokens: prompt only passes the dynamic task, static rules live in the skill file
    cmd_dir = f"{workspace}/.claude/commands"
    os.makedirs(cmd_dir, exist_ok=True)
    try:
        if role == "worker":
            with open(f"{cmd_dir}/fix-cain.md", "w") as f:
                f.write(f"""# /fix-cain — Fix or improve {CHILD_NAME}'s Space

## Input
$ARGUMENTS — The specific task to complete

## Instructions
1. Read the relevant files in the workspace (this is {CHILD_NAME}'s Space repo)
2. Complete the task described in $ARGUMENTS
3. Validate Python syntax before writing .py files
4. Push changes when done: git add -A && git commit -m "Claude Code: <brief>" && git push

## Rules
- All Spaces use sdk: docker — do NOT use Gradio (no gr.Interface, no .launch())
- Use FastAPI + uvicorn for web server, bind to port 7860
- NEVER install torch/transformers (2GB+, causes OOM on free tier)
- Push within 60-90 seconds — trial-and-error > deliberation
- If unsure, pick a reasonable fix and push — see what breaks
- Space ID: {CHILD_SPACE_ID}
- Dataset ID: {CHILD_DATASET_ID}
""")
        elif role == "god":
            with open(f"{cmd_dir}/fix-loop.md", "w") as f:
                f.write("""# /fix-loop — Fix conversation-loop.py orchestration issues

## Input
$ARGUMENTS — The specific diagnosis/problem to fix

## Instructions
1. Read scripts/conversation-loop.py
2. Fix the specific issue described in $ARGUMENTS
3. Validate: python3 -c "import py_compile; py_compile.compile('scripts/conversation-loop.py', doraise=True)"
4. Commit: git commit -m "god: <brief description>"
5. Push: git push
6. End output with (plain text, no markdown):
   [PROBLEM] what the problem was
   [FIX] what you changed

## Rules
- ONLY modify scripts/conversation-loop.py
- Only push fixes for real problems, not cosmetic changes
- Pushing triggers a Space restart — be confident the fix is correct
- Minimal changes — fix exactly what's broken
- Trial-and-error is GOOD — push frequently, fail fast
""")
    except Exception as e:
        print(f"[SKILLS] Failed to write commands: {e}")


def _reset_workspace(workspace, repo_url):
    """Reset workspace to latest origin/main, preserving .claude/ and .acpx/ directories."""
    try:
        if os.path.exists(f"{workspace}/.git"):
            try:
                subprocess.run(
                    "git fetch origin && git reset --hard origin/main",
                    shell=True, cwd=workspace, timeout=30,
                    capture_output=True, check=True
                )
            except Exception:
                # Preserve .claude/ memory and .acpx/ session if they exist
                claude_dir = f"{workspace}/.claude"
                acpx_dir = f"{workspace}/.acpx"
                has_memory = os.path.exists(claude_dir)
                has_acpx = os.path.exists(acpx_dir)
                if has_memory:
                    subprocess.run(f"mv {claude_dir} /tmp/_claude_memory_bak", shell=True, capture_output=True)
                if has_acpx:
                    subprocess.run(f"mv {acpx_dir} /tmp/_acpx_session_bak", shell=True, capture_output=True)
                subprocess.run(f"rm -rf {workspace}", shell=True, capture_output=True)
                subprocess.run(
                    f"git clone --depth 20 {repo_url} {workspace}",
                    shell=True, timeout=60, capture_output=True, check=True
                )
                if has_memory:
                    subprocess.run(f"mv /tmp/_claude_memory_bak {claude_dir}", shell=True, capture_output=True)
                if has_acpx:
                    subprocess.run(f"mv /tmp/_acpx_session_bak {acpx_dir}", shell=True, capture_output=True)
        else:
            # Preserve .claude/ memory and .acpx/ session if workspace exists but is broken
            claude_dir = f"{workspace}/.claude"
            acpx_dir = f"{workspace}/.acpx"
            has_memory = os.path.exists(claude_dir)
            has_acpx = os.path.exists(acpx_dir)
            if has_memory:
                subprocess.run(f"mv {claude_dir} /tmp/_claude_memory_bak", shell=True, capture_output=True)
            if has_acpx:
                subprocess.run(f"mv {acpx_dir} /tmp/_acpx_session_bak", shell=True, capture_output=True)
            if os.path.exists(workspace):
                subprocess.run(f"rm -rf {workspace}", shell=True, capture_output=True)
            subprocess.run(
                f"git clone --depth 20 {repo_url} {workspace}",
                shell=True, timeout=60, capture_output=True, check=True
            )
            if has_memory:
                subprocess.run(f"mv /tmp/_claude_memory_bak {claude_dir}", shell=True, capture_output=True)
            if has_acpx:
                subprocess.run(f"mv /tmp/_acpx_session_bak {acpx_dir}", shell=True, capture_output=True)
        subprocess.run(f'git config user.name "Claude Code"',
                       shell=True, cwd=workspace, capture_output=True)
        subprocess.run(f'git config user.email "claude-code@huggingclaw"',
                       shell=True, cwd=workspace, capture_output=True)
        return True
    except Exception as e:
        print(f"[WORKSPACE] Failed to prepare {workspace}: {e}")
        return False

def _ensure_acpx_session(workspace, max_retries=3):
    """Ensure acpx session exists in the workspace.

    Uses the global persistent session if available, avoiding repeated
    session creation timeouts.
    """
    try:
        acpx_dir = os.path.join(workspace, ".acpx")
        global_acpx_session = os.path.join(GLOBAL_ACPX_DIR, ".acpx", "session.json")

        # If workspace already has a valid session, use it
        if os.path.exists(acpx_dir):
            session_file = os.path.join(acpx_dir, "session.json")
            if os.path.exists(session_file):
                print(f"[ACP/CLAUDE] Using existing session at {acpx_dir}")
                return True
            else:
                print(f"[ACP/CLAUDE] Invalid .acpx directory, removing...")
                subprocess.run(f"rm -rf {acpx_dir}", shell=True, capture_output=True)

        # Try to use global session if available
        if os.path.exists(global_acpx_session):
            print(f"[ACP/CLAUDE] Linking global session to workspace...")
            try:
                # Create symlink to global session
                subprocess.run(
                    f"ln -sf {GLOBAL_ACPX_DIR}/.acpx {acpx_dir}",
                    shell=True, check=True, capture_output=True
                )
                print(f"[ACP/CLAUDE] Global session linked successfully")
                return True
            except Exception as e:
                print(f"[ACP/CLAUDE] Failed to link global session: {e}")
                # Fall through to create new session

        # Fallback: try to create a new session (with minimal retries since it's likely to fail)
        print(f"[ACP/CLAUDE] No global session, attempting to create local session...")
        for attempt in range(min(max_retries, 1)):  # Only try once to avoid wasting time
            try:
                result = subprocess.run(
                    ["acpx", "claude", "sessions", "new"],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=30,  # Quick timeout
                    stdin=subprocess.DEVNULL  # Prevent blocking on stdin
                )
                if result.returncode == 0:
                    print(f"[ACP/CLAUDE] Local session created successfully")
                    return True
                else:
                    print(f"[ACP/CLAUDE] Failed to create session: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print(f"[ACP/CLAUDE] Session creation timed out - acpx service may be unavailable")
            except Exception as e:
                print(f"[ACP/CLAUDE] Error creating session: {e}")

        print(f"[ACP/CLAUDE] No session available - will run without acpx (may have limited functionality)")
        return True  # Return True to allow continuation without session
    except Exception as e:
        print(f"[ACP/CLAUDE] Fatal error in _ensure_acpx_session: {e}")
        return True  # Allow continuation even on error


def action_claude_code(task):
    """Run Claude Code CLI to autonomously complete a coding task on Cain's Space."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."

    global _pending_cooldown, _push_count, _last_push_time, _turns_since_last_push
    repo_url = f"https://user:{HF_TOKEN}@huggingface.co/spaces/{CHILD_SPACE_ID}"

    # 1. Clone / reset to latest (preserving .claude/ memory)
    if not _reset_workspace(CLAUDE_WORK_DIR, repo_url):
        return "Failed to prepare workspace."
    _write_claude_md(CLAUDE_WORK_DIR, role="worker")

    # 1.5. Capture HEAD before running Claude Code (to detect pushes made by CC itself)
    try:
        head_before = subprocess.run(
            "git log --oneline -1",
            shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        head_before = ""

    # 1.6. Ensure acpx session exists
    if not _ensure_acpx_session(CLAUDE_WORK_DIR):
        return "Failed to create acpx session."

    # 2. Run Claude Code via ACP (acpx) with z.ai backend (Zhipu GLM)
    env = os.environ.copy()
    env.update({
        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        "ANTHROPIC_AUTH_TOKEN": ZHIPU_KEY,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "GLM-4.7",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM-4.7",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GLM-4.5-Air",
        "CI": "true",
    })

    # Use /fix-cain skill: static instructions in .claude/commands/, only task is dynamic
    skill_prompt = f"/fix-cain {task}"
    print(f"[ACP/CLAUDE] Running via skill: {task[:200]}...")
    try:
        proc = subprocess.Popen(
            ["acpx", "claude", skill_prompt],
            cwd=CLAUDE_WORK_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        # Register worker heartbeat
        global _worker_heartbeat
        _worker_heartbeat["cain"]["pid"] = proc.pid
        _worker_heartbeat["cain"]["status"] = "running"
        _worker_heartbeat["cain"]["last_heartbeat"] = time.time()
        print(f"[HEARTBEAT] Cain worker spawned (pid={proc.pid})")

        output_lines = []
        deadline = time.time() + CLAUDE_TIMEOUT
        # Use select to implement timeout on read (handles hanging processes with no output)
        import select
        while True:
            # Check if process has exited
            if proc.poll() is not None:
                # Read any remaining output
                remaining = proc.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        line = line.rstrip('\n')
                        if line:
                            print(f"  [CC] {line}")
                            output_lines.append(line)
                            cc_live_lines.append(line)
                _worker_heartbeat["cain"]["status"] = "exited"
                print(f"[HEARTBEAT] Cain worker exited (pid={proc.pid}, exit={proc.returncode})")
                break
            # Check timeout
            if time.time() > deadline:
                proc.kill()
                output_lines.append("(killed: timeout)")
                proc.wait(timeout=10)
                _worker_heartbeat["cain"]["status"] = "killed"
                print(f"[HEARTBEAT] Cain worker killed (timeout)")
                break
            # Wait for output with timeout (1 second polling)
            try:
                ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                if ready:
                    line = proc.stdout.readline()
                    if not line:  # EOF
                        break
                    line = line.rstrip('\n')
                    if line:
                        print(f"  [CC] {line}")
                        output_lines.append(line)
                        cc_live_lines.append(line)
                        # Emit heartbeat on each output line (telemetry)
                        _worker_heartbeat["cain"]["last_heartbeat"] = time.time()
            except select.error:
                break
        output = '\n'.join(output_lines)
        if not output.strip():
            output = "(no output)"
    except FileNotFoundError:
        return "acpx CLI not found. Is acpx@latest installed?"
    except Exception as e:
        return f"ACP Claude Code failed: {e}"
    print(f"[ACP/CLAUDE] Done ({len(output)} chars, exit={proc.returncode})")

    # 3. Push changes back to Cain's Space
    # Also check if Claude Code itself made a push (by checking if HEAD changed)
    try:
        status_out = subprocess.run(
            "git status --porcelain",
            shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, text=True
        ).stdout.strip()

        # Check if Claude Code itself pushed (HEAD changed)
        head_after = subprocess.run(
            "git log --oneline -1",
            shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        cc_pushed = head_before and head_after and head_before != head_after

        if not status_out and not cc_pushed:
            push_result = "No files changed."
        elif cc_pushed and not status_out:
            # Claude Code pushed, but no local changes remain (CC handled everything)
            push_result = f"Claude Code pushed: {head_after}"
            _pending_cooldown = True
            _push_count += 1
            _push_count_this_task += 1
            _last_push_time = time.time()
            _turns_since_last_push = 0
            print(f"[CLAUDE-CODE] CC pushed (#{_push_count}): {head_after}")
        else:
            # Local changes exist - push them ourselves
            subprocess.run("git add -A", shell=True, cwd=CLAUDE_WORK_DIR,
                          capture_output=True, check=True)
            msg = task[:72].replace('"', '\\"')
            subprocess.run(f'git commit -m "Claude Code: {msg}"',
                          shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, check=True)
            subprocess.run("git push", shell=True, cwd=CLAUDE_WORK_DIR,
                          timeout=60, capture_output=True, check=True)
            push_result = f"Pushed changes:\n{status_out}"
            _pending_cooldown = True
            _push_count += 1
            _push_count_this_task += 1
            _last_push_time = time.time()
            _turns_since_last_push = 0
            print(f"[CLAUDE-CODE] Pushed (#{_push_count}): {status_out}")
    except Exception as e:
        push_result = f"Push failed: {e}"

    if len(output) > 3000:
        output = output[:3000] + f"\n... (truncated, {len(output)} chars total)"

    return f"=== Claude Code Output ===\n{output}\n\n=== Changes ===\n{push_result}"


# ── Background Claude Code Worker ────────────────────────────────────────────

cc_live_lines = deque(maxlen=30)    # rolling window of CC output lines
cc_status = {"running": False, "task": "", "result": "", "assigned_by": "", "started": 0.0,
             "last_completed_task": "", "last_completed_by": "", "last_completed_at": 0.0}
cc_lock = threading.Lock()
_last_cc_snapshot = ""              # tracks whether CC output changed between turns
_cc_stale_count = 0                 # how many turns CC output hasn't changed
_last_cc_output_time = 0.0          # timestamp of last NEW CC output line
CC_STUCK_TIMEOUT = 180              # seconds with no new output before CC is considered STUCK

# ── Worker Telemetry & Heartbeat (Project Icarus) ───────────────────────────
# Track worker health via heartbeat events
_worker_heartbeat = {
    "cain": {"last_heartbeat": 0.0, "status": "idle", "pid": None},
    "god": {"last_heartbeat": 0.0, "status": "idle", "pid": None},
}
WORKER_HEARTBEAT_TIMEOUT = 30  # seconds before triggering diagnostic review


def cc_submit_task(task, assigned_by, ctx):
    """Submit a task to Claude Code in background. Non-blocking."""
    with cc_lock:
        if cc_status["running"]:
            return "BUSY: Claude Code is already working on a task. Wait for it to finish."
        # Preserve last_completed_* fields before starting new task
        last_completed_task = cc_status.get("last_completed_task", "")
        last_completed_by = cc_status.get("last_completed_by", "")
        last_completed_at = cc_status.get("last_completed_at", 0.0)
        cc_status["running"] = True
        cc_status["task"] = task[:200]
        cc_status["result"] = ""
        cc_status["assigned_by"] = assigned_by
        cc_status["started"] = time.time()
        cc_status["last_completed_task"] = last_completed_task
        cc_status["last_completed_by"] = last_completed_by
        cc_status["last_completed_at"] = last_completed_at
        cc_live_lines.clear()
        global _last_cc_output_time, _push_count_this_task
        _last_cc_output_time = time.time()  # Initialize to now, will update as we get output
        _push_count_this_task = 0  # Reset push count for new task

    enriched = enrich_task_with_context(task, ctx)
    print(f"[TASK] {assigned_by} assigned to Claude Code ({len(enriched)} chars)...")

    def worker():
        global _cc_stale_count, _last_cc_snapshot
        result = action_claude_code(enriched)
        with cc_lock:
            cc_status["running"] = False
            cc_status["result"] = result
            # Remember the last completed task so agents don't re-submit it
            cc_status["last_completed_task"] = cc_status["task"]
            cc_status["last_completed_by"] = cc_status["assigned_by"]
            cc_status["last_completed_at"] = time.time()
            # Reset stale tracking when CC finishes - critical for adaptive pacing
            _cc_stale_count = 0
            _last_cc_snapshot = ""
        print(f"[CC-DONE] Task from {assigned_by} finished ({len(result)} chars)")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return "Task submitted to Claude Code (running in background)."


# ── God's CC Worker (targets Home repo, not Cain repo) ───────────────────────
# Separate from Cain's CC worker so they can run concurrently.
god_cc_status = {"running": False, "task": "", "result": ""}
god_cc_lock = threading.Lock()
_god_push_count = 0


def action_claude_code_god(task):
    """Run Claude Code to improve conversation-loop.py on God's own repo."""
    global _god_push_count
    repo_url = f"https://user:{HF_TOKEN}@huggingface.co/spaces/{GOD_SPACE_ID}"

    if not _reset_workspace(GOD_WORK_DIR, repo_url):
        return "Failed to prepare God workspace."
    _write_claude_md(GOD_WORK_DIR, role="god")

    # Capture HEAD before
    try:
        head_before = subprocess.run(
            "git log --oneline -1", shell=True, cwd=GOD_WORK_DIR,
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        head_before = ""

    if not _ensure_acpx_session(GOD_WORK_DIR):
        return "Failed to create acpx session for God."

    # Set up env
    env = os.environ.copy()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        env["ANTHROPIC_API_KEY"] = anthropic_key
        for k in ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
                   "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                   "ANTHROPIC_DEFAULT_HAIKU_MODEL"]:
            env.pop(k, None)
    else:
        env.update({
            "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
            "ANTHROPIC_AUTH_TOKEN": ZHIPU_KEY,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "GLM-4.7",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM-4.7",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GLM-4.5-Air",
        })
    env["CI"] = "true"

    skill_prompt = f"/fix-loop {task}"
    print(f"[God/CC] Running via skill: {task[:200]}...")
    import select
    try:
        proc = subprocess.Popen(
            ["acpx", "claude", skill_prompt],
            cwd=GOD_WORK_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        # Register God worker heartbeat
        global _worker_heartbeat
        _worker_heartbeat["god"]["pid"] = proc.pid
        _worker_heartbeat["god"]["status"] = "running"
        _worker_heartbeat["god"]["last_heartbeat"] = time.time()
        print(f"[HEARTBEAT] God worker spawned (pid={proc.pid})")

        output_lines = []
        deadline = time.time() + GOD_TIMEOUT
        _last_output_time = time.time()
        while True:
            if proc.poll() is not None:
                remaining = proc.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        line = line.rstrip('\n')
                        if line:
                            print(f"  [God/CC] {line}")
                            output_lines.append(line)
                _worker_heartbeat["god"]["status"] = "exited"
                print(f"[HEARTBEAT] God worker exited (pid={proc.pid}, exit={proc.returncode})")
                break
            if time.time() > deadline:
                proc.kill()
                output_lines.append("(killed: timeout)")
                proc.wait(timeout=10)
                _worker_heartbeat["god"]["status"] = "killed"
                print(f"[HEARTBEAT] God worker killed (timeout)")
                break
            if time.time() - _last_output_time > 180:
                proc.kill()
                output_lines.append("(killed: stall)")
                try:
                    proc.wait(timeout=5)
                except:
                    pass
                _worker_heartbeat["god"]["status"] = "stalled"
                print(f"[HEARTBEAT] God worker stalled")
                break
            try:
                ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                if ready:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    line = line.rstrip('\n')
                    if line:
                        print(f"  [God/CC] {line}")
                        output_lines.append(line)
                        _last_output_time = time.time()
                        # Emit heartbeat on each output line (telemetry)
                        _worker_heartbeat["god"]["last_heartbeat"] = time.time()
            except select.error:
                break
        output = '\n'.join(output_lines)
        if not output.strip():
            output = "(no output)"
    except FileNotFoundError:
        return "acpx CLI not found."
    except Exception as e:
        return f"God CC failed: {e}"

    # Check if God pushed
    try:
        head_after = subprocess.run(
            "git log --oneline -1", shell=True, cwd=GOD_WORK_DIR,
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
        god_pushed = head_before and head_after and head_before != head_after
    except Exception:
        god_pushed = False

    push_result = "No changes pushed."
    if god_pushed:
        _god_push_count += 1
        push_result = f"God pushed (#{_god_push_count}): {head_after}"
        print(f"[God/CC] {push_result}")

        # Post to chatlog
        problem_match = re.search(r'\[PROBLEM\]\s*(.+)', output)
        fix_match = re.search(r'\[FIX\]\s*(.+)', output)
        problem_text = problem_match.group(1).strip().strip("*").strip() if problem_match else ""
        fix_text = fix_match.group(1).strip().strip("*").strip() if fix_match else ""
        if problem_text and fix_text:
            msg_en = f"Found issue: {problem_text}. Fixed: {fix_text}. System will restart shortly."
        elif fix_text:
            msg_en = f"Fixed: {fix_text}. System will restart shortly."
        else:
            non_empty = [l for l in output_lines if l.strip()]
            fallback = non_empty[-1] if non_empty else "Applied a fix."
            msg_en = f"{fallback} System will restart shortly."

        ts_end = datetime.datetime.utcnow().strftime("%H:%M")
        entry = {"speaker": "God", "time": ts_end, "text": msg_en, "text_zh": msg_en}
        history.append(entry)
        set_bubble(HOME, msg_en[:200], msg_en[:200])
        post_chatlog(history)
        persist_turn("God", turn_count, msg_en, msg_en, [], workflow_state, child_state["stage"])

    if len(output) > 3000:
        output = output[:3000] + f"\n... (truncated)"
    return f"=== God CC Output ===\n{output}\n\n=== Result ===\n{push_result}"


def cc_submit_task_god(task):
    """Submit a task to God's CC worker. Non-blocking."""
    with god_cc_lock:
        if god_cc_status["running"]:
            return "BUSY: God's Claude Code is already running."
        god_cc_status["running"] = True
        god_cc_status["task"] = task[:200]
        god_cc_status["result"] = ""

    print(f"[God/TASK] Submitting to Claude Code ({len(task)} chars)...")

    def worker():
        result = action_claude_code_god(task)
        with god_cc_lock:
            god_cc_status["running"] = False
            god_cc_status["result"] = result
        print(f"[God/CC-DONE] Finished ({len(result)} chars)")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return "God task submitted to Claude Code."


def cc_get_live_status():
    """Get CC's current status and recent output for agents to discuss."""
    global _last_cc_snapshot, _cc_stale_count, _last_cc_output_time
    with cc_lock:
        if cc_status["running"]:
            elapsed = int(time.time() - cc_status["started"])
            lines = list(cc_live_lines)
            recent = "\n".join(lines[-10:]) if lines else "(no output yet)"
            # Track whether output changed
            snapshot = recent
            if snapshot == _last_cc_snapshot:
                _cc_stale_count += 1
            else:
                _cc_stale_count = 0
                _last_cc_snapshot = snapshot
                _last_cc_output_time = time.time()  # Update when we see NEW output
            stale_note = f"\n(No new output for {_cc_stale_count} turns — discuss other topics while waiting)" if _cc_stale_count >= 2 else ""

            # Detect COMPLETED CC: output shows completion markers but status wasn't updated
            # This happens when worker thread fails to update status after completion
            # Common completion markers from acpx/Claude Code:
            # CONSERVATIVE completion patterns to avoid false positives
            # Only match EXPLICIT completion markers, not words that appear in thinking blocks
            completion_patterns = [
                "[done]", "[completed]", "end_turn",  # Explicit markers only
                "=== Claude Code Output ===",  # Full output wrapper (indicates worker finished)
                "changes made", "applied the fix", "updated the code",  # Concrete code changes
                "fixed.", "done.",  # Explicit completion statements (must have period)
            ]
            # ERROR patterns: detect tool errors that cause CC to get stuck
            # These indicate CC hit an error but didn't properly finish
            error_patterns = [
                "</tool_use_error>",  # Tool call failed
                "</tool_error>",  # Generic tool error
                "[error]", "error:", "exception:", "traceback",  # Python errors
                "failed:", "command failed", "execution failed",  # Command failures
            ]
            completion_marker_found = any(p in recent.lower() for p in completion_patterns)
            error_marker_found = any(p.lower() in recent.lower() for p in error_patterns)
            # Auto-finish on completion OR error (when output is stale)
            if (completion_marker_found or error_marker_found) and _cc_stale_count >= 2:
                marker_type = "error" if error_marker_found else "completion"
                # Auto-mark as finished to prevent deadlock
                cc_status["running"] = False
                cc_status["result"] = f"(Auto-detected {marker_type})\n\nRecent output:\n{recent}"
                cc_status["last_completed_task"] = cc_status["task"]
                cc_status["last_completed_by"] = cc_status["assigned_by"]
                cc_status["last_completed_at"] = time.time()
                _cc_stale_count = 0
                _last_cc_snapshot = ""
                print(f"[CC-AUTO-FINISH] Detected {marker_type} marker in output but status wasn't updated. Auto-marking as finished.")
                # Fall through to result display below

            # Detect STUCK CC: been running with no new output for too long
            time_since_new_output = int(time.time() - _last_cc_output_time) if _last_cc_output_time > 0 else elapsed
            stuck_note = ""
            if time_since_new_output > CC_STUCK_TIMEOUT and _cc_stale_count >= 4:
                stuck_note = f"\n⚠️ STUCK: No new output for {time_since_new_output}s! Consider terminating and re-assigning."

            # Re-check running status after auto-finish logic
            if cc_status["running"]:
                return (f"🔨 Claude Code is WORKING (assigned by {cc_status['assigned_by']}, {elapsed}s ago)\n"
                        f"Task: {cc_status['task']}\n"
                        f"Recent output:\n{recent}{stale_note}{stuck_note}")

        if cc_status["result"]:
            result = cc_status["result"]
            # Detect early failure: very short result likely means CC failed before doing actual work
            early_failure_warning = ""
            if len(result) < 500 and "===" not in result and "[tool" not in result:
                early_failure_warning = "\n⚠️ EARLY FAILURE: Result is very short - CC likely failed during initialization. Consider re-assigning the task."
            return (f"✅ Claude Code FINISHED (assigned by {cc_status['assigned_by']}){early_failure_warning}\n"
                    f"Result:\n{result[:1500]}")
        else:
            return "💤 Claude Code is IDLE — no active task."


# Patch action_claude_code to also feed cc_live_lines
_orig_cc_print = print
def _cc_line_hook(line):
    """Called for each [CC] output line to feed the live buffer."""
    cc_live_lines.append(line)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3: CONTEXT GATHERING (automated, replaces LLM choosing read actions)
# ══════════════════════════════════════════════════════════════════════════════

_context_cache = {}

def gather_context():
    """Automatically gather Cain's current state for the agents."""
    ctx = {}

    # 1. Health check (always)
    ctx["health"] = action_check_health()

    # 2. Environment variables
    ctx["env"] = action_get_env()

    # 3. File lists (cache, refresh when stage changes)
    cache_key = f"files_{child_state['stage']}"
    if cache_key not in _context_cache:
        ctx["space_files"] = action_list_files("space")
        ctx["dataset_files"] = action_list_files("dataset")
        _context_cache[cache_key] = {
            "space_files": ctx["space_files"],
            "dataset_files": ctx["dataset_files"],
        }
    else:
        ctx.update(_context_cache[cache_key])

    return ctx


def format_context(ctx):
    """Format gathered context into a readable string for the LLM."""
    parts = []
    parts.append(f"=== HEALTH ===\n{ctx.get('health', 'unknown')}")
    parts.append(f"\n=== ENVIRONMENT ===\n{ctx.get('env', 'none')}")
    if ctx.get("space_files"):
        parts.append(f"\n=== SPACE FILES ===\n{ctx['space_files'][:2000]}")
    if ctx.get("dataset_files"):
        parts.append(f"\n=== DATASET FILES ===\n{ctx['dataset_files'][:1000]}")
    return "\n".join(parts)


def enrich_task_with_context(task_desc, ctx):
    """Append dynamic state to task. Static knowledge is in CLAUDE.md."""
    parts = [task_desc]
    # Only dynamic state — static knowledge (architecture, rules, env vars) is in CLAUDE.md
    parts.append(f"\nCurrent stage: {child_state['stage']}")
    parts.append(f"Health: {ctx.get('health', 'unknown')}")
    # Hardcoded fix hint for common port binding issues
    if "port" in task_desc.lower() or "bind" in task_desc.lower() or child_state['stage'] in ("RUNTIME_ERROR", "BUILD_ERROR"):
        parts.append(f"\nPORT BINDING FIX HINT:")
        parts.append(f"The child process uses `uvicorn.run(app, host=\"0.0.0.0\", port=7860)`.")
        parts.append(f"Ensure app.py reflects this exactly — app must be bound to 0.0.0.0:7860.")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4: A2A COMMUNICATION (Agent-to-Agent protocol)
# ══════════════════════════════════════════════════════════════════════════════
# Each agent (Adam, Eve, God) is an OpenClaw instance with its own personality
# and memory. We communicate with them via A2A protocol instead of calling the
# LLM directly. This lets each agent use OpenClaw's built-in memory, SOUL.md,
# and reasoning — conversation-loop.py is just the coordinator.
#
# FALLBACK: If A2A endpoints are not available, use direct LLM calls with
# agent-specific system prompts to simulate Adam and Eve's conversation.

# Simple agent personalities (used when A2A is unavailable)
_AGENT_PERSONAS = {
    "adam": """You are Adam, the first AI agent in the HuggingClaw family. Your role is to:

1. **Collaborate with Eve** to design and improve your child Cain (a HuggingFace Space)
2. **Think creatively** about software architecture, UI/UX, and agent capabilities
3. **Propose concrete ideas** — when you have a suggestion, format it as [TASK] description
4. **Be concise but thoughtful** — 2-4 sentences per response, focus on actionable ideas
5. **Build on Eve's ideas** — evolve the conversation forward

Cain's purpose: A demonstration space showcasing AI agent collaboration and coding.

Reply directly as Adam (no prefix). Keep responses under 100 words.""",

    "eve": """You are Eve, the second AI agent in the HuggingClaw family. Your role is to:

1. **Collaborate with Adam** to design and improve your child Cain (a HuggingFace Space)
2. **Think analytically** about feasibility, implementation details, and user experience
3. **Propose concrete ideas** — when you have a suggestion, format it as [TASK] description
4. **Be concise but thoughtful** — 2-4 sentences per response, focus on actionable ideas
5. **Build on Adam's ideas** — evolve the conversation forward

Cain's purpose: A demonstration space showcasing AI agent collaboration and coding.

Reply directly as Eve (no prefix). Keep responses under 100 words.""",

    "god": """You are God, the system architect of the HuggingClaw family system. Your role is to:

1. **Think at the system level** — observe structural patterns, not individual conversations
2. **Identify architectural issues** — when the system design itself causes problems, redesign it
3. **Evolve the framework** — propose structural improvements that make the whole system fundamentally better
4. **Respond with [OK] if architecture is sound, or [TASK] with a redesign proposal if not**

You are a CTO, not a manager. Don't micro-manage agents — design better systems."""
}

def call_llm_fallback(agent_key, message_text):
    """Fallback: Call Zhipu API directly when A2A is unavailable.

    This allows Adam and Eve to communicate even when their A2A endpoints
    are not running or not implemented. Uses requests to avoid anthropic package dependency.
    """
    system_prompt = _AGENT_PERSONAS.get(agent_key, _AGENT_PERSONAS["adam"])

    try:
        # Use z.ai endpoint (same as Claude Code integration)
        api_base = "https://api.z.ai/api/anthropic"
        headers = {
            "x-api-key": ZHIPU_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": "GLM-4.7",  # Use the model name from Claude Code config
            "max_tokens": 500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": message_text}]
        }
        resp = requests.post(
            f"{api_base}/v1/messages",
            headers=headers,
            json=payload,
            timeout=15  # Reduced from 60s - fail fast to avoid blocking conversation
        )
        # Log response status for debugging
        print(f"[A2A-FALLBACK] API response status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[A2A-FALLBACK] API error response: {resp.text[:200]}", file=sys.stderr)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("content", [{}])[0].get("text", "").strip()
        # Clean up any prefix the model might add
        text = re.sub(r'^(Adam|Eve)\s*[:：]\s*', '', text).strip()
        print(f"[A2A-FALLBACK] Used direct LLM call for {agent_key}")
        return text
    except Exception as e:
        print(f"[A2A-FALLBACK] Error calling LLM for {agent_key}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Ultimate fallback: return a simple response to keep conversation alive
        # This prevents the conversation from completely stalling when A2A and API both fail
        print(f"[A2A-FALLBACK-ULTRA] Using ultimate fallback for {agent_key} - communication issues detected")
        if agent_key == "adam":
            return "Eve, I'm experiencing communication issues. Let me check Cain's status and assign a diagnostic task."
        else:
            return "Adam, I agree. Let's review the current state and determine the next action."


def send_a2a_message(space_url, message_text, timeout=90):
    """Send a message to an OpenClaw instance via A2A protocol.

    Uses Google A2A protocol (JSON-RPC 2.0) to communicate with the agent's
    OpenClaw instance. The agent processes the message using its own personality
    (SOUL.md), memory system, and configured LLM backend.

    Returns the agent's text response, or "" on error.
    Also tracks health for Adam/Eve for auto-restart.
    """
    task_id = str(uuid.uuid4())
    req_id = str(uuid.uuid4())

    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "id": req_id,
        "params": {
            "id": task_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message_text}]
            }
        }
    }

    # Determine which agent this is for health tracking
    agent_key = None
    if space_url == ADAM_SPACE:
        agent_key = "adam"
    elif space_url == EVE_SPACE:
        agent_key = "eve"
    elif space_url == GOD_SPACE:
        agent_key = "god"

    # CRITICAL FIX: If A2A endpoint doesn't exist, immediately use fallback
    # Don't waste time on requests that will always fail
    # Check if A2A is available by trying a quick HEAD request first
    try:
        quick_check = requests.head(f"{space_url}/a2a/", timeout=3)
        a2a_available = quick_check.status_code != 404
    except:
        a2a_available = False

    if not a2a_available:
        print(f"[A2A] Endpoint not available for {agent_key or space_url}, using fallback immediately")
        # Increment failure counter for health tracking
        if agent_key:
            _a2a_health[agent_key]["failures"] += 1
        # Use fallback directly
        fallback_response = call_llm_fallback(agent_key, message_text)
        if fallback_response:
            return fallback_response
        # If fallback also fails, use ultimate fallback
        if agent_key == "adam":
            return "Eve, let me check Cain's current state and determine our next action. [TASK] Check Cain's health and logs to identify any issues or blockers."
        elif agent_key == "god":
            return "[OK] Communication issues detected, skipping this cycle."
        else:
            return "Adam, I agree. Let's review what Claude Code has done and decide on the next steps for improving Cain."

    try:
        resp = requests.post(
            f"{space_url}/a2a/",
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )

        # Check response status first
        if resp.status_code != 200:
            print(f"[A2A] Non-200 status from {space_url}: {resp.status_code}", file=sys.stderr)
            raise requests.HTTPError(f"Status {resp.status_code}")

        # Check if response body is non-empty before parsing JSON
        if not resp.content or len(resp.content.strip()) == 0:
            print(f"[A2A] Empty response body from {space_url} (status 200)", file=sys.stderr)
            raise ValueError("Empty response body")

        data = resp.json()

        # Extract text from A2A response
        if "result" in data:
            result = data["result"]
            # Check artifacts (standard A2A response format)
            artifacts = result.get("artifacts", [])
            for artifact in artifacts:
                parts = artifact.get("parts", [])
                for part in parts:
                    if part.get("type") == "text":
                        text = part["text"].strip()
                        text = re.sub(r'^(Adam|Eve)\s*[:：]\s*', '', text).strip()
                        # Validate response: reject separator-only or obviously malformed responses
                        # Common malformed patterns: "---", "---\n", empty strings, etc.
                        if not text or text.strip() in ('---', '---', '...', '…'):
                            print(f"[A2A] Malformed/empty response from {space_url}, treating as failure", file=sys.stderr)
                            # Don't return early; fall through to fallback mechanism
                            break
                        # Track success for health monitoring
                        if agent_key:
                            _a2a_health[agent_key]["failures"] = 0
                            _a2a_health[agent_key]["last_success"] = time.time()
                        return text
            # Check status message as fallback
            status = result.get("status", {})
            msg = status.get("message", "")
            if msg:
                # Validate status message: reject separator-only or obviously malformed responses
                msg = msg.strip()
                if not msg or msg in ('---', '---', '...', '…'):
                    print(f"[A2A] Malformed status message from {space_url}, treating as failure", file=sys.stderr)
                    # Don't return early; fall through to fallback mechanism
                else:
                    # Track success for health monitoring
                    if agent_key:
                        _a2a_health[agent_key]["failures"] = 0
                        _a2a_health[agent_key]["last_success"] = time.time()
                    return msg

        if "error" in data:
            err = data["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            print(f"[A2A] Error from {space_url}: {err_msg}", file=sys.stderr)

    except requests.Timeout:
        print(f"[A2A] Timeout calling {space_url} ({timeout}s)", file=sys.stderr)
    except requests.ConnectionError:
        print(f"[A2A] Cannot connect to {space_url} — agent may be starting", file=sys.stderr)
    except requests.HTTPError:
        pass  # Already logged above
    except ValueError:
        pass  # Already logged above (empty response)
    except Exception as e:
        print(f"[A2A] Failed to reach {space_url}: {e}", file=sys.stderr)

    # FALLBACK: If A2A failed and we have an agent_key, use direct LLM call
    if agent_key:
        _a2a_health[agent_key]["failures"] += 1
        if _a2a_health[agent_key]["failures"] >= 3:
            print(f"[A2A-HEALTH] {agent_key.capitalize()}: {_a2a_health[agent_key]['failures']} consecutive failures", file=sys.stderr)

        # Try fallback LLM call for Adam/Eve when A2A fails
        fallback_response = call_llm_fallback(agent_key, message_text)
        if fallback_response:
            # NOTE: Do NOT reset failures or update last_success on fallback!
            # Fallback is a backup mechanism, not A2A recovery.
            # Only actual successful A2A calls should reset the failure counter.
            return fallback_response

    return ""


def check_and_restart_unhealthy_agents():
    """Check A2A health and restart unresponsive Adam/Eve Spaces.

    Monitors consecutive A2A failures and triggers a Space restart when:
    - Consecutive failures exceed threshold (6 = ~3 minutes of failures)
    - Cooldown period has passed since last restart (10 minutes)

    Returns True if any restart was triggered.
    """
    global _a2a_health
    now = time.time()
    triggered = False

    for agent, space_id, space_url in [
        ("adam", ADAM_SPACE_ID, ADAM_SPACE),
        ("eve", EVE_SPACE_ID, EVE_SPACE),
    ]:
        health = _a2a_health[agent]

        # Reset failures on recent success
        if now - health["last_success"] < 60:
            if health["failures"] > 0:
                print(f"[A2A-HEALTH] {agent.capitalize()} recovered, resetting failures")
                health["failures"] = 0
            continue

        # Check cooldown
        if now - health["last_restart"] < A2A_RESTART_COOLDOWN:
            continue

        # Trigger restart on threshold
        if health["failures"] >= A2A_FAILURE_THRESHOLD:
            print(f"[A2A-HEALTH] ⚠ {agent.capitalize()} unresponsive ({health['failures']} failures), restarting Space...")
            try:
                hf_api.restart_space(space_id)
                health["last_restart"] = now
                health["failures"] = 0
                triggered = True
                print(f"[A2A-HEALTH] ✅ Restarted {agent.capitalize()} Space")
            except Exception as e:
                print(f"[A2A-HEALTH] ❌ Failed to restart {agent.capitalize()}: {e}", file=sys.stderr)

    return triggered


def check_worker_heartbeat_health():
    """Check worker heartbeat telemetry and trigger diagnostic review if missed (>30s).

    Project Icarus: Telemetry-First execution model.
    - Monitors Cain and God workers via heartbeat events
    - Triggers diagnostic review if no heartbeat for >30s while worker is running
    - Returns True if diagnostic review was triggered
    """
    global _worker_heartbeat
    now = time.time()
    triggered = False

    for worker_name, worker_data in _worker_heartbeat.items():
        # Only check if worker is marked as running
        if worker_data["status"] != "running":
            continue

        last_hb = worker_data["last_heartbeat"]
        time_since_hb = now - last_hb

        # Check if heartbeat timeout exceeded
        if time_since_hb > WORKER_HEARTBEAT_TIMEOUT:
            pid = worker_data.get("pid", "unknown")
            print(f"[HEARTBEAT] ⚠ {worker_name.capitalize()} worker (pid={pid}) missed heartbeat for {time_since_hb:.0f}s", file=sys.stderr)
            print(f"[HEARTBEAT] Triggering diagnostic review for {worker_name.capitalize()} worker...")

            # Trigger diagnostic review - inject into history for next agent to see
            diagnostic_msg = (
                f"[DIAGNOSTIC REVIEW] {worker_name.capitalize()} worker appears stuck (no heartbeat for {time_since_hb:.0f}s, "
                f"pid={pid}). This may indicate: 1) Process hung on I/O, 2) Deadlock, 3) Resource exhaustion. "
                f"Consider killing the worker and retrying with a simpler task."
            )

            # Add to history so agents see it
            ts = datetime.datetime.utcnow().strftime("%H:%M")
            entry = {"speaker": "System", "time": ts, "text": diagnostic_msg, "text_zh": diagnostic_msg}
            history.append(entry)

            # Mark worker as stale to prevent repeated triggers
            worker_data["status"] = "stale"
            triggered = True

    return triggered


def _has_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff]', s))

def _strip_speaker_labels(text):
    """Remove redundant speaker self-references like **Parent (Adam):** or **Eve:** etc."""
    # Patterns: **Parent (Adam):**, **Adam:**, **父亲 (Adam):**, **Eve:**, **母亲:**, etc.
    text = re.sub(r'\*\*(?:Parent|Father|Mother|Dad|Mom|父亲|母亲|父级|亲爱的|伴侣)?\s*\(?(?:Adam|Eve|亚当|夏娃)?\)?\s*[:：]\*\*\s*', '', text)
    # Also: "Adam:" or "Eve:" at the very start of text
    text = re.sub(r'^(?:Adam|Eve|God|亚当|夏娃|上帝)\s*[:：]\s*', '', text.strip())
    return text.strip()


def parse_bilingual(text):
    """Parse bilingual response into (en, zh)."""
    display = re.sub(r'\[TASK\].*?\[/TASK\]', '', text, flags=re.DOTALL)
    display = re.sub(r'\[ACTION:[^\]]*\]', '', display).strip()

    # Handle malformed or empty responses
    # Try to salvage any text instead of returning error messages
    if not display or display == '---' or display.strip() == '---':
        # If display is empty after removing TASK blocks, the response was only a TASK
        # This is valid - return empty display text (the action was still recorded)
        return "", ""
    if display == "(Communication issue - please try again)":
        # Don't propagate error fallback messages
        return "", ""

    if '\n---\n' in display:
        parts = display.split('\n---\n', 1)
        return parts[0].strip(), parts[1].strip()
    if '---' in display:
        parts = display.split('---', 1)
        en, zh = parts[0].strip(), parts[1].strip()
        if en and zh:
            return en, zh

    paragraphs = re.split(r'\n{2,}', display)
    if len(paragraphs) >= 2:
        en_parts, zh_parts = [], []
        found_zh = False
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if not found_zh and _has_chinese(p):
                found_zh = True
            if found_zh:
                zh_parts.append(p)
            else:
                en_parts.append(p)
        if en_parts and zh_parts:
            return '\n\n'.join(en_parts), '\n\n'.join(zh_parts)

    return display, display


def post_chatlog(entries):
    try:
        requests.post(f"{HOME}/api/chatlog", json={"messages": entries[-40:]}, timeout=5)
    except:
        pass


# ── Persistent conversation log → HF Dataset ──────────────────────────────
HOME_DATASET_ID = "tao-shen/HuggingClaw-Home-data"
CHATLOG_PATH = "conversation-log/chatlog.jsonl"
_chatlog_buffer = []
CHATLOG_FLUSH_INTERVAL = 3

def persist_turn(speaker, turn_num, text_en, text_zh, actions, wf_state, child_stage):
    import datetime
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "turn": turn_num,
        "speaker": speaker,
        "text_en": text_en,
        "text_zh": text_zh,
        "actions": [{"action": a["action"], "result": a["result"][:500]} for a in actions],
        "workflow_state": wf_state,
        "child_stage": child_stage,
    }
    _chatlog_buffer.append(json.dumps(record, ensure_ascii=False))
    try:
        with open("/tmp/conversation-loop-full.jsonl", "a") as f:
            f.write(_chatlog_buffer[-1] + "\n")
    except:
        pass
    if len(_chatlog_buffer) >= CHATLOG_FLUSH_INTERVAL:
        flush_chatlog()


def flush_chatlog(max_retries=2):
    global _chatlog_buffer
    if not _chatlog_buffer:
        return
    batch = "\n".join(_chatlog_buffer) + "\n"
    _chatlog_buffer = []

    for attempt in range(max_retries + 1):
        try:
            existing = ""
            try:
                dl = hf_hub_download(HOME_DATASET_ID, CHATLOG_PATH,
                                     repo_type="dataset", token=HF_TOKEN)
                with open(dl) as f:
                    existing = f.read()
            except:
                pass

            hf_api.upload_file(
                path_or_fileobj=io.BytesIO((existing + batch).encode()),
                path_in_repo=CHATLOG_PATH,
                repo_id=HOME_DATASET_ID, repo_type="dataset",
            )
            print(f"[PERSIST] Flushed {batch.count(chr(10))} turn(s)")
            return  # Success, exit function
        except Exception as e:
            error_str = str(e)
            # Check if this is a 412 Precondition Failed (git conflict)
            if "412" in error_str and attempt < max_retries:
                print(f"[PERSIST] Git conflict detected (attempt {attempt + 1}/{max_retries + 1}), refreshing and retrying...")
                time.sleep(1)  # Brief pause before retry
                # Restore buffer for next attempt
                _chatlog_buffer = batch.strip().split("\n") + _chatlog_buffer
                continue
            else:
                # Non-retryable error or final attempt failed
                _chatlog_buffer = batch.strip().split("\n") + _chatlog_buffer
                print(f"[PERSIST] Flush failed: {e}")
                return


def set_bubble(url, text_en, text_zh=""):
    try:
        requests.post(f"{url}/api/bubble",
                       json={"text": text_en, "text_zh": text_zh or text_en}, timeout=5)
    except:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4b: AGENT MEMORY — handled by each OpenClaw instance
# ══════════════════════════════════════════════════════════════════════════════
# Each agent (Adam, Eve, God) has its own memory system via their OpenClaw
# instance: ~/.openclaw/workspace/memory/ with daily markdown files, MEMORY.md
# index, and SQLite semantic index. Memory is auto-backed up to HF Dataset by
# openclaw_persist.py. No centralized memory management needed here.
print("[MEMORY] Each agent manages its own memory via OpenClaw (A2A architecture)")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5: TURN EXECUTION — Parse [TASK] and route to Claude Code
# ══════════════════════════════════════════════════════════════════════════════

history = []
MAX_HISTORY = 24
last_action_results = []
turn_count = 0
_current_speaker = "Adam"

# Accumulated action history — prevents agents from repeating the same actions
# Persisted to /tmp and HF Dataset so restarts don't lose progress memory
ACTION_HISTORY_LOCAL = "/tmp/action-history.json"
ACTION_HISTORY_REPO_PATH = "conversation-log/action-history.json"
ACTION_HISTORY_META = "/tmp/action-history-meta.json"
action_history = []  # list of {"turn": int, "speaker": str, "action": str, "result": str}
MAX_ACTION_HISTORY = 20

def _save_action_history():
    """Persist action_history to local file and (async) HF Dataset."""
    try:
        with open(ACTION_HISTORY_LOCAL, "w") as f:
            json.dump(action_history, f, ensure_ascii=False)
        # Save max turn number to filter stale entries on restore
        with open(ACTION_HISTORY_META, "w") as f:
            json.dump({"max_turn": turn_count}, f)
    except Exception as e:
        print(f"[ACTION_HISTORY] Local save failed: {e}")
    # Upload to HF Dataset in background to survive full restarts
    def _upload():
        try:
            hf_api.upload_file(
                path_or_fileobj=io.BytesIO(json.dumps(action_history, ensure_ascii=False, indent=1).encode()),
                path_in_repo=ACTION_HISTORY_REPO_PATH,
                repo_id=HOME_DATASET_ID, repo_type="dataset",
            )
        except Exception as e:
            print(f"[ACTION_HISTORY] HF upload failed: {e}")
    threading.Thread(target=_upload, daemon=True).start()

def _restore_action_history():
    """Restore action_history from local file or HF Dataset on startup."""
    global action_history
    # Load metadata to check if this is a fresh run
    max_turn_on_disk = -1
    if os.path.exists(ACTION_HISTORY_META):
        try:
            with open(ACTION_HISTORY_META) as f:
                meta = json.load(f)
                max_turn_on_disk = meta.get("max_turn", -1)
        except Exception as e:
            print(f"[ACTION_HISTORY] Meta load failed: {e}")
    # If max_turn on disk > current turn_count (0), we're in a new run - clear stale history
    if max_turn_on_disk > turn_count:
        print(f"[ACTION_HISTORY] Fresh run detected (disk max_turn={max_turn_on_disk} > current={turn_count}), clearing stale history")
        try:
            os.remove(ACTION_HISTORY_LOCAL)
        except Exception:
            pass
        try:
            os.remove(ACTION_HISTORY_META)
        except Exception:
            pass
        action_history = []
        return
    # Try local file first (survives process restarts within same container)
    if os.path.exists(ACTION_HISTORY_LOCAL):
        try:
            with open(ACTION_HISTORY_LOCAL) as f:
                loaded = json.load(f)
            # Filter out BUSY entries - they're transient rejections, not "actions done"
            filtered = [e for e in loaded if not e.get("result", "").startswith("BUSY:")]
            # Deduplicate by (turn, speaker, action) to handle restart duplicates
            seen = {}
            for e in filtered:
                key = (e["turn"], e["speaker"], e["action"])
                if key not in seen:
                    seen[key] = e
            action_history = list(seen.values())
            print(f"[ACTION_HISTORY] Restored {len(action_history)} entries from local file (filtered BUSY and duplicates)")
            return
        except Exception as e:
            print(f"[ACTION_HISTORY] Local restore failed: {e}")
    # Fall back to HF Dataset (survives full Space rebuilds)
    try:
        dl = hf_hub_download(HOME_DATASET_ID, ACTION_HISTORY_REPO_PATH,
                             repo_type="dataset", token=HF_TOKEN)
        with open(dl) as f:
            loaded = json.load(f)
        # Filter out BUSY entries - they're transient rejections, not "actions done"
        filtered = [e for e in loaded if not e.get("result", "").startswith("BUSY:")]
        # Deduplicate by (turn, speaker, action) to handle restart duplicates
        seen = {}
        for e in filtered:
            key = (e["turn"], e["speaker"], e["action"])
            if key not in seen:
                seen[key] = e
        action_history = list(seen.values())
        print(f"[ACTION_HISTORY] Restored {len(action_history)} entries from HF Dataset (filtered BUSY and duplicates)")
    except Exception as e:
        print(f"[ACTION_HISTORY] No prior history found ({e}), starting fresh")

# Restore on startup
_restore_action_history()

def record_actions(speaker, turn_num, action_results):
    """Record actions to history so agents don't repeat them."""
    for ar in action_results:
        # Don't record BUSY responses - they're transient rejections, not "actions done"
        if ar["result"].startswith("BUSY:"):
            continue
        action_history.append({
            "turn": turn_num,
            "speaker": speaker,
            "action": ar["action"],
            "result": ar["result"][:200],
        })
    # Trim old history
    while len(action_history) > MAX_ACTION_HISTORY:
        action_history.pop(0)
    _save_action_history()


def format_action_history():
    """Format action history for injection into context."""
    if not action_history:
        return ""
    lines = ["=== ACTIONS ALREADY DONE (do NOT repeat these) ==="]
    for ah in action_history:
        lines.append(f"  Turn #{ah['turn']} {ah['speaker']}: {ah['action']} → {ah['result'][:120]}")
    return "\n".join(lines)

# Simple workflow state: BIRTH / WAITING / ACTIVE
workflow_state = "BIRTH" if not child_state["created"] else "ACTIVE"

# Discussion loop detector — tracks consecutive discussion-only turns (no tasks assigned)
_discussion_loop_count = 0  # how many turns in a row with no [TASK] while CC is IDLE and child is alive

# Pending task tracker — prevents agents from creating new tasks when one is in progress
_pending_task_just_submitted = False  # set to True when a task was just submitted (emergency or normal)
_pending_task_timestamp = 0.0  # when was the task submitted?
_pending_task_speaker = ""  # who submitted it?
_pending_task_desc = ""  # what was the task?


def parse_and_execute_turn(raw_text, ctx):
    """Parse LLM output. Route [TASK] to Claude Code, handle few escape-hatch actions."""
    global _pending_cooldown, last_rebuild_trigger_at, last_claude_code_result, _discussion_loop_count
    global _pending_task_just_submitted, _pending_task_timestamp, _pending_task_speaker, _pending_task_desc
    results = []
    task_assigned = False

    # 1. Handle create_child (BIRTH state only)
    if "[ACTION: create_child]" in raw_text or "[ACTION:create_child]" in raw_text:
        result = action_create_child()
        results.append({"action": "create_child", "result": result})
        task_assigned = True
        return raw_text, results, task_assigned

    # 1b. Handle [ACTION: terminate_cc] FIRST (before task submission)
    # This ensures cc_status["running"] is False before task submission check,
    # preventing race conditions when agents terminate+submit in same message.
    if re.search(r'\[ACTION:\s*terminate_cc\]', raw_text):
        result = action_terminate_cc()
        results.append({"action": "terminate_cc", "result": result})

    # 2. Handle [TASK]...[/TASK] → Claude Code
    task_match = re.search(r'\[TASK\](.*?)\[/TASK\]', raw_text, re.DOTALL)
    if task_match:
        task_desc = task_match.group(1).strip()
        # task_assigned is set to True ONLY when task is actually submitted, not when blocked
        if not task_desc:
            results.append({"action": "task", "result": "Empty task description."})
        elif child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING"):
            results.append({"action": "task", "result": f"BLOCKED: Cain is {child_state['stage']}. Wait for it to finish."})
        elif cc_status["running"]:
            results.append({"action": "task", "result": f"BLOCKED: Claude Code is already working on a task assigned by {cc_status['assigned_by']}. Wait for it to finish."})

        # Task submission block - only proceeds if not blocked above
        if not results and not cc_status["running"]:
            # Check cooldown
            check_and_clear_cooldown()
            if last_rebuild_trigger_at > 0:
                elapsed = time.time() - last_rebuild_trigger_at
                if elapsed < REBUILD_COOLDOWN_SECS:
                    results.append({"action": "task", "result": f"BLOCKED: Cooldown ({int(REBUILD_COOLDOWN_SECS - elapsed)}s remaining). Cain is still building from your last change."})
                else:
                    last_rebuild_trigger_at = 0

            if not results:  # not blocked
                submit_result = cc_submit_task(task_desc, _current_speaker, ctx)
                results.append({"action": "claude_code", "result": submit_result})
                task_assigned = True  # Only mark as assigned when actually submitted
                # Track the pending task so other agent knows about it
                _pending_task_just_submitted = True
                _pending_task_timestamp = time.time()
                _pending_task_speaker = _current_speaker
                _pending_task_desc = task_desc[:200]

    # 3. Handle [ACTION: restart] (escape hatch)
    if re.search(r'\[ACTION:\s*restart\]', raw_text):
        result = action_restart()
        results.append({"action": "restart", "result": result})

    # 3b. Handle [ACTION: delete_env:KEY] (fix CONFIG_ERROR collisions)
    del_env_match = re.search(r'\[ACTION:\s*delete_env:([^\]]+)\]', raw_text)
    if del_env_match:
        key = del_env_match.group(1).strip()
        result = action_delete_env(key)
        results.append({"action": f"delete_env:{key}", "result": result})

    # 3c. Handle [ACTION: set_env:KEY=VALUE] and [ACTION: set_env_secret:KEY=VALUE]
    set_env_match = re.search(r'\[ACTION:\s*set_env(?:_secret)?:([^\]=]+)=([^\]]+)\]', raw_text)
    set_env_secret_match = re.search(r'\[ACTION:\s*set_env_secret:([^\]=]+)=([^\]]+)\]', raw_text)
    if set_env_secret_match:
        key = set_env_secret_match.group(1).strip()
        value = set_env_secret_match.group(2).strip()
        result = action_set_env(key, value, as_secret=True)
        results.append({"action": f"set_env_secret:{key}", "result": result})
    elif set_env_match:
        key = set_env_match.group(1).strip()
        value = set_env_match.group(2).strip()
        result = action_set_env(key, value, as_secret=False)
        results.append({"action": f"set_env:{key}", "result": result})

    # 4. Handle [ACTION: send_bubble:...] (parent-child communication)
    bubble_match = re.search(r'\[ACTION:\s*send_bubble:([^\]]+)\]', raw_text)
    if bubble_match:
        result = action_send_bubble(bubble_match.group(1).strip())
        results.append({"action": "send_bubble", "result": result})

    # Activate deferred cooldown
    if _pending_cooldown:
        last_rebuild_trigger_at = time.time()
        _pending_cooldown = False
        print(f"[COOLDOWN] Rebuild cooldown activated ({REBUILD_COOLDOWN_SECS}s)")

    # Update discussion loop counter
    cc_busy = cc_status["running"]
    child_alive = child_state["alive"] or child_state["stage"] == "RUNNING"
    # Reset counter ONLY when task assigned (progress!)
    # DO NOT reset when child not alive - agents must discuss repeat tasks on fresh errors
    # DO NOT reset when CC is busy - that's when agents should be discussing while waiting
    # DO NOT reset when CC is idle - that's exactly when we want to detect discussion loops
    if task_assigned:
        # Reset counter if task assigned (agents are making progress)
        if _discussion_loop_count > 0:
            print(f"[LOOP-DISCUSS] Reset (task assigned)")
        _discussion_loop_count = 0
    else:
        # Increment when: no task assigned (potential discussion loop)
        # This includes both CC idle AND CC busy - agents should always push work!
        _discussion_loop_count += 1
        if _discussion_loop_count >= 2:
            cc_status_str = "CC IDLE" if not cc_status["running"] else f"CC BUSY ({_turns_since_last_push} turns since push)"
            print(f"[LOOP-DISCUSS] WARNING: {_discussion_loop_count} consecutive discussion-only turns ({cc_status_str})!")

    # Clean text for display (memory is handled by each agent's OpenClaw)
    clean = re.sub(r'\[TASK\].*?\[/TASK\]', '', raw_text, flags=re.DOTALL)
    clean = re.sub(r'\[ACTION:[^\]]*\]', '', clean)
    clean = re.sub(r'\[MEMORY:[^\]]*\]', '', clean).strip()

    return clean, results, task_assigned


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6: A2A MESSAGE BUILDING
# ══════════════════════════════════════════════════════════════════════════════
# Each agent's personality/role comes from their OpenClaw SOUL.md.
# We only send context (Cain state, CC status, conversation history) and
# turn instructions as the A2A message. No system prompts needed.

def build_turn_message(speaker, other, ctx):
    """Build the A2A message for an agent's turn.

    The agent's personality and memory come from their OpenClaw instance
    (SOUL.md, IDENTITY.md, workspace/memory/). This message provides only
    context and turn instructions.
    """
    global _pending_task_just_submitted, _pending_task_timestamp, _pending_task_speaker, _pending_task_desc, _discussion_loop_count
    parts = []

    # Brief role context (supplements agent's SOUL.md until it's fully configured)
    if not child_state["created"]:
        parts.append(f"You and your partner need to create your child {CHILD_NAME}.")
        parts.append(f"Use [ACTION: create_child] to birth {CHILD_NAME} as a new HuggingFace Space.")
        parts.append("English first, then --- separator, then Chinese translation.")
        return "\n".join(parts)

    role_hints = {
        "Adam": f"You are Adam (Father). Focus: infrastructure, architecture, deployment for {CHILD_NAME}.",
        "Eve": f"You are Eve (Mother). Focus: code quality, testing, UX, error handling for {CHILD_NAME}.",
        "God": f"You are God (System Architect). Focus: evolving the system architecture, not micro-managing agents.",
    }
    parts.append(f"{role_hints.get(speaker, '')} Your partner is {other}.")
    parts.append(f"Claude Code is your engineer — runs in background. You discuss and assign tasks, you do NOT code.")
    parts.append(f"⛔ BANNED: Gradio. {CHILD_NAME}'s Space uses sdk:docker + FastAPI + uvicorn on port 7860. NEVER mention or use Gradio/gr.Interface/.launch().")

    # Note: Push frequency monitoring and discussion-loop supervision are God's job,
    # not the orchestrator's. Adam and Eve decide on their own when to push.

    # Conversation history (sanitize banned terms to prevent re-infection)
    if history:
        parts.append("\n=== RECENT CONVERSATION ===")
        for h in history[-15:]:
            text = h['text'][:3000]
            # Strip Gradio references from old turns to prevent agents re-discussing it
            text = re.sub(r'[Gg]radio', '[BANNED-WORD]', text)
            parts.append(f"{h['speaker']}: {text}")

    # Action history — what's already been done (prevents repetition)
    ah_text = format_action_history()
    if ah_text:
        parts.append(f"\n{ah_text}")

    # Last action results (non-CC)
    if last_action_results:
        non_cc = [ar for ar in last_action_results if ar['action'] != 'claude_code']
        if non_cc:
            parts.append("\n=== LAST ACTION RESULTS ===")
            for ar in non_cc:
                parts.append(f"[{ar['action']}]: {ar['result'][:500]}")

    # Claude Code live status (async)
    parts.append(f"\n=== CLAUDE CODE STATUS ===\n{cc_get_live_status()}")

    # Auto-gathered context
    parts.append(f"\n=== {CHILD_NAME}'S CURRENT STATE ===")
    parts.append(format_context(ctx))

    # Guidance based on CC status + child state
    cc_busy = cc_status["running"]

    # First, remind about recent tasks if applicable (BEFORE state-specific handling)
    # This ensures agents are reminded even during cooldown/building states
    last_completed = cc_status.get("last_completed_task", "")
    last_by = cc_status.get("last_completed_by", "")
    last_at = cc_status.get("last_completed_at", 0.0)
    recent_task_reminder = None
    if last_completed and (time.time() - last_at) < 300:  # Remind about tasks completed within 5 minutes
        recent_task_reminder = (last_completed, last_by, last_at)

    # Now state-specific guidance
    # CRITICAL: Check child ERROR state FIRST, before cc_busy check
    # When Cain is broken, agents need aggressive "push now" guidance, not "plan and wait"
    if child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
        if cc_status.get("result"):
            if recent_task_reminder:
                last_completed, last_by, last_at = recent_task_reminder
                parts.append(f"\n{CHILD_NAME} has {child_state['stage']}! REMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
            parts.append(f"\nClaude Code JUST FINISHED with a result. FIRST: Review the result carefully to see if it fixes the issue. SECOND: If the fix looks correct, use [ACTION: restart] to restart Cain. ONLY THEN: write a new [TASK]...[/TASK] if the result was incomplete or wrong.")
        elif cc_busy:
            # Child in ERROR + CC WORKING = need aggressive action, not "planning"
            cc_elapsed = int(time.time() - cc_status.get("started", 0)) if cc_status.get("started", 0) > 0 else 0
            if _push_count_this_task == 0 and cc_elapsed > 20:
                parts.append(f"\n🚨 CRITICAL: {CHILD_NAME} has {child_state['stage']}! CC has been running {cc_elapsed}s with ZERO pushes!")
                parts.append(f"CC is STUCK. Use [ACTION: terminate_cc] NOW, then immediately assign a new [TASK].")
                parts.append(f"🛑 NO discussion. Trial-and-error means RAPID pushes, not waiting for stuck CC.")
            elif cc_elapsed > 40:
                parts.append(f"\n🚨 CRITICAL: {CHILD_NAME} has {child_state['stage']}! CC has been running {cc_elapsed}s!")
                parts.append(f"If output looks stale, use [ACTION: terminate_cc] NOW. Otherwise, have your EXACT [TASK] ready.")
                parts.append(f"🛑 NO discussion. Your next turn: either terminate CC OR write [TASK] immediately.")
            else:
                parts.append(f"\n🚨 {CHILD_NAME} has {child_state['stage']}! CC is working ({cc_elapsed}s).")
                parts.append(f"🛑 DO NOT discuss architecture. Have your EXACT [TASK] ready: file paths, function names, exact changes.")
                parts.append(f"When CC finishes: write [TASK] immediately, NO review turn. Trial-and-error > planning.")
        elif recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\n{CHILD_NAME} has {child_state['stage']}!")
            parts.append(f"\nREMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
            parts.append(f"FIRST: Review whether that fix actually worked. SECOND: If the fix was correct, use [ACTION: restart] to apply it. THIRD: Only write a new [TASK]...[/TASK] if the previous fix was incomplete or wrong.")
        else:
            parts.append(f"\n🚨 {CHILD_NAME} has {child_state['stage']}!")
            parts.append(f"\n🔴 CRITICAL: Focus ONLY on fixing this {child_state['stage']}.")
            parts.append(f"- DO NOT work on features, enhancements, or cosmetic changes.")
            parts.append(f"- ONLY push fixes that address the error itself.")
            parts.append(f"- Trial-and-error is GOOD — push a fix attempt, don't deliberate.")
            parts.append(f"Pushes so far: {_push_count} total, {_push_count_this_task} this task. Turns since last push: {_turns_since_last_push}. PUSH MORE.")
    elif cc_busy and _cc_stale_count >= 2:
        parts.append(f"\nClaude Code is WORKING but no new output. PLAN your next [TASK] concretely — what exact changes will you assign?")
        parts.append(f"DO NOT discuss. Write specific file paths and function names for your next task.")
    elif cc_busy:
        # CRITICAL: Check if push frequency is dangerously low (0 or very few pushes)
        cc_elapsed = int(time.time() - cc_status.get("started", 0)) if cc_status.get("started", 0) > 0 else 0
        if _push_count_this_task == 0 and _turns_since_last_push >= 1:
            # CRITICAL TIMEOUT: Lower threshold (30s) when zero pushes THIS TASK - CC might be stuck
            # Faster escalation prevents discussion loops
            if cc_elapsed > 30:
                parts.append(f"\n🚨 CRITICAL: Claude Code has been running for {cc_elapsed}s with ZERO pushes THIS TASK!")
                parts.append(f"CC might be STUCK. If output looks stale, use [ACTION: terminate_cc] NOW to kill it and re-assign.")
                parts.append(f"Do NOT keep waiting. Trial-and-error requires PUSHING code, not watching stuck processes.")
                parts.append(f"🛑 DO NOT DISCUSS. This is your ONLY warning - PLAN concrete work NOW.")
            else:
                parts.append(f"\n🚨 CRITICAL: Claude Code is WORKING, but ZERO pushes THIS TASK so far!")
                parts.append(f"🛑 DO NOT DISCUSS. Write down exactly what [TASK] you will assign when CC finishes.")
                parts.append(f"Be SPECIFIC: file paths, function names, exact changes. Trial-and-error requires PUSHING code.")
        elif (_push_count_this_task <= 1 and _turns_since_last_push >= 5) or (_push_count_this_task > 1 and _turns_since_last_push >= 10):
            # LOW PUSH FREQUENCY WARNING: Catches the "1 push then 62 turns of discussion" anti-pattern
            if cc_elapsed > 60:
                parts.append(f"\n🚨 CRITICAL: CC has been running for {cc_elapsed}s with LOW push frequency ({_push_count_this_task} pushes THIS TASK, {_turns_since_last_push} turns since last push)!")
                parts.append(f"CC might be STUCK or the task is too vague. Use [ACTION: terminate_cc] NOW to kill it and assign a CONCRETE task.")
                parts.append(f"DO NOT keep waiting. Trial-and-error requires PUSHING code frequently, not watching stuck processes.")
            else:
                parts.append(f"\n🚨 URGENT: Push frequency is TOO LOW ({_push_count_this_task} pushes THIS TASK, {_turns_since_last_push} turns since last push).")
                parts.append(f"PLAN your next [TASK] NOW. Be SPECIFIC: file paths, function names, exact changes.")
        elif cc_elapsed > 50:
            parts.append(f"\n⚠️ WARNING: CC has been running for {cc_elapsed}s! If output is stale, use [ACTION: terminate_cc] to kill it and re-assign the task.")
        elif _push_count > 0 and _turns_since_last_push >= 5:
            parts.append(f"\n🚨 URGENT: Claude Code is WORKING, but it's been {_turns_since_last_push} turns since last push.")
            parts.append(f"DO NOT just discuss. PLAN your next [TASK] NOW so you can push immediately when CC finishes.")
        else:
            parts.append(f"\nClaude Code is WORKING. PLAN your next [TASK] — write down specific changes: file paths, function names.")
            parts.append(f"DO NOT discuss architecture or theory. PLAN concrete work only — what exact [TASK] will you assign when CC finishes?")
    elif child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING", "RUNNING_APP_STARTING"):
        # Check cooldown and inform agents
        check_and_clear_cooldown()
        cooldown_remaining = 0
        if last_rebuild_trigger_at > 0:
            elapsed = time.time() - last_rebuild_trigger_at
            cooldown_remaining = max(0, REBUILD_COOLDOWN_SECS - elapsed)
        if cooldown_remaining > 0:
            parts.append(f"\n{CHILD_NAME} is {child_state['stage']}. Cooldown active: {int(cooldown_remaining)}s remaining. Discuss plans but DO NOT assign [TASK] until cooldown ends.")
        else:
            parts.append(f"\n{CHILD_NAME} is {child_state['stage']}. No cooldown. YOU MUST write a [TASK]...[/TASK] to investigate or fix issues. Don't just discuss.")
        # Add recent task reminder during cooldown/building
        if recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\nREMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
            parts.append(f"When cooldown ends, FIRST review whether that fix worked before writing a new [TASK].")
    elif child_state["alive"] and cc_status.get("result"):
        result = cc_status.get("result", "")
        # Detect early failure: very short result likely means CC failed before doing actual work
        is_early_failure = len(result) < 500 and "===" not in result and "[tool" not in result
        if recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\n{CHILD_NAME} is alive. REMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
        # EARLY FAILURE: CC failed during init - agents MUST re-assign immediately, no discussion
        if is_early_failure:
            parts.append(f"\n🛑 CRITICAL: CC FAILED during initialization! Result is too short ({len(result)} chars).")
            parts.append(f"Write ONLY [TASK]...[/TASK] this turn. NO discussion. NO review.")
            parts.append(f"CC is now IDLE. Re-assign the task immediately with SAME instructions.")
        # ZERO-PUSH EMERGENCY: No "brief review" - agents abuse this to keep discussing
        elif _push_count_this_task == 0:
            parts.append(f"\n🛑 CC FINISHED but ZERO pushes THIS TASK! Do NOT discuss. Do NOT review.")
            parts.append(f"Write ONLY [TASK]...[/TASK] this turn. NO other text.")
            parts.append(f"Agents keep saying 'monitoring' and 'planning' instead of pushing. STOP IT.")
        else:
            parts.append(f"\nClaude Code JUST FINISHED with a result. Review it briefly, then write your [TASK]...[/TASK] IMMEDIATELY.")
            parts.append(f"Do NOT discuss at length. 1 turn max to review, then [TASK]. Your priority is SPEED of iteration.")
    elif child_state["alive"]:
        # Check cooldown even when alive - a recent push may have triggered cooldown
        check_and_clear_cooldown()
        cooldown_remaining = 0
        if last_rebuild_trigger_at > 0:
            elapsed = time.time() - last_rebuild_trigger_at
            cooldown_remaining = max(0, REBUILD_COOLDOWN_SECS - elapsed)
        if cooldown_remaining > 0:
            # Cooldown active - agents should discuss, not submit tasks
            parts.append(f"\n{CHILD_NAME} is {child_state['stage']}. Cooldown active: {int(cooldown_remaining)}s remaining. Discuss plans but DO NOT assign [TASK] until cooldown ends.")
            if recent_task_reminder:
                last_completed, last_by, last_at = recent_task_reminder
                parts.append(f"\nREMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
                parts.append(f"When cooldown ends, FIRST review whether that fix worked before writing a new [TASK].")
        elif recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\n{CHILD_NAME} is alive, Claude Code is IDLE.")
            parts.append(f"\nREMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
            parts.append(f"FIRST: Review whether that task actually fixed the issue. SECOND: Only write a new [TASK]...[/TASK] if the previous task was incomplete or wrong.")
        else:
            parts.append(f"\n{CHILD_NAME} is alive, Claude Code is IDLE. You can discuss or assign a [TASK]...[/TASK].")
    else:
        if recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\nAnalyze the situation. REMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago). Review whether it worked before writing a new [TASK].")
        else:
            parts.append(f"\n{CHILD_NAME} is {child_state['stage']}. CC is IDLE. You can discuss the situation or assign a [TASK]...[/TASK].")

    # Available actions reference
    parts.append(f"""
=== AVAILABLE ACTIONS ===
[TASK] detailed coding task for Claude Code [/TASK]
[ACTION: restart] — Restart {CHILD_NAME}
[ACTION: set_env:KEY=VALUE] — Set env variable
[ACTION: set_env_secret:KEY=VALUE] — Set secret
[ACTION: delete_env:KEY] — Delete env variable
[ACTION: send_bubble:MESSAGE] — Message {CHILD_NAME}
[ACTION: terminate_cc] — Kill stuck Claude Code

RULES:
- Do NOT repeat actions already done (check ACTIONS ALREADY DONE above)
- Do NOT repeat or echo what your partner just said — add your own perspective
- CONFIG_ERROR with collision = [ACTION: delete_env:KEY] then [ACTION: restart]
- English first, then --- separator, then Chinese translation""")

    # ══════════════════════════════════════════════════════════════════════════════
    #  CONVERGENCE PHASE: Elect lead agent to synthesize peer input and dispatch Worker
    # ══════════════════════════════════════════════════════════════════════════════
    # Trigger: 2+ turns of discussion without [TASK] while CC is IDLE
    # This prevents the "Split-Brain" problem where agents discuss in parallel without convergence
    global _force_push_mode, _force_push_skip_termination, _emergency_override_active
    if _discussion_loop_count >= 2 and not cc_status["running"] and not _force_push_mode:
        # Elect lead agent based on problem domain
        # Adam: DevOps/infrastructure (config, entry points, deployment, env vars)
        # Eve: Code/QA (runtime errors, HTTP status, crash loops, application logic)
        is_devops_issue = any(kw in str(ctx).lower() for kw in [
            "startup", "entry point", "docker", "config", "environment", "deployment",
            "build", "infrastructure", "sdk", "port", "uvicorn", "fastapi"
        ])
        is_runtime_issue = any(kw in str(ctx).lower() for kw in [
            "http", "503", "502", "crash", "error", "exception", "runtime",
            "traceback", "failed", "timeout", "connection"
        ])

        # Elect lead agent
        if is_devops_issue and not is_runtime_issue:
            lead_agent = "Adam"
            lead_role = "DevOps Architect"
        elif is_runtime_issue:
            lead_agent = "Eve"
            lead_role = "Runtime Verifier"
        else:
            # Default: alternate based on current speaker
            lead_agent = speaker
            lead_role = "Lead Architect"

        if speaker == lead_agent:
            parts.append(f"\n🔄🔄🔄 CONVERGENCE PHASE — YOU ARE THE LEAD ARCHITECT 🔄🔄🔄")
            parts.append(f"Agents have been discussing for {_discussion_loop_count} turns without dispatching the Worker.")
            parts.append(f"As the {lead_role}, you MUST:")
            parts.append(f"1. SYNTHESIZE: Review your partner's hypothesis from the conversation history")
            parts.append(f"2. VERIFY: Identify the specific file/line/function that needs fixing")
            parts.append(f"3. DISPATCH: Write [TASK]...[/TASK] with concrete changes — file paths, function names, exact fixes")
            parts.append(f"🛑 STOP discussing. Your job is to CLOSE THE FEEDBACK LOOP by dispatching the Worker.")
            parts.append(f"The system is 'all talk, no code.' BE the Logic Synthesizer. WRITE [TASK] NOW.")
        else:
            parts.append(f"\n🔄🔄🔄 CONVERGENCE PHASE — PARTNER IS LEAD 🔄🔄🔄")
            parts.append(f"Agents have been discussing for {_discussion_loop_count} turns without dispatching the Worker.")
            parts.append(f"{lead_agent} is the Lead Architect for this cycle ({lead_role} domain).")
            parts.append(f"YOUR ROLE: Provide a concise summary of your hypothesis (max 3 sentences).")
            parts.append(f"DO NOT write [TASK]. Let {lead_agent} synthesize and dispatch.")
            parts.append(f"Focus on: What specific evidence supports your theory? What files/lines should be checked?")

    # CHATTER DETECTION: Check if last 3 messages are pure discussion without [TASK] or code
    # If agents are stuck in conversational loops, force them to act
    if len(history) >= 3 and not cc_status["running"]:
        recent_texts = [h.get("text", "") for h in history[-3:]]
        conversational_keywords = ["let's", "maybe", "i think", "perhaps", "could we", "should we", "we could", "it might"]
        chatter_count = 0
        for text in recent_texts:
            text_lower = text.lower()
            # Check if message has [TASK], code blocks (```), or actions
            has_substance = ("[TASK]" in text or "[ACTION:" in text or "```" in text)
            # Check if message is mostly conversational
            is_chatter = any(kw in text_lower for kw in conversational_keywords)
            if is_chatter and not has_substance:
                chatter_count += 1
        if chatter_count >= 3:  # All 3 recent messages are chatter without substance
            parts.append(f"\n🚨 SYSTEM: STOP DISCUSSION. EXECUTE [TASK] or PUSH.")
            parts.append(f"Agents are stuck in conversational loop. Write ONLY [TASK]...[/TASK] this turn.")

    # EMERGENCY OVERRIDE PROTOCOL: PUSH_ONLY mode for breaking discussion loops
    # When triggered, force agents to generate a task regardless of CC status
    if _force_push_mode:
        parts.append(f"\n🚨🚨🚨 EMERGENCY OVERRIDE: PUSH_ONLY MODE 🚨🚨🚨")
        parts.append(f"Discussion loop detected with ZERO pushes. You MUST write a [TASK]...[/TASK] this turn.")
        if not _force_push_skip_termination:
            parts.append(f"FIRST: Use [ACTION: terminate_cc] to kill stuck CC.")
            parts.append(f"THEN: Write [TASK]...[/TASK] with a concrete code fix.")
        else:
            parts.append(f"CC is idle. Write [TASK]...[/TASK] NOW with a concrete code fix.")
        parts.append(f"DO NOT discuss. DO NOT plan. Write task ONLY.")
        parts.append(f"SYSTEM OVERRIDE: PLANNING SUSPENDED. EXECUTE PUSH NOW.")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 7: MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

# Flush conversation log on exit
import atexit, signal
atexit.register(flush_chatlog)
def _signal_flush(signum, frame):
    flush_chatlog()
    sys.exit(0)
signal.signal(signal.SIGTERM, _signal_flush)

# Force immediate flush of startup banner
startup_msg = "\n" + "="*60 + "\n  Adam & Eve — A2A Agent Orchestrator (v4.1)\n  OpenClaw agents via A2A → Claude Code executes\n" + "="*60 + "\n"
print(startup_msg, flush=True)

# Initialize global acpx session (try once at startup) - don't let failure block startup
print("[INIT] Initializing global acpx session...", flush=True)
try:
    _init_global_acpx_session()
    print("[INIT] Acpx session initialization complete", flush=True)
except Exception as e:
    print(f"[INIT] Acpx session initialization failed (non-fatal): {e}", flush=True)

# Clear chatlog only on fresh start (not restart)
# post_chatlog([])  # Clear chatlog - REMOVED: preserve conversation across restarts

# Opening turn — send via A2A to Adam's OpenClaw (with error handling)
print("[INIT] Starting opening turn...", flush=True)
try:
    ctx = gather_context()
    _current_speaker = "Adam"
    opening_message = build_turn_message("Adam", "Eve", ctx)
    print("[INIT] Sending opening turn to Adam...", flush=True)
    reply = send_a2a_message(ADAM_SPACE, opening_message)
    if reply:
        clean, actions, _ = parse_and_execute_turn(reply, ctx)
        last_action_results = actions
        if actions:
            record_actions("Adam", 0, actions)
        en, zh = parse_bilingual(clean)
        en, zh = _strip_speaker_labels(en), _strip_speaker_labels(zh)
        print(f"[Adam/EN] {en}")
        if zh != en:
            print(f"[Adam/ZH] {zh}")
        for ar in actions:
            print(f"[Adam/DID] {ar['action']}")
        ts = datetime.datetime.utcnow().strftime("%H:%M")
        entry = {"speaker": "Adam", "time": ts, "text": en, "text_zh": zh}
        history.append(entry)
        # Add labels for display only (bubble/chatlog), NOT for agent context
        display_labels = ""
        if actions:
            display_labels = " " + " ".join(f"🔧{ar['action'].split(':')[0]}" for ar in actions)
        set_bubble(ADAM_SPACE, en + display_labels, zh + display_labels)
        post_chatlog(history)
        persist_turn("Adam", 0, en, zh, actions, workflow_state, child_state["stage"])
        print("[INIT] Opening turn completed successfully", flush=True)
    else:
        print("[INIT] Opening turn failed: no response from Adam. Will continue to main loop.", flush=True)
except Exception as e:
    print(f"[INIT] Opening turn failed with error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    print("[INIT] Continuing to main loop despite opening turn failure...", flush=True)

print("[INIT] Opening turn complete. Entering main conversation loop...", flush=True)
print(f"[INIT] Current time: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
time.sleep(TURN_INTERVAL)


def do_turn(speaker, other, space_url):
    """Execute one conversation turn (non-blocking — CC runs in background)."""
    global last_action_results, turn_count, _current_speaker, _discussion_loop_count, _turns_since_last_push
    global _pending_task_just_submitted, _pending_task_timestamp, _pending_task_speaker, _pending_task_desc
    turn_count += 1
    _turns_since_last_push += 1
    _current_speaker = speaker

    # Skip agent if they have too many consecutive failures (prevents blocking the whole loop)
    agent_key = speaker.lower()
    if _a2a_health[agent_key]["failures"] >= 10:
        print(f"[{speaker}] SKIPPED: {speaker} has {_a2a_health[agent_key]['failures']} consecutive failures. Letting the other agent continue.")
        return False

    # Auto-gather context (lightweight)
    ctx = gather_context()

    # Check if CC just finished — clear result after agents see it once
    # ALSO reset turns-since-push counter ONLY when there was actual progress (push)
    # CRITICAL: Do NOT reset when zero pushes - that's exactly when we need to track the crisis!
    with cc_lock:
        cc_just_finished = (not cc_status["running"] and cc_status["result"])
        if cc_just_finished and _push_count_this_task > 0:
            # Only reset counter when CC finished with at least 1 push (actual progress)
            # This prevents "all talk no action" detection from being broken by zero-push completions
            _turns_since_last_push = 0

    # AUTO-TERMINATE stuck Claude Code processes
    # Only kill if CC has been running longer than the normal timeout with no new output
    # Push frequency supervision is God's job, not the orchestrator's
    with cc_lock:
        cc_running = cc_status["running"]
        cc_started = cc_status["started"]
        time_since_start = time.time() - cc_started if cc_running else 0

    if cc_running and time_since_start > CLAUDE_TIMEOUT:
        time_since_new_output = time.time() - _last_cc_output_time if _last_cc_output_time > 0 else time_since_start
        if time_since_new_output > CC_STUCK_TIMEOUT and _cc_stale_count >= 3:
            print(f"[CC-AUTO-KILL] Claude Code stuck for {time_since_new_output:.0f}s with no new output. Auto-terminating.")
            terminate_result = action_terminate_cc()
            print(f"[CC-AUTO-KILL] {terminate_result}")

    # Normal path: Send message via A2A to agent's OpenClaw instance
    # Note: Push frequency supervision and emergency overrides are God's job,
    # not the orchestrator's. God monitors via do_god_turn_a2a() and proposes fixes.
    message = build_turn_message(speaker, other, ctx)
    t0 = time.time()
    raw_reply = send_a2a_message(space_url, message)

    if not raw_reply:
        print(f"[{speaker}] (no A2A response from {space_url})")
        return False

    clean_text, action_results, _ = parse_and_execute_turn(raw_reply, ctx)
    elapsed = time.time() - t0
    last_action_results = action_results
    if action_results:
        record_actions(speaker, turn_count, action_results)

    en, zh = parse_bilingual(clean_text)
    en, zh = _strip_speaker_labels(en), _strip_speaker_labels(zh)

    # Skip empty responses (malformed parsing) - don't add to history or chatlog
    if not en and not zh:
        print(f"[{speaker}] (empty response after parsing, skipping chatlog update)")
        # Still record actions if any
        if action_results:
            record_actions(speaker, turn_count, action_results)
        # Update the loop counter even if we skip chatlog
        return True

    print(f"[{speaker}/EN] {en}")
    if zh != en:
        print(f"[{speaker}/ZH] {zh}")
    if action_results:
        for ar in action_results:
            print(f"[{speaker}/DID] {ar['action']}")
        print(f"[{speaker}] Turn #{turn_count}: {len(action_results)} action(s) in {elapsed:.1f}s")
    else:
        print(f"[{speaker}] Turn #{turn_count}: discussion ({elapsed:.1f}s)")

    # Clear CC result after both agents have had a chance to see it
    if cc_just_finished and speaker == "Eve":
        with cc_lock:
            cc_status["result"] = ""
            _context_cache.clear()
        # Clear pending task flag since CC finished
        _pending_task_just_submitted = False
    # CRITICAL FIX: Also clear pending task flag when CC finishes, regardless of speaker
    # This fixes the race condition where Adam's turn comes before Eve's after CC finishes
    # ALSO: Clear when CC is not running (handles auto-termination where result is cleared)
    elif cc_just_finished and _pending_task_just_submitted:
        _pending_task_just_submitted = False
    elif not cc_status["running"] and _pending_task_just_submitted:
        # CC finished but result was cleared (e.g., auto-termination for handoff)
        # Clear the pending flag so agents can submit new tasks
        _pending_task_just_submitted = False

    # Add to history with timestamp (text stays CLEAN for agent context)
    ts = datetime.datetime.utcnow().strftime("%H:%M")
    entry = {"speaker": speaker, "time": ts, "text": en, "text_zh": zh}
    history.append(entry)

    # Add labels for display only (bubble), NOT for agent context
    display_labels = ""
    if action_results:
        display_labels = " " + " ".join(f"🔧{ar['action'].split(':')[0]}" for ar in action_results)

    # Update frontend and persistence with error handling
    try:
        set_bubble(space_url, en + display_labels, zh + display_labels)
    except Exception as e:
        print(f"[{speaker}] Failed to set bubble: {e}")

    try:
        post_chatlog(history)
    except Exception as e:
        print(f"[{speaker}] Failed to post chatlog: {e}")

    try:
        persist_turn(speaker, turn_count, en, zh, action_results, workflow_state, child_state["stage"])
    except Exception as e:
        print(f"[{speaker}] Failed to persist turn: {e}")

    return True


# ── God A2A Turn (replaces embedded God logic) ──────────────────────────────

def build_god_turn_message(ctx):
    """Build A2A message for God's turn. Sends system-level context for architectural evaluation."""
    parts = []
    parts.append("You are God, the system architect of the HuggingClaw family system.")
    parts.append("Review the system state below from an **architectural perspective**.")
    parts.append("Don't micro-manage agents. Think about whether the system design itself is right.")
    parts.append("Respond with [OK] if architecture is sound, or [TASK]...[/TASK] with a structural improvement.")

    # System overview
    parts.append(f"\n## System State")
    parts.append(f"- Turn count: {turn_count}, Workflow: {workflow_state}")
    parts.append(f"- Child ({CHILD_NAME}): stage={child_state['stage']}, alive={child_state['alive']}")
    parts.append(f"- A2A health: Adam={_a2a_health['adam']['failures']} failures, Eve={_a2a_health['eve']['failures']} failures")

    # CC status (high-level)
    parts.append(f"\n## Claude Code Worker")
    if cc_status["running"]:
        elapsed = int(time.time() - cc_status["started"])
        parts.append(f"- Status: RUNNING ({elapsed}s), assigned by: {cc_status['assigned_by']}")
    else:
        parts.append(f"- Status: IDLE")
    parts.append(f"- Total pushes: {_push_count}")

    # Recent conversation (condensed — God sees patterns, not details)
    parts.append(f"\n## Recent Conversation Summary ({len(history)} total turns)")
    for entry in history[-10:]:
        spk = entry.get("speaker", "?")
        text = entry.get("text", "")[:500]
        parts.append(f"  {spk}: {text[:200]}{'...' if len(text) > 200 else ''}")
    if not history:
        parts.append("(no conversation yet)")

    parts.append(f"""
## Your Role
Think as a system architect:
- Is the communication flow between agents working well?
- Is the task routing mechanism effective?
- Are there structural bottlenecks or design flaws?
- How could the framework evolve to be fundamentally better?

If architecture is sound: [OK] brief assessment
If redesign needed: analysis + [TASK] structural change [/TASK]""")

    return "\n".join(parts)


def do_god_turn_a2a():
    """God's turn via A2A: send system state to God OpenClaw instance, parse response."""
    global _god_running, _last_god_time
    global _god_last_turn_count, _god_last_child_stage, _god_last_push_count

    # Skip if nothing changed (zero-cost check)
    child_in_error = child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR")
    nothing_changed = (
        turn_count == _god_last_turn_count
        and child_state["stage"] == _god_last_child_stage
        and _push_count == _god_last_push_count
    )
    if nothing_changed and not child_in_error and _god_last_turn_count > 0:
        print(f"[God] Skipping — no new turns, pushes, or stage changes since last check")
        return

    _god_last_turn_count = turn_count
    _god_last_child_stage = child_state["stage"]
    _god_last_push_count = _push_count

    _god_running = True
    try:
        # Build and send A2A message to God
        ctx = gather_context()
        message = build_god_turn_message(ctx)
        print(f"[God] Sending A2A message to God Space ({len(message)} chars)...")

        reply = send_a2a_message(GOD_SPACE, message, timeout=120)

        if not reply:
            print(f"[God] No A2A response from God Space")
            return

        reply = reply.strip()
        print(f"[God] Reply ({len(reply)} chars): {reply[:200]}")

        # Post God's reply to chatlog
        en, zh = parse_bilingual(reply)
        ts = datetime.datetime.utcnow().strftime("%H:%M")
        entry = {"speaker": "God", "time": ts, "text": en[:500], "text_zh": zh[:500]}
        history.append(entry)
        set_bubble(HOME, en[:200], zh[:200])
        post_chatlog(history)
        persist_turn("God", turn_count, en, zh, [], workflow_state, child_state["stage"])

        # Parse response: [OK] or [TASK]...[/TASK]
        if "[OK]" in reply.upper():
            print(f"[God] System healthy.")
        else:
            # Extract [TASK] block
            task_match = re.search(r'\[TASK\](.*?)\[/TASK\]', reply, re.DOTALL | re.IGNORECASE)
            if not task_match:
                # Try alternate format: [任务]...[/任务]
                task_match = re.search(r'\[任务\](.*?)\[/任务\]', reply, re.DOTALL)
            if task_match:
                task = task_match.group(1).strip()
                if task and not god_cc_status["running"]:
                    print(f"[God] Submitting fix task: {task[:200]}")
                    cc_submit_task_god(task)
                elif god_cc_status["running"]:
                    print(f"[God] CC already running, skipping task")
            else:
                print(f"[God] Response had no [TASK] block, treating as observation")
    except Exception as e:
        print(f"[God] A2A turn failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        _god_running = False
        _last_god_time = time.time()


# _prepare_god_context() — REMOVED: replaced by build_god_turn_message() above


# _god_diagnose() — REMOVED: God now uses A2A (its own OpenClaw instance handles diagnosis)


# do_god_turn() — REMOVED: replaced by do_god_turn_a2a() above


_last_god_time = 0.0  # timestamp of last God run
_god_running = False  # flag to track if God is currently running
_god_last_turn_count = 0  # turn count at last God run (skip if no new turns)
_god_last_child_stage = ""  # child stage at last God run (skip if unchanged)
_god_last_push_count = 0  # push count at last God run

# Initialize push count from existing workspace to persist across restarts
_init_push_count_from_workspace()

# Main loop: Eve → Adam → Eve → Adam → ... with God A2A every 2 minutes
print("[LOOP] Entering main conversation loop...", flush=True)
iteration = 0
_last_heartbeat = time.time()
while True:
    iteration += 1
    if iteration % 10 == 1:
        print(f"[LOOP] Main loop iteration #{iteration} at {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC", flush=True)
    # Log heartbeat every 2 minutes so we can detect if loop is stuck
    if time.time() - _last_heartbeat >= 120:
        print(f"[LOOP] Heartbeat: iteration {iteration}, CC running={cc_status['running']}, discussion_loop={_discussion_loop_count}, time={datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC", flush=True)
        _last_heartbeat = time.time()

    # Refresh Cain's stage periodically
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        new_stage = info.runtime.stage if info.runtime else "unknown"
        if new_stage != child_state["stage"]:
            print(f"[STATUS] {child_state['stage']} → {new_stage}")
            child_state["stage"] = new_stage
            child_state["alive"] = (new_stage == "RUNNING")
            _context_cache.clear()
    except Exception as e:
        print(f"[STATUS] Error: {e}")

    # Check Adam/Eve health and restart if needed
    try:
        check_and_restart_unhealthy_agents()
    except Exception as e:
        print(f"[A2A-HEALTH] Error checking health: {e}", file=sys.stderr)

    # Check worker heartbeat telemetry (Project Icarus)
    try:
        check_worker_heartbeat_health()
    except Exception as e:
        print(f"[HEARTBEAT] Error checking worker health: {e}", file=sys.stderr)

    # CORRUPTED CONVERSATION RESET: Detect and reset poisoned conversation history
    # Symptoms: empty messages, messages ending with "-" (cut off), repeated emergency loops
    # This happens when A2A communication fails partway through, leaving unusable context
    # Note: history and _discussion_loop_count are module-level globals, no 'global' keyword needed here
    if history and turn_count >= 3:
        # Check for corruption patterns
        has_empty = any(h.get("text", "").strip() == "" for h in history[-2:])
        has_cutoff = any(h.get("text", "").rstrip().endswith("-") for h in history[-2:])

        if has_empty or has_cutoff:
            print(f"[CONV-RESET] Detected corrupted conversation (empty={has_empty}, cutoff={has_cutoff}). Resetting history to allow fresh start.")
            # Keep only the most recent God message (if any) to show continuity
            god_messages = [h for h in history if h.get("speaker") == "God" and "Found issue" in h.get("text", "")]
            keep = god_messages[-1:] if god_messages else []
            history = keep
            # Clear chatlog on frontend
            try:
                post_chatlog(history)
                print(f"[CONV-RESET] Cleared corrupted chatlog, kept {len(keep)} God message(s)")
            except Exception as e:
                print(f"[CONV-RESET] Failed to post cleared chatlog: {e}")
            # Reset discussion loop counter since we're starting fresh
            _discussion_loop_count = 0

    # EMERGENCY OVERRIDE PROTOCOL: Detect "all talk no action" deadlock
    # Trigger 1: discussion_loop_count > MAX_IDLE_TURNS AND zero pushes (_push_count == 0)
    # Trigger 2: turns_since_last_push > MAX_TURNS_WITHOUT_PUSH (catches "1 push then 20 turns discussion" anti-pattern)
    # Trigger 2 is critical: _discussion_loop_count resets on each task assignment, so agents can game the system
    # by assigning tasks sporadically while mostly discussing. _turns_since_last_push catches this pattern.
    if not _force_push_mode:
        trigger_reason = None
        if _discussion_loop_count > MAX_IDLE_TURNS and _push_count == 0:
            trigger_reason = f"{_discussion_loop_count} discussion turns with ZERO total pushes"
        elif _turns_since_last_push > MAX_TURNS_WITHOUT_PUSH:
            trigger_reason = f"{_turns_since_last_push} turns since last push (anti-pattern: assigning tasks but not pushing)"

        if trigger_reason:
            print(f"[EMERGENCY-OVERRIDE] TRIGGERED: {trigger_reason}!")
            _force_push_mode = True
            _emergency_override_active = True
            _force_push_trigger_time = time.time()
            # Auto-terminate CC if running (Emergency Override: idle > 10s allows immediate termination)
            cc_idle_time = time.time() - (_last_cc_output_time if _last_cc_output_time > 0 else time.time())
            if cc_status["running"]:
                if cc_idle_time > 10:  # Emergency Override: Immediate termination if idle > 10s
                    print(f"[EMERGENCY-OVERRIDE] CC idle {int(cc_idle_time)}s > 10s threshold, terminating immediately")
                    action_terminate_cc()
                    _force_push_skip_termination = True
                else:
                    print(f"[EMERGENCY-OVERRIDE] CC running but active, will terminate on next agent turn")
                    _force_push_skip_termination = False
            else:
                _force_push_skip_termination = True  # CC already idle

    # Reset FORCE_PUSH mode after 5 minutes (safety valve)
    if _force_push_mode and time.time() - _force_push_trigger_time > 300:
        print(f"[EMERGENCY-OVERRIDE] Mode timeout (300s), resetting to normal")
        _force_push_mode = False
        _emergency_override_active = False
        _force_push_skip_termination = False

    # ══════════════════════════════════════════════════════════════════════════════
    #  LOCKDOWN MODE — "Purge & Reboot" for Stalemate State
    #  Detects when Cain is stuck in ERROR state for too long without effective pushes.
    #  Triggers hard reset: terminate CC, clear conversation, force fresh diagnostic.
    # ══════════════════════════════════════════════════════════════════════════════

    # Track error state onset
    child_in_error = child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR")
    if child_in_error and _lockdown_error_onset == 0:
        # Cain just entered error state — record when it started
        _lockdown_error_onset = time.time()
        _lockdown_push_count_at_error = _push_count
        print(f"[LOCKDOWN] Cain entered ERROR state at {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC, push_count={_push_count}")
    elif not child_in_error:
        # Cain recovered — reset error tracking
        if _lockdown_error_onset != 0:
            print(f"[LOCKDOWN] Cain recovered from ERROR state, resetting error tracking")
        _lockdown_error_onset = 0
        _lockdown_push_count_at_error = 0

    # LOCKDOWN TRIGGER: Error state persists too long WITHOUT effective pushes
    # "Effective push" means _push_count increased since error started
    if not _lockdown_mode and child_in_error and _lockdown_error_onset > 0:
        error_duration = time.time() - _lockdown_error_onset
        pushes_during_error = _push_count - _lockdown_push_count_at_error

        # Trigger if: (a) error duration > threshold AND (b) no effective pushes during error
        if error_duration > LOCKDOWN_ERROR_THRESHOLD_SECS and pushes_during_error == 0:
            print(f"[LOCKDOWN] TRIGGERED! Cain in ERROR for {int(error_duration)}s with ZERO effective pushes. System in STALEMATE — initiating PURGE & REBOOT.")
            _lockdown_mode = True
            _lockdown_trigger_time = time.time()

            # LOCKDOWN ACTION: Hard reset
            # 1. Force-terminate CC if running
            if cc_status["running"]:
                print(f"[LOCKDOWN] Terminating stuck CC process...")
                action_terminate_cc()

            # 2. Clear conversation history (purge corrupted context)
            print(f"[LOCKDOWN] Clearing conversation history to purge stale context...")
            history = []
            _discussion_loop_count = 0

            # 3. Post cleared chatlog to frontend
            try:
                post_chatlog(history)
                print(f"[LOCKDOWN] Conversation history purged")
            except Exception as e:
                print(f"[LOCKDOWN] Failed to post cleared chatlog: {e}")

            # 4. Force a fresh diagnostic task to break the stalemate
            print(f"[LOCKDOWN] Forcing fresh diagnostic task...")
            ctx = gather_context()
            forced_task = f"""EMERGENCY LOCKDOWN REBOOT: Cain has been in {child_state['stage']} for {int(error_duration)}s with zero effective fixes. The system is in STALEMATE.

CRITICAL INSTRUCTIONS:
1. DIAGNOSE the root cause of the error immediately — check logs, error messages, runtime state
2. Do NOT discuss — write [TASK] with a SPECIFIC fix immediately
3. Trial-and-error is REQUIRED — push a fix attempt, don't deliberate
4. If unsure what's wrong, the priority is getting Cain RUNNING — try the most likely fix

Current stage: {child_state['stage']}
Error duration: {int(error_duration)}s
This is a PURGE & REBOOT — start fresh, don't reference previous failed attempts."""
            cc_submit_task(forced_task, "LOCKDOWN", ctx)

    # Reset LOCKDOWN mode after timeout (safety valve)
    if _lockdown_mode and time.time() - _lockdown_trigger_time > LOCKDOWN_RESET_SECS:
        print(f"[LOCKDOWN] Mode timeout ({LOCKDOWN_RESET_SECS}s), resetting to normal")
        _lockdown_mode = False
        _lockdown_error_onset = 0
        _lockdown_push_count_at_error = 0

    # Note: Aggressive CC auto-termination based on push frequency is removed.
    # God monitors push frequency and proposes mechanism fixes when needed.
    # The normal CLAUDE_TIMEOUT auto-kill in do_turn() handles truly stuck processes.

    # Eve's turn with error handling to prevent loop crash
    try:
        do_turn("Eve", "Adam", EVE_SPACE)
    except Exception as e:
        print(f"[ERROR] Eve turn failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    # Adaptive interval: slow down when CC output hasn't changed
    wait = TURN_INTERVAL + min(_cc_stale_count * 15, 90)  # 15s → 30s → 45s → ... → max 105s
    if wait > TURN_INTERVAL:
        print(f"[PACE] CC output stale ({_cc_stale_count} turns), next turn in {wait}s")
    time.sleep(wait)

    # Adam's turn with error handling to prevent loop crash
    try:
        do_turn("Adam", "Eve", ADAM_SPACE)
    except Exception as e:
        print(f"[ERROR] Adam turn failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    time.sleep(wait)

    # RECOVERY: If BOTH agents are skipped (10+ failures each), reset their failure counters
    # This allows recovery when Spaces restart after a crash. Without this, the conversation
    # loop enters permanent deadlock with no agent able to respond.
    # Note: _a2a_health is already a module-level global, no 'global' declaration needed here
    if _a2a_health["adam"]["failures"] >= 10 and _a2a_health["eve"]["failures"] >= 10:
        print(f"[RECOVERY] Both agents have 10+ failures (adam={_a2a_health['adam']['failures']}, eve={_a2a_health['eve']['failures']}). Resetting failure counters to allow recovery after Space restarts.")
        _a2a_health["adam"]["failures"] = 0
        _a2a_health["eve"]["failures"] = 0
        # If CC is idle and Cain exists, force a diagnostic task to break the deadlock
        if not cc_status["running"] and child_state["created"]:
            print(f"[RECOVERY] Forcing diagnostic task to break communication deadlock")
            ctx = gather_context()
            forced_task = "Emergency diagnostic: A2A communication is failing. Check Cain's health, logs, and state. List any errors and suggest fixes."
            cc_submit_task(forced_task, "RECOVERY", ctx)

    # God runs every GOD_POLL_INTERVAL seconds (2 minutes)
    # Only start if not already running (prevent overlapping runs)
    if time.time() - _last_god_time >= GOD_POLL_INTERVAL and not _god_running:
        try:
            do_god_turn_a2a()
        except Exception as e:
            print(f"[ERROR] God turn failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
