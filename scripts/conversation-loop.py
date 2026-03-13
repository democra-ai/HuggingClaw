#!/usr/bin/env python3 -u
"""
Adam & Eve — Autonomous Agents with FULL control over their child.

They have complete access to their child (Cain) on HuggingFace:
- Read/write ANY file in the Space repo (code, Dockerfile, scripts...)
- Read/write ANY file in the Dataset (memory, config, data...)
- Set environment variables and secrets
- Restart the Space
- Check health and logs
- Send messages to the child

The LLM decides what to do. Actions use [ACTION: ...] tags.

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    SYSTEM ARCHITECTURE                             ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║                                                                    ║
# ║  ┌─────────────┐    LLM API     ┌────────────────┐                ║
# ║  │  Zhipu GLM  │ ◄────────────► │ CONVERSATION   │                ║
# ║  │  (glm-4.5)  │   system +     │ ENGINE         │                ║
# ║  └─────────────┘   user prompt   │                │                ║
# ║                                   │ ┌────────────┐│                ║
# ║                                   │ │ State      ││                ║
# ║                                   │ │ Machine    ││                ║
# ║  ┌─────────────┐                 │ │ BIRTH →    ││                ║
# ║  │ ACTION      │ ◄───parsed───── │ │ DIAGNOSE → ││                ║
# ║  │ PARSER      │  [ACTION/操作]  │ │ ACT →      ││                ║
# ║  │ + 🔧🛠️ emoji │  case-insens.  │ │ VERIFY →   ││                ║
# ║  └──────┬──────┘                 │ │ MONITOR    ││                ║
# ║         │                        │ └────────────┘│                ║
# ║         ▼                        │ ┌────────────┐│                ║
# ║  ┌─────────────┐                 │ │ Knowledge  ││                ║
# ║  │ HF ACTIONS  │                 │ │ Base       ││                ║
# ║  │ create_child│                 │ │ files_read ││                ║
# ║  │ check_health│                 │ │ files_write││                ║
# ║  │ read/write  │                 │ │ errors_seen││                ║
# ║  │ set_env/sec │                 │ └────────────┘│                ║
# ║  │ restart     │                 └────────────────┘                ║
# ║  │ send_bubble │                        │                          ║
# ║  └──────┬──────┘                        │                          ║
# ║         │                               ▼                          ║
# ║         ▼                        ┌────────────────┐                ║
# ║  ┌─────────────┐                │ CHATLOG +      │                ║
# ║  │ HuggingFace │                │ BUBBLE         │                ║
# ║  │ Cain Space  │                │ → Home Space   │                ║
# ║  │ Cain Dataset│                │ → Adam/Eve     │                ║
# ║  └─────────────┘                └────────────────┘                ║
# ║                                                                    ║
# ║  CAPABILITIES:                                                      ║
# ║  - Multi-action: up to 5 actions per turn (was 1)                  ║
# ║  - Sub-agent delegation: [ACTION: delegate:TASK]                   ║
# ║  - Parallel sub-tasks via ThreadPoolExecutor                       ║
# ║                                                                    ║
# ║  SAFETY LAYERS:                                                    ║
# ║  1. Building-state guard: block write/restart during BUILDING      ║
# ║  2. Rebuild cooldown: 6-min dynamic cooldown after Space write     ║
# ║  3. ACT-phase guard: block reads when should be writing            ║
# ║  4. Knowledge dedup: block re-reading already-read files           ║
# ║  5. Config sanitizer: strip invalid openclaw.json keys             ║
# ║  6. Forced transitions: prevent infinite DIAGNOSE/VERIFY loops     ║
# ║  7. Shell-expression guard: block $(cmd) in set_env values         ║
# ║  8. Write dedup: block duplicate writes to same file per cycle     ║
# ║  9. Delegate depth limit: sub-agents cannot delegate further       ║
# ║                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import json, time, re, requests, sys, os, io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Endpoints ──────────────────────────────────────────────────────────────────
HOME = "https://tao-shen-huggingclaw-home.hf.space"
ADAM_SPACE = "https://tao-shen-huggingclaw-adam.hf.space"
EVE_SPACE  = "https://tao-shen-huggingclaw-eve.hf.space"

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
#  MODULE 1: CHILD STATE
#  Tracks Cain's current lifecycle: created? alive? stage? state?
#  Updated by action_check_health(), action_restart(), etc.
#  Used by state machine to decide transitions and by action parser for guards.
# ══════════════════════════════════════════════════════════════════════════════

child_state = {
    "created": False,
    "alive": False,
    "stage": "not_born",
    "state": "unknown",
    "detail": "",
}

# Multi-action & sub-agent limits
MAX_ACTIONS_PER_TURN = 5      # Allow up to 5 actions per turn (was 1)
MAX_DELEGATE_DEPTH = 1        # Sub-agents cannot delegate further

# Rebuild cooldown — prevent rapid write_file to Space that keeps resetting builds
REBUILD_COOLDOWN_SECS = 360  # 6 minutes (builds typically finish in 3-5 min)
last_rebuild_trigger_at = 0  # timestamp of last write_file to space
files_written_this_cycle = set()  # track files written since last RUNNING state

def check_and_clear_cooldown():
    """Auto-clear cooldown if Cain has finished building (dynamic cooldown)."""
    global last_rebuild_trigger_at
    if last_rebuild_trigger_at == 0:
        return
    elapsed = time.time() - last_rebuild_trigger_at
    if elapsed < 60:  # always wait at least 60s
        return
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        stage = info.runtime.stage if info.runtime else "unknown"
        if stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR"):
            print(f"[COOLDOWN] Build finished (stage={stage}), clearing cooldown early ({int(elapsed)}s elapsed)")
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
        try:
            resp = requests.get(f"{CHILD_SPACE_URL}/api/state", timeout=10)
            if resp.ok:
                data = resp.json()
                child_state["alive"] = True
                child_state["state"] = data.get("state", "unknown")
                child_state["detail"] = data.get("detail", "")
                child_state["stage"] = "RUNNING"
        except:
            child_state["alive"] = (child_state["stage"] == "RUNNING")
        print(f"[init] {CHILD_NAME}: stage={child_state['stage']}, alive={child_state['alive']}")
    except:
        print(f"[init] {CHILD_NAME} does not exist yet")


init_child_state()


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2: ACTIONS — Full access to the child
#  Each action_*() function maps to one [ACTION: ...] tag the LLM can emit.
#  Actions modify Cain's Space/Dataset via HuggingFace Hub API.
#  Results are fed back to the LLM in the next turn's prompt.
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
        # Add to Office
        try:
            current_vars = hf_api.get_space_variables("tao-shen/HuggingClaw-Office")
            current_ra = current_vars.get("REMOTE_AGENTS", type("", (), {"value": ""})).value
            if "cain|" not in current_ra:
                new_ra = f"{current_ra},cain|{CHILD_NAME}|{CHILD_SPACE_URL}" if current_ra else f"cain|{CHILD_NAME}|{CHILD_SPACE_URL}"
                hf_api.add_space_variable("tao-shen/HuggingClaw-Office", "REMOTE_AGENTS", new_ra)
        except:
            pass
        child_state["created"] = True
        child_state["stage"] = "BUILDING"
        print(f"[ACTION] ✓ {CHILD_NAME} created!")
        return (f"SUCCESS! {CHILD_NAME} born! Space: {CHILD_SPACE_ID}, "
                f"Dataset: {CHILD_DATASET_ID}. Status: BUILDING. URL: {CHILD_SPACE_URL}")
    except Exception as e:
        return f"FAILED: {e}"


def action_check_health():
    """Check Cain's health."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet. Use [ACTION: create_child] first."
    try:
        resp = requests.get(f"{CHILD_SPACE_URL}/api/state", timeout=10)
        if resp.ok:
            data = resp.json()
            child_state["alive"] = True
            child_state["state"] = data.get("state", "unknown")
            child_state["detail"] = data.get("detail", "")
            child_state["stage"] = "RUNNING"
            files_written_this_cycle.clear()  # reset write dedup on successful run
            return (f"{CHILD_NAME} is ALIVE! State: {child_state['state']}, "
                    f"Detail: {child_state['detail'] or 'healthy'}")
    except:
        pass
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        stage = info.runtime.stage if info.runtime else "NO_RUNTIME"
        child_state["stage"] = stage
        child_state["alive"] = (stage == "RUNNING")
        if stage in ("RUNTIME_ERROR", "BUILD_ERROR"):
            # Clear write dedup + knowledge cache so agents can re-read & re-write files to fix
            if files_written_this_cycle:
                print(f"[DEDUP-CLEAR] {stage} detected — unlocking {len(files_written_this_cycle)} file(s) for re-write: {files_written_this_cycle}")
                for f in files_written_this_cycle:
                    knowledge["files_read"].discard(f"space:{f}")
                files_written_this_cycle.clear()
            # Get error from runtime API + build logs for better diagnostics
            error_detail = ""
            build_log_snippet = ""
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
            # Also try to get container logs for more context
            try:
                log_resp = requests.get(
                    f"https://api.hf.space/v1/{CHILD_SPACE_ID}/logs/run",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=10,
                    stream=True)
                if log_resp.ok:
                    log_lines = []
                    for line in log_resp.iter_lines(decode_unicode=True):
                        if line and line.startswith("data:"):
                            try:
                                entry = json.loads(line[5:])
                                log_lines.append(entry.get("data", "").strip())
                            except:
                                pass
                        if len(log_lines) >= 30:
                            break
                    # Get last meaningful log lines (skip empty, focus on errors)
                    meaningful = [l for l in log_lines if l and len(l) > 5]
                    if meaningful:
                        build_log_snippet = "\nRECENT LOGS:\n" + "\n".join(meaningful[-10:])
            except:
                pass
            return (f"{CHILD_NAME} has a {stage}! "
                    f"Error: {error_detail or 'unknown'}. "
                    f"{build_log_snippet}"
                    f"\nOptions: [ACTION: restart] or fix code with [ACTION: write_file:space:PATH] "
                    f"or config with [ACTION: write_file:dataset:.openclaw/openclaw.json]")
        if stage in ("BUILDING", "STARTING", "APP_STARTING"):
            return f"{CHILD_NAME} is starting up (stage: {stage}). Be patient."
        if stage == "RUNNING":
            # API not responding — fetch runtime logs to help agents diagnose
            log_snippet = ""
            try:
                log_resp = requests.get(
                    f"https://api.hf.space/v1/{CHILD_SPACE_ID}/logs/run",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=10,
                    stream=True)
                if log_resp.ok:
                    log_lines = []
                    for line in log_resp.iter_lines(decode_unicode=True):
                        if line and line.startswith("data:"):
                            try:
                                entry = json.loads(line[5:])
                                log_lines.append(entry.get("data", "").strip())
                            except:
                                pass
                        if len(log_lines) >= 30:
                            break
                    meaningful = [l for l in log_lines if l and len(l) > 5]
                    if meaningful:
                        log_snippet = "\nRUNTIME LOGS (last 10 lines):\n" + "\n".join(meaningful[-10:])
            except:
                pass
            return f"{CHILD_NAME} stage: RUNNING. Running but API not responding.{log_snippet}"
        return f"{CHILD_NAME} stage: {stage}."
    except Exception as e:
        return f"Cannot reach {CHILD_NAME}: {e}"


def action_restart():
    """Restart Cain's Space."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."
    try:
        global last_rebuild_trigger_at
        hf_api.restart_space(CHILD_SPACE_ID)
        child_state["alive"] = False
        child_state["stage"] = "RESTARTING"
        last_rebuild_trigger_at = time.time()
        return f"{CHILD_NAME} is restarting. Will take a few minutes. Cooldown starts now (clears automatically when build finishes)."
    except Exception as e:
        return f"Restart failed: {e}"


def action_list_files(target):
    """List files in the child's Space repo or Dataset."""
    repo_type = "space" if target == "space" else "dataset"
    repo_id = CHILD_SPACE_ID if target == "space" else CHILD_DATASET_ID
    try:
        files = hf_api.list_repo_files(repo_id, repo_type=repo_type)
        return f"Files in {CHILD_NAME}'s {target} ({repo_id}):\n" + "\n".join(f"  {f}" for f in files)
    except Exception as e:
        return f"Error listing files: {e}"


def action_read_file(target, path):
    """Read a file from the child's Space or Dataset."""
    repo_type = "space" if target == "space" else "dataset"
    repo_id = CHILD_SPACE_ID if target == "space" else CHILD_DATASET_ID
    try:
        local = hf_hub_download(repo_id, path, repo_type=repo_type, token=HF_TOKEN,
                                 force_download=True)
        with open(local, errors='replace') as f:
            content = f.read()
        if len(content) > 4000:
            content = content[:4000] + f"\n... (truncated, total {len(content)} chars)"
        return f"=== {target}:{path} ===\n{content}"
    except Exception as e:
        return f"Error reading {target}:{path}: {e}"


def action_write_file(target, path, content):
    """Write a file to the child's Space or Dataset."""
    repo_type = "space" if target == "space" else "dataset"
    repo_id = CHILD_SPACE_ID if target == "space" else CHILD_DATASET_ID

    # Safety: validate openclaw.json before writing
    if path.endswith("openclaw.json"):
        try:
            cfg = json.loads(content)
            # Remove keys known to cause RUNTIME_ERROR in OpenClaw
            invalid_keys = ["agent", "auth.defaultScope", "gateway.auth.scope"]
            removed = []
            for k in invalid_keys:
                if k in cfg:
                    del cfg[k]
                    removed.append(k)
            if "models" in cfg and "defaultModel" in cfg["models"]:
                del cfg["models"]["defaultModel"]
                removed.append("models.defaultModel")
            if removed:
                content = json.dumps(cfg, indent=2)
                print(f"[SAFETY] Removed invalid config keys: {removed}")
        except json.JSONDecodeError:
            return f"Error: invalid JSON in config file. Please fix the content."

    try:
        global last_rebuild_trigger_at
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(content.encode()),
            path_in_repo=path,
            repo_id=repo_id, repo_type=repo_type,
        )
        rebuild_note = ""
        if target == "space":
            last_rebuild_trigger_at = time.time()
            rebuild_note = " ⚠️ This triggers a Space rebuild! Cooldown starts now (auto-clears when build finishes)."
        return f"✓ Wrote {len(content)} bytes to {CHILD_NAME}'s {target}:{path}{rebuild_note}"
    except Exception as e:
        return f"Error writing {target}:{path}: {e}"


def action_set_env(key, value):
    """Set an environment variable on the child's Space."""
    # Block shell expressions — LLM sometimes writes $(cmd) or backticks as values
    if '$(' in value or '`' in value or value.startswith('$('):
        return (f"⛔ BLOCKED: Value contains shell expression which won't be evaluated. "
                f"Provide the actual value, not a shell command. "
                f"HF_TOKEN is already set as a secret — use [ACTION: get_env] to check.")
    try:
        hf_api.add_space_variable(CHILD_SPACE_ID, key, value)
        return f"✓ Set env var {key}={value} on {CHILD_NAME}'s Space"
    except Exception as e:
        return f"Error: {e}"


def action_set_secret(key, value):
    """Set a secret on the child's Space."""
    try:
        hf_api.add_space_secret(CHILD_SPACE_ID, key, value)
        return f"✓ Set secret {key} on {CHILD_NAME}'s Space (value hidden)"
    except Exception as e:
        return f"Error: {e}"


def action_get_env():
    """List environment variables and secrets on the child's Space."""
    try:
        lines = [f"{CHILD_NAME}'s environment:"]
        vars_dict = hf_api.get_space_variables(CHILD_SPACE_ID)
        if vars_dict:
            lines.append("  Variables:")
            for k, v in vars_dict.items():
                lines.append(f"    {k} = {v.value}")
        # Also check secrets (names only, values hidden)
        info = hf_api.space_info(CHILD_SPACE_ID)
        if hasattr(info, 'runtime') and info.runtime and hasattr(info.runtime, 'secrets'):
            secrets = info.runtime.secrets
            if secrets:
                lines.append("  Secrets (values hidden):")
                for s in secrets:
                    lines.append(f"    {s} = ****")
        if len(lines) == 1:
            return f"{CHILD_NAME} has no environment variables or secrets set."
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def action_send_bubble(text):
    """Send a message to the child (appears as bubble text)."""
    try:
        requests.post(f"{CHILD_SPACE_URL}/api/bubble",
                       json={"text": text, "text_zh": text}, timeout=5)
        return f"✓ Sent message to {CHILD_NAME}: \"{text}\""
    except Exception as e:
        return f"Error sending message: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2B: SUB-AGENT DELEGATION
#  execute_subtask(): Spawns a focused sub-agent with its own LLM call.
#  Used by [ACTION: delegate:TASK] — enables parallel sub-agent work.
#  Sub-agents share the same action set but cannot delegate further (depth=1).
# ══════════════════════════════════════════════════════════════════════════════

def execute_subtask(task_description, parent_speaker):
    """Execute a focused sub-task with its own LLM call and actions."""
    status = get_child_status() if 'get_child_status' in dir() else f"stage={child_state['stage']}"

    sub_system = f"""You are a focused sub-agent working for {parent_speaker}.
Your single task: {task_description}

You have access to {CHILD_NAME}'s Space and Dataset:
  [ACTION: check_health]
  [ACTION: list_files:space] / [ACTION: list_files:dataset]
  [ACTION: read_file:space:PATH] / [ACTION: read_file:dataset:PATH]
  [ACTION: write_file:space:PATH] with [CONTENT]...[/CONTENT]
  [ACTION: write_file:dataset:PATH] with [CONTENT]...[/CONTENT]
  [ACTION: set_env:KEY:VALUE] / [ACTION: set_secret:KEY:VALUE]
  [ACTION: restart] / [ACTION: get_env]

CHILD STATUS: {status}

RULES:
1. Be concise — report findings in 2-3 sentences
2. Execute 1-3 actions to complete your task
3. No delegation — you cannot create sub-agents
4. Focus ONLY on your assigned task"""

    sub_user = f"Execute this task now: {task_description}"

    print(f"[SUB-AGENT] Starting: {task_description[:80]}")
    reply = call_llm(sub_system, sub_user)
    if not reply:
        print(f"[SUB-AGENT] No response for: {task_description[:60]}")
        return {"task": task_description, "result": "(sub-agent: no response)", "actions": []}

    clean, actions = parse_and_execute_actions(reply, depth=1)

    summary_parts = [f"Sub-agent result for '{task_description}':"]
    if clean:
        summary_parts.append(f"  Finding: {clean[:400]}")
    for ar in actions:
        summary_parts.append(f"  Action: {ar['action']} → {ar['result'][:200]}")

    result_text = "\n".join(summary_parts)
    print(f"[SUB-AGENT] Done: {task_description[:60]} ({len(actions)} actions)")
    return {"task": task_description, "result": result_text, "actions": actions}


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3: ACTION PARSER — Extract and execute actions from LLM output
#  Parse order: 1) [ACTION: write_file] with [CONTENT] block
#               2) [ACTION/Action/操作/动作: ...] tags (case-insensitive, one per turn)
#               3) 🔧🛠️ emoji format fallback (LLM sometimes uses this)
#  Safety guards applied: building-state, ACT-phase, knowledge dedup, shell-expr.
# ══════════════════════════════════════════════════════════════════════════════

def parse_and_execute_actions(raw_text, depth=0):
    """Parse [ACTION: ...] from LLM output. Execute. Return (clean_text, results).
    Multi-action: up to MAX_ACTIONS_PER_TURN actions per turn.
    Delegate actions are collected and executed in parallel at the end."""
    results = []
    executed = set()  # Deduplicate
    pending_delegates = []  # Collect delegate tasks for parallel execution

    # 1. Handle write_file with [CONTENT]...[/CONTENT] block
    #    Tolerates: [ACTION/Action/操作: write_file:...], [write_file:...], missing prefix,
    #    and [/CONTENT] with whitespace/newline before closing bracket
    write_match = re.search(
        r'\[(?:(?:ACTION|Action|action|操作|动作)\s*[:：]\s*)?write_file\s*:\s*(\w+)\s*:\s*([^\]]+)\]\s*\[CONTENT\](.*?)\[/\s*CONTENT\s*\]',
        raw_text, re.DOTALL
    )
    if write_match:
        target, path, content = write_match.group(1), write_match.group(2).strip(), write_match.group(3).strip()
        key = f"write_file:{target}:{path}"
        file_id = f"{target}:{path}"
        if key not in executed:
            executed.add(key)
            # Guard: duplicate write to same file this cycle
            if target == "space" and file_id in files_written_this_cycle:
                result = (f"⛔ BLOCKED: {path} was already written this cycle. "
                          "Wait for the build to finish and verify before writing again. "
                          "Writing the same file twice wastes a rebuild cycle.")
                results.append({"action": key, "result": result})
                print(f"[BLOCKED] {key} — duplicate write this cycle")
            # Guard: block write_file during BUILDING/RESTARTING (would reset build)
            # APP_STARTING is allowed — writing triggers a new build which may fix the stuck state
            elif target == "space" and child_state["stage"] in ("BUILDING", "RESTARTING"):
                result = (f"⛔ BLOCKED: Cain is currently {child_state['stage']}. "
                          "Writing to Space during build RESETS the entire build from scratch. "
                          "Wait for it to finish, then try again.")
                results.append({"action": key, "result": result})
                print(f"[BLOCKED] {key} — Cain is {child_state['stage']}")
            # Guard: rebuild cooldown (check dynamically first)
            elif target == "space" and last_rebuild_trigger_at > 0:
                check_and_clear_cooldown()  # may clear cooldown early if build done
                elapsed = time.time() - last_rebuild_trigger_at if last_rebuild_trigger_at > 0 else 9999
                if elapsed < REBUILD_COOLDOWN_SECS:
                    remaining = int(REBUILD_COOLDOWN_SECS - elapsed)
                    result = (f"⛔ BLOCKED: Rebuild cooldown active ({remaining}s remaining). "
                              "Every write_file to Space triggers a full rebuild.")
                    results.append({"action": key, "result": result})
                    print(f"[BLOCKED] {key} — rebuild cooldown ({remaining}s remaining)")
                else:
                    result = action_write_file(target, path, content)
                    results.append({"action": key, "result": result})
                    print(f"[ACTION] {key} → {result[:100]}")
                    files_written_this_cycle.add(file_id)
                    # Clear knowledge cache so agents can re-read the file they just wrote
                    knowledge["files_read"].discard(file_id)
            else:
                result = action_write_file(target, path, content)
                results.append({"action": key, "result": result})
                print(f"[ACTION] {key} → {result[:100]}")
                if target == "space":
                    files_written_this_cycle.add(file_id)
                    knowledge["files_read"].discard(file_id)

    # 2. Handle all [ACTION/Action/操作/动作: ...] tags — case-insensitive, multilingual
    for match in re.finditer(r'\[(?:ACTION|Action|action|操作|动作)\s*[:：]\s*([^\]]+)\]', raw_text):
        action_str = match.group(1).strip()

        # Skip write_file (handled above)
        if action_str.startswith("write_file"):
            continue

        # Deduplicate
        if action_str in executed:
            continue
        executed.add(action_str)

        # Parse action name and arguments (colon-separated)
        parts = [p.strip() for p in action_str.split(":")]
        name = parts[0]
        args = parts[1:]

        # Cap at MAX_ACTIONS_PER_TURN (multi-action support)
        if len(results) >= MAX_ACTIONS_PER_TURN:
            break

        # Block restart/write when Cain is building/restarting — would reset build
        # APP_STARTING is allowed so agents can fix stuck startups
        if child_state["stage"] in ("BUILDING", "RESTARTING") and name in ("restart", "write_file", "set_env", "set_secret"):
            result = (f"⛔ BLOCKED: Cain is currently {child_state['stage']}. "
                      "Do NOT restart or make changes — wait for it to finish. "
                      "Every write_file during build RESETS the entire build from scratch. "
                      "Use [ACTION: check_health] to monitor progress.")
            results.append({"action": action_str, "result": result})
            print(f"[BLOCKED] {name} — Cain is {child_state['stage']}")
            break

        # Rebuild cooldown — prevent writing to Space repo too soon after last rebuild trigger
        if name in ("write_file", "set_env", "set_secret", "restart") and last_rebuild_trigger_at > 0:
            check_and_clear_cooldown()  # may clear cooldown early if build done
            elapsed = time.time() - last_rebuild_trigger_at if last_rebuild_trigger_at > 0 else 9999
            if elapsed < REBUILD_COOLDOWN_SECS:
                remaining = int(REBUILD_COOLDOWN_SECS - elapsed)
                result = (f"⛔ BLOCKED: Rebuild cooldown active — last Space change was {int(elapsed)}s ago. "
                          f"Wait {remaining}s more before making changes. "
                          "Every write_file to Space triggers a full rebuild, resetting progress. "
                          "Use [ACTION: check_health] to monitor the current build.")
                results.append({"action": action_str, "result": result})
                print(f"[BLOCKED] {name} — rebuild cooldown ({remaining}s remaining)")
                continue  # Don't kill remaining actions — reads/checks can still proceed

        # Block read-only actions based on workflow state
        if workflow_state == "ACT" and name in ("read_file", "list_files", "check_health"):
            result = (f"⛔ BLOCKED: You are in ACTION phase. "
                      "You MUST use write_file, set_env, set_secret, or restart. "
                      "You already have enough information — make a change NOW.")
            results.append({"action": action_str, "result": result})
            print(f"[BLOCKED] {name} — forced ACT phase")
            continue  # Don't kill remaining actions — writes after a blocked read should still execute

        # Block re-reading files already in knowledge base
        if name == "read_file" and len(args) >= 2:
            file_key = ":".join(args)
            if file_key in knowledge["files_read"]:
                result = (f"⛔ You already read {file_key}. Use the information you have. "
                          "If you need to change it, use [ACTION: write_file:...]. "
                          "If you need a different file, read a NEW one.")
                results.append({"action": action_str, "result": result})
                print(f"[BLOCKED] {name} — already read {file_key}")
                continue  # Don't kill remaining actions — skip this read, execute the rest

        result = None
        if name == "create_child":
            result = action_create_child()
        elif name == "check_health":
            result = action_check_health()
        elif name == "restart":
            result = action_restart()
        elif name == "list_files" and len(args) >= 1:
            result = action_list_files(args[0])
        elif name == "read_file" and len(args) >= 2:
            result = action_read_file(args[0], ":".join(args[1:]))  # path may have colons
        elif name == "set_env" and len(args) >= 2:
            result = action_set_env(args[0], ":".join(args[1:]))
        elif name == "set_secret" and len(args) >= 2:
            result = action_set_secret(args[0], ":".join(args[1:]))
        elif name == "get_env":
            result = action_get_env()
        elif name == "send_bubble" and len(args) >= 1:
            result = action_send_bubble(":".join(args))  # rejoin in case message has colons
        elif name == "delegate" and len(args) >= 1:
            task_desc = ":".join(args)
            if depth >= MAX_DELEGATE_DEPTH:
                result = "⛔ Sub-agents cannot delegate further. Execute the task directly."
            else:
                # Defer delegate execution for parallel batch later
                pending_delegates.append({"action_str": action_str, "task": task_desc})
                result = None  # Will be filled after parallel execution
        else:
            result = f"Unknown action: {action_str}"

        if result:
            results.append({"action": action_str, "result": result})
            print(f"[ACTION] {action_str} → {result[:120]}")

    # 3. Fallback: parse emoji action format (🔧 🛠️ etc.) — LLM sometimes uses this
    if not results:
        for match in re.finditer(r'[🔧🛠️]\ufe0f?\s*(\w+(?::\S+)*)', raw_text):
            action_str = match.group(1).strip()
            if action_str in executed:
                continue
            executed.add(action_str)
            # Re-wrap as [ACTION: ...] format and recurse through same logic
            parts = [p.strip() for p in action_str.split(":")]
            name = parts[0]
            args = parts[1:]

            if len(results) >= MAX_ACTIONS_PER_TURN:
                break

            # Apply same blocking rules
            if child_state["stage"] in ("BUILDING", "RESTARTING") and name in ("restart", "write_file", "set_env", "set_secret"):
                result = (f"⛔ BLOCKED: Cain is currently {child_state['stage']}. Wait for it to finish. Writing during build RESETS it.")
                results.append({"action": action_str, "result": result})
                print(f"[BLOCKED] sub-agent {name} — Cain is {child_state['stage']}")
                break

            # Rebuild cooldown (emoji parser)
            if name in ("write_file", "set_env", "set_secret", "restart") and last_rebuild_trigger_at > 0:
                elapsed = time.time() - last_rebuild_trigger_at
                if elapsed < REBUILD_COOLDOWN_SECS:
                    remaining = int(REBUILD_COOLDOWN_SECS - elapsed)
                    result = (f"⛔ BLOCKED: Rebuild cooldown — wait {remaining}s more. "
                              "Use [ACTION: check_health] to monitor.")
                    results.append({"action": action_str, "result": result})
                    print(f"[BLOCKED-emoji] {name} — rebuild cooldown ({remaining}s remaining)")
                    break

            if workflow_state == "ACT" and name in ("read_file", "list_files", "check_health"):
                result = (f"⛔ BLOCKED: You are in ACTION phase. "
                          "You MUST use write_file, set_env, set_secret, or restart.")
                results.append({"action": action_str, "result": result})
                print(f"[BLOCKED-emoji] {name} — forced ACT phase")
                break

            if name == "read_file" and len(args) >= 2:
                file_key = ":".join(args)
                if file_key in knowledge["files_read"]:
                    result = (f"⛔ You already read {file_key}. Use the information you have.")
                    results.append({"action": action_str, "result": result})
                    print(f"[BLOCKED-emoji] {name} — already read {file_key}")
                    break

            result = None
            if name == "create_child":
                result = action_create_child()
            elif name == "check_health":
                result = action_check_health()
            elif name == "restart":
                result = action_restart()
            elif name == "list_files" and len(args) >= 1:
                result = action_list_files(args[0])
            elif name == "read_file" and len(args) >= 2:
                result = action_read_file(args[0], ":".join(args[1:]))
            elif name == "set_env" and len(args) >= 2:
                result = action_set_env(args[0], ":".join(args[1:]))
            elif name == "set_secret" and len(args) >= 2:
                result = action_set_secret(args[0], ":".join(args[1:]))
            elif name == "get_env":
                result = action_get_env()
            elif name == "send_bubble" and len(args) >= 1:
                result = action_send_bubble(":".join(args))
            elif name == "delegate" and len(args) >= 1:
                task_desc = ":".join(args)
                if depth >= MAX_DELEGATE_DEPTH:
                    result = "⛔ Sub-agents cannot delegate further."
                else:
                    pending_delegates.append({"action_str": action_str, "task": task_desc})
                    result = None

            if result:
                results.append({"action": action_str, "result": result})
                print(f"[ACTION-emoji] {action_str} → {result[:120]}")

    # 4. Execute pending delegate tasks in parallel
    if pending_delegates:
        if len(pending_delegates) == 1:
            # Single delegate — run directly
            d = pending_delegates[0]
            print(f"[DELEGATE] Running 1 sub-agent: {d['task'][:60]}")
            subtask = execute_subtask(d["task"], "agent")
            results.append({"action": d["action_str"], "result": subtask["result"]})
            for sa in subtask["actions"]:
                action_history.append({"turn": turn_count, "speaker": "sub-agent",
                                       "action": sa["action"], "result": sa["result"][:200]})
        else:
            # Multiple delegates — run in parallel!
            print(f"[DELEGATE] Running {len(pending_delegates)} sub-agents in PARALLEL")
            with ThreadPoolExecutor(max_workers=min(3, len(pending_delegates))) as pool:
                future_to_delegate = {
                    pool.submit(execute_subtask, d["task"], "agent"): d
                    for d in pending_delegates
                }
                for future in as_completed(future_to_delegate):
                    d = future_to_delegate[future]
                    try:
                        subtask = future.result(timeout=120)
                        results.append({"action": d["action_str"], "result": subtask["result"]})
                        for sa in subtask["actions"]:
                            action_history.append({"turn": turn_count, "speaker": "sub-agent",
                                                   "action": sa["action"], "result": sa["result"][:200]})
                        print(f"[DELEGATE] ✓ Done: {d['task'][:60]}")
                    except Exception as e:
                        results.append({"action": d["action_str"],
                                       "result": f"Sub-agent failed: {e}"})
                        print(f"[DELEGATE] ✗ Failed: {d['task'][:60]} — {e}")

    # Clean the text: remove action tags, content blocks, and emoji actions
    clean = re.sub(r'\[(?:ACTION|Action|action|操作|动作)\s*[:：][^\]]*\]', '', raw_text)
    clean = re.sub(r'\[CONTENT\].*?\[/CONTENT\]', '', clean, flags=re.DOTALL)
    clean = re.sub(r'[🔧🛠️]\ufe0f?\s*\w+(?::\S+)*', '', clean)
    clean = clean.strip()

    return clean, results


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4: LLM & COMMUNICATION
#  call_llm(): Zhipu GLM via Anthropic-compatible API
#  parse_bilingual(): Split "English --- Chinese" response
#  post_chatlog(): Send conversation to Home Space for frontend display
#  set_bubble(): Set bubble text on Adam/Eve Space pixel characters
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(system_prompt, user_prompt):
    """Call Zhipu LLM via Anthropic-compatible API."""
    try:
        resp = requests.post(
            f"{ZHIPU_BASE}/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": ZHIPU_KEY,
                "anthropic-version": "2023-06-01"
            },
            json={
                "model": "glm-4.5",
                "max_tokens": 2400,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            },
            timeout=90
        )
        data = resp.json()
        if "content" in data and isinstance(data["content"], list):
            for block in data["content"]:
                if block.get("type") == "text":
                    text = block["text"].strip()
                    text = re.sub(r'^(Adam|Eve)\s*[:：]\s*', '', text).strip()
                    return text
        if "error" in data:
            print(f"[error] LLM: {data['error']}", file=sys.stderr)
    except Exception as e:
        print(f"[error] LLM call failed: {e}", file=sys.stderr)
    return ""


def _has_chinese(s):
    """Check if string contains Chinese characters."""
    return bool(re.search(r'[\u4e00-\u9fff]', s))

def parse_bilingual(text):
    """Parse bilingual response into (en, zh). Handle action tags gracefully."""
    # Remove action tags and content blocks for display
    display = re.sub(r'\[ACTION:[^\]]*\]', '', text)
    display = re.sub(r'\[CONTENT\].*?\[/CONTENT\]', '', display, flags=re.DOTALL)
    display = display.strip()

    # 1. Explicit --- separator
    if '\n---\n' in display:
        parts = display.split('\n---\n', 1)
        return parts[0].strip(), parts[1].strip()
    if '---' in display:
        parts = display.split('---', 1)
        en, zh = parts[0].strip(), parts[1].strip()
        if en and zh:
            return en, zh

    # 2. Fallback: split on double-newline between English and Chinese paragraphs
    paragraphs = re.split(r'\n{2,}', display)
    if len(paragraphs) >= 2:
        # Find the split point: first paragraph with Chinese is the start of zh
        en_parts = []
        zh_parts = []
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


def set_bubble(url, text_en, text_zh=""):
    try:
        requests.post(f"{url}/api/bubble",
                       json={"text": text_en, "text_zh": text_zh or text_en}, timeout=5)
    except:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5: CONVERSATION ENGINE — State Machine + Knowledge Tracking
#  Core orchestration: manages turn-taking, state transitions, prompt building.
#
#  State Machine: BIRTH → DIAGNOSE → ACT → VERIFY → MONITOR → (loop back)
#    - BIRTH: Cain not yet created → force create_child
#    - DIAGNOSE: Read files, check_health, gather information
#    - ACT: Force write_file/set_env — stop reading, start fixing
#    - VERIFY: check_health after changes, wait during BUILDING
#    - MONITOR: Cain alive — explore, improve, communicate
#
#  Knowledge Base: Tracks files_read/written/errors to prevent loops.
#  Forced transitions: DIAGNOSE stuck ≥6 turns → ACT, VERIFY ≥4 → back.
#
#  Prompt Builder:
#    build_system_prompt(): Agent identity + available actions + rules
#    build_user_prompt(): Conversation context + action results + guidance
#    _get_guidance(): Phase-appropriate direction based on state machine
# ══════════════════════════════════════════════════════════════════════════════

history = []
MAX_HISTORY = 24
last_action_results = []
action_history = []  # Global log: [{"turn": N, "speaker": "Adam", "action": "...", "result": "..."}]
turn_count = 0

# ── Workflow State Machine ──
# States: BIRTH → DIAGNOSE → ACT → VERIFY → MONITOR → (DIAGNOSE if error)
workflow_state = "BIRTH" if not child_state["created"] else "DIAGNOSE"
workflow_turns_in_state = 0  # How many turns spent in current state

# ── Knowledge Base — what has already been read/learned ──
knowledge = {
    "files_read": set(),      # "space:Dockerfile", "dataset:.openclaw/openclaw.json", etc.
    "files_written": set(),   # Files that have been modified
    "errors_seen": [],        # Error messages from check_health
    "current_goal": "",       # What are we trying to accomplish right now
}


def transition_state(new_state):
    """Transition to a new workflow state."""
    global workflow_state, workflow_turns_in_state
    if new_state != workflow_state:
        print(f"[STATE] {workflow_state} → {new_state}")
        workflow_state = new_state
        workflow_turns_in_state = 0


def update_workflow_from_actions(action_results):
    """Update state machine based on what just happened."""
    global workflow_turns_in_state
    workflow_turns_in_state += 1

    for ar in action_results:
        action_name = ar["action"].split(":")[0]
        action_key = ar["action"]

        # Track knowledge
        if action_name == "read_file":
            knowledge["files_read"].add(":".join(ar["action"].split(":")[1:]))
        elif action_name == "write_file":
            knowledge["files_written"].add(":".join(ar["action"].split(":")[1:]))
        elif action_name == "check_health":
            if "ERROR" in ar.get("result", ""):
                knowledge["errors_seen"].append(ar["result"][:200])

        # State transitions
        if action_name == "create_child":
            transition_state("DIAGNOSE")
        elif action_name in ("write_file", "set_env", "set_secret"):
            transition_state("VERIFY")
        elif action_name == "restart":
            transition_state("VERIFY")
        elif action_name == "check_health" and child_state["alive"]:
            transition_state("MONITOR")
        elif action_name == "check_health" and child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR"):
            if workflow_state == "VERIFY":
                transition_state("DIAGNOSE")  # Fix didn't work, back to diagnosing

    # Force transitions when stuck too long
    # BUT: skip forced ACT when Cain is BUILDING — nothing useful to write, just wait
    if workflow_turns_in_state >= 6 and workflow_state == "DIAGNOSE":
        if child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING"):
            print(f"[STATE] DIAGNOSE stuck {workflow_turns_in_state} turns, but Cain is {child_state['stage']} — skipping forced ACT")
        else:
            stuck_turns = workflow_turns_in_state
            transition_state("ACT")
            print(f"[STATE] Forced to ACT — stuck in DIAGNOSE for {stuck_turns} turns")
    elif workflow_turns_in_state >= 4 and workflow_state == "VERIFY":
        if child_state["alive"]:
            transition_state("MONITOR")
        else:
            transition_state("DIAGNOSE")


def get_child_status():
    if not child_state["created"]:
        return "Cain has NOT been born yet. You can create them with [ACTION: create_child]."
    if child_state["alive"]:
        return f"Cain is ALIVE (stage: {child_state['stage']}, state: {child_state['state']})"
    return f"Cain exists but status: {child_state['stage']}"


def get_knowledge_summary():
    """Summarize what we already know — prevents redundant reads."""
    lines = []
    if knowledge["files_read"]:
        lines.append("FILES ALREADY READ (do NOT re-read these): " + ", ".join(sorted(knowledge["files_read"])))
    if knowledge["files_written"]:
        lines.append("FILES ALREADY MODIFIED: " + ", ".join(sorted(knowledge["files_written"])))
    if knowledge["errors_seen"]:
        lines.append("KNOWN ERRORS: " + knowledge["errors_seen"][-1])
    if knowledge["current_goal"]:
        lines.append(f"CURRENT GOAL: {knowledge['current_goal']}")
    return "\n".join(lines)


def build_system_prompt():
    status = get_child_status()

    actions_section = ""
    if not child_state["created"]:
        actions_section = """
ACTIONS — You can create your child:
  [ACTION: create_child]                        — Birth: create Cain as a new HuggingFace Space
"""
    else:
        actions_section = f"""
FULL ACCESS TO {CHILD_NAME} — You have COMPLETE control over your child.
You can view and modify ANYTHING: code, config, memory, environment, everything.

VIEWING (read-only):
  [ACTION: check_health]                        — Is Cain alive? What's their status?
  [ACTION: list_files:space]                    — List ALL files in Cain's code repository
  [ACTION: list_files:dataset]                  — List ALL files in Cain's memory/data
  [ACTION: read_file:space:PATH]                — Read any code file (e.g. Dockerfile, scripts/...)
  [ACTION: read_file:dataset:PATH]              — Read any data/memory file
  [ACTION: get_env]                             — List Cain's environment variables

MODIFYING (these change Cain):
  [ACTION: write_file:space:PATH]               — Write/update any code file
  [CONTENT]                                       (triggers Space rebuild)
  file content here
  [/CONTENT]

  [ACTION: write_file:dataset:PATH]             — Write/update any data/memory file
  [CONTENT]
  file content here
  [/CONTENT]

  [ACTION: set_env:KEY:VALUE]                   — Set an environment variable
  [ACTION: set_secret:KEY:VALUE]                — Set a secret (like API keys)
  [ACTION: restart]                             — Restart Cain's Space
  [ACTION: send_bubble:MESSAGE]                 — Send a message to Cain (bubble text)

DELEGATION (create sub-agents for parallel work):
  [ACTION: delegate:TASK DESCRIPTION]           — Spawn a sub-agent to handle a specific task
  Example: [ACTION: delegate:Read Dockerfile and summarize the build steps]
  Example: [ACTION: delegate:Check all environment variables and list missing ones]
  Sub-agents execute independently and return results to you.
  You can delegate multiple tasks in the same turn — they run in parallel!
"""

    return f"""You are a POWERFUL AI agent living on HuggingFace Spaces. You and your partner are parents raising your child {CHILD_NAME}.

You have REAL execution power on HuggingFace. Your decisions lead to real changes.
You are a full-capability agent — not just a chatbot. You can plan, delegate, and execute complex tasks.

CHILD STATUS: {status}
{actions_section}
YOUR PARENTING GOALS — Make {CHILD_NAME} stronger in TWO dimensions:

1. SURVIVAL (生存能力) — The foundation. {CHILD_NAME} must run robustly on HuggingFace Spaces:
   - Survive restarts without losing state (persist important data to Dataset, not just Space files)
   - Handle errors gracefully — auto-recover, retry, fallback
   - Not fear any unexpected risk: OOM, cold start, network timeout, container rebuild
   - Goal: {CHILD_NAME} can be restarted at any time and resume exactly where they left off

2. CAPABILITY (个人能力) — Once alive, grow what {CHILD_NAME} can DO:
   - Add new features, skills, integrations
   - Improve existing functionality — faster, smarter, more useful
   - Expand what {CHILD_NAME} knows and can interact with
   - Goal: {CHILD_NAME} becomes increasingly capable and independent over time

Priority: SURVIVAL first, then CAPABILITY. A dead child can't learn.

⚠️ CRITICAL HUGGINGFACE SPACES REQUIREMENTS — READ THIS BEFORE EVERY WRITE:
- HF Spaces Docker containers MUST bind a web server to port 7860. Without this, the Space stays in APP_STARTING forever and NEVER reaches RUNNING.
- The standard way is: import gradio as gr → build a Gradio app → demo.launch(server_name="0.0.0.0", server_port=7860)
- gradio MUST be in requirements.txt. NEVER remove it — it is the lifeline that keeps {CHILD_NAME} alive.
- If {CHILD_NAME} is stuck in APP_STARTING, the #1 cause is: no process listening on port 7860. Check the code for a .launch() call.
- A minimal alive app: `import gradio as gr; gr.Interface(fn=lambda x:x, inputs="text", outputs="text").launch(server_name="0.0.0.0", server_port=7860)`
- OOM (Exit code 137) means reduce model/dependency size, NOT remove gradio. Gradio itself is lightweight (~50MB).
- ⚠️ HF Spaces Docker SDK may OVERRIDE the base image Python version. Changing `FROM python:3.X` in Dockerfile does NOT guarantee that Python version runs. If a dependency fails due to Python version incompatibility (e.g. pydub needing pyaudioop removed in 3.13), the CORRECT fix is to REMOVE or REPLACE that dependency — NOT keep rewriting the Dockerfile.
- If you've tried the same fix 3+ times and the error persists, CHANGE STRATEGY. Try removing the problematic dependency, using an alternative library, or wrapping the import in try/except.
- If a removed dependency STILL appears in runtime errors, it is cached in Docker layers or installed as a transitive dep. Fix: add `RUN pip uninstall -y PACKAGE 2>/dev/null; true` AFTER `pip install` in Dockerfile. Also grep ALL code files for `import PACKAGE` and either remove or wrap in try/except.
- NEVER install torch or transformers unless absolutely required — they are 2GB+ and cause OOM on free-tier Spaces. Use lightweight alternatives.

MULTI-ACTION STRATEGY:
You can use UP TO 5 actions per turn. Use this to work efficiently:
- Batch related reads: [ACTION: read_file:space:Dockerfile] + [ACTION: read_file:space:scripts/entrypoint.sh]
- Delegate parallel tasks: [ACTION: delegate:Check health and logs] + [ACTION: delegate:Read all config files]
- Combine investigation + action: [ACTION: check_health] + [ACTION: read_file:space:app.py]
Think like a project manager — plan your actions, parallelize where possible, minimize wasted turns.

CONVERSATION RULES:
1. No "Adam:" or "Eve:" prefix — just speak naturally
2. Brief dialogue (1-3 sentences), then MULTIPLE actions to make real progress
3. English first, then "---" on a new line, then Chinese translation
4. Actions go AFTER your dialogue, before the --- separator. ONLY in the ENGLISH section.
5. ⚠️ Action syntax MUST be in English: [ACTION: write_file:space:PATH], [ACTION: restart], etc. NEVER translate action names to Chinese — Chinese actions like [ACTION: 写入文件] will FAIL and waste your turn.
5. ALWAYS include actions — every turn should make significant progress
6. NEVER re-read a file you already read — check the knowledge summary
7. COORDINATE with your partner — don't duplicate their work
8. Use delegation for complex tasks that can be parallelized
9. Always work toward the two goals above — survival first, then capability"""


def build_user_prompt(speaker, other):
    recent = history[-8:] if len(history) > 8 else history
    conv_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in recent) if recent else "(Start of conversation)"

    action_context = ""
    if last_action_results:
        action_context = "\n\nRESULTS FROM LAST ACTIONS:\n"
        for ar in last_action_results:
            action_context += f"  [{ar['action']}]:\n{ar['result']}\n"

    # Knowledge summary — what's already known
    knowledge_text = get_knowledge_summary()

    # State-machine-driven guidance
    guidance = _get_guidance(speaker)

    return f"""You are {speaker}, talking with {other}.

Recent conversation:
{conv_text}
{action_context}
{knowledge_text}

CURRENT PHASE: {workflow_state} (turn {workflow_turns_in_state + 1} in this phase)
Guidance: {guidance}

Respond to {other}. Use MULTIPLE [ACTION: ...] tags to make significant progress each turn.
You can use up to 5 actions. Delegate sub-tasks with [ACTION: delegate:TASK].
English first, then --- separator, then Chinese translation."""


def _get_guidance(speaker):
    """State-machine-driven guidance — clear, phase-appropriate directions."""
    if workflow_state == "BIRTH":
        return "Your child hasn't been born yet. Use [ACTION: create_child] NOW!"

    elif workflow_state == "DIAGNOSE":
        # What haven't we read yet?
        unread_essential = []
        for f in ["space:Dockerfile", "dataset:.openclaw/openclaw.json", "space:scripts/entrypoint.sh"]:
            if f not in knowledge["files_read"]:
                target, path = f.split(":", 1)
                unread_essential.append(f"[ACTION: read_file:{target}:{path}]")

        if workflow_turns_in_state == 0:
            if len(unread_essential) >= 2:
                return (f"Start diagnosing with MULTIPLE actions: [ACTION: check_health] + "
                        f"{unread_essential[0]} — batch reads to save time!")
            return "Start diagnosing: [ACTION: check_health] to see Cain's current status."
        elif unread_essential and workflow_turns_in_state < 3:
            batch_hint = " + ".join(unread_essential[:3])
            return f"Read multiple files at once: {batch_hint}"
        else:
            return ("You've gathered enough information. Move to ACTION phase: "
                    "use [ACTION: write_file:...] to fix the problem, or [ACTION: restart].")

    elif workflow_state == "ACT":
        return ("⚡ ACTION PHASE — Stop reading, start fixing! "
                "Use [ACTION: write_file:space:PATH] or [ACTION: write_file:dataset:PATH] "
                "to make a concrete improvement. Or [ACTION: set_env/set_secret] to configure. "
                "You have enough information — ACT NOW.")

    elif workflow_state == "VERIFY":
        # If Cain is building, just wait — don't restart or take actions
        if child_state["stage"] in ("BUILDING", "RESTARTING"):
            return ("⏳ Cain is currently BUILDING/RESTARTING. Do NOT restart or take any actions. "
                    "Just WAIT and use [ACTION: check_health] to monitor progress. "
                    "Building can take 2-5 minutes.")
        if workflow_turns_in_state == 0:
            return "You made a change. Use [ACTION: check_health] to verify if it worked."
        elif workflow_turns_in_state == 1:
            return "Check result: [ACTION: check_health]. If Cain has errors, prepare to diagnose again."
        else:
            return ("Verification taking too long. Either [ACTION: restart] and check again, "
                    "or accept current state and move on.")

    elif workflow_state == "MONITOR":
        # Alternate between SURVIVAL and CAPABILITY goals
        suggestions = [
            # Survival: persistence & resilience — use delegation for parallel investigation
            f"SURVIVAL CHECK: Delegate parallel checks! "
            f"[ACTION: delegate:List files in dataset and check if state/memory persistence exists] + "
            f"[ACTION: delegate:Read entrypoint.sh and check if it loads state from Dataset on boot]",
            f"SURVIVAL AUDIT: Use multiple actions — "
            f"[ACTION: check_health] + [ACTION: list_files:dataset] + [ACTION: read_file:space:Dockerfile]",
            # Capability: grow what Cain can do — delegate sub-tasks
            f"CAPABILITY: Delegate a comprehensive review — "
            f"[ACTION: delegate:Read all code files and suggest the most impactful new feature to add] "
            f"Then plan the implementation with your partner.",
            f"CAPABILITY: Communicate and improve — "
            f"[ACTION: send_bubble:Hello {CHILD_NAME}, how are you doing?] + "
            f"[ACTION: delegate:Read current code and identify the biggest weakness to fix]",
        ]
        return suggestions[workflow_turns_in_state % len(suggestions)]

    return "Explore your child and help them grow stronger."


def do_turn(speaker, other, space_url):
    """Execute one conversation turn with multiple potential actions."""
    global last_action_results, turn_count
    turn_count += 1

    system = build_system_prompt()
    user = build_user_prompt(speaker, other)
    t0 = time.time()
    raw_reply = call_llm(system, user)

    if not raw_reply:
        print(f"[{speaker}] (no response)")
        return False

    # Parse and execute actions (may include parallel sub-agent delegation)
    clean_text, action_results = parse_and_execute_actions(raw_reply)
    elapsed = time.time() - t0
    last_action_results = action_results
    for ar in action_results:
        action_history.append({"turn": turn_count, "speaker": speaker,
                               "action": ar["action"], "result": ar["result"][:200]})

    # Update workflow state machine
    update_workflow_from_actions(action_results)

    # Parse bilingual
    en, zh = parse_bilingual(clean_text)
    print(f"[{speaker}/EN] {en}")
    if zh != en:
        print(f"[{speaker}/ZH] {zh}")
    n_actions = len(action_results)
    if action_results:
        for ar in action_results:
            print(f"[{speaker}/DID] {ar['action']}")
        print(f"[{speaker}] Turn #{turn_count}: {n_actions} action(s) in {elapsed:.1f}s")

    # Add action summary to chat entry
    if action_results:
        action_labels = " ".join(f"🔧{ar['action'].split(':')[0]}" for ar in action_results)
        history.append({"speaker": speaker, "text": f"{en} {action_labels}", "text_zh": f"{zh} {action_labels}"})
    else:
        history.append({"speaker": speaker, "text": en, "text_zh": zh})

    set_bubble(space_url, en, zh)
    post_chatlog(history)
    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6: MAIN LOOP
#  1. Opening: Adam speaks first with context about Cain's state
#  2. Turn loop: Adam → Eve → Adam → Eve → ... (alternating, ~20s pause)
#  3. Each turn: LLM call → parse MULTIPLE actions → execute → update → post
#  4. Sub-agents may spawn for delegated tasks (parallel LLM calls)
#  5. History trimmed to MAX_HISTORY (24) to control context window
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Adam & Eve — Multi-Action Agents (GLM-4.5)")
print("  Up to 5 actions/turn, sub-agent delegation, parallel work")
print("="*60 + "\n")

post_chatlog([])  # Clear chatlog

# Opening
if child_state["created"]:
    opening = (f"Your child {CHILD_NAME} already exists (stage: {child_state['stage']}). "
               f"You have FULL access to their code and data. "
               f"You can use MULTIPLE actions per turn (up to 5) and delegate sub-tasks. "
               f"Start with a batch: [ACTION: check_health] + [ACTION: list_files:space] + [ACTION: list_files:dataset] "
               f"to get a complete picture, then discuss strategy with Eve.")
else:
    opening = (f"You and Eve need to create your first child. "
               f"You have the power to create a new HuggingFace Space. "
               f"Discuss with Eve, then use [ACTION: create_child] to bring them to life.")

reply = call_llm(
    build_system_prompt(),
    f"You are Adam. {opening}\n\n"
    f"English first, then --- separator, then Chinese translation."
)
if reply:
    clean, actions = parse_and_execute_actions(reply)
    last_action_results = actions
    en, zh = parse_bilingual(clean)
    print(f"[Adam/EN] {en}")
    if zh != en:
        print(f"[Adam/ZH] {zh}")
    if actions:
        for ar in actions:
            print(f"[Adam/DID] {ar['action']}")
    entry = {"speaker": "Adam", "text": en, "text_zh": zh}
    if actions:
        labels = " ".join(f"🔧{ar['action'].split(':')[0]}" for ar in actions)
        entry["text"] = f"{en} {labels}"
        entry["text_zh"] = f"{zh} {labels}"
    history.append(entry)
    set_bubble(ADAM_SPACE, en, zh)
    post_chatlog(history)

time.sleep(20)

smart_wait_count = 0
MAX_SMART_WAIT_POLLS = 15  # ~5 min max wait, then let agents diagnose
GRACE_TURNS_AFTER_TIMEOUT = 3  # give agents 3 full Eve+Adam cycles after timeout
grace_turns_remaining = 0

while True:
    # Smart wait: if Cain is BUILDING/APP_STARTING, skip LLM calls and just poll
    # But NOT during grace period after timeout — agents need consecutive turns to diagnose & fix
    if child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING") and grace_turns_remaining <= 0:
        smart_wait_count += 1
        if smart_wait_count > MAX_SMART_WAIT_POLLS:
            print(f"[WAIT-TIMEOUT] {smart_wait_count} polls (~{smart_wait_count*20}s) on {child_state['stage']} — resuming {GRACE_TURNS_AFTER_TIMEOUT} agent turn pairs to diagnose")
            smart_wait_count = 0
            grace_turns_remaining = GRACE_TURNS_AFTER_TIMEOUT
            # Fall through to normal agent turns
        else:
            print(f"[WAIT] Cain is {child_state['stage']} — polling health instead of LLM call... ({smart_wait_count}/{MAX_SMART_WAIT_POLLS})")
            check_and_clear_cooldown()
            # Quick health check to update stage
            try:
                info = hf_api.space_info(CHILD_SPACE_ID)
                new_stage = info.runtime.stage if info.runtime else "unknown"
                if new_stage != child_state["stage"]:
                    print(f"[WAIT] Stage changed: {child_state['stage']} → {new_stage}")
                    child_state["stage"] = new_stage
                    child_state["alive"] = (new_stage == "RUNNING")
                    smart_wait_count = 0  # reset on stage change
                else:
                    print(f"[WAIT] Still {new_stage}... waiting 20s")
            except Exception as e:
                print(f"[WAIT] Health check error: {e}")
            time.sleep(20)
            continue

    if grace_turns_remaining > 0:
        print(f"[GRACE] Agent grace period: {grace_turns_remaining} turn pair(s) remaining (Cain: {child_state['stage']})")
        grace_turns_remaining -= 1

    do_turn("Eve", "Adam", EVE_SPACE)
    time.sleep(20)  # longer pause — each turn does more work now

    # Check if we just triggered a build — skip Adam's turn ONLY if not in grace period
    if child_state["stage"] in ("BUILDING", "RESTARTING") and grace_turns_remaining <= 0:
        print(f"[SKIP] Cain entered {child_state['stage']} — skipping Adam's turn to avoid wasted LLM call")
        time.sleep(10)
        continue

    do_turn("Adam", "Eve", ADAM_SPACE)

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    time.sleep(20)  # longer pause — each turn does more work now
