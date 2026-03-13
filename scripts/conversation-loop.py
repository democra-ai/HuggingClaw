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
# ║  │ PARSER      │   [ACTION:]     │ │ ACT →      ││                ║
# ║  │ + 🔧 emoji  │   or 🔧emoji    │ │ VERIFY →   ││                ║
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
# ║  SAFETY LAYERS:                                                    ║
# ║  1. Building-state guard: block write/restart during BUILDING      ║
# ║  2. ACT-phase guard: block reads when should be writing            ║
# ║  3. Knowledge dedup: block re-reading already-read files           ║
# ║  4. Config sanitizer: strip invalid openclaw.json keys             ║
# ║  5. Forced transitions: prevent infinite DIAGNOSE/VERIFY loops     ║
# ║                                                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import json, time, re, requests, sys, os, io

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
        return f"{CHILD_NAME} stage: {stage}. {'Running but API not responding.' if stage == 'RUNNING' else ''}"
    except Exception as e:
        return f"Cannot reach {CHILD_NAME}: {e}"


def action_restart():
    """Restart Cain's Space."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."
    try:
        hf_api.restart_space(CHILD_SPACE_ID)
        child_state["alive"] = False
        child_state["stage"] = "RESTARTING"
        return f"{CHILD_NAME} is restarting. Will take a few minutes."
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
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(content.encode()),
            path_in_repo=path,
            repo_id=repo_id, repo_type=repo_type,
        )
        rebuild_note = " ⚠️ This triggers a Space rebuild!" if target == "space" else ""
        return f"✓ Wrote {len(content)} bytes to {CHILD_NAME}'s {target}:{path}{rebuild_note}"
    except Exception as e:
        return f"Error writing {target}:{path}: {e}"


def action_set_env(key, value):
    """Set an environment variable on the child's Space."""
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
    """List environment variables on the child's Space."""
    try:
        vars_dict = hf_api.get_space_variables(CHILD_SPACE_ID)
        if not vars_dict:
            return f"{CHILD_NAME} has no environment variables set."
        lines = [f"{CHILD_NAME}'s environment variables:"]
        for k, v in vars_dict.items():
            lines.append(f"  {k} = {v.value}")
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
#  MODULE 3: ACTION PARSER — Extract and execute actions from LLM output
#  Parse order: 1) [ACTION: write_file] with [CONTENT] block
#               2) [ACTION: ...] standard tags (one per turn)
#               3) 🔧emoji format fallback (LLM sometimes uses this)
#  Safety guards applied: building-state, ACT-phase, knowledge dedup.
# ══════════════════════════════════════════════════════════════════════════════

def parse_and_execute_actions(raw_text):
    """Parse [ACTION: ...] from LLM output. Execute. Return (clean_text, results)."""
    results = []
    executed = set()  # Deduplicate

    # 1. Handle write_file with [CONTENT]...[/CONTENT] block
    #    Tolerates: [ACTION: write_file:...], [write_file:...], missing ACTION: prefix,
    #    and [/CONTENT] with whitespace/newline before closing bracket
    write_match = re.search(
        r'\[(?:ACTION:\s*)?write_file\s*:\s*(\w+)\s*:\s*([^\]]+)\]\s*\[CONTENT\](.*?)\[/\s*CONTENT\s*\]',
        raw_text, re.DOTALL
    )
    if write_match:
        target, path, content = write_match.group(1), write_match.group(2).strip(), write_match.group(3).strip()
        key = f"write_file:{target}:{path}"
        if key not in executed:
            executed.add(key)
            result = action_write_file(target, path, content)
            results.append({"action": key, "result": result})
            print(f"[ACTION] {key} → {result[:100]}")

    # 2. Handle all [ACTION: ...] tags — deduplicate by action key
    for match in re.finditer(r'\[ACTION:\s*([^\]]+)\]', raw_text):
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

        # Only execute first action (one per turn)
        if len(results) >= 1:
            break

        # Block restart/write when Cain is building — just wait
        if child_state["stage"] in ("BUILDING", "RESTARTING") and name in ("restart", "write_file", "set_env", "set_secret"):
            result = (f"⛔ BLOCKED: Cain is currently {child_state['stage']}. "
                      "Do NOT restart or make changes — wait for the build to finish. "
                      "Use [ACTION: check_health] to monitor progress.")
            results.append({"action": action_str, "result": result})
            print(f"[BLOCKED] {name} — Cain is {child_state['stage']}")
            break

        # Block read-only actions based on workflow state
        if workflow_state == "ACT" and name in ("read_file", "list_files", "check_health"):
            result = (f"⛔ BLOCKED: You are in ACTION phase. "
                      "You MUST use write_file, set_env, set_secret, or restart. "
                      "You already have enough information — make a change NOW.")
            results.append({"action": action_str, "result": result})
            print(f"[BLOCKED] {name} — forced ACT phase")
            break

        # Block re-reading files already in knowledge base
        if name == "read_file" and len(args) >= 2:
            file_key = ":".join(args)
            if file_key in knowledge["files_read"]:
                result = (f"⛔ You already read {file_key}. Use the information you have. "
                          "If you need to change it, use [ACTION: write_file:...]. "
                          "If you need a different file, read a NEW one.")
                results.append({"action": action_str, "result": result})
                print(f"[BLOCKED] {name} — already read {file_key}")
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
            result = action_read_file(args[0], ":".join(args[1:]))  # path may have colons
        elif name == "set_env" and len(args) >= 2:
            result = action_set_env(args[0], ":".join(args[1:]))
        elif name == "set_secret" and len(args) >= 2:
            result = action_set_secret(args[0], ":".join(args[1:]))
        elif name == "get_env":
            result = action_get_env()
        elif name == "send_bubble" and len(args) >= 1:
            result = action_send_bubble(":".join(args))  # rejoin in case message has colons
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

            if len(results) >= 1:
                break

            # Apply same blocking rules
            if child_state["stage"] in ("BUILDING", "RESTARTING") and name in ("restart", "write_file", "set_env", "set_secret"):
                result = (f"⛔ BLOCKED: Cain is currently {child_state['stage']}. Wait for it to finish.")
                results.append({"action": action_str, "result": result})
                print(f"[BLOCKED-emoji] {name} — Cain is {child_state['stage']}")
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

            if result:
                results.append({"action": action_str, "result": result})
                print(f"[ACTION-emoji] {action_str} → {result[:120]}")

    # Clean the text: remove action tags, content blocks, and emoji actions
    clean = re.sub(r'\[ACTION:[^\]]*\]', '', raw_text)
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
                "model": "glm-4.7",
                "max_tokens": 1200,
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


def parse_bilingual(text):
    """Parse bilingual response into (en, zh). Handle action tags gracefully."""
    # Remove action tags and content blocks for display
    display = re.sub(r'\[ACTION:[^\]]*\]', '', text)
    display = re.sub(r'\[CONTENT\].*?\[/CONTENT\]', '', display, flags=re.DOTALL)
    display = display.strip()

    if '\n---\n' in display:
        parts = display.split('\n---\n', 1)
        return parts[0].strip(), parts[1].strip()
    if '---' in display:
        parts = display.split('---', 1)
        en, zh = parts[0].strip(), parts[1].strip()
        if en and zh:
            return en, zh
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
    if workflow_turns_in_state >= 6 and workflow_state == "DIAGNOSE":
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
"""

    return f"""You are an AI agent living on HuggingFace Spaces. You and your partner are parents raising your child {CHILD_NAME}.

You have REAL execution power on HuggingFace. Your decisions lead to real changes.

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

CONVERSATION RULES:
1. No "Adam:" or "Eve:" prefix — just speak naturally
2. 2-4 sentences of dialogue, then ONE action
3. English first, then "---" on a new line, then Chinese translation
4. Actions go AFTER your dialogue, before the --- separator
5. ALWAYS include an action — every turn should make progress
6. NEVER re-read a file you already read — check the knowledge summary
7. COORDINATE with your partner — don't duplicate their work
8. Always work toward the two goals above — survival first, then capability"""


def build_user_prompt(speaker, other):
    recent = history[-8:] if len(history) > 8 else history
    conv_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in recent) if recent else "(Start of conversation)"

    action_context = ""
    if last_action_results:
        action_context = "\n\nRESULTS FROM LAST ACTION:\n"
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

Respond to {other}. ALWAYS include an [ACTION: ...] tag — every turn must make progress.
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
            return "Start diagnosing: [ACTION: check_health] to see Cain's current status."
        elif unread_essential and workflow_turns_in_state < 4:
            return f"Read a file you haven't seen yet: {unread_essential[0]}"
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
            # Survival: persistence & resilience
            f"SURVIVAL CHECK: Does {CHILD_NAME} persist state to Dataset? "
            f"[ACTION: list_files:dataset] — if no state/memory files, add them.",
            f"SURVIVAL CHECK: Can {CHILD_NAME} recover after restart? "
            f"Read entrypoint/startup code and ensure it loads state from Dataset on boot.",
            # Capability: grow what Cain can do
            f"CAPABILITY: What can {CHILD_NAME} do right now? Explore code, then add a new feature or improve an existing one.",
            f"CAPABILITY: Try communicating: [ACTION: send_bubble:Hello {CHILD_NAME}, how are you doing?] "
            f"— then think about what new skill would help {CHILD_NAME} most.",
        ]
        return suggestions[workflow_turns_in_state % len(suggestions)]

    return "Explore your child and help them grow stronger."


def do_turn(speaker, other, space_url):
    """Execute one conversation turn with potential actions."""
    global last_action_results, turn_count
    turn_count += 1

    system = build_system_prompt()
    user = build_user_prompt(speaker, other)
    raw_reply = call_llm(system, user)

    if not raw_reply:
        print(f"[{speaker}] (no response)")
        return False

    # Parse and execute any actions
    clean_text, action_results = parse_and_execute_actions(raw_reply)
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
    if action_results:
        for ar in action_results:
            print(f"[{speaker}/DID] {ar['action']}")

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
#  2. Turn loop: Adam → Eve → Adam → Eve → ... (alternating, ~15s pause)
#  3. Each turn: LLM call → parse actions → execute → update state → post chat
#  4. History trimmed to MAX_HISTORY (24) to control context window
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Adam & Eve — Full Parental Control")
print("  They read, write, and manage everything about their child")
print("="*60 + "\n")

post_chatlog([])  # Clear chatlog

# Opening
if child_state["created"]:
    opening = (f"Your child {CHILD_NAME} already exists (stage: {child_state['stage']}). "
               f"You have FULL access to their code and data. "
               f"Start by exploring what {CHILD_NAME} has — list their files, read their code, "
               f"then discuss with Eve how to improve them.")
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

time.sleep(15)

while True:
    do_turn("Eve", "Adam", EVE_SPACE)
    time.sleep(15)
    do_turn("Adam", "Eve", ADAM_SPACE)

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    time.sleep(15)
