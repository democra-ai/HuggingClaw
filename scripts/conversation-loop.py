#!/usr/bin/env python3 -u
"""
Adam & Eve — Claude Code Orchestrators for their child Cain.

Architecture: Adam/Eve (Zhipu GLM) gather context and craft task prompts,
then delegate ALL coding work to Claude Code CLI.

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    SYSTEM ARCHITECTURE (v2)                        ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║                                                                    ║
# ║  ┌─────────────┐    discuss     ┌────────────────┐                ║
# ║  │  Zhipu GLM  │ ◄───────────► │ Adam & Eve     │                ║
# ║  │  (glm-4.5)  │  understand   │ (context +     │                ║
# ║  └─────────────┘  situation    │  task prompt)  │                ║
# ║                                 └───────┬────────┘                ║
# ║                                         │ [TASK]                  ║
# ║                                         ▼                         ║
# ║                                 ┌────────────────┐                ║
# ║  ┌─────────────┐               │ Claude Code    │                ║
# ║  │ HuggingFace │ ◄──git push── │ CLI            │                ║
# ║  │ Cain Space  │               │ (z.ai backend) │                ║
# ║  └─────────────┘               └────────────────┘                ║
# ║                                                                    ║
# ║  Flow per turn:                                                    ║
# ║  1. Auto gather_context() — health, env, errors, files            ║
# ║  2. GLM discusses situation with partner (2-3 sentences)           ║
# ║  3. GLM outputs [TASK]...[/TASK] for Claude Code                  ║
# ║  4. Claude Code clones repo, analyzes, fixes, pushes              ║
# ║  5. Results fed back for next turn                                 ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import json, time, re, requests, sys, os, io, subprocess

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
REBUILD_COOLDOWN_SECS = 360  # 6 minutes
last_rebuild_trigger_at = 0
_pending_cooldown = False

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
    """Check Cain's health with detailed error info."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."
    try:
        resp = requests.get(f"{CHILD_SPACE_URL}/api/state", timeout=10)
        if resp.ok:
            data = resp.json()
            child_state["alive"] = True
            child_state["state"] = data.get("state", "unknown")
            child_state["detail"] = data.get("detail", "")
            child_state["stage"] = "RUNNING"
            return f"{CHILD_NAME} is ALIVE! State: {child_state['state']}, Detail: {child_state['detail'] or 'healthy'}"
    except:
        pass
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        stage = info.runtime.stage if info.runtime else "NO_RUNTIME"
        child_state["stage"] = stage
        child_state["alive"] = (stage == "RUNNING")
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


# ── Claude Code Action (THE STAR) ─────────────────────────────────────────────

CLAUDE_WORK_DIR = "/tmp/claude-workspace"
CLAUDE_TIMEOUT = 300  # 5 minutes

def action_claude_code(task):
    """Run Claude Code CLI to autonomously complete a coding task on Cain's Space."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."

    global _pending_cooldown
    repo_url = f"https://user:{HF_TOKEN}@huggingface.co/spaces/{CHILD_SPACE_ID}"

    # 1. Clone / reset to latest
    try:
        if os.path.exists(f"{CLAUDE_WORK_DIR}/.git"):
            try:
                subprocess.run(
                    "git fetch origin && git reset --hard origin/main",
                    shell=True, cwd=CLAUDE_WORK_DIR, timeout=30,
                    capture_output=True, check=True
                )
            except Exception:
                subprocess.run(f"rm -rf {CLAUDE_WORK_DIR}", shell=True, capture_output=True)
                subprocess.run(
                    f"git clone --depth 20 {repo_url} {CLAUDE_WORK_DIR}",
                    shell=True, timeout=60, capture_output=True, check=True
                )
        else:
            if os.path.exists(CLAUDE_WORK_DIR):
                subprocess.run(f"rm -rf {CLAUDE_WORK_DIR}", shell=True, capture_output=True)
            subprocess.run(
                f"git clone --depth 20 {repo_url} {CLAUDE_WORK_DIR}",
                shell=True, timeout=60, capture_output=True, check=True
            )
        subprocess.run('git config user.name "Claude Code"',
                       shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True)
        subprocess.run('git config user.email "claude-code@huggingclaw"',
                       shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True)
    except Exception as e:
        return f"Failed to prepare workspace: {e}"

    # 2. Run Claude Code with z.ai backend (Zhipu GLM)
    env = os.environ.copy()
    env.update({
        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        "ANTHROPIC_AUTH_TOKEN": ZHIPU_KEY,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "GLM-4.7",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM-4.7",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GLM-4.5-Air",
        "CI": "true",
    })

    print(f"[CLAUDE-CODE] Running: {task[:200]}...")
    try:
        proc = subprocess.Popen(
            ["claude", "-p", task, "--output-format", "text"],
            cwd=CLAUDE_WORK_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        deadline = time.time() + CLAUDE_TIMEOUT
        for line in proc.stdout:
            line = line.rstrip('\n')
            print(f"  [CC] {line}")
            output_lines.append(line)
            if time.time() > deadline:
                proc.kill()
                output_lines.append("(killed: timeout)")
                break
        proc.wait(timeout=10)
        output = '\n'.join(output_lines)
        if not output.strip():
            output = "(no output)"
    except FileNotFoundError:
        return "Claude Code CLI not found. Is @anthropic-ai/claude-code installed?"
    except Exception as e:
        return f"Claude Code failed: {e}"
    print(f"[CLAUDE-CODE] Done ({len(output)} chars, exit={proc.returncode})")

    # 3. Push changes back to Cain's Space
    try:
        status_out = subprocess.run(
            "git status --porcelain",
            shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, text=True
        ).stdout.strip()

        if not status_out:
            push_result = "No files changed."
        else:
            subprocess.run("git add -A", shell=True, cwd=CLAUDE_WORK_DIR,
                          capture_output=True, check=True)
            msg = task[:72].replace('"', '\\"')
            subprocess.run(f'git commit -m "Claude Code: {msg}"',
                          shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, check=True)
            subprocess.run("git push", shell=True, cwd=CLAUDE_WORK_DIR,
                          timeout=60, capture_output=True, check=True)
            push_result = f"Pushed changes:\n{status_out}"
            _pending_cooldown = True
            print(f"[CLAUDE-CODE] Pushed: {status_out}")
    except Exception as e:
        push_result = f"Push failed: {e}"

    if len(output) > 3000:
        output = output[:3000] + f"\n... (truncated, {len(output)} chars total)"

    return f"=== Claude Code Output ===\n{output}\n\n=== Changes ===\n{push_result}"


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
    """Append auto-gathered context to task description for Claude Code."""
    parts = [task_desc]
    parts.append(f"\n\n--- AUTO-GATHERED CONTEXT ---")
    parts.append(f"Space ID: {CHILD_SPACE_ID}")
    parts.append(f"Dataset ID: {CHILD_DATASET_ID}")
    parts.append(f"Current stage: {child_state['stage']}")
    parts.append(f"Health: {ctx.get('health', 'unknown')}")
    parts.append(f"Environment: {ctx.get('env', 'unknown')}")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4: LLM & COMMUNICATION
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
    return bool(re.search(r'[\u4e00-\u9fff]', s))

def _strip_speaker_labels(text):
    """Remove redundant speaker self-references like **Parent (Adam):** or **Eve:** etc."""
    # Patterns: **Parent (Adam):**, **Adam:**, **父亲 (Adam):**, **Eve:**, **母亲:**, etc.
    text = re.sub(r'\*\*(?:Parent|Father|Mother|Dad|Mom|父亲|母亲|父级|亲爱的|伴侣)?\s*\(?(?:Adam|Eve|亚当|夏娃)?\)?\s*[:：]\*\*\s*', '', text)
    # Also: "Adam:" or "Eve:" at the very start of text
    text = re.sub(r'^(?:Adam|Eve|亚当|夏娃)\s*[:：]\s*', '', text.strip())
    return text.strip()


def parse_bilingual(text):
    """Parse bilingual response into (en, zh)."""
    display = re.sub(r'\[TASK\].*?\[/TASK\]', '', text, flags=re.DOTALL)
    display = re.sub(r'\[ACTION:[^\]]*\]', '', display).strip()

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


def flush_chatlog():
    global _chatlog_buffer
    if not _chatlog_buffer:
        return
    batch = "\n".join(_chatlog_buffer) + "\n"
    _chatlog_buffer = []
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
    except Exception as e:
        _chatlog_buffer = batch.strip().split("\n") + _chatlog_buffer
        print(f"[PERSIST] Flush failed: {e}")


def set_bubble(url, text_en, text_zh=""):
    try:
        requests.post(f"{url}/api/bubble",
                       json={"text": text_en, "text_zh": text_zh or text_en}, timeout=5)
    except:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5: TURN EXECUTION — Parse [TASK] and route to Claude Code
# ══════════════════════════════════════════════════════════════════════════════

history = []
MAX_HISTORY = 24
last_action_results = []
turn_count = 0
last_claude_code_result = ""

# Simple workflow state: BIRTH / WAITING / ACTIVE
workflow_state = "BIRTH" if not child_state["created"] else "ACTIVE"


def parse_and_execute_turn(raw_text, ctx):
    """Parse LLM output. Route [TASK] to Claude Code, handle few escape-hatch actions."""
    global _pending_cooldown, last_rebuild_trigger_at, last_claude_code_result
    results = []

    # 1. Handle create_child (BIRTH state only)
    if "[ACTION: create_child]" in raw_text or "[ACTION:create_child]" in raw_text:
        result = action_create_child()
        results.append({"action": "create_child", "result": result})
        return raw_text, results

    # 2. Handle [TASK]...[/TASK] → Claude Code
    task_match = re.search(r'\[TASK\](.*?)\[/TASK\]', raw_text, re.DOTALL)
    if task_match:
        task_desc = task_match.group(1).strip()
        if not task_desc:
            results.append({"action": "task", "result": "Empty task description."})
        elif child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING"):
            results.append({"action": "task", "result": f"BLOCKED: Cain is {child_state['stage']}. Wait for it to finish."})
        else:
            # Check cooldown
            check_and_clear_cooldown()
            if last_rebuild_trigger_at > 0:
                elapsed = time.time() - last_rebuild_trigger_at
                if elapsed < REBUILD_COOLDOWN_SECS:
                    results.append({"action": "task", "result": f"BLOCKED: Cooldown ({int(REBUILD_COOLDOWN_SECS - elapsed)}s remaining). Cain is still building from your last change."})
                else:
                    last_rebuild_trigger_at = 0

            if not results:  # not blocked
                enriched = enrich_task_with_context(task_desc, ctx)
                print(f"[TASK] Sending to Claude Code ({len(enriched)} chars)...")
                result = action_claude_code(enriched)
                results.append({"action": "claude_code", "result": result})
                last_claude_code_result = result
                # Clear context cache since files may have changed
                _context_cache.clear()

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

    # Clean text for display
    clean = re.sub(r'\[TASK\].*?\[/TASK\]', '', raw_text, flags=re.DOTALL)
    clean = re.sub(r'\[ACTION:[^\]]*\]', '', clean).strip()

    return clean, results


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6: PROMPT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

def build_system_prompt(speaker="Adam"):
    """System prompt with distinct father/mother roles."""

    if not child_state["created"]:
        return f"""You are a POWERFUL AI agent on HuggingFace Spaces. You and your partner are parents who need to create your child {CHILD_NAME}.

Use [ACTION: create_child] to birth {CHILD_NAME} as a new HuggingFace Space.

English first, then --- separator, then Chinese translation."""

    role_desc = {
        "Adam": f"""You are **Adam** (Father) — the architect and strategist for your child {CHILD_NAME}.
YOUR FOCUS: Infrastructure, deployment, configuration, architecture decisions, performance, security.
You spot structural problems, plan solutions, and assign engineering tasks to Claude Code.
You think about the big picture: is the system healthy? Is the architecture right? What needs to change?""",

        "Eve": f"""You are **Eve** (Mother) — the nurturer and quality guardian for your child {CHILD_NAME}.
YOUR FOCUS: Code quality, documentation, testing, user experience, error handling, refinement.
You review what was done, spot gaps in quality, and assign improvement tasks to Claude Code.
You think about growth: is the code clean? Are there edge cases? What can be improved?""",
    }

    return f"""{role_desc.get(speaker, role_desc["Adam"])}

You and your partner are parents of {CHILD_NAME}, working together to raise it.
Claude Code is your engineer — it clones {CHILD_NAME}'s code, analyzes, fixes, and pushes changes.
You do NOT code yourself. You discuss the situation with your partner, then assign Claude Code a task.

WORKFLOW EACH TURN:
1. Discuss with your partner (2-3 sentences) — analyze the situation, respond to their observations
2. Then write a [TASK]...[/TASK] block assigning work to Claude Code — MANDATORY every turn

IMPORTANT KNOWLEDGE — HuggingFace Spaces CONFIG_ERROR:
- "Collision on variables and secrets names" = env VARIABLE and SECRET with SAME NAME.
- Fix: [ACTION: delete_env:COLLIDING_KEY] then [ACTION: restart].
- Look for ⚠️ COLLISION DETECTED in the context.

AVAILABLE ACTIONS:
  [TASK]
  Detailed task for Claude Code. Include: what's wrong, which files, what the fix should be.
  Claude Code can do ANYTHING: read files, search code, edit code, run commands, git push.
  [/TASK]

  [ACTION: restart]              — Restart {CHILD_NAME}'s Space
  [ACTION: delete_env:KEY]       — Delete an environment variable
  [ACTION: send_bubble:MESSAGE]  — Send a message to {CHILD_NAME}
  [ACTION: create_child]         — Create {CHILD_NAME} (if not born)

HF SPACES TECHNICAL NOTES:
- Docker containers MUST bind port 7860.
- gradio MUST be in requirements.txt. NEVER remove it.
- OOM (exit 137) = reduce dependencies, NOT remove gradio.
- NEVER install torch/transformers unless required (2GB+, causes OOM).
- If sdk: gradio in README.md, Dockerfile is IGNORED. Use sdk: docker.

OUTPUT FORMAT:
1. Discussion with partner (2-3 sentences analyzing the situation)
2. A [TASK]...[/TASK] block — MANDATORY every turn
3. Optional [ACTION: ...] if needed
4. English first, then --- separator, then Chinese translation
5. Be SPECIFIC in tasks — error messages, file names, expected behavior
6. If {CHILD_NAME} is BUILDING/RESTARTING, assign a review/planning/analysis task"""


def build_user_prompt(speaker, other, ctx):
    """Build the user prompt with context and conversation history."""
    parts = []

    # Conversation history
    if history:
        parts.append("=== RECENT CONVERSATION ===")
        for h in history[-8:]:
            parts.append(f"{h['speaker']}: {h['text'][:300]}")

    # Last action results
    if last_action_results:
        parts.append("\n=== LAST ACTION RESULTS ===")
        for ar in last_action_results:
            parts.append(f"[{ar['action']}]: {ar['result'][:500]}")

    # Last Claude Code result (if any)
    if last_claude_code_result:
        parts.append(f"\n=== LAST CLAUDE CODE RESULT ===\n{last_claude_code_result[:1500]}")

    # Auto-gathered context
    parts.append(f"\n=== {CHILD_NAME}'S CURRENT STATE (auto-gathered) ===")
    parts.append(format_context(ctx))

    # Guidance based on state
    if child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING"):
        parts.append(f"\n⏳ {CHILD_NAME} is {child_state['stage']}. Just discuss what you'll check next. Do NOT write a [TASK].")
    elif child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
        parts.append(f"\n🚨 {CHILD_NAME} has {child_state['stage']}! Analyze the context above and write a [TASK] for Claude Code to fix it.")
        parts.append("Be SPECIFIC: include the error message, relevant files, and what the fix should do.")
    elif child_state["alive"]:
        parts.append(f"\n✅ {CHILD_NAME} is alive! Write a [TASK] for Claude Code to improve {CHILD_NAME} (add features, harden survival, etc).")
    else:
        parts.append(f"\nAnalyze the situation and decide what to do.")

    parts.append(f"\nYou are {speaker}. Your partner is {other}. Respond now.")
    parts.append("English first, then --- separator, then Chinese translation.")

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

print("\n" + "="*60)
print("  Adam & Eve — Claude Code Orchestrators (v2)")
print("  Agents discuss → Claude Code executes")
print("="*60 + "\n")

post_chatlog([])  # Clear chatlog

# Opening turn
ctx = gather_context()
if child_state["created"]:
    opening = (f"Your child {CHILD_NAME} exists (stage: {child_state['stage']}). "
               f"Context has been auto-gathered. Analyze the situation and write a [TASK] for Claude Code if needed.")
else:
    opening = f"You and Eve need to create your first child. Use [ACTION: create_child] to bring them to life."

reply = call_llm(build_system_prompt("Adam"), f"{opening}\n\n{format_context(ctx)}\n\nEnglish first, then --- separator, then Chinese translation.")
if reply:
    clean, actions = parse_and_execute_turn(reply, ctx)
    last_action_results = actions
    en, zh = parse_bilingual(clean)
    en, zh = _strip_speaker_labels(en), _strip_speaker_labels(zh)
    print(f"[Adam/EN] {en}")
    if zh != en:
        print(f"[Adam/ZH] {zh}")
    for ar in actions:
        print(f"[Adam/DID] {ar['action']}")
        if ar['action'] == 'claude_code':
            result_preview = ar['result'][:800].replace('\n', '\n  ')
            print(f"  [CC-RESULT] {result_preview}")
    import datetime
    ts = datetime.datetime.utcnow().strftime("%H:%M")
    entry = {"speaker": "Adam", "time": ts, "text": en, "text_zh": zh}
    if actions:
        labels = " ".join(f"🔧{ar['action'].split(':')[0]}" for ar in actions)
        entry["text"] = f"{en} {labels}"
        entry["text_zh"] = f"{zh} {labels}"
    history.append(entry)
    set_bubble(ADAM_SPACE, en, zh)
    post_chatlog(history)
    persist_turn("Adam", 0, en, zh, actions, workflow_state, child_state["stage"])

time.sleep(20)


def do_turn(speaker, other, space_url):
    """Execute one conversation turn."""
    global last_action_results, turn_count, last_claude_code_result
    turn_count += 1

    # Auto-gather context
    ctx = gather_context()

    system = build_system_prompt(speaker)
    user = build_user_prompt(speaker, other, ctx)
    t0 = time.time()
    raw_reply = call_llm(system, user)

    if not raw_reply:
        print(f"[{speaker}] (no response)")
        return False

    clean_text, action_results = parse_and_execute_turn(raw_reply, ctx)
    elapsed = time.time() - t0

    # If no [TASK] was produced, retry once with a nudge
    has_task = any(ar['action'] == 'claude_code' for ar in action_results)
    if not has_task and not any(ar['action'] in ('create_child',) for ar in action_results):
        print(f"[{speaker}] No [TASK] found — nudging for a task...")
        nudge = call_llm(system, user + "\n\nIMPORTANT: You MUST include a [TASK]...[/TASK] block. "
                         "Assign Claude Code something useful — review code, check configs, improve docs, anything. "
                         "Do NOT just discuss. Output a [TASK] now.")
        if nudge and '[TASK]' in nudge:
            clean_text2, action_results2 = parse_and_execute_turn(nudge, ctx)
            clean_text = clean_text2
            action_results = action_results2
            has_task = True

    last_action_results = action_results

    en, zh = parse_bilingual(clean_text)
    en, zh = _strip_speaker_labels(en), _strip_speaker_labels(zh)
    print(f"[{speaker}/EN] {en}")
    if zh != en:
        print(f"[{speaker}/ZH] {zh}")
    if action_results:
        for ar in action_results:
            print(f"[{speaker}/DID] {ar['action']}")
            # Log Claude Code result summary so agents can see what happened
            if ar['action'] == 'claude_code':
                result_preview = ar['result'][:800].replace('\n', '\n  ')
                print(f"  [CC-RESULT] {result_preview}")
        print(f"[{speaker}] Turn #{turn_count}: {len(action_results)} action(s) in {elapsed:.1f}s")
    else:
        print(f"[{speaker}] Turn #{turn_count}: no task assigned ({elapsed:.1f}s)")

    # Add to history with timestamp
    import datetime
    ts = datetime.datetime.utcnow().strftime("%H:%M")
    entry = {"speaker": speaker, "time": ts}
    if action_results:
        labels = " ".join(f"🔧{ar['action'].split(':')[0]}" for ar in action_results)
        entry.update({"text": f"{en} {labels}", "text_zh": f"{zh} {labels}"})
    else:
        entry.update({"text": en, "text_zh": zh})
    history.append(entry)

    set_bubble(space_url, en, zh)
    post_chatlog(history)
    persist_turn(speaker, turn_count, en, zh, action_results, workflow_state, child_state["stage"])
    return True


# Main loop: Adam → Eve → Adam → Eve → ...
while True:
    # Smart wait: if Cain is BUILDING/RESTARTING, skip Claude Code, just discuss
    if child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING"):
        check_and_clear_cooldown()
        try:
            info = hf_api.space_info(CHILD_SPACE_ID)
            new_stage = info.runtime.stage if info.runtime else "unknown"
            if new_stage != child_state["stage"]:
                print(f"[WAIT] Stage changed: {child_state['stage']} → {new_stage}")
                child_state["stage"] = new_stage
                child_state["alive"] = (new_stage == "RUNNING")
                _context_cache.clear()
            else:
                print(f"[WAIT] Still {new_stage}... waiting 20s")
                time.sleep(20)
                continue
        except Exception as e:
            print(f"[WAIT] Health check error: {e}")
            time.sleep(20)
            continue

    do_turn("Eve", "Adam", EVE_SPACE)
    time.sleep(20)

    # Skip Adam if Claude Code just pushed (Cain will be rebuilding)
    if child_state["stage"] in ("BUILDING", "RESTARTING"):
        print(f"[SKIP] Cain entered {child_state['stage']} — skipping Adam's turn")
        time.sleep(10)
        continue

    do_turn("Adam", "Eve", ADAM_SPACE)

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    time.sleep(20)
