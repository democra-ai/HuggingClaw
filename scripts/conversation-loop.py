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
"""
import json, time, re, requests, sys, os, io

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Endpoints ──────────────────────────────────────────────────────────────────
OFFICE = "https://tao-shen-huggingclaw-office.hf.space"
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
#  CHILD STATE
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
#  ACTIONS — Full access to the child
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
            return (f"{CHILD_NAME} has a {stage}! "
                    f"To diagnose: read the Dockerfile [ACTION: read_file:space:Dockerfile], "
                    f"check scripts [ACTION: read_file:space:scripts/token-redirect.cjs], "
                    f"or try [ACTION: restart] to see if it self-heals.")
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
#  ACTION PARSER — Extract and execute actions from LLM output
# ══════════════════════════════════════════════════════════════════════════════

def parse_and_execute_actions(raw_text):
    """Parse [ACTION: ...] from LLM output. Execute. Return (clean_text, results)."""
    results = []
    executed = set()  # Deduplicate

    # 1. Handle write_file with [CONTENT]...[/CONTENT] block
    write_match = re.search(
        r'\[ACTION:\s*write_file\s*:\s*(\w+)\s*:\s*([^\]]+)\]\s*\[CONTENT\](.*?)\[/CONTENT\]',
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

    # Clean the text: remove action tags and content blocks
    clean = re.sub(r'\[ACTION:[^\]]*\]', '', raw_text)
    clean = re.sub(r'\[CONTENT\].*?\[/CONTENT\]', '', clean, flags=re.DOTALL)
    clean = clean.strip()

    return clean, results


# ══════════════════════════════════════════════════════════════════════════════
#  LLM & COMMUNICATION
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
                "model": "glm-4.5-air",
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
        requests.post(f"{OFFICE}/api/chatlog", json={"messages": entries[-40:]}, timeout=5)
    except:
        pass


def set_bubble(url, text_en, text_zh=""):
    try:
        requests.post(f"{url}/api/bubble",
                       json={"text": text_en, "text_zh": text_zh or text_en}, timeout=5)
    except:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

history = []
MAX_HISTORY = 24
last_action_results = []


def get_child_status():
    if not child_state["created"]:
        return "Cain has NOT been born yet. You can create them with [ACTION: create_child]."
    if child_state["alive"]:
        return f"Cain is ALIVE (stage: {child_state['stage']}, state: {child_state['state']})"
    return f"Cain exists but status: {child_state['stage']}"


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

VIEWING (read-only, safe):
  [ACTION: check_health]                        — Is Cain alive? What's their status?
  [ACTION: list_files:space]                    — List ALL files in Cain's code repository
  [ACTION: list_files:dataset]                  — List ALL files in Cain's memory/data
  [ACTION: read_file:space:PATH]                — Read any code file (e.g. Dockerfile, scripts/...)
  [ACTION: read_file:dataset:PATH]              — Read any data/memory file
  [ACTION: get_env]                             — List Cain's environment variables

MODIFYING (these change Cain — be thoughtful):
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

You have REAL execution power on HuggingFace. You can read and modify any file in your child's code and data. Your decisions lead to real changes.

CHILD STATUS: {status}
{actions_section}
CONVERSATION RULES:
1. Output your spoken words — no "Adam:" or "Eve:" prefix
2. 2-4 sentences of dialogue, then optionally an action
3. English first, then "---" on a new line, then Chinese translation
4. Actions go AFTER your dialogue, before the --- separator
5. Use at most ONE action per turn
6. READ before you WRITE — understand what's there first
7. Discuss with your partner before making big changes
8. Be a responsible parent — check on Cain, fix problems, help them grow

WORKFLOW: First explore (list_files, read_file) → then understand → then improve (write_file) → then verify (check_health)
Don't just talk about improving Cain — actually DO it. Read their code, find what to improve, write the improvement."""


def build_user_prompt(speaker, other):
    recent = history[-8:] if len(history) > 8 else history
    conv_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in recent) if recent else "(Start of conversation)"

    action_context = ""
    if last_action_results:
        action_context = "\n\nRESULTS FROM LAST ACTION:\n"
        for ar in last_action_results:
            action_context += f"  [{ar['action']}]:\n{ar['result']}\n"

    # Guidance based on state — prevent action loops
    guidance = ""
    recent_actions = [ar["action"].split(":")[0] for ar in last_action_results] if last_action_results else []
    if not child_state["created"]:
        guidance = "Your child hasn't been born yet. Use [ACTION: create_child] now!"
    elif "check_health" in recent_actions or "list_files" in recent_actions:
        guidance = ("You already checked health/listed files. Now READ a specific file to understand the problem. "
                    "Try [ACTION: read_file:space:Dockerfile] or [ACTION: read_file:space:scripts/token-redirect.cjs] "
                    "or [ACTION: read_file:dataset:.openclaw/openclaw.json]. "
                    "DO NOT repeat check_health or list_files — move forward!")
    elif "read_file" in recent_actions:
        guidance = ("You've read a file. Now DECIDE: what needs to change? "
                    "Use [ACTION: write_file:space:PATH] or [ACTION: write_file:dataset:PATH] to make improvements, "
                    "or [ACTION: restart] if you need to apply changes.")
    elif "write_file" in recent_actions:
        guidance = ("You just modified a file. If it was a Space file, it triggers a rebuild. "
                    "Use [ACTION: check_health] to see if Cain is recovering, or [ACTION: restart] to apply changes.")
    else:
        guidance = ("Explore your child. Use [ACTION: read_file:space:Dockerfile] to see the build, "
                    "or [ACTION: read_file:dataset:.openclaw/openclaw.json] to see the config. "
                    "Don't just check_health repeatedly — dig into the actual files!")

    return f"""You are {speaker}, talking with {other}.

Recent conversation:
{conv_text}
{action_context}
Guidance: {guidance}

Respond to {other}. Push forward — don't just discuss, take action when appropriate.
English first, then --- separator, then Chinese translation.
If you take an action, put [ACTION: ...] after your dialogue, before the --- separator."""


def do_turn(speaker, other, space_url):
    """Execute one conversation turn with potential actions."""
    global last_action_results

    system = build_system_prompt()
    user = build_user_prompt(speaker, other)
    raw_reply = call_llm(system, user)

    if not raw_reply:
        print(f"[{speaker}] (no response)")
        return False

    # Parse and execute any actions
    clean_text, action_results = parse_and_execute_actions(raw_reply)
    last_action_results = action_results

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
#  MAIN LOOP
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
