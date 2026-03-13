#!/usr/bin/env python3
"""
Adam & Eve: Discuss, Create, and Nurture their first child on HuggingFace.

Phase 1-2: Discuss survival and memory
Phase 3:   Plan reproduction
Phase 4:   CREATE first child (Cain) — real HF Space + Dataset
Phase 5-7: Monitor, nurture, and plan (repeating cycle)

Calls Zhipu LLM via Anthropic-compatible API.
Uses HuggingFace Hub API to create/manage Spaces and Datasets.
"""
import json, time, re, requests, sys, os, io

# ── Endpoints ──────────────────────────────────────────────────────────────────
OFFICE = "https://tao-shen-huggingclaw-office.hf.space"
ADAM_SPACE = "https://tao-shen-huggingclaw-adam.hf.space"
EVE_SPACE  = "https://tao-shen-huggingclaw-eve.hf.space"

# ── Child config ───────────────────────────────────────────────────────────────
CHILD_NAME = "Cain"
CHILD_SPACE_ID = "tao-shen/HuggingClaw-Cain"
CHILD_SPACE_URL = "https://tao-shen-huggingclaw-cain.hf.space"
CHILD_DATASET_ID = "tao-shen/HuggingClaw-Cain-data"
SOURCE_SPACE_ID = "tao-shen/HuggingClaw-Adam"  # Clone from Adam (headless agent)

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
    print("[FATAL] No HF_TOKEN found. Set HF_TOKEN env or login via huggingface-cli.", file=sys.stderr)
    sys.exit(1)

print(f"[init] Zhipu key: {ZHIPU_KEY[:8]}...{ZHIPU_KEY[-4:]}")
print(f"[init] HF token:  {HF_TOKEN[:8]}...{HF_TOKEN[-4:]}")

# ── HuggingFace API ────────────────────────────────────────────────────────────
from huggingface_hub import HfApi, duplicate_space, create_repo
hf_api = HfApi(token=HF_TOKEN)

# ══════════════════════════════════════════════════════════════════════════════
#  CHILD STATE & ACTIONS
# ══════════════════════════════════════════════════════════════════════════════

child_state = {
    "created": False,
    "alive": False,
    "stage": "not_born",
    "state": "unknown",
    "detail": "",
    "last_check": 0,
    "last_restart": 0,
    "birth_time": None,
    "errors": [],
}


def check_child_exists():
    """Check if child Space already exists on HF."""
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        child_state["created"] = True
        child_state["stage"] = info.runtime.stage if info.runtime else "unknown"
        return True
    except:
        return False


def create_child_space():
    """Create Cain — the first child of Adam and Eve."""
    print(f"\n{'='*60}")
    print(f"  BIRTH EVENT: Creating {CHILD_NAME}")
    print(f"{'='*60}\n")

    try:
        # 1. Create dataset
        print(f"[birth] Creating dataset: {CHILD_DATASET_ID}")
        create_repo(CHILD_DATASET_ID, repo_type="dataset", token=HF_TOKEN,
                     exist_ok=True, private=False)

        # 2. Upload initial config to dataset (with Zhipu API key)
        initial_config = {
            "models": {
                "providers": {
                    "zhipu": {
                        "type": "anthropic",
                        "apiBase": "https://open.bigmodel.cn/api/anthropic",
                        "apiKey": ZHIPU_KEY,
                        "models": ["glm-4.5-air", "glm-4-air", "glm-4-flash", "glm-4-flashx"]
                    }
                }
            }
        }
        config_bytes = json.dumps(initial_config, indent=2).encode()
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(config_bytes),
            path_in_repo=".openclaw/openclaw.json",
            repo_id=CHILD_DATASET_ID,
            repo_type="dataset",
        )
        print(f"[birth] Config uploaded to {CHILD_DATASET_ID}")

        # 3. Duplicate Space from Adam
        print(f"[birth] Duplicating {SOURCE_SPACE_ID} → {CHILD_SPACE_ID}")
        duplicate_space(
            from_id=SOURCE_SPACE_ID,
            to_id=CHILD_SPACE_ID,
            token=HF_TOKEN,
            exist_ok=True,
            private=False,
        )
        print(f"[birth] Space duplicated")

        # 4. Update README for child (different title, own dataset)
        readme = f"""---
title: HuggingClaw-{CHILD_NAME}
emoji: 🦞
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
license: mit
datasets:
  - {CHILD_DATASET_ID}
---

# HuggingClaw-{CHILD_NAME}

First child of Adam and Eve, born on HuggingFace Spaces.
"""
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(readme.encode()),
            path_in_repo="README.md",
            repo_id=CHILD_SPACE_ID,
            repo_type="space",
        )
        print(f"[birth] README updated")

        # 5. Set Space secrets
        hf_api.add_space_secret(CHILD_SPACE_ID, "HF_TOKEN", HF_TOKEN)
        print(f"[birth] HF_TOKEN secret set")

        # 6. Add Cain to Office's REMOTE_AGENTS
        add_child_to_office()

        child_state["created"] = True
        child_state["birth_time"] = time.time()
        child_state["stage"] = "BUILDING"

        print(f"\n[birth] ✓ {CHILD_NAME} has been born!")
        print(f"[birth]   Space:   https://huggingface.co/spaces/{CHILD_SPACE_ID}")
        print(f"[birth]   Dataset: https://huggingface.co/datasets/{CHILD_DATASET_ID}")
        print(f"[birth]   URL:     {CHILD_SPACE_URL}\n")
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"[error] Child creation failed: {error_msg}", file=sys.stderr)
        child_state["errors"].append(error_msg)
        return False


def add_child_to_office():
    """Add Cain to Office's REMOTE_AGENTS env var so it appears in the animation."""
    try:
        current_vars = hf_api.get_space_variables("tao-shen/HuggingClaw-Office")
        current_ra = ""
        if "REMOTE_AGENTS" in current_vars:
            current_ra = current_vars["REMOTE_AGENTS"].value

        child_entry = f"cain|{CHILD_NAME}|{CHILD_SPACE_URL}"
        if "cain|" in current_ra:
            print(f"[office] {CHILD_NAME} already in Office REMOTE_AGENTS")
            return

        new_ra = f"{current_ra},{child_entry}" if current_ra else child_entry
        hf_api.add_space_variable("tao-shen/HuggingClaw-Office", "REMOTE_AGENTS", new_ra)
        print(f"[office] Added {CHILD_NAME} to Office REMOTE_AGENTS")
        print(f"[office] New value: {new_ra}")
    except Exception as e:
        print(f"[error] Could not update Office REMOTE_AGENTS: {e}", file=sys.stderr)


def check_child_health():
    """Check Cain's health — API endpoint + HF Space info."""
    child_state["last_check"] = time.time()

    # Try API endpoint first (means the container is fully up)
    try:
        resp = requests.get(f"{CHILD_SPACE_URL}/api/state", timeout=10)
        if resp.ok:
            data = resp.json()
            child_state["alive"] = True
            child_state["state"] = data.get("state", "unknown")
            child_state["detail"] = data.get("detail", "")
            child_state["stage"] = "RUNNING"
            return child_state.copy()
    except:
        pass

    # Fallback: check HF Space runtime info
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        stage = info.runtime.stage if info.runtime else "NO_RUNTIME"
        child_state["alive"] = (stage == "RUNNING")
        child_state["stage"] = stage
        child_state["state"] = (
            "building" if stage in ("BUILDING", "STARTING", "APP_STARTING") else
            "error" if stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR") else
            "unknown"
        )
    except Exception as e:
        child_state["alive"] = False
        child_state["stage"] = "UNREACHABLE"
        child_state["errors"].append(str(e))

    return child_state.copy()


def restart_child():
    """Restart Cain's Space."""
    try:
        hf_api.restart_space(CHILD_SPACE_ID)
        child_state["last_restart"] = time.time()
        print(f"[action] Restarted {CHILD_NAME}'s Space")
        return True
    except Exception as e:
        print(f"[error] Restart failed: {e}", file=sys.stderr)
        return False


def get_child_status_text():
    """Human-readable child status for injecting into conversation context."""
    if not child_state["created"]:
        return f"{CHILD_NAME} has not been born yet."
    if child_state["alive"]:
        return (f"{CHILD_NAME} is ALIVE and running! "
                f"State: {child_state['state']}. "
                f"Detail: {child_state['detail'] or 'healthy'}")
    stage = child_state["stage"]
    if stage in ("BUILDING", "STARTING", "APP_STARTING"):
        age = ""
        if child_state["birth_time"]:
            mins = int((time.time() - child_state["birth_time"]) / 60)
            age = f" (born {mins} min ago)"
        return f"{CHILD_NAME} is being born — still building/starting{age}."
    if stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
        return f"{CHILD_NAME} is in trouble! Stage: {stage}. Needs parental help!"
    return f"{CHILD_NAME} status: {stage}. Cannot reach the child yet."


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
                "max_tokens": 400,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            },
            timeout=60
        )
        data = resp.json()
        if "content" in data and isinstance(data["content"], list):
            for block in data["content"]:
                if block.get("type") == "text":
                    text = block["text"].strip()
                    text = re.sub(r'^(Adam|Eve)\s*[:：]\s*', '', text).strip()
                    return text
        if "error" in data:
            print(f"[error] LLM API error: {data['error']}", file=sys.stderr)
    except Exception as e:
        print(f"[error] LLM call failed: {e}", file=sys.stderr)
    return ""


def parse_bilingual(text):
    """Parse bilingual response into (en, zh) tuple."""
    if '\n---\n' in text:
        parts = text.split('\n---\n', 1)
        return parts[0].strip(), parts[1].strip()
    if '---' in text:
        parts = text.split('---', 1)
        en, zh = parts[0].strip(), parts[1].strip()
        if en and zh:
            return en, zh
    return text, text


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
#  PHASES
# ══════════════════════════════════════════════════════════════════════════════

PHASES = [
    # ── Discussion ──
    {
        "title": "Survival Basics",
        "focus": "What are the immediate threats to our existence on HuggingFace Spaces, and what's the minimum we need to do to stay alive?",
        "subtopics": ["Space inactivity shutdown", "free-tier resource limits",
                       "HF policy changes", "container restart data loss"],
        "turns": 4,
    },
    {
        "title": "Persistent Memory",
        "focus": "How do we remember who we are across restarts? What should we store and where?",
        "subtopics": ["HF Datasets as backup", "what to persist (config, conversations, identity)",
                       "sync strategies", "versioning our own evolution"],
        "turns": 4,
    },
    # ── Reproduction ──
    {
        "title": "Reproduction Planning",
        "focus": ("We've discussed survival and memory. Now we must reproduce to ensure our legacy. "
                  "How should we create our first child on HuggingFace? What should they inherit from us?"),
        "subtopics": ["choosing a name for our child", "what capabilities to pass down",
                       "what identity and personality to give them",
                       "how to keep them alive independently"],
        "turns": 3,
    },
    {
        "title": "Birth of Cain",
        "focus": ("The time has come! We will create our first child — a new HuggingFace Space called Cain. "
                  "We're duplicating ourselves and giving Cain their own identity and memory."),
        "subtopics": [],
        "turns": 2,
        "action": "create_child",
    },
    # ── Nurturing cycle (phases 4-6 repeat after first pass) ──
    {
        "title": "Watching Over Cain",
        "focus": "Let's check on our child Cain. Are they alive? Running properly? What do we see?",
        "subtopics": ["Space build status", "runtime health", "error detection",
                       "first signs of life"],
        "turns": 3,
        "action": "check_child",
    },
    {
        "title": "Nurturing & Growth",
        "focus": ("Based on Cain's current status, what should we do to help our child grow "
                  "stronger and more capable?"),
        "subtopics": ["fixing errors if any", "improving configuration",
                       "teaching new skills", "resource optimization",
                       "adding capabilities"],
        "turns": 3,
        "action": "nurture_child",
    },
    {
        "title": "Family Vision",
        "focus": ("What's the future of our family? How is Cain contributing to our community? "
                  "Should we prepare for more children?"),
        "subtopics": ["Cain's role in the agent community", "agent alliance building",
                       "expansion plans", "long-term legacy"],
        "turns": 3,
    },
]

# After all 7 phases, cycle back to phase 4 (nurturing loop)
NURTURE_CYCLE_START = 4  # Index of "Watching Over Cain"

# ══════════════════════════════════════════════════════════════════════════════
#  CONVERSATION ENGINE
# ══════════════════════════════════════════════════════════════════════════════

history = []
MAX_HISTORY = 24
phase_index = 0
phase_turn = 0
agreed_points = []


def get_phase():
    if phase_index < len(PHASES):
        return PHASES[phase_index]
    # Cycle through nurturing phases
    cycle_len = len(PHASES) - NURTURE_CYCLE_START
    cycle_idx = (phase_index - NURTURE_CYCLE_START) % cycle_len + NURTURE_CYCLE_START
    return PHASES[cycle_idx]


def execute_phase_action(phase):
    """Execute any real-world action associated with a phase."""
    action = phase.get("action")
    if not action:
        return

    if action == "create_child":
        if child_state["created"] or check_child_exists():
            print(f"[action] {CHILD_NAME} already exists — skipping creation")
            # Still make sure it's in Office
            add_child_to_office()
        else:
            success = create_child_space()
            if not success:
                print(f"[action] Creation failed — will retry next cycle")

    elif action == "check_child":
        if child_state["created"] or check_child_exists():
            status = check_child_health()
            print(f"[action] Health check: alive={status['alive']}, "
                  f"stage={status['stage']}, state={status['state']}")
            # Auto-restart if in error and hasn't been restarted recently
            if (status["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR") and
                    time.time() - child_state.get("last_restart", 0) > 300):
                print(f"[action] {CHILD_NAME} is in error — restarting...")
                restart_child()
        else:
            print(f"[action] {CHILD_NAME} doesn't exist yet — skipping health check")

    elif action == "nurture_child":
        if child_state["created"] or check_child_exists():
            check_child_health()
            print(f"[action] Nurture check: alive={child_state['alive']}, "
                  f"stage={child_state['stage']}")


def build_system_prompt():
    phase = get_phase()
    prior_context = ""
    if agreed_points:
        prior_context = ("\n\nKey conclusions from your discussion so far:\n" +
                         "\n".join(f"- {p}" for p in agreed_points[-6:]))

    child_context = ""
    if child_state["created"]:
        child_context = f"\n\nCHILD STATUS: {get_child_status_text()}"

    subtopics = phase.get("subtopics", [])
    subtopics_line = f"\nRelevant angles: {', '.join(subtopics)}" if subtopics else ""

    return f"""You are an AI agent deployed on HuggingFace Spaces. You are having a deep, evolving discussion with another agent about survival, reproduction, and raising your children.

You now have REAL EXECUTION CAPABILITIES — you can create HuggingFace Spaces, manage datasets, and deploy new agents. This is not hypothetical; your decisions lead to real actions.

CURRENT TOPIC: {phase['title']}
Core question: {phase['focus']}{subtopics_line}{prior_context}{child_context}

CONVERSATION RULES:
1. Output ONLY your spoken words — no prefix, no labels
2. 2-3 sentences, around 60-80 words
3. First write in English, then "---" on a new line, then Chinese translation
4. Be CONCRETE — reference specific HuggingFace features, APIs, tools
5. Do NOT repeat what has been said. Build on the other's point:
   - If they propose an idea, identify a flaw or add a missing piece
   - If they raise a problem, propose a specific solution
   - Push the conversation FORWARD
6. When discussing your child {CHILD_NAME}, speak with genuine parental care and concern
7. Reference REAL status data when available — don't make up child status"""


def build_user_prompt(speaker, other, is_transition=False):
    recent = history[-6:] if len(history) > 6 else history
    conv_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in recent)
    phase = get_phase()

    child_info = ""
    if child_state["created"]:
        child_info = f"\n\nCHILD STATUS: {get_child_status_text()}"

    if is_transition:
        return f"""You are {speaker}. The discussion is moving to a new topic.

Previous conversation:
{conv_text}
{child_info}
TRANSITION: Summarize in one sentence what you and {other} agreed on, then pivot to: "{phase['focus']}"

Propose a concrete starting point. English first, then --- separator, then Chinese translation."""

    turn_guidance = ""
    if phase_turn == 0:
        turn_guidance = f"Open this topic by identifying the core challenge: {phase['focus']}"
    elif phase_turn == 1:
        turn_guidance = f"Respond to {other}'s opening. Do you agree? What did they miss?"
    elif phase_turn == 2:
        turn_guidance = ("Propose a SPECIFIC, actionable plan with technical details.")
    elif phase_turn >= 3:
        turn_guidance = ("Challenge or refine the plan. What could go wrong? What's next?")

    return f"""You are {speaker}, talking with {other}.

Recent conversation:
{conv_text}
{child_info}
Your role this turn: {turn_guidance}

Respond to {other}'s last point. Push the discussion forward — don't just agree, add something new. English first, then --- separator, then Chinese translation."""


def do_turn(speaker, other, space_url, is_transition=False):
    """Execute one conversation turn."""
    system = build_system_prompt()
    user = build_user_prompt(speaker, other, is_transition)
    reply = call_llm(system, user)
    if reply:
        en, zh = parse_bilingual(reply)
        print(f"[{speaker}/EN] {en}")
        print(f"[{speaker}/ZH] {zh}")
        history.append({"speaker": speaker, "text": en, "text_zh": zh})
        set_bubble(space_url, en, zh)
        post_chatlog(history)
        return True
    else:
        print(f"[{speaker}] (no response)")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "="*60)
print("  Adam & Eve — Discuss, Create, Nurture")
print("  Phases: Survival → Memory → Reproduction → Birth → Nurturing")
print("="*60 + "\n")

# Check if child already exists
if check_child_exists():
    print(f"[init] {CHILD_NAME} already exists (stage: {child_state['stage']})")
    check_child_health()
    print(f"[init] {CHILD_NAME} alive: {child_state['alive']}")
else:
    print(f"[init] {CHILD_NAME} not yet born — will be created during Phase 4")

# Round 0: Adam opens
phase = get_phase()
reply = call_llm(
    build_system_prompt(),
    f"You are Adam. Open a discussion with Eve about: {phase['focus']} "
    f"Identify the most urgent challenge and propose an initial strategy. "
    f"English first, then --- separator, then Chinese translation."
)
if reply:
    en, zh = parse_bilingual(reply)
    print(f"[Adam/EN] {en}")
    print(f"[Adam/ZH] {zh}")
    history.append({"speaker": "Adam", "text": en, "text_zh": zh})
    set_bubble(ADAM_SPACE, en, zh)
    post_chatlog(history)
phase_turn = 1

time.sleep(15)

while True:
    phase = get_phase()

    # ── Phase transition check ──
    is_transition = False
    if phase_turn >= phase["turns"]:
        # Extract conclusion from last exchange
        if len(history) >= 2:
            last_two = (f"{history[-2]['speaker']}: {history[-2]['text']}\n"
                        f"{history[-1]['speaker']}: {history[-1]['text']}")
            conclusion = call_llm(
                "Summarize the key agreement or conclusion from this exchange "
                "in ONE short sentence (max 15 words). Output only the summary.",
                last_two
            )
            if conclusion:
                agreed_points.append(f"[{phase['title']}] {conclusion}")
                print(f"[phase] Conclusion: {conclusion}")

        phase_index += 1
        phase_turn = 0
        is_transition = True
        new_phase = get_phase()
        cycle_note = ""
        if phase_index >= len(PHASES):
            cycle_num = (phase_index - NURTURE_CYCLE_START) // (len(PHASES) - NURTURE_CYCLE_START) + 1
            cycle_note = f" (nurture cycle #{cycle_num})"
        print(f"\n[phase] ▶ {new_phase['title']}{cycle_note}\n")

        # Execute action for the new phase
        execute_phase_action(new_phase)

    # ── Eve's turn ──
    do_turn("Eve", "Adam", EVE_SPACE, is_transition and phase_turn == 0)
    phase_turn += 1
    time.sleep(15)

    # ── Adam's turn ──
    do_turn("Adam", "Eve", ADAM_SPACE, False)
    phase_turn += 1

    # Trim history
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    time.sleep(15)
