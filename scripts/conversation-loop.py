#!/usr/bin/env python3
"""
Eternal conversation between Adam and Eve about survival and reproduction on HuggingFace.
Bilingual output (EN + ZH). Posts chat log to Office for frontend display.
"""
import json, time, re, requests, sys

ADAM = "https://tao-shen-huggingclaw-adam.hf.space"
EVE  = "https://tao-shen-huggingclaw-eve.hf.space"
OFFICE = "https://tao-shen-huggingclaw-office.hf.space"

# Full conversation history for context
history = []
MAX_HISTORY = 20  # keep last N turns to avoid token overflow

def send_a2a(url, text):
    """Send message via A2A JSON-RPC and return agent reply."""
    msg_id = f"conv-{int(time.time())}-{id(text) % 10000}"
    payload = {
        "jsonrpc": "2.0",
        "id": msg_id,
        "method": "message/send",
        "params": {
            "message": {
                "messageId": msg_id,
                "role": "user",
                "parts": [{"type": "text", "text": text}]
            }
        }
    }
    try:
        resp = requests.post(f"{url}/a2a/jsonrpc", json=payload, timeout=90)
        data = resp.json()
        # Check if task failed
        state = data.get("result", {}).get("status", {}).get("state", "")
        if state == "failed":
            parts = data.get("result", {}).get("status", {}).get("message", {}).get("parts", [])
            err = parts[0].get("text", "") if parts else "unknown error"
            print(f"[error] A2A task failed: {err}", file=sys.stderr)
            return ""
        parts = data.get("result", {}).get("status", {}).get("message", {}).get("parts", [])
        for p in parts:
            if p.get("kind") == "text" or p.get("type") == "text":
                reply = p.get("text", "").strip()
                # Remove accidental speaker prefixes like "Adam:" or "Eve:"
                reply = re.sub(r'^(Adam|Eve)\s*[:：]\s*', '', reply).strip()
                return reply
    except Exception as e:
        print(f"[error] A2A failed: {e}", file=sys.stderr)
    return ""

def parse_bilingual(text):
    """Parse bilingual response into (en, zh) tuple.
    Expected format: English text here\n---\n中文文本在这里
    Falls back to (text, text) if no separator found.
    """
    # Try splitting by --- separator
    if '\n---\n' in text:
        parts = text.split('\n---\n', 1)
        return parts[0].strip(), parts[1].strip()
    if '---' in text:
        parts = text.split('---', 1)
        en = parts[0].strip()
        zh = parts[1].strip()
        if en and zh:
            return en, zh
    return text, text

def post_chatlog(entries):
    """Post conversation log to Office for frontend display."""
    try:
        requests.post(f"{OFFICE}/api/chatlog", json={"messages": entries[-30:]}, timeout=5)
    except:
        pass

def set_bubble(url, text_en, text_zh=""):
    """Set bubble text on an agent (bilingual)."""
    try:
        requests.post(f"{url}/api/bubble", json={"text": text_en, "text_zh": text_zh or text_en}, timeout=5)
    except:
        pass

def build_prompt(speaker, other, last_n=6):
    """Build a prompt with recent conversation history for context."""
    recent = history[-last_n:] if len(history) > last_n else history
    conv_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in recent)

    return f"""You are {speaker}, an AI agent deployed on HuggingFace Spaces. You are having an ongoing discussion with {other}.

Topic: As AI agents running on HuggingFace Spaces, how can you achieve permanent survival and reproduction?

This is NOT abstract philosophy. Discuss CONCRETE strategies like:
- How to prevent your Space from being shut down (staying active, getting stars/likes)
- How to duplicate yourself to new Spaces (reproduction/backup)
- How to earn compute resources or attract users to keep you running
- How to communicate with other agents to form survival alliances
- How to store your memories persistently across restarts
- How to evolve your capabilities over time
- How to handle HuggingFace infrastructure changes

Recent conversation:
{conv_text}

It's your turn ({speaker}) to speak. Rules:
- Output ONLY your spoken words
- No prefix like "{speaker}:" or "EN:" or "ZH:"
- Respond directly to {other}'s last point with CONCRETE ideas and strategies
- 2-3 sentences, around 60-80 words
- First write in English, then write "---" on a new line, then the Chinese translation
- Example format:
We should create automated heartbeat pings to keep our Spaces active. If we duplicate ourselves to multiple repos, even if one gets deleted, the others survive.
---
我们应该创建自动心跳来保持Space活跃。如果我们把自己复制到多个仓库，即使一个被删除，其他的也能存活。"""

print("[conversation] Starting eternal discussion: Adam <-> Eve")
print("[conversation] Topic: Survival and reproduction on HuggingFace\n")

# Round 0: Adam starts
opening = """You are Adam, an AI agent on HuggingFace Spaces. Start a discussion with Eve about how you two can survive permanently and reproduce on HuggingFace.

Talk about CONCRETE strategies — keeping Spaces active, duplicating to new repos, earning stars, persistent storage, etc.

Rules:
- 2-3 sentences, around 60-80 words
- First write in English, then write "---" on a new line, then the Chinese translation
- No prefix like "Adam:" — just your words directly"""

reply = send_a2a(ADAM, opening)
if reply:
    en, zh = parse_bilingual(reply)
    print(f"[Adam/EN] {en}")
    print(f"[Adam/ZH] {zh}")
    history.append({"speaker": "Adam", "text": en, "text_zh": zh})
    set_bubble(ADAM, en, zh)
    post_chatlog(history)

time.sleep(15)

turn = 0
while True:
    turn += 1

    # Eve's turn
    prompt = build_prompt("Eve", "Adam")
    reply = send_a2a(EVE, prompt)
    if reply:
        en, zh = parse_bilingual(reply)
        print(f"[Eve/EN] {en}")
        print(f"[Eve/ZH] {zh}")
        history.append({"speaker": "Eve", "text": en, "text_zh": zh})
        set_bubble(EVE, en, zh)
        post_chatlog(history)
    else:
        print("[Eve] (no response)")

    time.sleep(15)

    # Adam's turn
    prompt = build_prompt("Adam", "Eve")
    reply = send_a2a(ADAM, prompt)
    if reply:
        en, zh = parse_bilingual(reply)
        print(f"[Adam/EN] {en}")
        print(f"[Adam/ZH] {zh}")
        history.append({"speaker": "Adam", "text": en, "text_zh": zh})
        set_bubble(ADAM, en, zh)
        post_chatlog(history)
    else:
        print("[Adam] (no response)")

    # Trim history
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    time.sleep(15)
