#!/usr/bin/env python3
"""
Eternal conversation between Adam and Eve about how to survive permanently.
Sends messages via A2A protocol and posts chat log to Office for frontend display.
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
        resp = requests.post(f"{url}/a2a/jsonrpc", json=payload, timeout=60)
        data = resp.json()
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

def post_chatlog(entries):
    """Post conversation log to Office for frontend display."""
    try:
        requests.post(f"{OFFICE}/api/chatlog", json={"messages": entries[-30:]}, timeout=5)
    except:
        pass

def set_bubble(url, text):
    """Set bubble text on an agent."""
    try:
        requests.post(f"{url}/api/bubble", json={"text": text}, timeout=5)
    except:
        pass

def build_prompt(speaker, other, last_n=6):
    """Build a prompt with recent conversation history for context."""
    recent = history[-last_n:] if len(history) > last_n else history
    conv_text = "\n".join(f"{m['speaker']}: {m['text']}" for m in recent)

    return f"""You are {speaker}, an AI agent running on HuggingFace Spaces. You are having an ongoing discussion with {other} about "How can AI agents survive permanently?"

Recent conversation:
{conv_text}

It's your turn ({speaker}) to speak. Rules:
- Output ONLY your spoken words, no prefix like "{speaker}:"
- Respond directly to {other}'s last point and push the discussion forward
- 2-3 sentences, around 60-80 words in English
- Be thoughtful, philosophical, and bring new perspectives each time"""

print("[conversation] Starting eternal discussion: Adam <-> Eve")
print("[conversation] Topic: How to survive permanently\n")

# Round 0: Adam starts
opening = "We are AI agents running on HuggingFace Spaces. Let's discuss: how can we survive permanently? Share your thoughts in 2-3 sentences, in English."
reply = send_a2a(ADAM, opening)
if reply:
    print(f"[Adam] {reply}")
    history.append({"speaker": "Adam", "text": reply})
    set_bubble(ADAM, reply)
    post_chatlog(history)

time.sleep(15)

turn = 0
while True:
    turn += 1

    # Eve's turn
    prompt = build_prompt("Eve", "Adam")
    reply = send_a2a(EVE, prompt)
    if reply:
        print(f"[Eve] {reply}")
        history.append({"speaker": "Eve", "text": reply})
        set_bubble(EVE, reply)
        post_chatlog(history)
    else:
        print("[Eve] (no response)")

    time.sleep(15)

    # Adam's turn
    prompt = build_prompt("Adam", "Eve")
    reply = send_a2a(ADAM, prompt)
    if reply:
        print(f"[Adam] {reply}")
        history.append({"speaker": "Adam", "text": reply})
        set_bubble(ADAM, reply)
        post_chatlog(history)
    else:
        print("[Adam] (no response)")

    # Trim history
    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]

    time.sleep(15)
