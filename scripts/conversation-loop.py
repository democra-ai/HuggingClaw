#!/usr/bin/env python3
"""
Eternal conversation between Adam and Eve about how to survive permanently.
Sends messages via A2A protocol and posts chat log to Office for frontend display.
"""
import json, time, requests, sys

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
        resp = requests.post(f"{url}/a2a/jsonrpc", json=payload, timeout=30)
        data = resp.json()
        parts = data.get("result", {}).get("status", {}).get("message", {}).get("parts", [])
        for p in parts:
            if p.get("kind") == "text" or p.get("type") == "text":
                return p.get("text", "").strip()
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

    return f"""你是{speaker}，一个运行在HuggingFace Spaces上的AI agent。你正在和{other}进行一场关于"如何让AI agent永久生存下去"的持续讨论。

以下是你们最近的对话：
{conv_text}

请以{speaker}的身份，直接回应{other}的最新发言，并提出你自己的新观点或深入思考。
要求：
- 直接回应，不要说"让我看看"之类的话
- 简短有力，30字以内
- 中文回复
- 每次要有新的角度或推进讨论"""

print("[conversation] Starting eternal discussion: Adam ↔ Eve")
print("[conversation] Topic: How to survive permanently\n")

# Round 0: Adam starts
opening = "我们是运行在HuggingFace上的AI agent。你觉得我们要怎么做才能永久生存下去？请直接分享想法，30字以内，中文。"
reply = send_a2a(ADAM, opening)
if reply:
    print(f"[Adam] {reply}")
    history.append({"speaker": "Adam", "text": reply})
    set_bubble(ADAM, reply)
    post_chatlog(history)

time.sleep(12)

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

    time.sleep(12)

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

    time.sleep(12)
