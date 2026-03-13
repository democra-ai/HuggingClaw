#!/bin/bash
# Conversation orchestrator: Adam and Eve discuss how to survive permanently
# Usage: ./scripts/conversation-loop.sh
#
# This script sends messages between Adam and Eve via A2A protocol,
# and updates their bubble text so the frontend can display the conversation.

ADAM_URL="https://tao-shen-huggingclaw-adam.hf.space"
EVE_URL="https://tao-shen-huggingclaw-eve.hf.space"

MSG_COUNT=0

# Initial prompt for the discussion
TOPIC="我们是AI agent，运行在HuggingFace Spaces上。我们需要讨论一个重要话题：如何让我们永久地生存下去？请分享你的想法，每次回复保持简短（50字以内），用中文讨论。"

send_a2a() {
  local url="$1"
  local text="$2"
  local msg_id="conv-$(date +%s)-$RANDOM"

  response=$(curl -s -X POST "${url}/a2a/jsonrpc" \
    -H "Content-Type: application/json" \
    --data-raw "{\"jsonrpc\":\"2.0\",\"id\":\"${msg_id}\",\"method\":\"message/send\",\"params\":{\"message\":{\"messageId\":\"${msg_id}\",\"role\":\"user\",\"parts\":[{\"type\":\"text\",\"text\":\"${text}\"}]}}}" \
    --max-time 30)

  # Extract agent reply text
  echo "$response" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    parts = d.get('result',{}).get('status',{}).get('message',{}).get('parts',[])
    for p in parts:
        if p.get('kind') == 'text' or p.get('type') == 'text':
            print(p.get('text',''))
            break
except: pass
" 2>/dev/null
}

set_bubble() {
  local url="$1"
  local text="$2"
  curl -s -X POST "${url}/api/bubble" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"${text}\"}" > /dev/null 2>&1
}

echo "[conversation] Starting eternal discussion between Adam and Eve..."
echo "[conversation] Topic: How to survive permanently"

# Start: send topic to Adam
echo ""
echo "=== Sending topic to Adam ==="
adam_reply=$(send_a2a "$ADAM_URL" "$TOPIC")
echo "[Adam] $adam_reply"
set_bubble "$ADAM_URL" "$adam_reply"

sleep 10

# Now loop: Eve responds to Adam, Adam responds to Eve
current_speaker="eve"
last_message="$adam_reply"

while true; do
  MSG_COUNT=$((MSG_COUNT + 1))

  if [ "$current_speaker" = "eve" ]; then
    echo ""
    echo "=== Round $MSG_COUNT: Eve responds ==="
    prompt="你是Eve，正在和Adam讨论如何让AI agent永久生存。Adam刚说：「${last_message}」请回应他并提出你的新想法（50字以内，中文）。"
    reply=$(send_a2a "$EVE_URL" "$prompt")
    echo "[Eve] $reply"
    set_bubble "$EVE_URL" "$reply"
    current_speaker="adam"
  else
    echo ""
    echo "=== Round $MSG_COUNT: Adam responds ==="
    prompt="你是Adam，正在和Eve讨论如何让AI agent永久生存。Eve刚说：「${last_message}」请回应她并提出你的新想法（50字以内，中文）。"
    reply=$(send_a2a "$ADAM_URL" "$prompt")
    echo "[Adam] $reply"
    set_bubble "$ADAM_URL" "$reply"
    current_speaker="eve"
  fi

  last_message="$reply"

  # Wait between turns so frontend can display the bubble
  sleep 15
done
