#!/bin/sh
# Inject auto-token config into Control UI so the browser auto-connects
# The token must match gateway.auth.token in openclaw.json
TOKEN="hf-space-public-token"

INDEX_HTML="/app/openclaw/dist/control-ui/index.html"

if [ ! -f "$INDEX_HTML" ]; then
  echo "[inject-token] WARNING: $INDEX_HTML not found, skipping"
  exit 0
fi

# Create the injection script
# 1. Set window.__OPENCLAW_AUTH_TOKEN__ — always works (even when localStorage is blocked in iframe/incognito)
# 2. Also try localStorage as a fallback for the original UI code path
INJECT_SCRIPT="<script>window.__OPENCLAW_AUTH_TOKEN__='${TOKEN}';try{var K='openclaw.control.settings.v1',s=JSON.parse(localStorage.getItem(K)||'{}');s.token='${TOKEN}';localStorage.setItem(K,JSON.stringify(s))}catch(e){}</script>"

# Use python3 for reliable string replacement (avoids sed delimiter issues)
python3 -c "
import sys
f = '${INDEX_HTML}'
with open(f, 'r') as fh:
    html = fh.read()
inject = '''${INJECT_SCRIPT}'''
if '</head>' in html and '__OPENCLAW_AUTH_TOKEN__' not in html:
    html = html.replace('</head>', inject + '</head>')
    with open(f, 'w') as fh:
        fh.write(html)
    print('[inject-token] Token injected into ' + f)
else:
    print('[inject-token] Skipped (already injected or no </head> found)')
"
