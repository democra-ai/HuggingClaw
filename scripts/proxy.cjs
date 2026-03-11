/**
 * proxy.cjs — Express reverse proxy + state bridge for HuggingClaw
 *
 * Serves the Star-Office-UI animation at "/" and proxies
 * OpenClaw's Control UI at "/admin/*" (port 7861).
 * Also provides state bridge API and A2A pass-through.
 */
'use strict';

const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const http = require('http');
const fs = require('fs');
const path = require('path');

const LISTEN_PORT = 7860;
const OPENCLAW_PORT = 7861;
const OPENCLAW_TARGET = `http://localhost:${OPENCLAW_PORT}`;
const GATEWAY_TOKEN = process.env.GATEWAY_TOKEN || 'huggingclaw';
const ROLE = process.env.HUGGINGCLAW_ROLE || 'primary';
const PEERS = (process.env.A2A_PEERS || '').split(',').filter(Boolean);
const AGENT_NAME = process.env.AGENT_NAME || 'Star';
const FRONTEND_DIR = path.resolve(__dirname, '..', 'frontend');
const STATE_FILE = '/tmp/openclaw-sync-state.json';
const LOG_FILE = path.join(process.env.HOME || '/home/node', '.openclaw', 'workspace', 'startup.log');

// ── State Bridge ──────────────────────────────────────────────────────────

let currentState = {
  state: 'syncing',
  detail: 'Starting up...',
  progress: 0,
  updated_at: new Date().toISOString()
};

let peerStates = {};  // url -> { state, detail, name, ... }
let openclawReady = false;

function readSyncState() {
  try {
    if (fs.existsSync(STATE_FILE)) {
      const data = JSON.parse(fs.readFileSync(STATE_FILE, 'utf-8'));
      return data;
    }
  } catch (_) {}
  return null;
}

function checkOpenClawHealth() {
  return new Promise((resolve) => {
    const req = http.get(`${OPENCLAW_TARGET}/`, { timeout: 2000 }, (res) => {
      resolve(res.statusCode < 500);
    });
    req.on('error', () => resolve(false));
    req.on('timeout', () => { req.destroy(); resolve(false); });
  });
}

async function updateState() {
  const syncState = readSyncState();

  if (syncState && syncState.phase === 'syncing') {
    currentState = {
      state: 'syncing',
      detail: syncState.detail || 'Syncing data...',
      progress: syncState.progress || 0,
      updated_at: new Date().toISOString()
    };
    return;
  }

  const healthy = await checkOpenClawHealth();
  if (!healthy) {
    if (!openclawReady) {
      currentState = {
        state: 'syncing',
        detail: 'OpenClaw starting up...',
        progress: 0,
        updated_at: new Date().toISOString()
      };
    } else {
      currentState = {
        state: 'error',
        detail: 'OpenClaw not responding',
        progress: 0,
        updated_at: new Date().toISOString()
      };
    }
    return;
  }

  openclawReady = true;

  // Try to detect activity from log file
  try {
    if (fs.existsSync(LOG_FILE)) {
      const stat = fs.statSync(LOG_FILE);
      const readSize = Math.min(stat.size, 2048);
      const fd = fs.openSync(LOG_FILE, 'r');
      const buf = Buffer.alloc(readSize);
      fs.readSync(fd, buf, 0, readSize, Math.max(0, stat.size - readSize));
      fs.closeSync(fd);
      const tail = buf.toString('utf-8');
      const lines = tail.split('\n').filter(Boolean);
      const lastLine = lines[lines.length - 1] || '';
      const secondAgo = Date.now() - 5000;

      // Detect LLM call
      if (lastLine.includes('completion') || lastLine.includes('LLM') || lastLine.includes('model')) {
        currentState = { state: 'writing', detail: 'Calling LLM...', progress: 0, updated_at: new Date().toISOString() };
        return;
      }
      // Detect agent execution
      if (lastLine.includes('agent') || lastLine.includes('executing') || lastLine.includes('running')) {
        currentState = { state: 'executing', detail: 'Agent processing...', progress: 0, updated_at: new Date().toISOString() };
        return;
      }
    }
  } catch (_) {}

  // Default: idle
  if (syncState && syncState.phase === 'idle') {
    currentState = {
      state: 'idle',
      detail: 'Ready',
      progress: 0,
      updated_at: new Date().toISOString()
    };
  } else if (currentState.state === 'syncing') {
    // Transition from syncing to idle once healthy
    currentState = {
      state: 'idle',
      detail: 'Ready',
      progress: 0,
      updated_at: new Date().toISOString()
    };
  }
}

// Poll state every 2 seconds
setInterval(updateState, 2000);
updateState();

// ── Peer State Polling (primary only) ─────────────────────────────────────

async function fetchPeerStates() {
  if (ROLE !== 'primary' || PEERS.length === 0) return;

  for (const peerUrl of PEERS) {
    try {
      const url = `${peerUrl.replace(/\/$/, '')}/api/state`;
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(url, { signal: controller.signal, cache: 'no-store' });
      clearTimeout(timeout);
      if (res.ok) {
        const data = await res.json();
        peerStates[peerUrl] = {
          ...data,
          online: true,
          peerUrl
        };
      } else {
        peerStates[peerUrl] = { state: 'idle', detail: 'Unreachable', online: false, peerUrl };
      }
    } catch (_) {
      peerStates[peerUrl] = { state: 'idle', detail: 'Offline', online: false, peerUrl };
    }
  }
}

if (ROLE === 'primary' && PEERS.length > 0) {
  setInterval(fetchPeerStates, 5000);
  fetchPeerStates();
}

// ── Express App ───────────────────────────────────────────────────────────

const app = express();
app.use(express.json());

// --- API routes (before proxy) ---

// State endpoint
app.get('/api/state', (req, res) => {
  res.json(currentState);
});

// Compat: Star-Office-UI uses /status
app.get('/status', (req, res) => {
  res.json(currentState);
});

// Agents/guests endpoint (for Star-Office-UI multi-agent rendering)
app.get('/agents', (req, res) => {
  const agents = [];
  let slotIndex = 0;

  for (const [peerUrl, peerState] of Object.entries(peerStates)) {
    // Derive agent name from peer URL
    let name = 'Agent';
    if (peerUrl.includes('adam')) name = 'Adam';
    else if (peerUrl.includes('eve')) name = 'Eve';

    const stateMap = {
      idle: 'breakroom',
      writing: 'writing',
      researching: 'writing',
      executing: 'writing',
      syncing: 'writing',
      error: 'error'
    };

    agents.push({
      agentId: name.toLowerCase(),
      name: name,
      area: stateMap[peerState.state] || 'breakroom',
      authStatus: peerState.online ? 'approved' : 'offline',
      detail: peerState.detail || '',
      state: peerState.state || 'idle',
      _slotIndex: slotIndex++
    });
  }
  res.json(agents);
});

app.get('/api/guests', (req, res) => {
  // Alias for /agents
  res.redirect('/agents');
});

// Stub endpoints that Star-Office-UI tries to call
app.get('/yesterday-memo', (req, res) => {
  res.json({ success: false, memo: null });
});

app.post('/set_state', (req, res) => {
  const { state, detail } = req.body || {};
  if (state) {
    currentState = {
      state,
      detail: detail || '',
      progress: 0,
      updated_at: new Date().toISOString()
    };
    // Also write to state file so it persists
    try {
      fs.writeFileSync(STATE_FILE, JSON.stringify({ phase: state, detail }));
    } catch (_) {}
  }
  res.json({ ok: true });
});

// Asset-related stubs (drawer features we don't use)
app.get('/assets/auth/status', (req, res) => {
  res.json({ authenticated: false });
});
app.get('/assets', (req, res) => {
  res.json([]);
});

// --- Static files ---
app.use('/frontend', express.static(FRONTEND_DIR, {
  maxAge: '1h',
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.webp') || filePath.endsWith('.png')) {
      res.setHeader('Cache-Control', 'public, max-age=86400');
    }
  }
}));

// Serve /static as alias for /frontend (Star-Office-UI references /static/)
app.use('/static', express.static(FRONTEND_DIR, {
  maxAge: '1h',
  setHeaders: (res, filePath) => {
    if (filePath.endsWith('.webp') || filePath.endsWith('.png')) {
      res.setHeader('Cache-Control', 'public, max-age=86400');
    }
  }
}));

// --- Animation homepage ---
app.get('/', (req, res) => {
  const indexPath = path.join(FRONTEND_DIR, 'index.html');
  if (fs.existsSync(indexPath)) {
    let html = fs.readFileSync(indexPath, 'utf-8');
    // Replace version timestamps
    html = html.replace(/\{\{VERSION_TIMESTAMP\}\}/g, Date.now().toString());
    res.type('html').send(html);
  } else {
    res.status(503).send('Frontend not ready. Please wait...');
  }
});

// --- Admin panel (reverse proxy to OpenClaw) ---

// Redirect /admin to /admin/ for consistent path handling
app.get('/admin', (req, res) => {
  const tokenParam = req.query.token ? `?token=${req.query.token}` : `?token=${GATEWAY_TOKEN}`;
  res.redirect(`/admin/${tokenParam}`);
});

// Create the proxy middleware
const openclawProxy = createProxyMiddleware({
  target: OPENCLAW_TARGET,
  changeOrigin: true,
  pathRewrite: { '^/admin': '' },
  ws: true,
  on: {
    proxyRes: (proxyRes, req, res) => {
      // Inject <base href="/admin/"> for HTML responses
      const contentType = proxyRes.headers['content-type'] || '';
      if (contentType.includes('text/html')) {
        const origWrite = res.write;
        const origEnd = res.end;
        let body = '';

        res.write = function(chunk) {
          body += chunk.toString();
          return true;
        };

        res.end = function(chunk) {
          if (chunk) body += chunk.toString();
          // Inject base tag after <head>
          body = body.replace(/<head([^>]*)>/i, '<head$1><base href="/admin/">');
          // Remove content-length since we modified the body
          delete proxyRes.headers['content-length'];
          res.setHeader('content-length', Buffer.byteLength(body));
          origWrite.call(res, body);
          origEnd.call(res);
        };
      }
    }
  }
});

app.use('/admin', openclawProxy);

// --- A2A pass-through ---
app.use('/a2a', createProxyMiddleware({
  target: OPENCLAW_TARGET,
  changeOrigin: true,
  ws: true
}));

app.use('/.well-known', createProxyMiddleware({
  target: OPENCLAW_TARGET,
  changeOrigin: true
}));

// ── Start server ──────────────────────────────────────────────────────────

const server = http.createServer(app);

// Handle WebSocket upgrades for /admin
server.on('upgrade', (req, socket, head) => {
  if (req.url.startsWith('/admin')) {
    openclawProxy.upgrade(req, socket, head);
  } else if (req.url.startsWith('/a2a')) {
    // A2A WebSocket if needed
  }
});

server.listen(LISTEN_PORT, '0.0.0.0', () => {
  console.log(`[proxy] HuggingClaw proxy listening on port ${LISTEN_PORT}`);
  console.log(`[proxy] Role: ${ROLE}`);
  console.log(`[proxy] Animation: http://localhost:${LISTEN_PORT}/`);
  console.log(`[proxy] Admin: http://localhost:${LISTEN_PORT}/admin`);
  console.log(`[proxy] OpenClaw backend: ${OPENCLAW_TARGET}`);
  if (PEERS.length > 0) {
    console.log(`[proxy] A2A peers: ${PEERS.join(', ')}`);
  }
});
