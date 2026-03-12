/**
 * a2a-proxy.cjs — Reverse proxy on port 7860
 *
 * Routes:
 *   /.well-known/*  → A2A gateway (port 18800)
 *   /a2a/*          → A2A gateway (port 18800)
 *   /api/state      → local state JSON (for Office frontend polling)
 *   /agents         → merged agent list (OpenClaw + remote agents)
 *   everything else → OpenClaw (port 7861)
 */
'use strict';

const http = require('http');
const url = require('url');

const LISTEN_PORT = 7860;
const OPENCLAW_PORT = 7861;
const A2A_PORT = 18800;
const AGENT_NAME = process.env.AGENT_NAME || 'Agent';

// Remote agents to monitor (comma-separated URLs)
// e.g. REMOTE_AGENTS=adam|Adam|https://tao-shen-huggingclaw-adam.hf.space,eve|Eve|https://tao-shen-huggingclaw-eve.hf.space
const REMOTE_AGENTS_RAW = process.env.REMOTE_AGENTS || '';
const remoteAgents = REMOTE_AGENTS_RAW
  ? REMOTE_AGENTS_RAW.split(',').map(entry => {
      const [id, name, baseUrl] = entry.trim().split('|');
      return { id, name, baseUrl };
    }).filter(a => a.id && a.name && a.baseUrl)
  : [];

let currentState = {
  state: 'syncing',
  detail: `${AGENT_NAME} is starting...`,
  progress: 0,
  updated_at: new Date().toISOString()
};

// Track A2A activity — when an A2A message is being processed,
// temporarily switch state to 'writing' so frontends can see it
let a2aActiveRequests = 0;
let a2aIdleTimer = null;
const A2A_IDLE_DELAY = 8000; // stay "writing" for 8s after last A2A request ends

function markA2AActive() {
  a2aActiveRequests++;
  if (a2aIdleTimer) { clearTimeout(a2aIdleTimer); a2aIdleTimer = null; }
  currentState = {
    state: 'writing',
    detail: `${AGENT_NAME} is communicating...`,
    progress: 100,
    updated_at: new Date().toISOString()
  };
}

function markA2ADone() {
  a2aActiveRequests = Math.max(0, a2aActiveRequests - 1);
  if (a2aActiveRequests === 0) {
    if (a2aIdleTimer) clearTimeout(a2aIdleTimer);
    a2aIdleTimer = setTimeout(() => {
      a2aIdleTimer = null;
      pollOpenClawHealth();
    }, A2A_IDLE_DELAY);
  }
}

// Remote agent states (polled periodically)
const remoteAgentStates = new Map();

async function pollRemoteAgent(agent) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const resp = await fetch(`${agent.baseUrl}/api/state`, {
      signal: controller.signal
    });
    clearTimeout(timeout);
    if (resp.ok) {
      const data = await resp.json();
      remoteAgentStates.set(agent.id, {
        agentId: agent.id,
        name: agent.name,
        state: data.state || 'idle',
        detail: data.detail || '',
        area: (data.state === 'idle') ? 'breakroom'
            : (data.state === 'error') ? 'error'
            : 'writing',
        authStatus: 'approved',
        updated_at: data.updated_at
      });
    }
  } catch (_) {
    // Keep last known state or mark as offline
    if (!remoteAgentStates.has(agent.id)) {
      remoteAgentStates.set(agent.id, {
        agentId: agent.id,
        name: agent.name,
        state: 'syncing',
        detail: `${agent.name} is starting...`,
        area: 'door',
        authStatus: 'approved'
      });
    }
  }
}

function pollAllRemoteAgents() {
  for (const agent of remoteAgents) {
    pollRemoteAgent(agent);
  }
}

if (remoteAgents.length > 0) {
  setInterval(pollAllRemoteAgents, 5000);
  pollAllRemoteAgents();
  console.log(`[a2a-proxy] Monitoring ${remoteAgents.length} remote agent(s): ${remoteAgents.map(a => a.name).join(', ')}`);
}

// Poll OpenClaw health to track state
async function pollOpenClawHealth() {
  // Don't overwrite active A2A state
  if (a2aActiveRequests > 0 || a2aIdleTimer) return;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const resp = await fetch(`http://127.0.0.1:${OPENCLAW_PORT}/`, {
      signal: controller.signal
    });
    clearTimeout(timeout);
    currentState = {
      state: resp.ok ? 'idle' : 'error',
      detail: resp.ok ? `${AGENT_NAME} is running` : `HTTP ${resp.status}`,
      progress: resp.ok ? 100 : 0,
      updated_at: new Date().toISOString()
    };
  } catch (_) {
    currentState = {
      state: 'syncing',
      detail: `${AGENT_NAME} is starting...`,
      progress: 0,
      updated_at: new Date().toISOString()
    };
  }
}

setInterval(pollOpenClawHealth, 5000);
pollOpenClawHealth();

// Fetch agents from OpenClaw and merge with remote agents
async function getMergedAgents() {
  let openClawAgents = [];
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const resp = await fetch(`http://127.0.0.1:${OPENCLAW_PORT}/agents`, {
      signal: controller.signal
    });
    clearTimeout(timeout);
    if (resp.ok) {
      openClawAgents = await resp.json();
      if (!Array.isArray(openClawAgents)) openClawAgents = [];
    }
  } catch (_) {}

  // Merge: OpenClaw agents + remote agents (deduplicate by agentId)
  const existingIds = new Set(openClawAgents.map(a => a.agentId));
  const merged = [...openClawAgents];
  let slotIndex = openClawAgents.length;
  for (const [id, agentState] of remoteAgentStates) {
    if (!existingIds.has(id)) {
      merged.push({ ...agentState, _slotIndex: slotIndex++ });
    }
  }
  return merged;
}

function proxyRequest(req, res, targetPort) {
  const options = {
    hostname: '127.0.0.1',
    port: targetPort,
    path: req.url,
    method: req.method,
    headers: { ...req.headers, host: `127.0.0.1:${targetPort}` }
  };

  const proxy = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });

  proxy.on('error', (err) => {
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'Backend unavailable', target: targetPort }));
    }
  });

  req.pipe(proxy, { end: true });
}

const server = http.createServer((req, res) => {
  const pathname = url.parse(req.url).pathname;

  // A2A routes → A2A gateway
  if (pathname.startsWith('/.well-known/') || pathname.startsWith('/a2a/')) {
    // Track POST requests (message/send) as active communication
    if (req.method === 'POST') {
      markA2AActive();
      res.on('finish', markA2ADone);
    }
    return proxyRequest(req, res, A2A_PORT);
  }

  // State endpoint for Office frontend polling
  if (pathname === '/api/state' || pathname === '/status') {
    res.writeHead(200, {
      'Content-Type': 'application/json',
      'Access-Control-Allow-Origin': '*'
    });
    return res.end(JSON.stringify(currentState));
  }

  // Agents endpoint — merge OpenClaw agents with remote agents
  if (pathname === '/agents' && req.method === 'GET') {
    getMergedAgents().then(agents => {
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      });
      res.end(JSON.stringify(agents));
    }).catch(() => {
      // Fallback: just return remote agents
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      });
      res.end(JSON.stringify([...remoteAgentStates.values()]));
    });
    return;
  }

  // Everything else → OpenClaw
  proxyRequest(req, res, OPENCLAW_PORT);
});

// Handle WebSocket upgrades
server.on('upgrade', (req, socket, head) => {
  const pathname = url.parse(req.url).pathname;
  const targetPort = (pathname.startsWith('/.well-known/') || pathname.startsWith('/a2a/'))
    ? A2A_PORT
    : OPENCLAW_PORT;

  const options = {
    hostname: '127.0.0.1',
    port: targetPort,
    path: req.url,
    method: req.method,
    headers: { ...req.headers, host: `127.0.0.1:${targetPort}` }
  };

  const proxy = http.request(options);
  proxy.on('upgrade', (proxyRes, proxySocket, proxyHead) => {
    socket.write(
      `HTTP/1.1 101 Switching Protocols\r\n` +
      Object.entries(proxyRes.headers).map(([k, v]) => `${k}: ${v}`).join('\r\n') +
      '\r\n\r\n'
    );
    proxySocket.write(head);
    proxySocket.pipe(socket);
    socket.pipe(proxySocket);
  });
  proxy.on('error', () => socket.end());
  proxy.end();
});

server.listen(LISTEN_PORT, '0.0.0.0', () => {
  console.log(`[a2a-proxy] Listening on port ${LISTEN_PORT}`);
  console.log(`[a2a-proxy] OpenClaw → :${OPENCLAW_PORT}, A2A → :${A2A_PORT}`);
  console.log(`[a2a-proxy] Agent: ${AGENT_NAME}`);
});
