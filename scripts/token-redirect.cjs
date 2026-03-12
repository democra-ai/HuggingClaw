/**
 * token-redirect.cjs — Node.js preload script (enhanced)
 *
 * Loaded via NODE_OPTIONS --require before OpenClaw starts.
 * Intercepts OpenClaw's HTTP server to:
 *   1. Redirect GET / to /?token=GATEWAY_TOKEN (auto-fill token)
 *   2. Proxy A2A requests (/.well-known/*, /a2a/*) to gateway port 18800
 *   3. Serve /api/state and /agents for Office frontends
 *   4. Fix iframe embedding (strip X-Frame-Options, fix CSP)
 *   5. Serve Office frontend when OFFICE_MODE=1
 */
'use strict';

const http = require('http');
const url = require('url');
const fs = require('fs');
const path = require('path');

const GATEWAY_TOKEN = process.env.GATEWAY_TOKEN || 'huggingclaw';
const AGENT_NAME = process.env.AGENT_NAME || 'HuggingClaw';
const A2A_PORT = 18800;
const OFFICE_MODE = process.env.OFFICE_MODE === '1';

// Frontend directory for Office mode
const FRONTEND_DIR = fs.existsSync('/home/node/frontend')
  ? '/home/node/frontend'
  : path.join(__dirname, '..', 'frontend');

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript',
  '.css': 'text/css',
  '.json': 'application/json',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.webp': 'image/webp',
  '.gif': 'image/gif',
  '.svg': 'image/svg+xml',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.ttf': 'font/ttf',
  '.ico': 'image/x-icon',
  '.mp3': 'audio/mpeg',
  '.ogg': 'audio/ogg',
  '.md': 'text/markdown; charset=utf-8',
};

function serveStaticFile(res, filePath) {
  const resolved = path.resolve(filePath);
  if (!resolved.startsWith(path.resolve(FRONTEND_DIR))) {
    res.writeHead(403);
    return res.end('Forbidden');
  }
  fs.readFile(resolved, (err, data) => {
    if (err) {
      res.writeHead(404, { 'Content-Type': 'text/plain' });
      return res.end('Not Found');
    }
    const ext = path.extname(resolved).toLowerCase();
    const contentType = MIME_TYPES[ext] || 'application/octet-stream';
    res.writeHead(200, {
      'Content-Type': contentType,
      'Cache-Control': (ext === '.html') ? 'no-cache' : 'public, max-age=86400',
      'Access-Control-Allow-Origin': '*'
    });
    res.end(data);
  });
}

// Remote agents polling
const REMOTE_AGENTS_RAW = process.env.REMOTE_AGENTS || '';
const remoteAgents = REMOTE_AGENTS_RAW
  ? REMOTE_AGENTS_RAW.split(',').map(entry => {
      const [id, name, baseUrl] = entry.trim().split('|');
      return { id, name, baseUrl };
    }).filter(a => a.id && a.name && a.baseUrl)
  : [];

const remoteAgentStates = new Map();

async function pollRemoteAgent(agent) {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 5000);
    const resp = await fetch(`${agent.baseUrl}/api/state`, { signal: controller.signal });
    clearTimeout(timeout);
    if (resp.ok) {
      const data = await resp.json();
      remoteAgentStates.set(agent.id, {
        agentId: agent.id, name: agent.name,
        state: data.state || 'idle',
        detail: data.detail || '',
        area: (data.state === 'idle') ? 'breakroom' : (data.state === 'error') ? 'error' : 'writing',
        authStatus: 'approved',
        updated_at: data.updated_at
      });
    }
  } catch (_) {
    if (!remoteAgentStates.has(agent.id)) {
      remoteAgentStates.set(agent.id, {
        agentId: agent.id, name: agent.name,
        state: 'syncing', detail: `${agent.name} is starting...`,
        area: 'door', authStatus: 'approved'
      });
    }
  }
}

if (remoteAgents.length > 0) {
  setInterval(() => remoteAgents.forEach(a => pollRemoteAgent(a)), 5000);
  remoteAgents.forEach(a => pollRemoteAgent(a));
  console.log(`[token-redirect] Monitoring ${remoteAgents.length} remote agent(s)`);
}

// State tracking
let currentState = {
  state: 'syncing', detail: `${AGENT_NAME} is starting...`,
  progress: 0, updated_at: new Date().toISOString()
};

// Once OpenClaw starts listening, mark as idle
setTimeout(() => {
  if (currentState.state === 'syncing') {
    currentState = {
      state: 'idle', detail: `${AGENT_NAME} is running`,
      progress: 100, updated_at: new Date().toISOString()
    };
  }
}, 30000);

function proxyToA2A(req, res) {
  const options = {
    hostname: '127.0.0.1', port: A2A_PORT,
    path: req.url, method: req.method,
    headers: { ...req.headers, host: `127.0.0.1:${A2A_PORT}` }
  };
  const proxy = http.request(options, (proxyRes) => {
    res.writeHead(proxyRes.statusCode, proxyRes.headers);
    proxyRes.pipe(res, { end: true });
  });
  proxy.on('error', () => {
    if (!res.headersSent) {
      res.writeHead(502, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ error: 'A2A gateway unavailable' }));
    }
  });
  req.pipe(proxy, { end: true });
}

const origEmit = http.Server.prototype.emit;

http.Server.prototype.emit = function (event, ...args) {
  if (event === 'request') {
    const [req, res] = args;

    // Only intercept on the main OpenClaw server (port 7860), not A2A gateway (18800)
    const serverPort = this.address && this.address() && this.address().port;
    if (serverPort && serverPort !== 7860) {
      return origEmit.apply(this, [event, ...args]);
    }

    // Fix iframe embedding — must be applied BEFORE any early returns
    const origWriteHead = res.writeHead;
    res.writeHead = function (statusCode, ...whArgs) {
      if (res.getHeader) {
        res.removeHeader('x-frame-options');
        const csp = res.getHeader('content-security-policy');
        if (csp && typeof csp === 'string') {
          res.setHeader('content-security-policy',
            csp.replace(/frame-ancestors\s+'none'/i,
              "frame-ancestors 'self' https://huggingface.co https://*.hf.space"));
        }
      }
      return origWriteHead.apply(this, [statusCode, ...whArgs]);
    };

    const parsed = url.parse(req.url, true);
    const pathname = parsed.pathname;

    // A2A routes → proxy to A2A gateway on 18800
    if (pathname.startsWith('/.well-known/') || pathname.startsWith('/a2a/')) {
      proxyToA2A(req, res);
      return true;
    }

    // /api/state → return current state
    if (pathname === '/api/state' || pathname === '/status') {
      // Update state to idle once we're handling requests
      if (currentState.state === 'syncing') {
        currentState = {
          state: 'idle', detail: `${AGENT_NAME} is running`,
          progress: 100, updated_at: new Date().toISOString()
        };
      }
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      });
      res.end(JSON.stringify({
        ...currentState,
        officeName: `${AGENT_NAME}'s Office`
      }));
      return true;
    }

    // /agents → return remote agent list
    if (pathname === '/agents' && req.method === 'GET') {
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      });
      res.end(JSON.stringify([...remoteAgentStates.values()]));
      return true;
    }

    // Office mode: serve frontend at /, static at /static/*, admin proxies to OpenClaw
    if (OFFICE_MODE) {
      if (pathname === '/' && req.method === 'GET' && !req.headers.upgrade) {
        serveStaticFile(res, path.join(FRONTEND_DIR, 'index.html'));
        return true;
      }
      if (pathname.startsWith('/static/')) {
        serveStaticFile(res, path.join(FRONTEND_DIR, pathname.slice('/static/'.length).split('?')[0]));
        return true;
      }
      if (pathname === '/admin' || pathname === '/admin/') {
        // Rewrite to root with token and let OpenClaw handle it
        req.url = GATEWAY_TOKEN ? `/?token=${GATEWAY_TOKEN}` : '/';
        return origEmit.apply(this, [event, ...args]);
      }
    } else {
      // Default mode: rewrite URL to inject token (no redirect needed)
      if (req.method === 'GET' && !req.headers.upgrade) {
        if (pathname === '/' && !parsed.query.token) {
          req.url = `/?token=${GATEWAY_TOKEN}`;
          return origEmit.apply(this, [event, ...args]);
        }
      }
    }
  }

  return origEmit.apply(this, [event, ...args]);
};

// Also handle WebSocket upgrades for A2A
const origServerEmit = http.Server.prototype.emit;
// Already patched above, A2A WS upgrades handled via 'upgrade' event in OpenClaw

console.log(`[token-redirect] Active: token=${GATEWAY_TOKEN}, agent=${AGENT_NAME}, office=${OFFICE_MODE}`);
