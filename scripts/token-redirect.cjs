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
      const prev = remoteAgentStates.get(agent.id) || {};
      remoteAgentStates.set(agent.id, {
        agentId: agent.id, name: agent.name,
        state: data.state || 'idle',
        detail: data.detail || '',
        area: (data.state === 'idle') ? 'breakroom' : (data.state === 'error') ? 'error' : 'writing',
        authStatus: 'approved',
        updated_at: data.updated_at,
        bubbleText: data.bubbleText || prev.bubbleText || '',
        bubbleTextZh: data.bubbleTextZh || prev.bubbleTextZh || ''
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
let currentBubbleText = '';
let currentBubbleTextZh = '';
let chatLog = []; // {speaker, text, text_zh, time}

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

/**
 * A2A Bridge — bypass A2A gateway's scope issue by sending messages
 * directly to OpenClaw via WebSocket (which has proper auth context).
 *
 * Intercepts POST /a2a/jsonrpc with method "message/send",
 * connects to OpenClaw WS on localhost:7860, sends the message,
 * waits for the agent response, and returns it as A2A JSON-RPC.
 */
function handleA2ABridge(req, res) {
  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    let rpc;
    try {
      rpc = JSON.parse(body);
    } catch (e) {
      res.writeHead(400, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ jsonrpc: '2.0', id: null, error: { code: -32700, message: 'Parse error' } }));
      return;
    }

    // Only handle message/send — forward everything else to A2A gateway
    if (rpc.method !== 'message/send') {
      // Re-create request to A2A gateway
      const options = {
        hostname: '127.0.0.1', port: A2A_PORT,
        path: req.url, method: 'POST',
        headers: { 'Content-Type': 'application/json', host: `127.0.0.1:${A2A_PORT}` }
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
      proxy.end(body);
      return;
    }

    const msgParts = (rpc.params && rpc.params.message && rpc.params.message.parts) || [];
    const messageText = msgParts.map(p => p.text || '').join('\n').trim();
    const messageId = (rpc.params && rpc.params.message && rpc.params.message.messageId) || '';

    if (!messageText) {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        jsonrpc: '2.0', id: rpc.id,
        error: { code: -32602, message: 'Empty message text' }
      }));
      return;
    }

    // Connect to OpenClaw via WebSocket
    const WebSocket = (() => {
      try { return require('ws'); } catch (e) { return null; }
    })();

    if (!WebSocket) {
      // Fallback: try native fetch to OpenClaw HTTP API
      console.log('[a2a-bridge] ws module not available, trying HTTP fallback');
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        jsonrpc: '2.0', id: rpc.id,
        error: { code: -32603, message: 'WebSocket module not available' }
      }));
      return;
    }

    const wsUrl = `ws://127.0.0.1:7860/?token=${GATEWAY_TOKEN}`;
    const ws = new WebSocket(wsUrl);
    let responded = false;
    let agentText = '';
    const timeout = setTimeout(() => {
      if (!responded) {
        responded = true;
        ws.close();
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          jsonrpc: '2.0', id: rpc.id,
          error: { code: -32000, message: 'Agent response timeout' }
        }));
      }
    }, 120000); // 2 minute timeout

    ws.on('open', () => {
      // Send message via OpenClaw's RPC protocol
      ws.send(JSON.stringify({
        type: 'rpc',
        method: 'sessions.send',
        params: {
          agentId: 'main',
          text: messageText,
        },
        id: 'a2a-' + Date.now()
      }));
    });

    ws.on('message', (data) => {
      try {
        const msg = JSON.parse(data.toString());

        // Collect agent text responses
        if (msg.type === 'event' && msg.event === 'agent.message') {
          const text = (msg.data && msg.data.text) || '';
          if (text) agentText += text;
        }

        // Agent finished responding
        if (msg.type === 'event' && (msg.event === 'agent.done' || msg.event === 'session.done' || msg.event === 'turn.done')) {
          if (!responded) {
            responded = true;
            clearTimeout(timeout);
            ws.close();
            sendA2AResponse(res, rpc.id, messageId, agentText || '(no response)');
          }
        }

        // RPC response (might contain the reply directly)
        if (msg.type === 'rpc_response' || msg.type === 'rpc-response') {
          if (msg.result && typeof msg.result === 'object') {
            const text = msg.result.text || msg.result.message || '';
            if (text && !responded) {
              responded = true;
              clearTimeout(timeout);
              ws.close();
              sendA2AResponse(res, rpc.id, messageId, text);
            }
          }
        }

        // Error response
        if (msg.type === 'error' || (msg.error && !responded)) {
          const errMsg = (msg.error && msg.error.message) || msg.message || 'Unknown error';
          console.log(`[a2a-bridge] WS error: ${errMsg}`);
          // Don't immediately fail — wait for agent text that may have already arrived
        }
      } catch (e) {
        // Non-JSON message, ignore
      }
    });

    ws.on('close', () => {
      if (!responded) {
        responded = true;
        clearTimeout(timeout);
        if (agentText) {
          sendA2AResponse(res, rpc.id, messageId, agentText);
        } else {
          res.writeHead(200, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({
            jsonrpc: '2.0', id: rpc.id,
            error: { code: -32000, message: 'WebSocket closed without response' }
          }));
        }
      }
    });

    ws.on('error', (err) => {
      console.log(`[a2a-bridge] WS error: ${err.message}`);
      if (!responded) {
        responded = true;
        clearTimeout(timeout);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
          jsonrpc: '2.0', id: rpc.id,
          error: { code: -32000, message: `WebSocket error: ${err.message}` }
        }));
      }
    });
  });
}

function sendA2AResponse(res, rpcId, messageId, text) {
  const respMsgId = require('crypto').randomUUID();
  res.writeHead(200, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify({
    jsonrpc: '2.0',
    id: rpcId,
    result: {
      kind: 'task',
      id: require('crypto').randomUUID(),
      status: {
        state: 'completed',
        message: {
          kind: 'message',
          messageId: respMsgId,
          role: 'agent',
          parts: [{ kind: 'text', text: text }]
        },
        timestamp: new Date().toISOString()
      }
    }
  }));
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

    // A2A routes
    if (pathname.startsWith('/.well-known/')) {
      proxyToA2A(req, res);
      return true;
    }
    if (pathname.startsWith('/a2a/')) {
      // POST /a2a/jsonrpc → use bridge (bypasses scope issue)
      if (req.method === 'POST' && pathname === '/a2a/jsonrpc') {
        handleA2ABridge(req, res);
        return true;
      }
      // Everything else (GET agent-card etc) → A2A gateway
      proxyToA2A(req, res);
      return true;
    }

    // /api/state → return God's remote state (God is the star/main character)
    if (pathname === '/api/state' || pathname === '/status') {
      const godState = remoteAgentStates.get('god');
      const state = godState
        ? { state: godState.state || 'idle', detail: godState.detail || 'God is watching',
            progress: 100, updated_at: godState.updated_at || new Date().toISOString() }
        : currentState;
      // Fall back to local state while God hasn't been polled yet
      if (!godState && currentState.state === 'syncing') {
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
        ...state,
        bubbleText: (godState && godState.bubbleText) || currentBubbleText,
        bubbleTextZh: (godState && godState.bubbleTextZh) || currentBubbleTextZh,
        officeName: `${AGENT_NAME}'s Home`
      }));
      return true;
    }

    // GET /api/conv-loop-log → tail conversation-loop log for remote diagnostics
    if (pathname === '/api/conv-loop-log' && req.method === 'GET') {
      const logPath = '/tmp/conversation-loop.log';
      fs.readFile(logPath, 'utf8', (err, data) => {
        const lines = err ? [`No log file: ${err.code}`] : data.split('\n').slice(-50);
        res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
        res.end(JSON.stringify({ lines, officeMode: OFFICE_MODE }));
      });
      return true;
    }

    // POST /api/bubble → set bubble text (used by conversation orchestrator)
    if (pathname === '/api/bubble' && req.method === 'POST') {
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => {
        try {
          const { text, text_zh } = JSON.parse(body);
          currentBubbleText = text || '';
          currentBubbleTextZh = text_zh || text || '';
          // Auto-clear bubble after 8 seconds
          const clearText = text;
          setTimeout(() => { if (currentBubbleText === clearText) { currentBubbleText = ''; currentBubbleTextZh = ''; } }, 8000);
          res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
          res.end(JSON.stringify({ ok: true }));
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: e.message }));
        }
      });
      return true;
    }

    // GET /api/chatlog → return conversation log
    if (pathname === '/api/chatlog' && req.method === 'GET') {
      res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
      res.end(JSON.stringify({ messages: chatLog }));
      return true;
    }

    // POST /api/chatlog → update conversation log (from orchestrator)
    if (pathname === '/api/chatlog' && req.method === 'POST') {
      let body = '';
      req.on('data', chunk => body += chunk);
      req.on('end', () => {
        try {
          const { messages } = JSON.parse(body);
          if (Array.isArray(messages)) {
            chatLog = messages.slice(-50); // keep last 50 messages
          }
          res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
          res.end(JSON.stringify({ ok: true }));
        } catch (e) {
          res.writeHead(400, { 'Content-Type': 'application/json' });
          res.end(JSON.stringify({ ok: false, error: e.message }));
        }
      });
      return true;
    }

    // /agents → return remote agent list (exclude God — God is the star/main character)
    if (pathname === '/agents' && req.method === 'GET') {
      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': '*'
      });
      const agents = [...remoteAgentStates.values()].filter(a => a.agentId !== 'god');
      res.end(JSON.stringify(agents));
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
      // Default mode: 302 redirect to inject token into browser URL
      // (must be a redirect, not a rewrite, so frontend JS can read the token)
      if (req.method === 'GET' && !req.headers.upgrade) {
        if (pathname === '/' && !parsed.query.token) {
          res.writeHead(302, { Location: `/?token=${GATEWAY_TOKEN}` });
          res.end();
          return true;
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
