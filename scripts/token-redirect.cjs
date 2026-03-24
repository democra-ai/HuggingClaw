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

    // Connect to OpenClaw gateway via raw WebSocket and use the "agent" RPC
    // method (same as the a2a-gateway plugin uses internally).
    const crypto = require('crypto');
    const wsKey = crypto.randomBytes(16).toString('base64');
    let responded = false;
    let agentText = '';
    let challengeNonce = '';
    let wsSocket = null;
    const contextId = rpc.params?.message?.messageId || crypto.randomUUID();
    const sessionKey = `agent:main:a2a:${contextId}`;

    const wsTimeout = setTimeout(() => {
      if (!responded) {
        responded = true;
        try { if (wsSocket) wsSocket.destroy(); } catch (_) {}
        sendA2AResponse(res, rpc.id, messageId, agentText || '(timeout)');
      }
    }, 180000);

    function finish(text) {
      if (responded) return;
      responded = true;
      clearTimeout(wsTimeout);
      try { if (wsSocket) wsSocket.destroy(); } catch (_) {}
      sendA2AResponse(res, rpc.id, messageId, text || '(no response)');
    }

    function finishError(msg) {
      if (responded) return;
      responded = true;
      clearTimeout(wsTimeout);
      try { if (wsSocket) wsSocket.destroy(); } catch (_) {}
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({ jsonrpc: '2.0', id: rpc.id, error: { code: -32000, message: msg } }));
    }

    // Connect to OpenClaw's internal gateway port.
    // token-redirect.cjs hooks http.Server.prototype.emit('request') but NOT
    // 'upgrade', so WebSocket upgrades to 7860 reach OpenClaw directly.
    // However, OpenClaw may close with 1002 if missing required headers.
    const wsReq = http.request({
      hostname: '127.0.0.1', port: 7860,
      path: `/?token=${GATEWAY_TOKEN}`,
      method: 'GET',
      headers: {
        'Upgrade': 'websocket',
        'Connection': 'Upgrade',
        'Sec-WebSocket-Key': wsKey,
        'Sec-WebSocket-Version': '13',
        'Origin': 'http://127.0.0.1:7860',
      }
    });

    wsReq.on('upgrade', (upgradeRes, socket, head) => {
      wsSocket = socket;
      console.log(`[a2a-bridge] WS connected, head=${head.length} bytes`);

      let frameBuf = Buffer.alloc(0);

      socket.on('data', (chunk) => {
        frameBuf = Buffer.concat([frameBuf, chunk]);
        // Debug: log raw data size
        if (frameBuf.length > 0 && frameBuf.length < 500) {
          console.log(`[a2a-bridge] Raw data: len=${frameBuf.length} first_bytes=[${frameBuf[0]},${frameBuf[1]}] opcode=${frameBuf[0] & 0x0f}`);
        }
        while (frameBuf.length >= 2) {
          const fin = (frameBuf[0] & 0x80) !== 0;
          const opcode = frameBuf[0] & 0x0f;
          const masked = (frameBuf[1] & 0x80) !== 0;
          let payloadLen = frameBuf[1] & 0x7f;
          let offset = 2;
          if (payloadLen === 126) {
            if (frameBuf.length < 4) return;
            payloadLen = frameBuf.readUInt16BE(2); offset = 4;
          } else if (payloadLen === 127) {
            if (frameBuf.length < 10) return;
            payloadLen = Number(frameBuf.readBigUInt64BE(2)); offset = 10;
          }
          if (masked) offset += 4;
          if (frameBuf.length < offset + payloadLen) return;

          let payload = frameBuf.slice(offset, offset + payloadLen);
          if (masked) {
            const mask = frameBuf.slice(offset - 4, offset);
            for (let i = 0; i < payload.length; i++) payload[i] ^= mask[i % 4];
          }
          frameBuf = frameBuf.slice(offset + payloadLen);

          if (opcode === 0x08 && payloadLen >= 2) {
            const closeCode = payload.readUInt16BE(0);
            const closeReason = payloadLen > 2 ? payload.slice(2).toString('utf8') : '';
            console.log(`[a2a-bridge] WS Close: code=${closeCode} reason=${closeReason}`);
          } else {
            console.log(`[a2a-bridge] Frame: opcode=${opcode} fin=${fin} len=${payloadLen} preview=${payload.toString('utf8').slice(0, 200)}`);
          }

          if (opcode === 0x01) {
            try { handleWsMessage(JSON.parse(payload.toString('utf8'))); }
            catch (e) { console.log(`[a2a-bridge] JSON parse error: ${e.message}`); }
          } else if (opcode === 0x08) { socket.end(); }
          else if (opcode === 0x09) { wsSend(socket, '', 0x0a); }
        }
      });

      socket.on('close', () => finish(agentText));
      socket.on('error', (err) => finishError(`WS error: ${err.message}`));

      // Process head buffer AFTER data handler is registered
      if (head && head.length > 0) {
        socket.emit('data', head);
      }
    });

    wsReq.on('error', (err) => finishError(`WS connect failed: ${err.message}`));
    wsReq.end();

    function handleWsMessage(msg) {
      console.log(`[a2a-bridge] WS msg: type=${msg.type} ${msg.event || msg.method || msg.id || ''} ok=${msg.ok} keys=${Object.keys(msg).join(',')}`);

      // Step 1: Wait for connect.challenge event
      if (msg.type === 'event' && msg.event === 'connect.challenge') {
        challengeNonce = (msg.payload && msg.payload.nonce) || '';
        console.log('[a2a-bridge] Got challenge, sending connect...');
        // Step 2: Send connect request with auth
        wsSend(wsSocket, JSON.stringify({
          type: 'req', id: 'connect-' + Date.now(), method: 'connect',
          params: {
            auth: { token: GATEWAY_TOKEN },
            client: { id: 'openclaw-control-ui', platform: 'web', mode: 'ui', version: '1.0.0' },
            scopes: ['operator.read', 'operator.write', 'operator.admin', 'operator.approvals', 'operator.pairing'],
            minProtocol: 3,
            maxProtocol: 3,
          }
        }));
        return;
      }

      // Step 3: On connect response, send agent request
      if (msg.type === 'res' && msg.id && msg.id.startsWith('connect-')) {
        if (msg.ok) {
          console.log(`[a2a-bridge] Connected OK, scopes=${JSON.stringify(msg.payload)}`);
          console.log('[a2a-bridge] Sending agent request...');
          wsSend(wsSocket, JSON.stringify({
            type: 'req', id: 'agent-' + Date.now(), method: 'agent',
            params: {
              agentId: 'main',
              message: messageText,
              deliver: false,
              idempotencyKey: crypto.randomUUID(),
              sessionKey: sessionKey,
            }
          }));
        } else {
          const errMsg = (msg.error && msg.error.message) || JSON.stringify(msg.error || msg.payload || {}).slice(0, 300);
          console.log(`[a2a-bridge] Connect failed: ${errMsg}`);
          finishError(`Gateway connect failed: ${errMsg}`);
        }
        return;
      }

      // Step 4: Agent RPC responses — first "accepted", then "final" with summary
      if (msg.type === 'res' && msg.id && msg.id.startsWith('agent-')) {
        const p = msg.payload || {};
        if (msg.ok) {
          if (p.status === 'accepted') {
            console.log(`[a2a-bridge] Agent accepted, waiting for final response...`);
            return; // Wait for the second response with the actual result
          }
          // Final response — extract text or fallback to chat.history
          console.log(`[a2a-bridge] Agent final: status=${p.status} summary=${(p.summary||'').slice(0,200)} agentText=${agentText.slice(0,100)}`);
          if (p.summary) agentText = p.summary;
          if (p.text) agentText = p.text;

          if (agentText) {
            finish(agentText);
          } else {
            // Fallback: fetch latest reply from chat.history (like a2a-gateway does)
            console.log('[a2a-bridge] No text yet, fetching chat.history...');
            wsSend(wsSocket, JSON.stringify({
              type: 'req', id: 'history-' + Date.now(), method: 'chat.history',
              params: { sessionKey: sessionKey, limit: 5 }
            }));
            // Will be handled by the history response handler below
          }
        } else {
          const errMsg = (msg.error && msg.error.message) || JSON.stringify(msg.error || msg.payload || {}).slice(0, 300);
          console.log(`[a2a-bridge] Agent RPC failed: ${errMsg}`);
          finishError(`Agent dispatch failed: ${errMsg}`);
        }
        return;
      }

      // Step 4b: chat.history response — extract latest assistant reply
      if (msg.type === 'res' && msg.id && msg.id.startsWith('history-')) {
        if (msg.ok && msg.payload) {
          const messages = Array.isArray(msg.payload) ? msg.payload :
                          (msg.payload.messages || msg.payload.history || []);
          // Find last assistant message
          for (let i = messages.length - 1; i >= 0; i--) {
            const m = messages[i];
            if (m.role === 'assistant' || m.role === 'agent') {
              const text = m.content || m.text || m.message || '';
              if (text && typeof text === 'string') {
                console.log(`[a2a-bridge] Got text from history: ${text.slice(0, 200)}`);
                finish(text);
                return;
              }
            }
          }
        }
        finish(agentText || '(no reply in history)');
        return;
      }

      // Step 5: Collect streaming agent response events
      if (msg.type === 'event') {
        const p = msg.payload || {};
        const ev = msg.event || '';

        // Log ALL events for debugging
        const pkeys = Object.keys(p).join(',');
        console.log(`[a2a-bridge] Event: ${ev} keys=[${pkeys}] summary=${(p.summary||'').slice(0,100)} status=${p.status||''}`);

        // agent event — collect text from data.text or data.content
        if (ev === 'agent' && p.data) {
          const d = typeof p.data === 'object' ? p.data : {};
          if (d.text) agentText += d.text;
          if (d.content) agentText += d.content;
          if (d.delta) agentText += d.delta;
        }
        if (ev === 'agent') {
          if (p.summary) agentText = p.summary;
          if (p.text) agentText = p.text;
        }

        // chat event — message field may be string or object
        if (ev === 'chat') {
          if (typeof p.message === 'string') agentText += p.message;
          else if (p.message && typeof p.message === 'object') {
            // Structured message: {role, content, ...}
            const c = p.message.content || p.message.text || '';
            if (c) agentText += c;
          }
          if (p.text && typeof p.text === 'string') agentText += p.text;
          if (p.content && typeof p.content === 'string') agentText += p.content;
          if (p.delta && typeof p.delta === 'string') agentText += p.delta;
        }

        // session.message — structured message
        if (ev === 'session.message' && p.role === 'assistant') {
          const content = p.content || p.text || '';
          if (content && typeof content === 'string') agentText += content;
        }

        // Generic delta
        if (p.delta && typeof p.delta === 'string' && ev !== 'agent' && ev !== 'chat') agentText += p.delta;
      }
    }
  });
}

function wsSend(socket, data, opcode = 0x01) {
  // Send a WebSocket frame (unmasked, server-to-client style)
  const payload = Buffer.from(data, 'utf8');
  let header;
  if (payload.length < 126) {
    header = Buffer.alloc(2);
    header[0] = 0x80 | opcode; // FIN + opcode
    header[1] = payload.length;
  } else if (payload.length < 65536) {
    header = Buffer.alloc(4);
    header[0] = 0x80 | opcode;
    header[1] = 126;
    header.writeUInt16BE(payload.length, 2);
  } else {
    header = Buffer.alloc(10);
    header[0] = 0x80 | opcode;
    header[1] = 127;
    header.writeBigUInt64BE(BigInt(payload.length), 2);
  }
  // Client frames must be masked
  const mask = require('crypto').randomBytes(4);
  const masked = Buffer.alloc(payload.length);
  for (let i = 0; i < payload.length; i++) masked[i] = payload[i] ^ mask[i % 4];
  const maskHeader = Buffer.alloc(header.length);
  header.copy(maskHeader);
  maskHeader[1] |= 0x80; // Set mask bit
  try {
    socket.write(Buffer.concat([maskHeader, mask, masked]));
  } catch (_) {}
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
    // Debug: log all /a2a requests
    if (pathname.startsWith('/a2a')) {
      console.log(`[token-redirect] A2A request: ${req.method} ${pathname} port=${serverPort}`);
    }

    if (pathname.startsWith('/.well-known/')) {
      proxyToA2A(req, res);
      return true;
    }
    if (pathname.startsWith('/a2a/')) {
      // POST /a2a/jsonrpc → use bridge (bypasses scope issue)
      if (req.method === 'POST' && (pathname === '/a2a/jsonrpc' || pathname === '/a2a/jsonrpc/')) {
        console.log(`[a2a-bridge] Intercepted POST ${pathname}`);
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
