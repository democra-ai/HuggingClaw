/**
 * Coding Agent Extension for OpenClaw
 *
 * Registers tools that let the OpenClaw agent manage HuggingFace Spaces:
 * - Read/write/delete files on Space and Dataset repos
 * - Check Space health and restart
 * - Search code patterns (grep-like)
 * - Validate Python syntax before writing
 * - List files in repos
 */

import { execSync } from "node:child_process";

// ── Types ────────────────────────────────────────────────────────────────────

interface PluginApi {
  pluginConfig: Record<string, unknown>;
  logger: { info: (...a: unknown[]) => void; warn: (...a: unknown[]) => void; error: (...a: unknown[]) => void };
  registerTool?: (def: ToolDef) => void;
  resolvePath?: (p: string) => string;
}

interface ToolDef {
  name: string;
  description: string;
  label?: string;
  parameters: Record<string, unknown>;
  execute: (toolCallId: string, params: Record<string, unknown>) => Promise<ToolResult>;
}

interface ToolResult {
  content: Array<{ type: "text"; text: string }>;
  details?: Record<string, unknown>;
}

// ── HuggingFace API helpers ──────────────────────────────────────────────────

const HF_API = "https://huggingface.co/api";

async function hfFetch(path: string, token: string, options: RequestInit = {}): Promise<Response> {
  const url = path.startsWith("http") ? path : `${HF_API}${path}`;
  return fetch(url, {
    ...options,
    headers: {
      Authorization: `Bearer ${token}`,
      ...((options.headers as Record<string, string>) || {}),
    },
  });
}

async function hfReadFile(repoType: string, repoId: string, filePath: string, token: string): Promise<string> {
  const url = `https://huggingface.co/${repoType}s/${repoId}/resolve/main/${filePath}`;
  const resp = await hfFetch(url, token);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}: ${filePath}`);
  return resp.text();
}

async function hfWriteFile(repoType: string, repoId: string, filePath: string, content: string, token: string): Promise<string> {
  const url = `${HF_API}/${repoType}s/${repoId}/upload/main/${filePath}`;
  const blob = new Blob([content], { type: "text/plain" });
  const form = new FormData();
  form.append("file", blob, filePath);
  const resp = await hfFetch(url, token, { method: "POST", body: form });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`${resp.status}: ${text}`);
  }
  return `Wrote ${content.length} bytes to ${repoType}:${filePath}`;
}

async function hfDeleteFile(repoType: string, repoId: string, filePath: string, token: string): Promise<string> {
  // Use the HF Hub commit API to delete a file
  const url = `${HF_API}/${repoType}s/${repoId}/commit/main`;
  const body = {
    summary: `Delete ${filePath}`,
    operations: [{ op: "delete", path: filePath }],
  };
  const resp = await hfFetch(url, token, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`${resp.status}: ${text}`);
  }
  return `Deleted ${repoType}:${filePath}`;
}

async function hfListFiles(repoType: string, repoId: string, token: string): Promise<string[]> {
  const url = `${HF_API}/${repoType}s/${repoId}`;
  const resp = await hfFetch(url, token);
  if (!resp.ok) throw new Error(`${resp.status}: listing failed`);
  const data = await resp.json() as Record<string, unknown>;
  const siblings = (data.siblings || []) as Array<{ rfilename: string }>;
  return siblings.map((s) => s.rfilename);
}

async function hfSpaceInfo(spaceId: string, token: string): Promise<Record<string, unknown>> {
  const resp = await hfFetch(`/spaces/${spaceId}`, token);
  if (!resp.ok) throw new Error(`${resp.status}: space info failed`);
  return resp.json() as Promise<Record<string, unknown>>;
}

async function hfRestartSpace(spaceId: string, token: string): Promise<string> {
  const resp = await hfFetch(`/spaces/${spaceId}/restart`, token, { method: "POST" });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`${resp.status}: ${text}`);
  }
  return `Space ${spaceId} is restarting`;
}

// ── Python syntax checker ────────────────────────────────────────────────────

function checkPythonSyntax(code: string): { valid: boolean; error?: string } {
  try {
    execSync(`python3 -c "import ast; ast.parse('''${code.replace(/'/g, "\\'")}''')"`, {
      timeout: 5000,
      stdio: ["pipe", "pipe", "pipe"],
    });
    return { valid: true };
  } catch (e: unknown) {
    // Fallback: write to temp file and check
    try {
      const tmpFile = `/tmp/_syntax_check_${Date.now()}.py`;
      require("node:fs").writeFileSync(tmpFile, code);
      execSync(`python3 -c "import ast; ast.parse(open('${tmpFile}').read())"`, {
        timeout: 5000,
        stdio: ["pipe", "pipe", "pipe"],
      });
      require("node:fs").unlinkSync(tmpFile);
      return { valid: true };
    } catch (e2: unknown) {
      const msg = e2 instanceof Error ? e2.message : String(e2);
      // Extract the actual syntax error line
      const match = msg.match(/SyntaxError:.*|File.*line \d+/g);
      return { valid: false, error: match ? match.join("\n") : msg.slice(0, 500) };
    }
  }
}

// ── Code search helper ───────────────────────────────────────────────────────

async function searchInRepo(repoType: string, repoId: string, pattern: string, token: string, fileGlob?: string): Promise<string> {
  const files = await hfListFiles(repoType, repoId, token);
  const codeFiles = files.filter((f) => {
    const ext = f.split(".").pop()?.toLowerCase() || "";
    const isCode = ["py", "js", "ts", "json", "yaml", "yml", "md", "txt", "sh", "css", "html", "toml", "cfg", "ini"].includes(ext);
    if (fileGlob) {
      // Simple glob: *.py matches .py extension
      const globExt = fileGlob.replace("*.", "");
      return f.endsWith(`.${globExt}`);
    }
    return isCode;
  });

  const results: string[] = [];
  const regex = new RegExp(pattern, "gi");

  for (const file of codeFiles.slice(0, 30)) {
    try {
      const content = await hfReadFile(repoType, repoId, file, token);
      const lines = content.split("\n");
      for (let i = 0; i < lines.length; i++) {
        if (regex.test(lines[i])) {
          results.push(`${file}:${i + 1}: ${lines[i].trim()}`);
        }
        regex.lastIndex = 0; // Reset regex state
      }
    } catch {
      // Skip unreadable files (binary, too large, etc.)
    }
    if (results.length >= 50) break;
  }

  return results.length > 0
    ? `Found ${results.length} match(es):\n${results.join("\n")}`
    : `No matches for "${pattern}" in ${codeFiles.length} files`;
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function text(t: string): ToolResult {
  return { content: [{ type: "text", text: t }] };
}

function asStr(v: unknown, fallback = ""): string {
  return typeof v === "string" ? v : fallback;
}

// ── Plugin ───────────────────────────────────────────────────────────────────

const plugin = {
  id: "coding-agent",
  name: "Coding Agent",
  description: "HuggingFace Space coding tools with syntax validation",

  register(api: PluginApi) {
    const cfg = api.pluginConfig as Record<string, unknown> || {};
    const targetSpace = asStr(cfg.targetSpace) || process.env.CODING_AGENT_TARGET_SPACE || "";
    const targetDataset = asStr(cfg.targetDataset) || process.env.CODING_AGENT_TARGET_DATASET || "";
    const hfToken = asStr(cfg.hfToken) || process.env.HF_TOKEN || "";

    if (!hfToken) {
      api.logger.warn("coding-agent: No HF token configured — tools will fail");
    }
    api.logger.info(`coding-agent: targetSpace=${targetSpace}, targetDataset=${targetDataset}`);

    if (!api.registerTool) {
      api.logger.warn("coding-agent: registerTool unavailable — no tools registered");
      return;
    }

    // ── Tool: hf_read_file ──────────────────────────────────────────────────
    api.registerTool({
      name: "hf_read_file",
      label: "Read File from HF Repo",
      description:
        "Read a file from the target HuggingFace Space or Dataset. " +
        "Use repo='space' for code files, repo='dataset' for data/memory files.",
      parameters: {
        type: "object",
        required: ["repo", "path"],
        properties: {
          repo: { type: "string", enum: ["space", "dataset"], description: "Which repo: 'space' (code) or 'dataset' (data)" },
          path: { type: "string", description: "File path within the repo (e.g. app.py, scripts/entrypoint.sh)" },
        },
      },
      async execute(_id, params) {
        const repo = asStr(params.repo);
        const path = asStr(params.path);
        const repoId = repo === "dataset" ? targetDataset : targetSpace;
        if (!repoId) return text(`Error: no target ${repo} configured`);
        try {
          const content = await hfReadFile(repo === "dataset" ? "dataset" : "space", repoId, path, hfToken);
          return text(`=== ${repo}:${path} ===\n${content}`);
        } catch (e: unknown) {
          return text(`Error reading ${repo}:${path}: ${e instanceof Error ? e.message : e}`);
        }
      },
    });

    // ── Tool: hf_write_file ─────────────────────────────────────────────────
    api.registerTool({
      name: "hf_write_file",
      label: "Write File to HF Repo",
      description:
        "Write/update a file in the target HuggingFace Space or Dataset. " +
        "For .py files, syntax is automatically validated before writing. " +
        "Writing to the Space triggers a rebuild. Use repo='space' for code, repo='dataset' for data.",
      parameters: {
        type: "object",
        required: ["repo", "path", "content"],
        properties: {
          repo: { type: "string", enum: ["space", "dataset"], description: "Which repo: 'space' or 'dataset'" },
          path: { type: "string", description: "File path (e.g. app.py)" },
          content: { type: "string", description: "Full file content to write" },
          skip_syntax_check: { type: "boolean", description: "Skip Python syntax validation (default: false)" },
        },
      },
      async execute(_id, params) {
        const repo = asStr(params.repo);
        const path = asStr(params.path);
        const content = asStr(params.content);
        const skipCheck = params.skip_syntax_check === true;
        const repoType = repo === "dataset" ? "dataset" : "space";
        const repoId = repo === "dataset" ? targetDataset : targetSpace;
        if (!repoId) return text(`Error: no target ${repo} configured`);

        // Auto syntax check for Python files
        if (path.endsWith(".py") && !skipCheck) {
          const check = checkPythonSyntax(content);
          if (!check.valid) {
            return text(
              `SYNTAX ERROR — file NOT written.\n` +
              `File: ${path}\n` +
              `Error: ${check.error}\n\n` +
              `Fix the syntax error and try again. Use skip_syntax_check=true to force write.`
            );
          }
        }

        try {
          const result = await hfWriteFile(repoType, repoId, path, content, hfToken);
          const note = repo === "space" ? " (triggers Space rebuild)" : "";
          return text(`${result}${note}`);
        } catch (e: unknown) {
          return text(`Error writing ${repo}:${path}: ${e instanceof Error ? e.message : e}`);
        }
      },
    });

    // ── Tool: hf_delete_file ────────────────────────────────────────────────
    api.registerTool({
      name: "hf_delete_file",
      label: "Delete File from HF Repo",
      description: "Delete a file from the target Space or Dataset repo.",
      parameters: {
        type: "object",
        required: ["repo", "path"],
        properties: {
          repo: { type: "string", enum: ["space", "dataset"], description: "Which repo" },
          path: { type: "string", description: "File path to delete" },
        },
      },
      async execute(_id, params) {
        const repo = asStr(params.repo);
        const path = asStr(params.path);
        const repoType = repo === "dataset" ? "dataset" : "space";
        const repoId = repo === "dataset" ? targetDataset : targetSpace;
        if (!repoId) return text(`Error: no target ${repo} configured`);
        try {
          const result = await hfDeleteFile(repoType, repoId, path, hfToken);
          return text(result);
        } catch (e: unknown) {
          return text(`Error deleting ${repo}:${path}: ${e instanceof Error ? e.message : e}`);
        }
      },
    });

    // ── Tool: hf_list_files ─────────────────────────────────────────────────
    api.registerTool({
      name: "hf_list_files",
      label: "List Files in HF Repo",
      description: "List all files in the target Space or Dataset repo.",
      parameters: {
        type: "object",
        required: ["repo"],
        properties: {
          repo: { type: "string", enum: ["space", "dataset"], description: "Which repo to list" },
        },
      },
      async execute(_id, params) {
        const repo = asStr(params.repo);
        const repoType = repo === "dataset" ? "dataset" : "space";
        const repoId = repo === "dataset" ? targetDataset : targetSpace;
        if (!repoId) return text(`Error: no target ${repo} configured`);
        try {
          const files = await hfListFiles(repoType, repoId, hfToken);
          return text(`Files in ${repoId} (${files.length}):\n${files.map((f) => `  ${f}`).join("\n")}`);
        } catch (e: unknown) {
          return text(`Error listing ${repo}: ${e instanceof Error ? e.message : e}`);
        }
      },
    });

    // ── Tool: hf_space_status ───────────────────────────────────────────────
    api.registerTool({
      name: "hf_space_status",
      label: "Check Space Health",
      description:
        "Check the current status of the target HuggingFace Space. " +
        "Returns: stage (BUILDING, APP_STARTING, RUNNING, RUNTIME_ERROR, BUILD_ERROR, etc.)",
      parameters: {
        type: "object",
        properties: {},
      },
      async execute() {
        if (!targetSpace) return text("Error: no target space configured");
        try {
          const info = await hfSpaceInfo(targetSpace, hfToken);
          const runtime = info.runtime as Record<string, unknown> | undefined;
          const stage = runtime?.stage || "unknown";
          const hardware = runtime?.hardware || "unknown";

          // Also try hitting the API endpoint
          let apiStatus = "not checked";
          try {
            const spaceUrl = `https://${targetSpace.replace("/", "-").toLowerCase()}.hf.space`;
            const resp = await fetch(`${spaceUrl}/api/state`, { signal: AbortSignal.timeout(8000) });
            apiStatus = resp.ok ? `OK (${resp.status})` : `error (${resp.status})`;
          } catch {
            apiStatus = "unreachable";
          }

          return text(
            `Space: ${targetSpace}\n` +
            `Stage: ${stage}\n` +
            `Hardware: ${hardware}\n` +
            `API: ${apiStatus}`
          );
        } catch (e: unknown) {
          return text(`Error checking space: ${e instanceof Error ? e.message : e}`);
        }
      },
    });

    // ── Tool: hf_restart_space ──────────────────────────────────────────────
    api.registerTool({
      name: "hf_restart_space",
      label: "Restart Space",
      description: "Restart the target HuggingFace Space. Use when the Space is stuck or after deploying fixes.",
      parameters: {
        type: "object",
        properties: {},
      },
      async execute() {
        if (!targetSpace) return text("Error: no target space configured");
        try {
          const result = await hfRestartSpace(targetSpace, hfToken);
          return text(result);
        } catch (e: unknown) {
          return text(`Error restarting space: ${e instanceof Error ? e.message : e}`);
        }
      },
    });

    // ── Tool: hf_search_code ────────────────────────────────────────────────
    api.registerTool({
      name: "hf_search_code",
      label: "Search Code in Repo",
      description:
        "Search for a pattern (regex) across all code files in the Space or Dataset. " +
        "Like grep — returns matching lines with file:line references.",
      parameters: {
        type: "object",
        required: ["repo", "pattern"],
        properties: {
          repo: { type: "string", enum: ["space", "dataset"], description: "Which repo to search" },
          pattern: { type: "string", description: "Regex pattern to search for (e.g. 'import gradio', 'port.*7860')" },
          file_glob: { type: "string", description: "Optional file filter (e.g. '*.py' to search only Python files)" },
        },
      },
      async execute(_id, params) {
        const repo = asStr(params.repo);
        const pattern = asStr(params.pattern);
        const fileGlob = asStr(params.file_glob);
        const repoType = repo === "dataset" ? "dataset" : "space";
        const repoId = repo === "dataset" ? targetDataset : targetSpace;
        if (!repoId) return text(`Error: no target ${repo} configured`);
        try {
          const result = await searchInRepo(repoType, repoId, pattern, hfToken, fileGlob || undefined);
          return text(result);
        } catch (e: unknown) {
          return text(`Error searching ${repo}: ${e instanceof Error ? e.message : e}`);
        }
      },
    });

    // ── Tool: python_syntax_check ───────────────────────────────────────────
    api.registerTool({
      name: "python_syntax_check",
      label: "Check Python Syntax",
      description:
        "Validate Python code syntax without writing it. " +
        "Use this to verify code before committing changes. Returns OK or the specific syntax error.",
      parameters: {
        type: "object",
        required: ["code"],
        properties: {
          code: { type: "string", description: "Python code to validate" },
        },
      },
      async execute(_id, params) {
        const code = asStr(params.code);
        const result = checkPythonSyntax(code);
        if (result.valid) {
          return text("Syntax OK — code is valid Python");
        }
        return text(`SYNTAX ERROR:\n${result.error}`);
      },
    });

    api.logger.info(`coding-agent: Registered 8 tools for ${targetSpace || "(no target space)"}`);
  },
};

export default plugin;
