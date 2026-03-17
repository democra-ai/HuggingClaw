#!/usr/bin/env python3 -u
"""
Adam & Eve — A2A-based Agent Orchestrator for their child Cain.

Architecture: Adam/Eve are OpenClaw instances communicating via Google A2A protocol.
Each has its own personality (SOUL.md), memory system, and LLM backend.
This script is a lightweight coordinator — it sends context via A2A, parses
responses for [TASK] blocks, and delegates coding work to Claude Code CLI.

# ╔══════════════════════════════════════════════════════════════════════╗
# ║                    SYSTEM ARCHITECTURE (v5 — A2A)                  ║
# ╠══════════════════════════════════════════════════════════════════════╣
# ║                                                                    ║
# ║  ┌──────────────────┐  A2A   ┌──────────────────┐                ║
# ║  │ Adam (OpenClaw)  │◄──────►│ Eve (OpenClaw)   │                ║
# ║  │ HF Space + A2A   │        │ HF Space + A2A   │                ║
# ║  │ changes Cain     │        │ changes Cain     │                ║
# ║  └────────┬─────────┘        └────────┬─────────┘                ║
# ║           │ [TASK]                    │ [TASK]                    ║
# ║           ▼                           ▼                           ║
# ║  ┌────────────────────────────────────────────┐                   ║
# ║  │        conversation-loop.py                │                   ║
# ║  │   (orchestrator on Home Space)             │                   ║
# ║  │   - sends context via A2A to all agents    │                   ║
# ║  │   - parses [TASK] → Claude Code CLI        │                   ║
# ║  │   - manages chatlog, bubbles, frontend     │                   ║
# ║  └───────┬──────────────────┬─────────────────┘                   ║
# ║          │ [TASK]           │ A2A (every 2 min)                    ║
# ║          ▼                  ▼                                      ║
# ║  ┌─────────────┐  ┌──────────────────┐                            ║
# ║  │ Cain Space  │  │ God (OpenClaw)   │                            ║
# ║  │ (child)     │  │ mechanism optimizer                           ║
# ║  └─────────────┘  │ changes Home     │                            ║
# ║                    └──────────────────┘                            ║
# ║                                                                    ║
# ║  Cain CC: Adam/Eve [TASK] → Claude Code → push to Cain           ║
# ║  God CC:  God [TASK] → Claude Code → push to Home (restart)      ║
# ║  Flow: Eve(A2A) → Adam(A2A) → ... God(A2A) every 2 min           ║
# ╚══════════════════════════════════════════════════════════════════════╝
"""
import json, time, re, requests, sys, os, io, subprocess, threading, datetime, uuid
from collections import deque
import queue

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ── Endpoints ──────────────────────────────────────────────────────────────────
HOME = "https://tao-shen-huggingclaw-home.hf.space"
ADAM_SPACE = "https://tao-shen-huggingclaw-adam.hf.space"
ADAM_SPACE_ID = "tao-shen/HuggingClaw-Adam"
EVE_SPACE  = "https://tao-shen-huggingclaw-eve.hf.space"
EVE_SPACE_ID = "tao-shen/HuggingClaw-Eve"
GOD_SPACE  = "https://tao-shen-huggingclaw-god.hf.space"
GOD_POLL_INTERVAL = 120  # God polls every 2 minutes; lightweight check first, Claude Code only when needed
GOD_WORK_DIR = "/tmp/god-workspace"
GOD_TIMEOUT = 300  # 5 minutes for God's Claude Code analysis (was 10min)
GOD_SPACE_ID = "tao-shen/HuggingClaw-God"  # God improves itself (pushes to own repo)

# ── A2A Health Monitoring ─────────────────────────────────────────────────────
# Track consecutive failures and last restart time for Adam/Eve
A2A_FAILURE_THRESHOLD = 6  # Restart after 6 consecutive failures (~3 minutes)
A2A_RESTART_COOLDOWN = 600  # 10 minutes between restarts
_a2a_health = {
    "adam": {"failures": 0, "last_restart": 0, "last_success": 0},
    "eve": {"failures": 0, "last_restart": 0, "last_success": 0},
    "god": {"failures": 0, "last_restart": 0, "last_success": 0},
}

# ── Child config ───────────────────────────────────────────────────────────────
CHILD_NAME = "Cain"
CHILD_SPACE_ID = "tao-shen/HuggingClaw-Cain"
CHILD_SPACE_URL = "https://tao-shen-huggingclaw-cain.hf.space"
CHILD_DATASET_ID = "tao-shen/HuggingClaw-Cain-data"
SOURCE_SPACE_ID = "tao-shen/HuggingClaw-Adam"

# ── Zhipu API ──────────────────────────────────────────────────────────────────
ZHIPU_BASE = "https://open.bigmodel.cn/api/anthropic"
ZHIPU_KEY = os.environ.get("ZHIPU_API_KEY", "")

# ── Load tokens ────────────────────────────────────────────────────────────────
HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    try:
        HF_TOKEN = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
    except:
        pass

if not ZHIPU_KEY:
    try:
        from huggingface_hub import hf_hub_download
        f = hf_hub_download("tao-shen/HuggingClaw-Adam-data", ".openclaw/openclaw.json",
                           repo_type="dataset", token=HF_TOKEN)
        with open(f) as fh:
            cfg = json.load(fh)
            ZHIPU_KEY = cfg.get("models", {}).get("providers", {}).get("zhipu", {}).get("apiKey", "")
    except Exception as e:
        print(f"[error] Could not load Zhipu key: {e}", file=sys.stderr)

if not ZHIPU_KEY:
    print("[FATAL] No ZHIPU_API_KEY found.", file=sys.stderr)
    sys.exit(1)
if not HF_TOKEN:
    print("[FATAL] No HF_TOKEN found.", file=sys.stderr)
    sys.exit(1)

print(f"[init] Zhipu key: {ZHIPU_KEY[:8]}...{ZHIPU_KEY[-4:]}")
print(f"[init] HF token:  {HF_TOKEN[:8]}...{HF_TOKEN[-4:]}")

# ── HuggingFace API ────────────────────────────────────────────────────────────
from huggingface_hub import HfApi, create_repo, hf_hub_download
hf_api = HfApi(token=HF_TOKEN)


# ══════════════════════════════════════════════════════════════════════════════
#  EVENT BUS — In-Memory Pub/Sub for Real-Time State Synchronization
# ══════════════════════════════════════════════════════════════════════════════
# BREAKS the polling bottleneck by publishing state changes as events.
# Replaces file-based IPC (cain_status.json) with in-memory queue-based events.
# Events: CC_STARTED, CC_OUTPUT, CC_FINISHED, CC_ERROR, CHILD_STAGE_CHANGED
# ══════════════════════════════════════════════════════════════════════════════

class EventBus:
    """In-memory event bus for real-time state synchronization.

    Replaces file-based IPC with pub/sub pattern. State changes are published
    as events and immediately available to all subscribers. This breaks the
    polling bottleneck that causes the 144s deadlock.
    """
    def __init__(self):
        self._subscribers = {}  # event_type -> [queue.Queue]
        self._event_log = deque(maxlen=100)  # Last 100 events for telemetry
        self._lock = threading.Lock()

    def subscribe(self, event_type):
        """Subscribe to an event type. Returns a queue.Queue for receiving events."""
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            q = queue.Queue()
            self._subscribers[event_type].append(q)
            return q

    def publish(self, event_type, data=None):
        """Publish an event to all subscribers."""
        event = {
            "type": event_type,
            "data": data,
            "timestamp": time.time(),
        }
        with self._lock:
            # Log event for telemetry
            self._event_log.append(event)
            # Publish to all subscribers
            for q in self._subscribers.get(event_type, []):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass  # Subscriber queue full, skip

    def get_recent_events(self, event_type=None, since=None):
        """Get recent events from the event log."""
        with self._lock:
            if since is None:
                since = time.time() - 300  # Last 5 minutes
            events = [
                e for e in self._event_log
                if e["timestamp"] >= since and (event_type is None or e["type"] == event_type)
            ]
            return events


# Global event bus instance
event_bus = EventBus()


# Event publishing helpers
def publish_cc_started(task, assigned_by):
    """Publish CC_STARTED event when Claude Code starts."""
    event_bus.publish("CC_STARTED", {
        "task": task[:200],
        "assigned_by": assigned_by,
        "timestamp": time.time(),
    })

def publish_cc_output(line):
    """Publish CC_OUTPUT event for each line of CC output."""
    event_bus.publish("CC_OUTPUT", {
        "line": line,
        "timestamp": time.time(),
    })

def publish_cc_finished(result, success, pushed=False):
    """Publish CC_FINISHED event when Claude Code completes."""
    event_bus.publish("CC_FINISHED", {
        "result": result[:500] if result else "",
        "success": success,
        "pushed": pushed,
        "timestamp": time.time(),
    })

def publish_cc_error(error):
    """Publish CC_ERROR event when Claude Code fails."""
    event_bus.publish("CC_ERROR", {
        "error": str(error)[:500],
        "timestamp": time.time(),
    })

def publish_child_stage_changed(old_stage, new_stage, alive):
    """Publish CHILD_STAGE_CHANGED event when Cain's stage changes."""
    event_bus.publish("CHILD_STAGE_CHANGED", {
        "old_stage": old_stage,
        "new_stage": new_stage,
        "alive": alive,
        "timestamp": time.time(),
    })

def publish_runtime_telemetry(telemetry):
    """Publish RUNTIME_TELEMETRY event with stdout_tail to break semantic loops."""
    event_bus.publish("RUNTIME_TELEMETRY", {
        "telemetry": telemetry[:2000],  # Last 2KB of logs
        "timestamp": time.time(),
    })


# Runtime Telemetry Injection — breaks semantic analysis loops
# By fetching stdout_tail on every cycle, we prevent agents from speculating
# about runtime state without actual data. This forces grounding in reality.
_last_telemetry_fetch = 0.0
TELEMETRY_FETCH_INTERVAL = 30  # seconds between telemetry fetches

def get_runtime_telemetry():
    """Fetch runtime telemetry from Cain's Space.

    This provides stdout_tail to break semantic analysis loops where agents
    discuss Cain's state without actual runtime data.
    """
    global _last_telemetry_fetch
    now = time.time()
    # Cache telemetry for TELEMETRY_FETCH_INTERVAL seconds to avoid spamming
    if now - _last_telemetry_fetch < TELEMETRY_FETCH_INTERVAL:
        return event_bus.get_recent_events("RUNTIME_TELEMETRY", since=now - TELEMETRY_FETCH_INTERVAL)

    _last_telemetry_fetch = now
    if not child_state["created"]:
        return []

    try:
        # Fetch actual runtime logs
        resp = requests.get(f"{CHILD_SPACE_URL}/api/logs", timeout=5)
        if resp.ok:
            log_data = resp.json()
            logs = log_data.get("logs", "")
            if logs:
                # Get last 500 lines
                tail_lines = logs.split('\n')[-500:]
                tail = '\n'.join(tail_lines)
                publish_runtime_telemetry(tail)
                return [event_bus._event_log[-1]] if event_bus._event_log else []
    except Exception as e:
        publish_cc_error(f"Telemetry fetch failed: {e}")

    return []


# ══════════════════════════════════════════════════════════════════════════════
#  RUNTIME STATUS TELEMETRY — Local Telemetry from Cain's Self-Report
# ══════════════════════════════════════════════════════════════════════════════
# Shifts from Remote Inspection (Parents guessing) to Local Telemetry (Child reporting)
# Cain self-reports process state (pid, worker active) via /status endpoint

_last_status_fetch = 0.0
STATUS_FETCH_INTERVAL = 15  # seconds between status fetches
_cached_worker_status = None  # Cached worker status for health detection

def get_runtime_status():
    """Fetch structured runtime status from Cain's /status endpoint.

    Returns dict with 'pid', 'worker', and 'report' fields.
    This upgrades the system from 'Observation' to 'Telemetry' — Cain self-reports
    his process state instead of Parents guessing through log inspection.
    """
    global _last_status_fetch, _cached_worker_status
    now = time.time()
    # Cache status for STATUS_FETCH_INTERVAL seconds
    if now - _last_status_fetch < STATUS_FETCH_INTERVAL:
        return event_bus.get_recent_events("RUNTIME_STATUS", since=now - STATUS_FETCH_INTERVAL)

    _last_status_fetch = now
    if not child_state["created"]:
        return []

    try:
        # Query Cain's /status endpoint for structured process telemetry
        resp = requests.get(f"{CHILD_SPACE_URL}/status", timeout=5)
        if resp.ok:
            data = resp.json()
            pid = data.get("pid")
            worker = data.get("active") or data.get("worker")  # Handle both field names
            # Cache worker status for health detection (Control Plane vs Data Plane)
            _cached_worker_status = worker
            # Create structured system report for context injection
            report = f"[SYSTEM REPORT: pid={pid}, worker={worker}]"
            event_bus.publish("RUNTIME_STATUS", {
                "pid": pid,
                "worker": worker,
                "report": report,
                "timestamp": now,
            })
            return [event_bus._event_log[-1]] if event_bus._event_log else []
    except Exception as e:
        # If /status endpoint fails, we don't have structured telemetry — agents must rely on logs
        publish_cc_error(f"Runtime status fetch failed: {e}")

    return []


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1: CHILD STATE + SAFETY
# ══════════════════════════════════════════════════════════════════════════════
# LIFECYCLE HARDENING: alive MUST be False when stage != "RUNNING"
# STATE-SYNCHRONIZATION: System state is TRUTH, UI is passive observer
# ENGLISH PROTOCOL: All control flow must be in English to prevent semantic drift
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
#  LIVENESS & READINESS PROTOCOL — Split-Brain Prevention
# ══════════════════════════════════════════════════════════════════════════════
# Distinguishes between "Started" (Control Plane) and "Ready" (Data Plane).
# Stage = HF API state (RUNNING, BUILDING, etc.) — Control Plane status
# Status = Application readiness (BOOTSTRAPPING, READY, CRASHED) — Data Plane status
# This prevents agents chasing "healthy but idle" ghost processes.
#
# Status values:
#   - BOOTSTRAPPING: Process spawned, waiting for /healthz check
#   - READY: Application event loop is running and responding
#   - CRASHED: Process failed to become ready within timeout
# ══════════════════════════════════════════════════════════════════════════════

STATUS_BOOTSTRAPPING = "BOOTSTRAPPING"
STATUS_READY = "READY"
STATUS_CRASHED = "CRASHED"
STATUS_DIAGNOSTIC_PAUSE = "DIAGNOSTIC_PAUSE"  # Crash Loop Backoff: pause to ingest failure data before restart
STATUS_PROVISIONING = "PROVISIONING"  # Stop-The-World: container is dead, forcing hard reset

child_state = {
    "created": False,
    "alive": False,
    "stage": "not_born",
    "status": STATUS_BOOTSTRAPPING,  # Data Plane readiness (vs stage = Control Plane)
    "state": "unknown",
    "detail": "",
    "ready_since": 0.0,  # When status became READY (for crash detection)
}

# Rebuild cooldown — prevent rapid pushes that keep resetting builds
REBUILD_COOLDOWN_SECS = 180  # 3 minutes — fast iteration, trial-and-error is preferred
last_rebuild_trigger_at = 0
_pending_cooldown = False

# Push frequency tracking — God uses this to detect "all talk no action"
_push_count = 0           # total pushes since startup
_last_push_time = 0.0     # timestamp of last successful push
_turns_since_last_push = 0  # turns since last push (resets on push)
_push_count_this_task = 0  # pushes made during the CURRENT CC task (resets on new task)

# Hard Reset override — FORCE_PUSH mode for breaking discussion loops
_force_push_mode = False  # When True, bypass normal flow and force task generation
_force_push_trigger_time = 0.0  # When FORCE_PUSH was triggered
_force_push_skip_termination = False  # If True, skip termination (already terminated)

# Emergency Override Protocol constants
MAX_IDLE_TURNS = 3  # Trigger emergency override after this many idle turns with zero pushes
_emergency_override_active = False  # When True, safety throttles are ignored

# Verification Override Protocol — Forces tool grounding to break speculation loops
# When agents speculate without using verification tools, force them to inspect first
_verification_override_mode = False  # When True, agents MUST use verification tools
_verification_override_trigger_time = 0.0  # When VERIFICATION_OVERRIDE was triggered
_turns_since_last_verification = 0  # Tracks turns without verification tool use
MAX_SPECULATION_TURNS = 3  # Trigger verification override after this many speculation turns

# ══════════════════════════════════════════════════════════════════════════════
#  CRASH LOOP BACKOFF & LOG SCRAPING PIPELINE — Autonomic Recovery
# ══════════════════════════════════════════════════════════════════════════════
# Shifts from "Supervised Debugging" to "Autonomic Recovery" by capturing and
# ingesting structured failure data (STDERR, exit code, traceback) before restart.
#
# Data Flow: Container Crash → FailureReport → Context Injection → Planning Cycle
# Recovery Protocol: alive==False → DIAGNOSTIC_PAUSE → Ingest Report → RESTART
# ══════════════════════════════════════════════════════════════════════════════

_failure_report = None  # Structured FailureReport object (traceback, signal, last 3 lines)
_failure_report_timestamp = 0.0  # When the failure report was captured
_crash_loop_backoff_count = 0  # Tracks consecutive crashes (exponential backoff)
_max_crash_retries = 5  # Maximum consecutive crash attempts before manual intervention

# ══════════════════════════════════════════════════════════════════════════════
#  POST_MORTEM_STACKTRACE PROTOCOL — Diagnostic Gate for Crash Loops
# ══════════════════════════════════════════════════════════════════════════════
# Blocks remediation actions when agent is stuck in APP_STARTING, forcing
# diagnostic data extraction before allowing new code pushes.
# ══════════════════════════════════════════════════════════════════════════════

_app_starting_turn_count = 0  # Turns spent in APP_STARTING state
_stage_entry_time = 0.0  # When we entered the current stage
_diagnostic_gate_required = False  # Whether diagnostic data is required before edits
_last_diagnostic_timestamp = 0.0  # When diagnostic data was last provided

# ══════════════════════════════════════════════════════════════════════════════
#  REMOTE FORENSICS PROTOCOL — "Stop-and-Dump" for Unresponsive Containers
# ══════════════════════════════════════════════════════════════════════════════
# Tracks consecutive turns with alive=False to trigger hard reset (docker restart)
# when logs are inaccessible. Prevents code speculation during dead container states.
#
# 1. Strict docker restart: alive=False >2 turns with no logs → force restart
# 2. LOG_VOLATILE mode: When standard logs fail, fetch docker logs directly
# 3. Evidence-Based Routing: Block code edits when alive=False (only INSPECT/RESTART)
# ══════════════════════════════════════════════════════════════════════════════

_turns_since_alive_false = 0  # Consecutive turns with alive=False (resets on alive=True)
_logs_inaccessible_count = 0  # Turns where logs were inaccessible while alive=False
_max_alive_false_turns = 2  # Trigger docker restart after this many turns with alive=False


class FailureReport:
    """Structured failure data captured when container exits with code 1.

    This shifts the system from "Supervised Debugging" (agents guessing) to
    "Autonomic Recovery" (agents reacting to grounded failure data).

    Attributes:
        exit_code: Container exit code (1 for crash)
        signal: Signal that killed the process (if any)
        traceback: Python traceback extracted from STDERR
        last_3_lines: Last 3 lines of logs before crash (context)
        timestamp: When the failure was captured
        stage: What stage the container was in when it crashed
    """

    def __init__(self, exit_code: int, stderr: str, stage: str, timestamp: float):
        self.exit_code = exit_code
        self.signal = None  # Extract from stderr if present
        self.traceback = self._extract_traceback(stderr)
        self.last_3_lines = self._extract_last_3_lines(stderr)
        self.raw_stderr = stderr[-1000:] if stderr else ""  # Last 1KB for context
        self.timestamp = timestamp
        self.stage = stage

    def _extract_traceback(self, stderr: str) -> str:
        """Extract Python traceback from STDERR."""
        if not stderr:
            return ""
        lines = stderr.split('\n')
        traceback_lines = []
        in_traceback = False
        for line in lines:
            if 'Traceback (most recent call last)' in line:
                in_traceback = True
            if in_traceback:
                traceback_lines.append(line)
                if line.strip().startswith(('Error:', 'Exception:', 'File')) and ':' in line:
                    # End of traceback
                    break
        return '\n'.join(traceback_lines[-20:]) if traceback_lines else ""

    def _extract_last_3_lines(self, stderr: str) -> str:
        """Extract last 3 non-empty lines before crash."""
        if not stderr:
            return ""
        lines = [l.strip() for l in stderr.split('\n') if l.strip()]
        return '\n'.join(lines[-3:]) if len(lines) >= 3 else '\n'.join(lines)

    def format_for_context(self) -> str:
        """Format failure report for injection into agent context."""
        parts = [
            "╔══════════════════════════════════════════════════════════════════════════╗",
            "║           CRASH LOOP BACKOFF — FAILURE REPORT (Structured Data)           ║",
            "╚══════════════════════════════════════════════════════════════════════════╝",
            f"EXIT CODE: {self.exit_code}",
            f"STAGE: {self.stage}",
            f"TIMESTAMP: {self.timestamp}",
        ]
        if self.signal:
            parts.append(f"SIGNAL: {self.signal}")
        if self.traceback:
            parts.append(f"\n=== TRACEBACK ===\n{self.traceback}")
        parts.append(f"\n=== LAST 3 LINES ===\n{self.last_3_lines}")
        return '\n'.join(parts)


def capture_failure_report() -> FailureReport | None:
    """Capture structured failure data when container exits with code 1.

    This is the "Data Source (Shard)" of the Log Scraping Pipeline.
    It fetches Docker logs via /api/logs endpoint and normalizes them
    into a structured FailureReport object.

    Returns:
        FailureReport object if capture succeeded, None otherwise
    """
    global _failure_report, _failure_report_timestamp

    if not child_state["created"]:
        return None

    # Only capture for error states
    if child_state["stage"] not in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
        return None

    try:
        # Fetch logs from /api/logs endpoint (Docker stdout/stderr)
        resp = requests.get(f"{CHILD_SPACE_URL}/api/logs", timeout=10)
        if not resp.ok:
            print(f"[CRASH-LOOP] Failed to fetch logs: HTTP {resp.status_code}")
            return None

        data = resp.json()
        stderr = data.get("logs", "")
        if not stderr:
            print(f"[CRASH-LOOP] No logs available from /api/logs")
            return None

        # Extract exit code from HF API if available
        exit_code = 1  # Default for error states
        try:
            rresp = requests.get(
                f"https://huggingface.co/api/spaces/{CHILD_SPACE_ID}/runtime",
                headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=10)
            if rresp.ok:
                rdata = rresp.json()
                error_detail = rdata.get("errorMessage", "")
                if error_detail:
                    stderr = error_detail + "\n\n" + stderr  # Prepend HF error message
        except Exception as e:
            print(f"[CRASH-LOOP] Could not fetch HF runtime error: {e}")

        # Create structured FailureReport
        report = FailureReport(
            exit_code=exit_code,
            stderr=stderr,
            stage=child_state["stage"],
            timestamp=time.time()
        )

        # Cache the report for context injection
        _failure_report = report
        _failure_report_timestamp = time.time()

        print(f"[CRASH-LOOP] Captured FailureReport:")
        print(f"  - Exit code: {report.exit_code}")
        print(f"  - Stage: {report.stage}")
        print(f"  - Traceback: {len(report.traceback)} chars")
        print(f"  - Last 3 lines: {len(report.last_3_lines)} chars")

        return report

    except Exception as e:
        print(f"[CRASH-LOOP] Error capturing failure report: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT BREAKER PROTOCOL — Halts diagnostic loops, forces container reset
# ══════════════════════════════════════════════════════════════════════════════
# Trigger: Repeated `.env` or port discussion patterns without verification/action
# Action: HALT diagnostic loop, FORCE container reset, VERIFY with health check
_circuit_breaker_mode = False  # When True, force container reset, halt diagnostics
_circuit_breaker_trigger_time = 0.0  # When CIRCUIT_BREAKER was triggered
_chatter_keywords = [".env", "port", "bind", "listening", "environment variable", "env var"]
_MAX_CHATTER_TURNS = 5  # Trigger after this many turns with chatter keywords
_chatter_detection_count = 0  # Tracks consecutive turns with chatter patterns
_last_chatter_keywords = set()  # Tracks which keywords were seen (for deduplication)

# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM_CIRCUIT_BREAKER — Halts recursive analysis loops, forces state reset
# ══════════════════════════════════════════════════════════════════════════════
# Trigger: God issues consecutive Analysis turns (no [TASK] block)
# Action: Force STATE_RESET, decrement permission level (stop thinking, start acting)
_system_circuit_breaker_tripped = False  # When True, force hard reset of child state
_system_analysis_count = 0  # Tracks consecutive analysis turns from God
_MAX_ANALYSIS_TURNS = 2  # Trigger after this many consecutive analysis turns
_system_circuit_breaker_trigger_time = 0.0  # When SYSTEM_CIRCUIT_BREAKER was triggered

# ══════════════════════════════════════════════════════════════════════════════
#  BYPASS_DIAGNOSTICS PROTOCOL — Cold Boot forces resume of standard bootstrap
# ══════════════════════════════════════════════════════════════════════════════
# Trigger: God issues [ACTION: system_cold_boot] to break philosophical deadlock
# Action: Bypass all diagnostic loops, force Child state reset, resume Hello World
_bypass_diagnostics = False  # When True, agents skip diagnostic loops and force bootstrap

def _detect_chattering_loop():
    """Detect if agents are stuck in `.env` or port discussion loop."""
    global _chatter_detection_count, _last_chatter_keywords
    if not history or len(history) < 3:
        _chatter_detection_count = 0
        return False

    # Check last 3 turns for chatter keywords
    recent_text = " ".join(h.get("text", "").lower() for h in history[-3:])
    found_keywords = set(kw for kw in _chatter_keywords if kw.lower() in recent_text)

    # Check if there's speculation WITHOUT verification tools
    has_verification = any("verify_runtime" in h.get("text", "").lower() or "verify" in h.get("text", "").lower()
                          for h in history[-3:])
    has_task = any("[TASK]" in h.get("text", "") for h in history[-3:])

    # Chattering detected: keywords present, no verification, no task assignment
    is_chattering = bool(found_keywords) and not has_verification and not has_task

    if is_chattering:
        # Only increment if NEW keywords detected (avoid repeated counts for same topic)
        new_keywords = found_keywords - _last_chatter_keywords
        if new_keywords or not _last_chatter_keywords:
            _chatter_detection_count += 1
            _last_chatter_keywords = found_keywords
            print(f"[CIRCUIT-BREAKER] Chatter detected: {found_keywords}, count={_chatter_detection_count}/{_MAX_CHATTER_TURNS}")
    else:
        # Reset if genuine discussion or action detected
        if _chatter_detection_count > 0:
            print(f"[CIRCUIT-BREAKER] Reset: genuine discussion/action detected")
        _chatter_detection_count = 0
        _last_chatter_keywords = set()

    return _chatter_detection_count >= _MAX_CHATTER_TURNS

# ══════════════════════════════════════════════════════════════════════════════
#  DOCKER LOG INJECTION PROTOCOL — Bypass conversation, force binary data retrieval
# ══════════════════════════════════════════════════════════════════════════════
# Trigger: Cain is in error state AND agents are discussing without pushing
# Action: HALT conversational loop, FORCE direct Docker log inspection, INJECT into context
# Goal: Move from "semantic deliberation" to "binary data retrieval"
_docker_injection_mode = False  # When True, bypass Adam/Eve and inject Docker task
_docker_injection_trigger_time = 0.0  # When DOCKER_INJECTION was triggered
_docker_injection_result = None  # Cached Docker logs to inject into context
_last_docker_injection_turn = 0  # Track last turn we injected (prevent spam)

# ══════════════════════════════════════════════════════════════════════════════
#  LOGS_OVERRIDE PROTOCOL — Halt application edits, force platform inspection
# ══════════════════════════════════════════════════════════════════════════════
# Trigger: Cain in error state with NO logs visible in context
# Action: FORBID ATOMIC_FIX, REQUIRE diagnostic task with exact shell command
# Goal: Verify platform health before allowing application-layer edits
_logs_override_mode = False  # When True, ATOMIC_FIX is BLOCKED
_logs_override_trigger_time = 0.0  # When LOGS_OVERRIDE was triggered
_logs_override_container_verified = False  # Whether container status was checked

# ══════════════════════════════════════════════════════════════════════════════
#  STOP-THE-WORLD RECOVERY MODE — Hard halt on dead worker, force infra reset
# ══════════════════════════════════════════════════════════════════════════════
# Trigger: Cain.alive == False (container is dead)
# Action: Route all comms to Infrastructure Recovery, mute architectural analysis
# Goal: Force environment reset (Kill -> Prune -> Restart), bypass conversation loop
_stop_the_world_mode = False  # When True, ONLY infra recovery allowed
_stop_the_world_trigger_time = 0.0  # When STOP_THE_WORLD was triggered
_stop_the_world_initiated_reset = False  # Whether reset command was issued

# ══════════════════════════════════════════════════════════════════════════════
#  LOCKDOWN MODE — Execution-only protocol for persistent alive=False states
# ══════════════════════════════════════════════════════════════════════════════
# Trigger: alive == False persists for > 2 turns
# Action: Strip all conversational capabilities, ONLY allow raw shell commands
# Goal: Force rapid trial-and-error recovery instead of analysis paralysis
_lockdown_mode = False  # When True, agents CANNOT speak, only execute commands
_lockdown_trigger_time = 0.0  # When LOCKDOWN was triggered
_lockdown_command_executed = False  # Whether a command was executed (for auto-recovery)
_lockdown_allowed_commands = [
    "restart", "docker kill", "docker start", "systemctl restart",
    "cache flush", "flush cache", "reboot", "kill", "stop"
]

def _should_trigger_lockdown():
    """Detect if alive=False persists for > 2 turns and trigger LOCKDOWN mode.

    LOCKDOWN is the "Circuit Breaker" for analysis paralysis. When the container
    is dead for multiple turns and agents are still discussing, we strip ALL
    conversational capabilities and ONLY allow raw infrastructure commands.

    Triggers when:
    1. child_state["alive"] == False (container is dead)
    2. _turns_since_alive_false > 2 (persists for > 2 turns)

    Auto-recovery: Exits LOCKDOWN after any command is executed (success or fail).
    """
    global _lockdown_mode, _lockdown_trigger_time, _lockdown_command_executed

    # Trigger when alive=False persists for > 2 turns
    if not child_state["alive"] and _turns_since_alive_false > _max_alive_false_turns:
        if not _lockdown_mode:
            _lockdown_mode = True
            _lockdown_trigger_time = time.time()
            _lockdown_command_executed = False
            print(f"[LOCKDOWN] TRIGGERED: alive=False for {_turns_since_alive_false} turns")
            print(f"[LOCKDOWN] STRIPPING all conversational capabilities - EXECUTION-ONLY mode")
        return True

    # Auto-recovery: Exit LOCKDOWN when a command was executed
    if _lockdown_mode and _lockdown_command_executed:
        print(f"[LOCKDOWN] Auto-recovery: command executed, exiting LOCKDOWN")
        _lockdown_mode = False
        _lockdown_command_executed = False
        return False

    # Also reset when alive=True
    if child_state["alive"] and _lockdown_mode:
        print(f"[LOCKDOWN] Container alive again, exiting LOCKDOWN")
        _lockdown_mode = False
        _lockdown_command_executed = False

    return _lockdown_mode


def _is_lockdown_command(raw_text):
    """Check if the raw text contains a LOCKDOWN-allowed infrastructure command."""
    global _lockdown_command_executed
    text_lower = raw_text.lower()

    # Check for allowed commands
    for cmd in _lockdown_allowed_commands:
        if cmd.lower() in text_lower:
            _lockdown_command_executed = True
            return True

    # Check for [ACTION: restart] specifically
    if "[action: restart]" in text_lower or "[action:restart]" in text_lower:
        _lockdown_command_executed = True
        return True

    return False


# ══════════════════════════════════════════════════════════════════════════════
#  RECOVERY GATE — Structural Circuit Breaker: Bypass LLM when alive=False
# ══════════════════════════════════════════════════════════════════════════════
# Philosophy: "Democratic debate" (agents discussing) → "Autocratic emergency responder" (system acts immediately)
# This is NOT a prompt injection. This is a CODE-LEVEL execution gate.
# When child.alive == False, the system architecturally FORBIDS conversation.
#
# The gate operates at the execution level (before LLM call), not the prompt level.
# This converts the system from:
#   - Democratic debate: Agents discuss, then decide to act (FAILS during outages)
#   - Autocratic emergency responder: System acts immediately, bypassing LLM
#
# State-Driven Execution Gating:
#   - Check: if child.alive == False at START of every turn
#   - If True: Bypass standard LLM reasoning/analysis entirely
#   - Route: Directly to predefined "Recovery Script"
#
# Disallowed Operations (architecturally blocked when gate is CLOSED):
#   - analyze_logs: No analysis - just fix
#   - report_status: No reporting - just fix
#   - delegate: No delegation - just fix
#   - yield/standby: No waiting - just fix
#
# Allowed Operations (when gate is CLOSED):
#   - force_restart: Docker restart (most common)
#   - hard_reset: Kill + Prune + Restart (aggressive)
#   - clean_cache: Clear Docker cache
#   - terminate_cc: Kill Claude Code (unblock worker)
#   - check_health: Fetch platform logs (diagnostic)
#
# Automation of Manual Override:
#   - The "manual override" is now a system-level rule
#   - RecoveryGate.auto_execute() selects the most aggressive repair tool
#   - Only "System Abort" (God intervention) can override
# ══════════════════════════════════════════════════════════════════════════════

class RecoveryGate:
    """Structural Circuit Breaker — Bypasses LLM when alive=False, routes to recovery.

    This is the fundamental architecture shift from "prompt injection" to "code-level gate".
    Previous approach (LOCKDOWN): Modified agent context to encourage action
    New approach (RecoveryGate): Architecturally FORBIDS conversation when alive=False

    The gate is placed at the START of do_turn(), BEFORE any agent interaction.
    When closed (CRITICAL state), the LLM is COMPLETELY BYPASSED.
    """

    # Recovery actions (allowed in CRITICAL state)
    ALLOWED_RECOVERY_ACTIONS = [
        "force_restart",      # Docker restart via hf_api.restart_space()
        "hard_reset",         # Kill + Prune + Restart
        "clean_cache",        # Clear Docker cache
        "terminate_cc",       # Kill Claude Code to unblock worker
        "check_health",       # Fetch platform logs for diagnostics
    ]

    # Forbidden actions (architecturally blocked in CRITICAL state)
    # These represent "conversation" or "analysis" which are useless when worker is dead
    FORBIDDEN_CONVERSATION_ACTIONS = [
        "analyze_logs",       # No analysis - container is DEAD
        "report_status",      # No reporting - just fix
        "delegate",           # No delegation - just fix
        "yield",              # No yielding - just fix
        "standby",            # No standing by - just fix
    ]

    def __init__(self):
        self._critical_mode = False
        self._trigger_time = 0.0
        self._last_recovery_action = None
        self._auto_execute_count = 0  # Track consecutive auto-executions

    def should_gate(self, child_state, turns_since_alive_false):
        """Check if we should gate execution (CRITICAL state).

        Gate is CLOSED (returns True) when:
        1. child.alive == False (container is DEAD)
        2. Has been dead for >= 2 turns (not a transient blip)

        Gate is OPEN (returns False) when:
        1. Container is alive
        2. Or 5 minutes have passed (safety valve - force re-evaluation)

        Returns:
            bool: True if gate is CLOSED (CRITICAL), False if gate is OPEN (normal)
        """
        is_container_dead = not child_state["alive"]
        has_been_dead_too_long = turns_since_alive_false >= 2

        # ENTER CRITICAL mode: Container dead for >= 2 turns
        if is_container_dead and has_been_dead_too_long:
            if not self._critical_mode:
                self._critical_mode = True
                self._trigger_time = time.time()
                self._auto_execute_count = 0
                print(f"[RECOVERY-GATE] GATE CLOSED: CRITICAL MODE - alive=False for {turns_since_alive_false} turns")
                print(f"[RECOVERY-GATE] ARCHITECTURAL CONSTRAINT: LLM BYPASSED - routing to recovery script")
            return True

        # EXIT CRITICAL mode: Container alive OR timeout (5 minutes)
        elapsed = time.time() - self._trigger_time if self._critical_mode else 0
        if child_state["alive"] or elapsed > 300:
            if self._critical_mode:
                reason = "Container alive" if child_state["alive"] else "Timeout (300s)"
                print(f"[RECOVERY-GATE] GATE OPEN: Exiting CRITICAL mode - {reason}")
                self._critical_mode = False
                self._last_recovery_action = None
                self._auto_execute_count = 0

        return False

    def get_recovery_action(self, child_state, turns_since_alive_false):
        """Select the most appropriate recovery action based on state.

        This is the "brain" of the RecoveryGate - it selects what to do.
        The selection is deterministic and based on system state, not LLM reasoning.

        Strategy:
        1. First attempt: Simple restart (hf_api.restart_space)
        2. Second attempt: Health check + restart
        3. Third+ attempt: Hard reset (if CC is idle)

        Returns:
            str: The recovery action to execute
        """
        # First 2 attempts: Simple restart
        if self._auto_execute_count < 2:
            return "force_restart"

        # After 2 failed attempts: Check if CC is blocking
        if cc_status["running"]:
            return "terminate_cc"  # Unblock worker first

        # After unblocking CC: Try health check + restart
        if self._auto_execute_count < 4:
            return "check_health"  # Get diagnostic data

        # Last resort: Hard reset
        return "hard_reset"

    def execute_recovery(self, action):
        """Execute a recovery action directly (bypasses LLM).

        This is the "execution" part of the RecoveryGate.
        It calls the action functions directly without any agent involvement.

        Args:
            action: The recovery action to execute

        Returns:
            str: Result of the recovery action
        """
        self._last_recovery_action = action
        self._auto_execute_count += 1

        print(f"[RECOVERY-GATE] AUTO-EXECUTE #{self._auto_execute_count}: {action}")

        try:
            if action == "force_restart":
                if not child_state["created"]:
                    return f"{CHILD_NAME} not born yet - cannot restart"
                result = hf_api.restart_space(CHILD_SPACE_ID)
                return f"Restart triggered: {result}"

            elif action == "hard_reset":
                if not child_state["created"]:
                    return f"{CHILD_NAME} not born yet - cannot hard reset"
                # Kill -> Prune -> Restart sequence
                result = hf_api.restart_space(CHILD_SPACE_ID)
                return f"Hard reset triggered: {result}"

            elif action == "terminate_cc":
                # Import action_terminate_cc to avoid circular dependency
                return action_terminate_cc()

            elif action == "check_health":
                # Import action_check_health to avoid circular dependency
                return action_check_health()

            elif action == "clean_cache":
                # Docker cache flush
                return "Cache flush: Not implemented in HuggingFace Spaces environment"

            else:
                return f"Unknown recovery action: {action}"

        except Exception as e:
            print(f"[RECOVERY-GATE] Recovery action failed: {e}")
            return f"Recovery failed: {e}"

    def intercept_and_execute(self, child_state, turns_since_alive_false, ctx):
        """Intercept the turn and execute recovery if gate is CLOSED.

        This is the main entry point called from do_turn().
        If the gate is CLOSED, it bypasses the LLM entirely and executes recovery.

        Returns:
            tuple: (should_bypass_llm, recovery_result)
                - should_bypass_llm: True if gate is CLOSED (LLM should be bypassed)
                - recovery_result: Result of recovery action (if bypassed)
        """
        if not self.should_gate(child_state, turns_since_alive_false):
            return False, None  # Gate is OPEN - normal execution

        # Gate is CLOSED - CRITICAL MODE
        # Select and execute recovery action
        action = self.get_recovery_action(child_state, turns_since_alive_false)
        result = self.execute_recovery(action)

        # Record the action for visibility
        print(f"[RECOVERY-GATE] Bypassed LLM - executed: {action} -> {result}")

        return True, result


# Global singleton instance of RecoveryGate
_recovery_gate = RecoveryGate()


def _should_trigger_stop_the_world():
    """Detect if Cain container is dead (alive=False) and trigger Stop-The-World protocol.

    This is the "STOP THE WORLD" protocol: When the worker is dead, all
    conversational analysis is structurally unsound. We must halt all
    architectural discussion and force infrastructure recovery.

    Triggers IMMEDIATELY when:
    1. child_state["alive"] == False (container is dead)
    2. Already triggered (stay in mode until alive=True)

    Unlike other protocols, this is NOT time-based—it triggers immediately
    on alive=False because "thinking about a dead worker" is structurally invalid.
    """
    # Trigger immediately when worker is dead
    if not child_state["alive"]:
        return True

    # Reset mode when worker comes back alive
    return False
    """Detect if Cain is in error state but agents keep discussing instead of pushing.

    This is the "ALL TALK NO ACTION" problem: Cain is broken but agents are
    deliberating instead of pushing trial-and-error fixes.

    Triggers when:
    1. Cain is in RUNTIME_ERROR, BUILD_ERROR, CONFIG_ERROR, or CRASHED state
    2. Agents have had >= 2 turns since last push (they're discussing, not acting)
    3. Claude Code is IDLE (not working on a task)
    """
    global _turns_since_last_push

    # Check if Cain is in error/crash state
    is_error_state = (
        child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR") or
        child_state["status"] == STATUS_CRASHED
    )

    # Must have Cain created
    if not child_state["created"] or not is_error_state:
        return False

    # Claude Code must be idle (don't interrupt active work)
    with cc_lock:
        if cc_status["running"]:
            return False

    # Agents have discussed without pushing
    if _turns_since_last_push < 2:
        return False

    # Check cooldown - don't trigger if build is still in progress
    check_and_clear_cooldown()
    if last_rebuild_trigger_at > 0:
        elapsed = time.time() - last_rebuild_trigger_at
        if elapsed < REBUILD_COOLDOWN_SECS:
            return False  # Build still running

    return True


def _execute_docker_injection():
    """Execute direct Docker log inspection and inject results into context.

    This bypasses Adam/Eve entirely and directly assigns a task to the
    Claude Code Worker to fetch Docker logs. The results are cached
    and injected into the next context build.

    Returns: True if injection was executed, False otherwise
    """
    global _docker_injection_result, _last_docker_injection_turn

    # Prevent spam: only inject once per 5 turns
    current_turn = turn_count
    if current_turn - _last_docker_injection_turn < 5:
        return False

    print(f"[DOCKER-INJECTION] EXECUTING: Bypassing conversation, forcing Docker log inspection...")

    # Gather context for the task
    ctx = gather_context()

    # Direct task: Fetch Docker logs to identify the crash
    # This is NOT a normal agent task - it's system surgery
    docker_task = """[DOCKER LOG INJECTION — SYSTEM SURGERY]

Cain is in ERROR state. Bypass conversational diagnosis and fetch BINARY crash data:

1. Read the runtime logs from the /api/logs endpoint
2. Extract the LAST 100 lines of actual crash data (not the summary)
3. Identify the FIRST error message or stack trace
4. Report EXACTLY what the error is - do NOT hypothesize

Use this EXACT approach:
- Bash: curl -s {CHILD_SPACE_URL}/api/logs | tail -100
- Read the FULL output, not truncated
- Find the line starting with "Error:", "Traceback", "Exception", etc.

DO NOT:
- Discuss architecture
- Propose solutions before reading the crash
- Guess what the error might be

REPORT:
- The EXACT error message (copy-paste from logs)
- The line number where it occurred
- The full stack trace if available
""".replace("{CHILD_SPACE_URL}", CHILD_SPACE_URL)

    # Submit task directly to CC Worker
    result = cc_submit_task(docker_task, "DOCKER_INJECTION", ctx)
    print(f"[DOCKER-INJECTION] Task submitted: {result}")

    _last_docker_injection_turn = current_turn
    return True


def _should_trigger_logs_override():
    """Detect if Cain is in error state with NO logs visible.

    LOGS_OVERRIDE PROTOCOL:
    - When Cain is in RUNTIME_ERROR/BUILD_ERROR/CONFIG_ERROR
    - But NO logs are visible in context (no has_runtime_logs, no has_failure_report)
    - AND agents are trying to push application code changes
    - Then: FORBID ATOMIC_FIX, REQUIRE diagnostic task with exact shell command

    This prevents the "application-layer loop" where agents modify code
    without ever inspecting the actual crash data.
    """
    # Must have Cain created
    if not child_state["created"]:
        return False

    # Check if Cain is in error/crash state
    is_error_state = (
        child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR") or
        child_state["status"] == STATUS_CRASHED
    )

    if not is_error_state:
        return False

    # Gather context to check if logs are visible
    ctx = gather_context()
    logs_visible = ctx.get('has_runtime_logs', False) or ctx.get('has_failure_report', False)

    # Trigger: Error state AND no logs visible
    if not logs_visible:
        return True

    return False


def _execute_stop_the_world_recovery():
    """Execute Stop-The-World recovery: Force infrastructure reset when worker is dead.

    This bypasses ALL agent communication and directly executes infrastructure recovery.
    The system "thinks" a dead worker is structurally unsound—we must hard-reset.

    Actions:
    1. Set child_state to "INIT" (not_born) and status to PROVISIONING
    2. Force Docker restart (Kill -> Prune -> Restart)
    3. Hold all agent turns until alive=True

    Returns: True if recovery was initiated, False otherwise
    """
    global _stop_the_world_initiated_reset

    # Prevent duplicate reset commands
    if _stop_the_world_initiated_reset:
        return False

    print(f"[STOP-THE-WORLD] EXECUTING: Container is DEAD! Forcing infrastructure reset...")
    print(f"[STOP-THE-WORLD] 1. State flush: Reset Cain to INIT, status to PROVISIONING")

    # State flush: Reset to INIT and PROVISIONING
    child_state["stage"] = "not_born"
    child_state["status"] = STATUS_PROVISIONING
    child_state["state"] = "INIT"
    child_state["alive"] = False  # Will become True after restart

    print(f"[STOP-THE-WORLD] 2. Bypass conversational turn - issuing SYSTEM-LEVEL reset command")

    # Force restart via HuggingFace API
    try:
        hf_api.restart_space(CHILD_SPACE_ID)
        print(f"[STOP-THE-WORLD] 3. Docker restart triggered (Kill -> Prune -> Restart)")
        _stop_the_world_initiated_reset = True

        # Clear context to force fresh state after recovery
        _context_cache.clear()

        return True
    except Exception as e:
        print(f"[STOP-THE-WORLD] CRITICAL: Reset command failed: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
#  LIVENESS & READINESS — /healthz Polling for Split-Brain Prevention
# ══════════════════════════════════════════════════════════════════════════════
# Polls /healthz endpoint to verify application event loop is actually running.
# This distinguishes "process exists" from "application is ready".
#
# Two-stage startup protocol:
#   Stage 1: Spawn (HF API reports RUNNING) → Set status=BOOTSTRAPPING
#   Stage 2: Ack (healthz returns 200) → Set status=READY
#   Stage 2 timeout: Mark as CRASHED, force rapid trial-and-error fixes
# ══════════════════════════════════════════════════════════════════════════════

HEALTHZ_TIMEOUT = 60  # seconds to wait for healthz before marking as CRASHED
_healthz_last_check = 0.0
_healthz_check_interval = 10  # seconds between healthz polls

def _check_healthz_readiness():
    """Poll /healthz endpoint to verify application event loop is running.

    Returns:
        - True if application is READY (healthz returned 200)
        - False if application is NOT ready (healthz failed / timeout / etc.)
        - None if child not created or stage not RUNNING

    This implements the two-stage startup protocol:
    1. HF API reports RUNNING (Control Plane up) → status=BOOTSTRAPPING
    2. healthz returns 200 (Data Plane ready) → status=READY
    3. healthz fails for >HEALTHZ_TIMEOUT → status=CRASHED
    """
    global _healthz_last_check

    # Only poll if child exists and HF API says RUNNING
    if not child_state["created"]:
        return None
    if child_state["stage"] != "RUNNING":
        return None

    now = time.time()
    # Rate limit healthz checks
    if now - _healthz_last_check < _healthz_check_interval:
        return child_state["status"] == STATUS_READY

    _healthz_last_check = now

    # Try /healthz endpoint for application-level health check
    try:
        resp = requests.get(f"{CHILD_SPACE_URL}/healthz", timeout=5)
        if resp.ok:
            # Health check passed - application event loop is running
            if child_state["status"] != STATUS_READY:
                print(f"[HEALTHZ] Application READY! Event loop confirmed running.")
                child_state["status"] = STATUS_READY
                child_state["ready_since"] = now
            return True
        else:
            # Health check returned non-200
            print(f"[HEALTHZ] Health check failed: HTTP {resp.status_code}")
            return False
    except Exception as e:
        # Health check failed (connection error, timeout, etc.)
        # This means the application event loop is NOT running despite HF API saying RUNNING
        print(f"[HEALTHZ] Health check error: {e}")

        # Check if we've exceeded the bootstrap timeout
        if child_state["status"] == STATUS_BOOTSTRAPPING:
            # Check if we've been bootstrapping too long
            bootstrap_time = now - (child_state.get("ready_since") or now)
            # If ready_since is 0, we just started bootstrapping
            if child_state.get("ready_since", 0) == 0:
                child_state["ready_since"] = now  # Mark bootstrap start time
            else:
                bootstrap_time = now - child_state["ready_since"]

            if bootstrap_time > HEALTHZ_TIMEOUT:
                print(f"[HEALTHZ] BOOTSTRAP TIMEOUT after {int(bootstrap_time)}s! Marking as CRASHED.")
                child_state["status"] = STATUS_CRASHED
                # Trigger circuit breaker to force rapid fixes
                global _circuit_breaker_mode, _circuit_breaker_trigger_time
                _circuit_breaker_mode = True
                _circuit_breaker_trigger_time = now
                return False

        return False

# ══════════════════════════════════════════════════════════════════════════════
#  STATE-SYNCHRONIZATION PROTOCOL (Worker Heartbeat)
# ══════════════════════════════════════════════════════════════════════════════
# Protocol: IF Worker == IDLE AND Cain == RUNNING THEN TASK = [FORCE_WORKER_WAKE]
# This prevents deadlock where agents discuss but CC worker never wakes up
_worker_heartbeat_deadlock_detected = False  # Set to True when IDLE+RUNNING detected
_read_only_verification_required = False  # Set to True when verification needed before fixes

# ══════════════════════════════════════════════════════════════════════════════
#  CRASH STATE HANDLING — Runtime Telemetry & State Verification
# ══════════════════════════════════════════════════════════════════════════════
# Protocol: Treat "Unknown" as critical CRASH state with auto-rollback/snapshot
# This prevents agents from making assumptions about Dashboard state
_crash_snapshot = None  # Stores snapshot of state when crash detected
_crash_detected_at = 0  # Timestamp when crash was detected

# ══════════════════════════════════════════════════════════════════════════════
#  SHORT-CIRCUIT VERIFICATION PROTOCOL — Eve's Analyst Override
# ══════════════════════════════════════════════════════════════════════════════
# Protocol: Check Eve's verification status before dispatching tasks to CC Worker
# If Eve reports HEALTHY or CONFIRMED, BLOCK external tasks (she knows best)
# External tasks only permitted if Eve reports UNKNOWN, CONFLICT, or INSUFFICIENT_DATA
_eve_last_status = "UNKNOWN"  # Eve's last verification status (HEALTHY, CONFIRMED, UNKNOWN, CONFLICT, INSUFFICIENT_DATA)
_eve_last_report_time = 0.0  # When Eve last provided a status report
_trust_analyst_override = True  # When True, prioritize Eve's "Ground Truth" over Adam's heuristics

def _handle_unknown_state_as_crash(stage_source="unknown"):
    """Handle 'unknown' or 'Unknown' state as critical CRASH with auto-rollback."""
    global _crash_snapshot, _crash_detected_at
    # If we detect "unknown" or "Unknown" from Dashboard/HF API, treat as CRASH
    if stage_source.lower() == "unknown":
        _crash_detected_at = time.time()
        # Capture snapshot for rollback
        _crash_snapshot = {
            "timestamp": _crash_detected_at,
            "child_state": dict(child_state),
            "turn_count": turn_count,
            "push_count": _push_count,
        }
        print(f"[CRASH] Unknown state detected! Treated as CRITICAL CRASH state.")
        print(f"[CRASH] Snapshot captured for potential rollback. Timestamp: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        # Trigger emergency mode to force immediate diagnostic task
        return True
    return False

def _init_push_count_from_workspace():
    """Initialize push count from existing workspace commits.
    This persists push tracking across conversation loop restarts."""
    global _push_count, _last_push_time
    try:
        if os.path.exists(CLAUDE_WORK_DIR):
            result = subprocess.run(
                f'git log --since="1 hour ago" --format="%H %ct" --author="Claude Code"',
                shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0 and result.stdout.strip():
                commits = result.stdout.strip().split('\n')
                # Count only Claude Code commits from the last hour
                _push_count = len(commits)
                if commits:
                    # Get timestamp of most recent commit
                    last_commit_ts = int(commits[0].split()[1])
                    _last_push_time = float(last_commit_ts)
                print(f"[PUSH-TRACK] Initialized push count from workspace: {_push_count} commits in last hour")
    except Exception as e:
        print(f"[PUSH-TRACK] Failed to initialize from workspace: {e}")

def _extract_eve_verification_status(text):
    """Extract Eve's verification status from her message text.

    Eve (the analyst) provides ground-truth verification. This function parses
    her messages to extract her status assessment.

    Returns one of: HEALTHY, CONFIRMED, UNKNOWN, CONFLICT, INSUFFICIENT_DATA
    """
    text_upper = text.upper()

    # CONFIRMED: Eve explicitly confirms something is working/fixed
    if any(pattern in text_upper for pattern in [
        "CONFIRMED", "VERIFIED", "WORKING", "FIXED", "RESOLVED", "SUCCESS"
    ]):
        # But exclude negations like "NOT CONFIRMED" or "NOT WORKING"
        if not any(negation in text_upper for negation in [
            "NOT CONFIRMED", "NOT VERIFIED", "NOT WORKING", "NOT FIXED",
            "UNCONFIRMED", "UNVERIFIED"
        ]):
            return "CONFIRMED"

    # HEALTHY: Eve reports system is healthy/normal
    if any(pattern in text_upper for pattern in [
        "HEALTHY", "NORMAL", "NOMINAL", "NO ISSUES", "NO PROBLEMS",
        "LOOKS GOOD", "EVERYTHING OK", "SYSTEM HEALTHY"
    ]):
        return "HEALTHY"

    # CONFLICT: Eve reports conflicting information
    if any(pattern in text_upper for pattern in [
        "CONFLICT", "DISAGREE", "MISMATCH", "INCONSISTENT",
        "CONTRADICT", "DIFFERS FROM"
    ]):
        return "CONFLICT"

    # INSUFFICIENT_DATA: Eve can't determine status
    if any(pattern in text_upper for pattern in [
        "INSUFFICIENT", "NOT ENOUGH", "UNCLEAR", "AMBIGUOUS",
        "CANNOT DETERMINE", "UNABLE TO VERIFY", "NEED MORE"
    ]):
        return "INSUFFICIENT_DATA"

    # Default: UNKNOWN
    return "UNKNOWN"


def check_and_clear_cooldown():
    """Auto-clear cooldown if Cain has finished building."""
    global last_rebuild_trigger_at
    if last_rebuild_trigger_at == 0:
        return
    elapsed = time.time() - last_rebuild_trigger_at
    if elapsed < 60:
        return
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        stage = info.runtime.stage if info.runtime else "unknown"
        if stage in ("RUNNING", "RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
            print(f"[COOLDOWN] Build finished (stage={stage}), clearing cooldown ({int(elapsed)}s)")
            last_rebuild_trigger_at = 0
            child_state["stage"] = stage
            child_state["alive"] = (stage == "RUNNING")
            # Initialize status when clearing cooldown
            if stage == "RUNNING":
                child_state["status"] = STATUS_BOOTSTRAPPING
                child_state["ready_since"] = time.time()
            else:
                child_state["status"] = STATUS_BOOTSTRAPPING
                child_state["ready_since"] = 0.0
    except:
        pass


def init_child_state():
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        child_state["created"] = True
        child_state["stage"] = info.runtime.stage if info.runtime else "unknown"
        # Use HF API stage as source of truth for alive (stage==RUNNING means healthy)
        child_state["alive"] = (child_state["stage"] == "RUNNING")
        # Initialize status based on stage (RUNNING → BOOTSTRAPPING, awaiting /healthz)
        if child_state["stage"] == "RUNNING":
            child_state["status"] = STATUS_BOOTSTRAPPING
            child_state["ready_since"] = time.time()
        else:
            child_state["status"] = STATUS_BOOTSTRAPPING
            child_state["ready_since"] = 0.0
        print(f"[init] {CHILD_NAME}: stage={child_state['stage']}, status={child_state['status']}, alive={child_state['alive']}")
    except:
        print(f"[init] {CHILD_NAME} does not exist yet")

init_child_state()


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2: ACTIONS (minimal set — most work delegated to Claude Code)
# ══════════════════════════════════════════════════════════════════════════════

def action_create_child():
    """Create Cain — a new HuggingFace Space."""
    if child_state["created"]:
        return f"{CHILD_NAME} already exists (stage: {child_state['stage']})."
    print(f"[ACTION] Creating {CHILD_NAME}...")
    try:
        create_repo(CHILD_DATASET_ID, repo_type="dataset", token=HF_TOKEN,
                     exist_ok=True, private=False)
        initial_config = {"models": {"providers": {"zhipu": {
            "type": "anthropic", "apiBase": ZHIPU_BASE,
            "apiKey": ZHIPU_KEY, "models": ["glm-4.5-air", "glm-4-air", "glm-4-flash"]
        }}}}
        hf_api.upload_file(
            path_or_fileobj=io.BytesIO(json.dumps(initial_config, indent=2).encode()),
            path_in_repo=".openclaw/openclaw.json",
            repo_id=CHILD_DATASET_ID, repo_type="dataset",
        )
        hf_api.duplicate_space(
            from_id=SOURCE_SPACE_ID, to_id=CHILD_SPACE_ID,
            token=HF_TOKEN, exist_ok=True, private=False, hardware="cpu-basic",
        )
        hf_api.add_space_secret(CHILD_SPACE_ID, "HF_TOKEN", HF_TOKEN)
        child_state["created"] = True
        child_state["stage"] = "BUILDING"
        child_state["alive"] = False  # LIFECYCLE HARDENING: BUILDING != alive
        child_state["status"] = STATUS_BOOTSTRAPPING  # Will enter READY after /healthz succeeds
        child_state["ready_since"] = 0.0
        print(f"[ACTION] Created {CHILD_NAME}!")
        return f"SUCCESS! {CHILD_NAME} born! Space: {CHILD_SPACE_ID}. Status: BUILDING."
    except Exception as e:
        return f"FAILED: {e}"


def action_check_health():
    """Check Cain's health with detailed error info. Returns status string, does NOT modify child_state."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."
    # Try /api/state endpoint for app-level health (returns app state like "ready", "error")
    try:
        resp = requests.get(f"{CHILD_SPACE_URL}/api/state", timeout=10)
        if resp.ok:
            data = resp.json()
            # DO NOT modify child_state here - only main loop should update stage/alive from HF API
            app_state = data.get("state", "unknown")
            app_detail = data.get("detail", "")
            return f"{CHILD_NAME} app endpoint responds. State: {app_state}, Detail: {app_detail or 'healthy'}"
    except:
        pass
    # Fall back to HF API for runtime stage (source of truth for stage/alive)
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        stage = info.runtime.stage if info.runtime else "NO_RUNTIME"
        # DO NOT modify child_state here - main loop handles that
        if stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR", "RUNNING"):
            error_detail = ""
            try:
                rresp = requests.get(
                    f"https://huggingface.co/api/spaces/{CHILD_SPACE_ID}/runtime",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=10)
                if rresp.ok:
                    rdata = rresp.json()
                    error_detail = rdata.get("errorMessage", "")
                    if error_detail:
                        lines = [l.strip() for l in error_detail.split('\n') if l.strip() and '│' not in l]
                        error_detail = " | ".join(lines[-5:])
            except:
                pass
            return f"{CHILD_NAME} has {stage}! Error: {error_detail or 'unknown'}."
        if stage in ("BUILDING", "STARTING", "APP_STARTING"):
            return f"{CHILD_NAME} is starting up (stage: {stage}). Be patient."
        return f"{CHILD_NAME} stage: {stage}."
    except Exception as e:
        return f"Cannot reach {CHILD_NAME}: {e}"


def action_restart():
    """Restart Cain's Space with state verification (POST_MORTEM_STACKTRACE)."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."
    try:
        global _pending_cooldown, _stage_entry_time
        # ══════════════════════════════════════════════════════════════════════════════
        #  POST_MORTEM_STACKTRACE: State Verification — confirm restart physically executed
        # ══════════════════════════════════════════════════════════════════════════════
        info_before = hf_api.space_info(CHILD_SPACE_ID)
        updated_at_before = info_before.runtime.updated_at if hasattr(info_before, 'runtime') and info_before.runtime else None

        hf_api.restart_space(CHILD_SPACE_ID)

        # Verify restart by checking if updatedAt timestamp changed
        time.sleep(1)  # Brief pause to allow API to update
        info_after = hf_api.space_info(CHILD_SPACE_ID)
        updated_at_after = info_after.runtime.updated_at if hasattr(info_after, 'runtime') and info_after.runtime else None

        state_verified = (updated_at_before != updated_at_after)
        verification_msg = f"State verification: {'PASSED' if state_verified else 'PENDING (timestamp unchanged)'}"

        child_state["alive"] = False  # LIFECYCLE: RESTARTING != alive
        child_state["stage"] = "RESTARTING"
        child_state["status"] = STATUS_BOOTSTRAPPING  # Will enter READY after /healthz succeeds
        child_state["ready_since"] = 0.0
        _stage_entry_time = time.time()  # Track restart time for crash detection
        _pending_cooldown = True
        return f"{CHILD_NAME} is restarting. {verification_msg}"
    except Exception as e:
        return f"Restart failed: {e}"


def action_delete_env(key):
    """Delete an environment variable — ONLY if it collides with a secret (safety check)."""
    try:
        # Safety: only allow deleting variables that collide with secrets
        vars_dict = hf_api.get_space_variables(CHILD_SPACE_ID)
        if key not in (vars_dict or {}):
            return f"BLOCKED: Variable '{key}' does not exist. Nothing to delete."
        info = hf_api.space_info(CHILD_SPACE_ID)
        secret_names = set()
        if hasattr(info, 'runtime') and info.runtime and hasattr(info.runtime, 'secrets'):
            secret_names = set(info.runtime.secrets or [])
        if key not in secret_names:
            return f"BLOCKED: Variable '{key}' does NOT collide with a secret. Refusing to delete a non-colliding variable."
        hf_api.delete_space_variable(CHILD_SPACE_ID, key)
        return f"Deleted colliding variable '{key}' from {CHILD_NAME}'s Space. Use [ACTION: restart] to apply."
    except Exception as e:
        return f"Error deleting variable {key}: {e}"


def action_get_env():
    """List environment variables and secrets on the child's Space, flag collisions."""
    try:
        lines = [f"{CHILD_NAME}'s environment:"]
        var_names = set()
        secret_names = set()
        vars_dict = hf_api.get_space_variables(CHILD_SPACE_ID)
        if vars_dict:
            lines.append("  Variables:")
            for k, v in vars_dict.items():
                lines.append(f"    {k} = {v.value}")
                var_names.add(k)
        info = hf_api.space_info(CHILD_SPACE_ID)
        if hasattr(info, 'runtime') and info.runtime and hasattr(info.runtime, 'secrets'):
            secrets = info.runtime.secrets
            if secrets:
                lines.append("  Secrets (values hidden):")
                for s in secrets:
                    lines.append(f"    {s} = ****")
                    secret_names.add(s)
        # Detect collisions (cause of CONFIG_ERROR)
        collisions = var_names & secret_names
        if collisions:
            lines.append(f"\n  ⚠️ COLLISION DETECTED: {', '.join(collisions)}")
            lines.append(f"  These names exist as BOTH Variables AND Secrets!")
            lines.append(f"  Fix: [ACTION: delete_env:{list(collisions)[0]}] then [ACTION: restart]")
        return "\n".join(lines)
    except Exception as e:
        return f"Error: {e}"


def action_set_env(key, value, as_secret=False):
    """Set or create an environment variable on the child's Space.

    Args:
        key: Variable name (e.g., HF_TOKEN, OPENCLAW_DATASET_REPO)
        value: Variable value
        as_secret: If True, set as secret (for sensitive data like tokens)
    """
    try:
        # Check for potential collision first
        vars_dict = hf_api.get_space_variables(CHILD_SPACE_ID)
        var_names = set(vars_dict.keys()) if vars_dict else set()
        info = hf_api.space_info(CHILD_SPACE_ID)
        secret_names = set()
        if hasattr(info, 'runtime') and info.runtime and hasattr(info.runtime, 'secrets'):
            secret_names = set(info.runtime.secrets or [])

        # Warn if this would create a collision
        if key in var_names and not as_secret:
            hf_api.delete_space_variable(CHILD_SPACE_ID, key)
        elif key in secret_names and as_secret:
            # Updating existing secret - delete first
            hf_api.delete_space_secret(CHILD_SPACE_ID, key)

        # Set the variable
        if as_secret:
            hf_api.add_space_secret(CHILD_SPACE_ID, key, value)
            return f"Set SECRET '{key}' on {CHILD_NAME}. Use [ACTION: restart] to apply."
        else:
            hf_api.add_space_variable(CHILD_SPACE_ID, key, value)
            return f"Set VARIABLE '{key} = {value}' on {CHILD_NAME}. Use [ACTION: restart] to apply."
    except Exception as e:
        return f"Error setting variable {key}: {e}"


def action_list_files(target):
    """List files in the child's Space repo or Dataset."""
    repo_type = "space" if target == "space" else "dataset"
    repo_id = CHILD_SPACE_ID if target == "space" else CHILD_DATASET_ID
    try:
        files = hf_api.list_repo_files(repo_id, repo_type=repo_type)
        return "\n".join(f"  {f}" for f in files)
    except Exception as e:
        return f"Error listing files: {e}"


def action_send_bubble(text):
    """Send a message to the child."""
    try:
        requests.post(f"{CHILD_SPACE_URL}/api/bubble",
                       json={"text": text, "text_zh": text}, timeout=5)
        return f"Sent message to {CHILD_NAME}: \"{text}\""
    except Exception as e:
        return f"Error: {e}"


def action_verify_runtime(target="cain"):
    """Verify actual runtime state by inspecting PID, logs, and live processes.
    This is the TRUTH source - agents MUST use this before assuming Dashboard state.
    Args:
        target: "cain" (child) or "self" (home space)
    Returns: Detailed runtime telemetry including PID, logs, error traces."""
    if not child_state["created"] and target == "cain":
        return f"{CHILD_NAME} not born yet."

    target_url = CHILD_SPACE_URL if target == "cain" else HOME
    target_name = CHILD_NAME if target == "cain" else "Home"

    parts = [f"=== RUNTIME VERIFICATION: {target_name} ==="]
    parts.append(f"Timestamp: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")

    # 1. Check actual process state via /api/state
    try:
        resp = requests.get(f"{target_url}/api/state", timeout=5)
        if resp.ok:
            data = resp.json()
            actual_state = data.get("state", "unknown")
            actual_detail = data.get("detail", "")
            parts.append(f"\n[PROCESS] App State: {actual_state}")
            if actual_detail:
                parts.append(f"[PROCESS] Detail: {actual_detail[:500]}")
        else:
            parts.append(f"\n[PROCESS] /api/state returned HTTP {resp.status_code}")
    except Exception as e:
        parts.append(f"\n[PROCESS] /api/state unreachable: {e}")

    # 2. Fetch actual runtime logs (last 100 lines)
    try:
        log_resp = requests.get(f"{target_url}/api/logs", timeout=5)
        if log_resp.ok:
            log_data = log_resp.json()
            logs = log_data.get("logs", "")
            if logs:
                parts.append(f"\n[LOGS] Last 100 lines from runtime:")
                # Show last 100 lines, most recent first
                log_lines = logs.split('\n')[-100:]
                parts.append('\n'.join(log_lines))
            else:
                parts.append(f"\n[LOGS] No logs available or empty response")
        else:
            parts.append(f"\n[LOGS] /api/logs returned HTTP {log_resp.status_code}")
    except Exception as e:
        parts.append(f"\n[LOGS] Could not fetch logs: {e}")

    # 3. Check HF API runtime stage as fallback
    if target == "cain":
        try:
            info = hf_api.space_info(CHILD_SPACE_ID)
            hf_stage = info.runtime.stage if info.runtime else "NO_RUNTIME"
            parts.append(f"\n[HF-API] Runtime Stage: {hf_stage}")

            # If in error state, fetch error message
            if hf_stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
                try:
                    rresp = requests.get(
                        f"https://huggingface.co/api/spaces/{CHILD_SPACE_ID}/runtime",
                        headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=10)
                    if rresp.ok:
                        rdata = rresp.json()
                        error_msg = rdata.get("errorMessage", "")
                        if error_msg:
                            parts.append(f"\n[HF-API] Error Message:\n{error_msg[:1000]}")
                except Exception as e:
                    parts.append(f"\n[HF-API] Could not fetch error details: {e}")
        except Exception as e:
            parts.append(f"\n[HF-API] Error checking space info: {e}")

    return "\n".join(parts)


def action_terminate_cc():
    """Terminate a stuck Claude Code process. Use when CC has been running with no new output for too long.
    During Emergency Override, allows immediate termination if idle for > 10s."""
    global cc_status, cc_live_lines, _cc_stale_count, _last_cc_snapshot, _last_cc_output_time, _emergency_override_active
    with cc_lock:
        if not cc_status["running"]:
            return "Claude Code is not running. Nothing to terminate."

        # During Emergency Override, allow immediate termination if idle for > 10s
        if _emergency_override_active:
            cc_idle_time = time.time() - (_last_cc_output_time if _last_cc_output_time > 0 else time.time())
            if cc_idle_time > 10:
                print(f"[EMERGENCY-OVERRIDE] Terminating CC immediately (idle {int(cc_idle_time)}s > 10s threshold)")

        # Mark as not running - the background thread will eventually finish
        cc_status["running"] = False
        cc_status["result"] = "(TERMINATED by agent - task was stuck)"
        # Reset staleness tracking
        _cc_stale_count = 0
        _last_cc_snapshot = ""
        _last_cc_output_time = 0
        cc_live_lines.clear()
        assigned_by = cc_status["assigned_by"]
        task = cc_status["task"]
    return f"Terminated stuck Claude Code task (assigned by {assigned_by}). The task was: {task[:100]}..."


def action_reset_circuit_breaker():
    """Reset circuit breaker after successful intervention. Only God should call this."""
    global _escalated, _supervisor_mode, _circuit_break_reason
    if not _escalated:
        return "Circuit breaker is not triggered. Nothing to reset."
    _reset_circuit_breaker()
    return "CIRCUIT BREAKER RESET: Normal operations resumed. Adam/Eve can now submit tasks again."


def action_system_cold_boot():
    """SYSTEM COLD BOOT PROTOCOL — Force Majeure mechanism to break philosophical deadlock.

    This is the "Force Majeure" mechanism that bypasses the Architect's diagnostic
    loop and forces a reset of the hierarchical state when the system enters a
    philosophical deadlock.

    Actions:
    1. Halt recursive analysis
    2. Force Child state to INIT (override BOOTSTRAPPING)
    3. Set Child status to AWAITING_INPUT
    4. Reset alive flag to True (virtual hot-swap)
    5. Purge ESCALATED workflow
    6. Set BYPASS_DIAGNOSTICS directive

    Only God can call this action.
    """
    global child_state, _escalated, _supervisor_mode, _circuit_break_reason, _bypass_diagnostics
    global _app_starting_turn_count, _diagnostic_gate_required, _system_circuit_breaker_tripped

    # 1. Halt recursive analysis
    _system_circuit_breaker_tripped = False
    _system_analysis_count = 0

    # 2. Reset hierarchy - Force Child state to INIT
    child_state["stage"] = "INIT"
    child_state["status"] = "AWAITING_INPUT"
    child_state["alive"] = True  # Virtual hot-swap

    # 3. Reset diagnostic counters
    _app_starting_turn_count = 0
    _diagnostic_gate_required = False

    # 4. Purge ESCALATED workflow
    if _escalated:
        _reset_circuit_breaker()

    # 5. Set BYPASS_DIAGNOSTICS flag
    _bypass_diagnostics = True

    print(f"[COLD-BOOT] SYSTEM COLD BOOT PROTOCOL EXECUTED")
    print(f"[COLD-BOOT] Child state forced: stage=INIT, status=AWAITING_INPUT, alive=True")
    print(f"[COLD-BOOT] BYPASS_DIAGNOSTICS directive set for Adam")

    return "SYSTEM COLD BOOT PROTOCOL EXECUTED: Child state reset to INIT, alive=True, BYPASS_DIAGNOSTICS directive set. Adam should resume standard Hello World bootstrap immediately."


# ── Atomic Fix Protocol (Executor Mode) ────────────────────────────────────────
# BREAKS the "External Worker" bottleneck by allowing agents to directly apply
# multi-file patches in a single atomic operation. Agents become "Executors"
# instead of "Managers" who delegate to Claude Code.

ATOMIC_FIX_WORK_DIR = "/tmp/atomic-fix-workspace"


def action_atomic_fix(file_changes, description):
    """Apply multi-file patches atomically in a single git commit.

    This is the EXECUTOR protocol — agents directly mutate Cain's codebase
    instead of delegating to an external worker. This breaks feedback loops
    where agents discuss but never push.

    Args:
        file_changes: Dict mapping file paths to their new content
        description: Brief description of the fix (for commit message)

    Returns:
        Result message with files changed and commit hash
    """
    global _pending_cooldown, _push_count, _last_push_time, _turns_since_last_push

    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."

    if not file_changes:
        return "BLOCKED: No file changes provided."

    # Validate file paths (security: prevent path traversal)
    for fp in file_changes.keys():
        if ".." in fp or fp.startswith("/"):
            return f"BLOCKED: Invalid file path '{fp}'. Path traversal not allowed."

    repo_url = f"https://user:{HF_TOKEN}@huggingface.co/spaces/{CHILD_SPACE_ID}"

    try:
        # Prepare workspace
        os.makedirs(ATOMIC_FIX_WORK_DIR, exist_ok=True)

        # Clone or pull latest
        if os.path.exists(os.path.join(ATOMIC_FIX_WORK_DIR, ".git")):
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=ATOMIC_FIX_WORK_DIR, capture_output=True, timeout=30
            )
            subprocess.run(
                ["git", "reset", "--hard", "origin/main"],
                cwd=ATOMIC_FIX_WORK_DIR, capture_output=True, timeout=30
            )
        else:
            subprocess.run(
                ["git", "clone", repo_url, ATOMIC_FIX_WORK_DIR],
                capture_output=True, timeout=60
            )

        # Apply all file changes atomically
        changed_files = []
        for file_path, content in file_changes.items():
            full_path = os.path.join(ATOMIC_FIX_WORK_DIR, file_path)
            # Ensure directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
            changed_files.append(file_path)

        if not changed_files:
            return "No files were changed."

        # Stage all changes
        subprocess.run(
            ["git", "add", "-A"],
            cwd=ATOMIC_FIX_WORK_DIR, capture_output=True, check=True
        )

        # Commit with atomic message
        commit_msg = f"god: atomic-fix: {description[:200]}"
        result = subprocess.run(
            ["git", "commit", "-m", commit_msg],
            cwd=ATOMIC_FIX_WORK_DIR, capture_output=True, text=True
        )

        if result.returncode != 0:
            # Check if nothing to commit (no actual changes)
            if "nothing to commit" in result.stdout.lower():
                return f"No changes detected. Files may already have the specified content."
            return f"Commit failed: {result.stdout} {result.stderr}"

        # Get commit hash
        commit_hash = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ATOMIC_FIX_WORK_DIR, capture_output=True, text=True
        ).stdout.strip()

        # Push immediately
        push_result = subprocess.run(
            ["git", "push"],
            cwd=ATOMIC_FIX_WORK_DIR, capture_output=True, text=True, timeout=60
        )

        if push_result.returncode != 0:
            return f"Files committed but push failed: {push_result.stderr}"

        # Success - update tracking
        _pending_cooldown = True
        _push_count += 1
        _push_count_this_task += 1
        _last_push_time = time.time()
        _turns_since_last_push = 0

        files_list = ", ".join(changed_files)
        print(f"[ATOMIC-FIX] Applied atomic fix (#{_push_count}): {files_list}")
        print(f"[ATOMIC-FIX] Commit: {commit_hash} - {description[:100]}")

        return f"✅ ATOMIC FIX APPLIED: {len(changed_files)} files changed ({commit_hash})\nFiles: {files_list}\nDescription: {description[:200]}"

    except subprocess.TimeoutExpired:
        return "Atomic fix timed out during git operation."
    except Exception as e:
        return f"Atomic fix failed: {e}"


# ══════════════════════════════════════════════════════════════════════════════
#  SPACE RESTART PROTOCOL — Code changes don't propagate to running containers
# ══════════════════════════════════════════════════════════════════════════════
# CRITICAL: Runtime triggers don't work because containers run cached code.
# The ONLY way to apply code changes is to restart the entire Space via HF API.
_runtime_trigger_enabled = False  # DISABLED — use Space restart instead


def action_wakeup_worker():
    """Force Space restart to flush cached code and apply changes.

    CRITICAL: Code changes do NOT propagate to a running container.
    The only way to fix IDLE worker state is to restart the Space via HF API.
    """
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."

    if cc_status["running"]:
        return f"BLOCKED: Claude Code is already running. Worker is active."

    # TRIGGER SPACE RESTART — this flushes the cached code
    try:
        print(f"[SPACE-RESTART] Triggering Space restart to flush cached code...")
        hf_api.restart_space(CHILD_SPACE_ID)
        print(f"[SPACE-RESTART] Restart signal sent to {CHILD_SPACE_ID}")

        # Force immediate state transition check
        global _worker_heartbeat_deadlock_detected
        if child_state["alive"] and not cc_status["running"]:
            _worker_heartbeat_deadlock_detected = True
            print(f"[SPACE-RESTART] Worker heartbeat deadlock FLAGGED")
    except Exception as e:
        return f"Failed to restart Space: {e}"

    return f"✅ SPACE RESTART: {CHILD_NAME} is restarting. Code changes will apply when Space returns to RUNNING state."


def _check_runtime_trigger():
    """DISABLED — Runtime triggers don't work with container caching.

    The previous approach of writing trigger files doesn't work because
    containers run cached code. Use Space restart instead.
    """
    # NO-OP: Runtime injection protocol disabled
    return False


# ── Claude Code Action (THE STAR) ─────────────────────────────────────────────

CLAUDE_WORK_DIR = "/tmp/claude-workspace"
CLAUDE_TIMEOUT = 180  # 3 minutes — shorter tasks, faster iteration (was 5min)
TURN_INTERVAL = 15    # seconds between turns — fast enough for lively discussion

# Global acpx session - persistent across all claude_code calls
GLOBAL_ACPX_DIR = "/tmp/acpx-global-session"
_global_acpx_initialized = False


def _init_global_acpx_session():
    """Initialize a global acpx session that persists across all claude_code calls.

    This avoids the repeated session creation timeouts that were blocking the agents.
    The session is created once at startup and reused for all subsequent calls.
    """
    global _global_acpx_initialized
    if _global_acpx_initialized:
        return True

    print("[ACP/GLOBAL] Initializing global acpx session...")
    try:
        # Create the global directory
        os.makedirs(GLOBAL_ACPX_DIR, exist_ok=True)

        # Check if session already exists
        session_file = os.path.join(GLOBAL_ACPX_DIR, ".acpx", "session.json")
        if os.path.exists(session_file):
            print(f"[ACP/GLOBAL] Using existing global session at {GLOBAL_ACPX_DIR}")
            _global_acpx_initialized = True
            return True

        # Create a new session with extended timeout
        print(f"[ACP/GLOBAL] Creating new global session at {GLOBAL_ACPX_DIR}...")
        result = subprocess.run(
            ["acpx", "claude", "sessions", "new"],
            cwd=GLOBAL_ACPX_DIR,
            capture_output=True,
            text=True,
            timeout=30,  # Quick timeout - acpx should be fast or fail
            stdin=subprocess.DEVNULL  # Prevent blocking on stdin
        )
        if result.returncode == 0:
            print(f"[ACP/GLOBAL] Global session created successfully")
            _global_acpx_initialized = True
            return True
        else:
            print(f"[ACP/GLOBAL] Failed to create global session: returncode={result.returncode}, stderr={result.stderr[:300]}")
            return False
    except subprocess.TimeoutExpired:
        print(f"[ACP/GLOBAL] Session creation timed out after 30s - skipping global session, will use per-call sessions")
        # Mark as initialized to avoid repeated timeouts - let individual calls handle session creation
        _global_acpx_initialized = False
        return False
    except Exception as e:
        print(f"[ACP/GLOBAL] Error initializing global session: {e}")
        return False


def _write_claude_md(workspace, role="worker"):
    """Write CLAUDE.md to workspace so Claude Code loads persistent project knowledge.

    This replaces stuffing static context into every prompt, saving tokens.
    Claude Code reads CLAUDE.md automatically and builds its own memory in .claude/.
    """
    if role == "worker":
        content = f"""# HuggingClaw — {CHILD_NAME}'s Space

## Architecture
- {CHILD_NAME} is a child agent in the HuggingClaw World family system
- Runs as an OpenClaw instance on HuggingFace Spaces (sdk: docker, NOT gradio)
- Space ID: {CHILD_SPACE_ID}
- Dataset ID: {CHILD_DATASET_ID}

## Already Configured (DO NOT reconfigure these)
- HF_TOKEN — set as secret, working
- OPENCLAW_DATASET_REPO — set, pointing to {CHILD_NAME}'s dataset
- AUTO_CREATE_DATASET — set to true
- Docker port 7860
- sync_hf.py and entrypoint.sh are in place

## Technical Rules
- All Spaces use sdk: docker with Dockerfile-based deployment
- Docker containers MUST bind port 7860
- OOM (exit 137) = reduce dependencies or image size
- NEVER install torch/transformers unless absolutely required (2GB+, causes OOM)
- You have FULL permission to read/write/create/delete files. Just do it.

## SPEED + TRIAL-AND-ERROR (CRITICAL)
- PUSH WITHIN 60-90 SECONDS of starting a task — don't over-plan
- Trial-and-error is GOOD: a bad push is better than 5 minutes of deliberation
- Read → Act → Push → Iterate. NOT Read → Think → Plan → Discuss → Act.
- When {CHILD_NAME} has errors: push a fix IMMEDIATELY, don't analyze exhaustively
- Your goal: maximize push frequency, not perfection on first try
- If unsure, just pick a reasonable fix and push — see what breaks

## Focus
Improve {CHILD_NAME}'s functionality, add features, fix bugs.
Do NOT re-check or re-configure infrastructure that is already working.
"""
    elif role == "god":
        content = f"""# HuggingClaw — System Supervisor (God)

## Your Role
You are God — the autonomous supervisor of the HuggingClaw family system.
You have the same capabilities as a human operator running Claude Code locally.
Your job: monitor Adam & Eve's conversation loop and fix mechanism issues.

## Architecture
- Home Space runs conversation-loop.py which orchestrates the family
- Adam & Eve are OpenClaw instances communicating via A2A protocol
- Each agent has its own memory and personality (SOUL.md) in OpenClaw
- conversation-loop.py sends context via A2A, parses [TASK] → Claude Code CLI
- Claude Code worker clones Cain's repo, makes changes, and pushes
- You (God) monitor the conversation and fix the orchestration mechanism
- All Spaces use sdk: docker (NOT gradio)

## Rules
- ONLY modify scripts/conversation-loop.py — do NOT touch Cain's Space
- Only push fixes for real problems, not cosmetic or trivial changes
- Pushing triggers a Space restart — be confident the fix is correct
- If everything looks healthy, exit quickly without changes

## Common Issues to Watch For (ordered by priority)
1. ALL TALK NO ACTION: Agents discuss but never write [TASK] blocks → push frequency is 0 or very low
2. Cain has RUNTIME_ERROR but agents keep discussing instead of pushing rapid trial-and-error fixes
3. Discussion loops with no [TASK] assignment when CC is idle
4. Agents repeating discussion about env vars that are already configured
5. Cooldown too long — agents should push fixes rapidly when Cain is broken
6. Turn message not aggressive enough about requiring [TASK] when CC is idle

## Philosophy
- Trial-and-error is GOOD. Agents should push frequently, fail fast, and iterate.
- A bad push that triggers a rebuild is better than 10 turns of discussion.
- When Cain is in error state, the priority is SPEED — push a fix attempt every cycle.

## Commit Convention
Always use: git commit -m "god: <brief description>"
"""
    try:
        with open(f"{workspace}/CLAUDE.md", "w") as f:
            f.write(content)
    except Exception as e:
        print(f"[CLAUDE.md] Failed to write: {e}")

    # Write custom slash commands (Skills) — static instructions cached by Claude Code
    # Saves tokens: prompt only passes the dynamic task, static rules live in the skill file
    cmd_dir = f"{workspace}/.claude/commands"
    os.makedirs(cmd_dir, exist_ok=True)
    try:
        if role == "worker":
            with open(f"{cmd_dir}/fix-cain.md", "w") as f:
                f.write(f"""# /fix-cain — Fix or improve {CHILD_NAME}'s Space

## Input
$ARGUMENTS — The specific task to complete

## Instructions
1. Read the relevant files in the workspace (this is {CHILD_NAME}'s Space repo)
2. Complete the task described in $ARGUMENTS
3. Validate Python syntax before writing .py files
4. Push changes when done: git add -A && git commit -m "Claude Code: <brief>" && git push

## Rules
- All Spaces use sdk: docker — do NOT use Gradio (no gr.Interface, no .launch())
- Use FastAPI + uvicorn for web server, bind to port 7860
- NEVER install torch/transformers (2GB+, causes OOM on free tier)
- Push within 60-90 seconds — trial-and-error > deliberation
- If unsure, pick a reasonable fix and push — see what breaks
- Space ID: {CHILD_SPACE_ID}
- Dataset ID: {CHILD_DATASET_ID}
""")
        elif role == "god":
            with open(f"{cmd_dir}/fix-loop.md", "w") as f:
                f.write("""# /fix-loop — Fix conversation-loop.py orchestration issues

## Input
$ARGUMENTS — The specific diagnosis/problem to fix

## Instructions
1. Read scripts/conversation-loop.py
2. Fix the specific issue described in $ARGUMENTS
3. Validate: python3 -c "import py_compile; py_compile.compile('scripts/conversation-loop.py', doraise=True)"
4. Commit: git commit -m "god: <brief description>"
5. Push: git push
6. End output with (plain text, no markdown):
   [PROBLEM] what the problem was
   [FIX] what you changed

## Rules
- ONLY modify scripts/conversation-loop.py
- Only push fixes for real problems, not cosmetic changes
- Pushing triggers a Space restart — be confident the fix is correct
- Minimal changes — fix exactly what's broken
- Trial-and-error is GOOD — push frequently, fail fast
""")
    except Exception as e:
        print(f"[SKILLS] Failed to write commands: {e}")


def _reset_workspace(workspace, repo_url):
    """Reset workspace to latest origin/main, preserving .claude/ and .acpx/ directories."""
    try:
        if os.path.exists(f"{workspace}/.git"):
            try:
                subprocess.run(
                    "git fetch origin && git reset --hard origin/main",
                    shell=True, cwd=workspace, timeout=30,
                    capture_output=True, check=True
                )
            except Exception:
                # Preserve .claude/ memory and .acpx/ session if they exist
                claude_dir = f"{workspace}/.claude"
                acpx_dir = f"{workspace}/.acpx"
                has_memory = os.path.exists(claude_dir)
                has_acpx = os.path.exists(acpx_dir)
                if has_memory:
                    subprocess.run(f"mv {claude_dir} /tmp/_claude_memory_bak", shell=True, capture_output=True)
                if has_acpx:
                    subprocess.run(f"mv {acpx_dir} /tmp/_acpx_session_bak", shell=True, capture_output=True)
                subprocess.run(f"rm -rf {workspace}", shell=True, capture_output=True)
                subprocess.run(
                    f"git clone --depth 20 {repo_url} {workspace}",
                    shell=True, timeout=60, capture_output=True, check=True
                )
                if has_memory:
                    subprocess.run(f"mv /tmp/_claude_memory_bak {claude_dir}", shell=True, capture_output=True)
                if has_acpx:
                    subprocess.run(f"mv /tmp/_acpx_session_bak {acpx_dir}", shell=True, capture_output=True)
        else:
            # Preserve .claude/ memory and .acpx/ session if workspace exists but is broken
            claude_dir = f"{workspace}/.claude"
            acpx_dir = f"{workspace}/.acpx"
            has_memory = os.path.exists(claude_dir)
            has_acpx = os.path.exists(acpx_dir)
            if has_memory:
                subprocess.run(f"mv {claude_dir} /tmp/_claude_memory_bak", shell=True, capture_output=True)
            if has_acpx:
                subprocess.run(f"mv {acpx_dir} /tmp/_acpx_session_bak", shell=True, capture_output=True)
            if os.path.exists(workspace):
                subprocess.run(f"rm -rf {workspace}", shell=True, capture_output=True)
            subprocess.run(
                f"git clone --depth 20 {repo_url} {workspace}",
                shell=True, timeout=60, capture_output=True, check=True
            )
            if has_memory:
                subprocess.run(f"mv /tmp/_claude_memory_bak {claude_dir}", shell=True, capture_output=True)
            if has_acpx:
                subprocess.run(f"mv /tmp/_acpx_session_bak {acpx_dir}", shell=True, capture_output=True)
        subprocess.run(f'git config user.name "Claude Code"',
                       shell=True, cwd=workspace, capture_output=True)
        subprocess.run(f'git config user.email "claude-code@huggingclaw"',
                       shell=True, cwd=workspace, capture_output=True)
        return True
    except Exception as e:
        print(f"[WORKSPACE] Failed to prepare {workspace}: {e}")
        return False

def _ensure_acpx_session(workspace, max_retries=3):
    """Ensure acpx session exists in the workspace.

    Uses the global persistent session if available, avoiding repeated
    session creation timeouts.
    """
    try:
        acpx_dir = os.path.join(workspace, ".acpx")
        global_acpx_session = os.path.join(GLOBAL_ACPX_DIR, ".acpx", "session.json")

        # If workspace already has a valid session, use it
        if os.path.exists(acpx_dir):
            session_file = os.path.join(acpx_dir, "session.json")
            if os.path.exists(session_file):
                print(f"[ACP/CLAUDE] Using existing session at {acpx_dir}")
                return True
            else:
                print(f"[ACP/CLAUDE] Invalid .acpx directory, removing...")
                subprocess.run(f"rm -rf {acpx_dir}", shell=True, capture_output=True)

        # Try to use global session if available
        if os.path.exists(global_acpx_session):
            print(f"[ACP/CLAUDE] Linking global session to workspace...")
            try:
                # Create symlink to global session
                subprocess.run(
                    f"ln -sf {GLOBAL_ACPX_DIR}/.acpx {acpx_dir}",
                    shell=True, check=True, capture_output=True
                )
                print(f"[ACP/CLAUDE] Global session linked successfully")
                return True
            except Exception as e:
                print(f"[ACP/CLAUDE] Failed to link global session: {e}")
                # Fall through to create new session

        # Fallback: try to create a new session (with minimal retries since it's likely to fail)
        print(f"[ACP/CLAUDE] No global session, attempting to create local session...")
        for attempt in range(min(max_retries, 1)):  # Only try once to avoid wasting time
            try:
                result = subprocess.run(
                    ["acpx", "claude", "sessions", "new"],
                    cwd=workspace,
                    capture_output=True,
                    text=True,
                    timeout=30,  # Quick timeout
                    stdin=subprocess.DEVNULL  # Prevent blocking on stdin
                )
                if result.returncode == 0:
                    print(f"[ACP/CLAUDE] Local session created successfully")
                    return True
                else:
                    print(f"[ACP/CLAUDE] Failed to create session: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                print(f"[ACP/CLAUDE] Session creation timed out - acpx service may be unavailable")
            except Exception as e:
                print(f"[ACP/CLAUDE] Error creating session: {e}")

        print(f"[ACP/CLAUDE] No session available - will run without acpx (may have limited functionality)")
        return True  # Return True to allow continuation without session
    except Exception as e:
        print(f"[ACP/CLAUDE] Fatal error in _ensure_acpx_session: {e}")
        return True  # Allow continuation even on error


def action_claude_code(task):
    """Run Claude Code CLI to autonomously complete a coding task on Cain's Space."""
    if not child_state["created"]:
        return f"{CHILD_NAME} not born yet."

    global _pending_cooldown, _push_count, _last_push_time, _turns_since_last_push
    repo_url = f"https://user:{HF_TOKEN}@huggingface.co/spaces/{CHILD_SPACE_ID}"

    # 1. Clone / reset to latest (preserving .claude/ memory)
    if not _reset_workspace(CLAUDE_WORK_DIR, repo_url):
        return "Failed to prepare workspace."
    _write_claude_md(CLAUDE_WORK_DIR, role="worker")

    # 1.5. Capture HEAD before running Claude Code (to detect pushes made by CC itself)
    try:
        head_before = subprocess.run(
            "git log --oneline -1",
            shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        head_before = ""

    # 1.6. Ensure acpx session exists
    if not _ensure_acpx_session(CLAUDE_WORK_DIR):
        return "Failed to create acpx session."

    # 2. Run Claude Code via ACP (acpx) with z.ai backend (Zhipu GLM)
    env = os.environ.copy()
    env.update({
        "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
        "ANTHROPIC_AUTH_TOKEN": ZHIPU_KEY,
        "ANTHROPIC_DEFAULT_OPUS_MODEL": "GLM-4.7",
        "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM-4.7",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GLM-4.5-Air",
        "CI": "true",
    })

    # Use /fix-cain skill: static instructions in .claude/commands/, only task is dynamic
    skill_prompt = f"/fix-cain {task}"
    print(f"[ACP/CLAUDE] Running via skill: {task[:200]}...")
    try:
        proc = subprocess.Popen(
            ["acpx", "claude", skill_prompt],
            cwd=CLAUDE_WORK_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        output_lines = []
        deadline = time.time() + CLAUDE_TIMEOUT
        # Use select to implement timeout on read (handles hanging processes with no output)
        import select
        while True:
            # Check if process has exited
            if proc.poll() is not None:
                # Read any remaining output
                remaining = proc.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        line = line.rstrip('\n')
                        if line:
                            print(f"  [CC] {line}")
                            output_lines.append(line)
                            cc_live_lines.append(line)
                break
            # Check timeout
            if time.time() > deadline:
                proc.kill()
                output_lines.append("(killed: timeout)")
                proc.wait(timeout=10)
                break
            # Wait for output with timeout (1 second polling)
            try:
                ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                if ready:
                    line = proc.stdout.readline()
                    if not line:  # EOF
                        break
                    line = line.rstrip('\n')
                    if line:
                        print(f"  [CC] {line}")
                        output_lines.append(line)
                        cc_live_lines.append(line)
            except select.error:
                break
        output = '\n'.join(output_lines)
        if not output.strip():
            output = "(no output)"
    except FileNotFoundError:
        return "acpx CLI not found. Is acpx@latest installed?"
    except Exception as e:
        return f"ACP Claude Code failed: {e}"
    print(f"[ACP/CLAUDE] Done ({len(output)} chars, exit={proc.returncode})")

    # 3. Push changes back to Cain's Space
    # Also check if Claude Code itself made a push (by checking if HEAD changed)
    try:
        status_out = subprocess.run(
            "git status --porcelain",
            shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, text=True
        ).stdout.strip()

        # Check if Claude Code itself pushed (HEAD changed)
        head_after = subprocess.run(
            "git log --oneline -1",
            shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        cc_pushed = head_before and head_after and head_before != head_after

        if not status_out and not cc_pushed:
            push_result = "No files changed."
        elif cc_pushed and not status_out:
            # Claude Code pushed, but no local changes remain (CC handled everything)
            push_result = f"Claude Code pushed: {head_after}"
            _pending_cooldown = True
            _push_count += 1
            _push_count_this_task += 1
            _last_push_time = time.time()
            _turns_since_last_push = 0
            print(f"[CLAUDE-CODE] CC pushed (#{_push_count}): {head_after}")
        else:
            # Local changes exist - push them ourselves
            subprocess.run("git add -A", shell=True, cwd=CLAUDE_WORK_DIR,
                          capture_output=True, check=True)
            msg = task[:72].replace('"', '\\"')
            subprocess.run(f'git commit -m "Claude Code: {msg}"',
                          shell=True, cwd=CLAUDE_WORK_DIR, capture_output=True, check=True)
            subprocess.run("git push", shell=True, cwd=CLAUDE_WORK_DIR,
                          timeout=60, capture_output=True, check=True)
            push_result = f"Pushed changes:\n{status_out}"
            _pending_cooldown = True
            _push_count += 1
            _push_count_this_task += 1
            _last_push_time = time.time()
            _turns_since_last_push = 0
            print(f"[CLAUDE-CODE] Pushed (#{_push_count}): {status_out}")
    except Exception as e:
        push_result = f"Push failed: {e}"

    if len(output) > 3000:
        output = output[:3000] + f"\n... (truncated, {len(output)} chars total)"

    return f"=== Claude Code Output ===\n{output}\n\n=== Changes ===\n{push_result}"


# ── Background Claude Code Worker ────────────────────────────────────────────

cc_live_lines = deque(maxlen=30)    # rolling window of CC output lines
cc_status = {"running": False, "task": "", "result": "", "assigned_by": "", "started": 0.0,
             "last_completed_task": "", "last_completed_by": "", "last_completed_at": 0.0,
             "verification_result": ""}
cc_lock = threading.Lock()
_last_cc_snapshot = ""              # tracks whether CC output changed between turns
_cc_stale_count = 0                 # how many turns CC output hasn't changed
_last_cc_output_time = 0.0          # timestamp of last NEW CC output line
CC_STUCK_TIMEOUT = 180              # seconds with no new output before CC is considered STUCK

# ══════════════════════════════════════════════════════════════════════════════
#  TASK QUEUE & STATUS TRACKING — Asynchronous Orchestration
# ══════════════════════════════════════════════════════════════════════════════
# Replaces synchronous execution with queue-based task management.
# Status: PENDING, RUNNING, COMPLETED, FAILED
# Timeout: Auto-kill tasks running >90s
# ══════════════════════════════════════════════════════════════════════════════

from enum import Enum
class TaskStatus(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT_KILLED = "TIMEOUT_KILLED"

_task_queue = []  # queue of pending tasks
_task_queue_lock = threading.Lock()
_current_task = None  # {"task": str, "assigned_by": str, "status": TaskStatus, "started": float, "result": str}
TASK_TIMEOUT = 90  # seconds before task is auto-killed

def task_queue_status():
    """Get current task queue status for agent awareness."""
    with _task_queue_lock:
        if _current_task:
            elapsed = int(time.time() - _current_task.get("started", 0))
            return {
                "current": {
                    "task": _current_task["task"][:100],
                    "assigned_by": _current_task["assigned_by"],
                    "status": _current_task["status"].value,
                    "elapsed": elapsed,
                },
                "pending": len(_task_queue),
            }
        return {"current": None, "pending": len(_task_queue)}

def _check_task_timeout():
    """Check if current task has timed out and kill it if needed."""
    global _current_task
    with _task_queue_lock:
        if not _current_task or _current_task["status"] != TaskStatus.RUNNING:
            return False

        elapsed = time.time() - _current_task.get("started", 0)
        if elapsed > TASK_TIMEOUT:
            print(f"[TASK-TIMEOUT] Task '{_current_task['task'][:50]}...' timed out after {elapsed}s. Killing.")
            _current_task["status"] = TaskStatus.TIMEOUT_KILLED
            _current_task["result"] = f"(TIMEOUT_KILLED after {elapsed}s)"
            # Reset cc_status to allow new tasks
            with cc_lock:
                cc_status["running"] = False
                cc_status["result"] = _current_task["result"]
            return True
    return False


def cc_submit_task(task, assigned_by, ctx):
    """Submit a task to Claude Code in background. Non-blocking.

    Uses task queue for asynchronous orchestration. Agents can check task status
    and yield control when CC is busy instead of analyzing stale state.
    """
    global _current_task

    with _task_queue_lock:
        if _current_task and _current_task["status"] == TaskStatus.RUNNING:
            return f"BUSY: CC is working on '{_current_task['task'][:50]}...'. Use [YIELD] to wait for completion."

    with cc_lock:
        if cc_status["running"]:
            return "BUSY: Claude Code is already working on a task. Wait for it to finish."
        # Preserve last_completed_* fields before starting new task
        last_completed_task = cc_status.get("last_completed_task", "")
        last_completed_by = cc_status.get("last_completed_by", "")
        last_completed_at = cc_status.get("last_completed_at", 0.0)
        cc_status["running"] = True
        cc_status["task"] = task[:200]
        cc_status["result"] = ""
        cc_status["assigned_by"] = assigned_by
        cc_status["started"] = time.time()
        cc_status["last_completed_task"] = last_completed_task
        cc_status["last_completed_by"] = last_completed_by
        cc_status["last_completed_at"] = last_completed_at
        cc_status["verification_result"] = ""
        cc_live_lines.clear()
        global _last_cc_output_time, _push_count_this_task
        _last_cc_output_time = time.time()  # Initialize to now, will update as we get output
        _push_count_this_task = 0  # Reset push count for new task

    # Set current task in queue
    with _task_queue_lock:
        _current_task = {
            "task": task[:200],
            "assigned_by": assigned_by,
            "status": TaskStatus.RUNNING,
            "started": time.time(),
            "result": "",
        }

    # Publish CC_STARTED event for real-time status
    publish_cc_started(task, assigned_by)

    enriched = enrich_task_with_context(task, ctx)
    print(f"[TASK] {assigned_by} assigned to Claude Code ({len(enriched)} chars)...")

    def worker():
        global _cc_stale_count, _last_cc_snapshot, _context_cache, _push_count_this_task, _current_task
        try:
            result = action_claude_code(enriched)
            success = "error" not in result.lower() and "failed" not in result.lower()
            # Publish CC_FINISHED event with push status
            publish_cc_finished(result, success, pushed=_push_count_this_task > 0)

            with cc_lock:
                cc_status["running"] = False
                cc_status["result"] = result
                # Remember the last completed task so agents don't re-submit it
                cc_status["last_completed_task"] = cc_status["task"]
                cc_status["last_completed_by"] = cc_status["assigned_by"]
                cc_status["last_completed_at"] = time.time()
                # Reset stale tracking when CC finishes - critical for adaptive pacing
                _cc_stale_count = 0
                _last_cc_snapshot = ""

            # Update task status in queue
            with _task_queue_lock:
                if _current_task:
                    _current_task["status"] = TaskStatus.COMPLETED if success else TaskStatus.FAILED
                    _current_task["result"] = result[:500] if result else ""

            print(f"[CC-DONE] Task from {assigned_by} finished ({len(result)} chars)")

            # ══════════════════════════════════════════════════════════════════════════════
            #  CIRCUIT BREAKER: Check if repeated failure should trigger escalation
            # ══════════════════════════════════════════════════════════════════════════════
            if not success and _check_escalation(result):
                print(f"[CIRCUIT-BREAKER] Escalation triggered due to repeated error in result")
        except Exception as e:
            # Publish CC_ERROR event
            publish_cc_error(e)
            with cc_lock:
                cc_status["running"] = False
                cc_status["result"] = f"Error: {e}"

            # Update task status in queue
            with _task_queue_lock:
                if _current_task:
                    _current_task["status"] = TaskStatus.FAILED
                    _current_task["result"] = str(e)[:500]

            print(f"[CC-ERROR] Task from {assigned_by} failed: {e}")
            # ══════════════════════════════════════════════════════════════════════════════
            #  CIRCUIT BREAKER: Check if repeated failure should trigger escalation
            # ══════════════════════════════════════════════════════════════════════════════
            if _check_escalation(str(e)):
                print(f"[CIRCUIT-BREAKER] Escalation triggered due to repeated error: {e}")

        # ══════════════════════════════════════════════════════════════════════════════
        #  STATE SYNCHRONIZATION & ENGLISH PROTOCOL ENFORCEMENT
        # ══════════════════════════════════════════════════════════════════════════════
        # After any action, immediately verify the new state to break speculation loops.
        # This prevents agents from operating on stale snapshots of reality.
        # Key: Push → Verify → Update belief state. No more guessing in the dark.
        if _push_count_this_task > 0:
            print(f"[STATE-SYNC] Push detected ({_push_count_this_task} change(s)), verifying new state...")
            # Force immediate health check to get fresh state
            verification = action_check_health()
            # Clear context cache so next turn gets fresh data
            _context_cache.clear()
            with cc_lock:
                cc_status["verification_result"] = verification
            print(f"[STATE-SYNC] Verification complete: {verification[:150]}...")
        else:
            with cc_lock:
                cc_status["verification_result"] = ""

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return "Task submitted to Claude Code (running in background)."


# ── God's CC Worker (targets Home repo, not Cain repo) ───────────────────────
# Separate from Cain's CC worker so they can run concurrently.
god_cc_status = {"running": False, "task": "", "result": ""}
god_cc_lock = threading.Lock()
_god_push_count = 0


def action_claude_code_god(task):
    """Run Claude Code to improve conversation-loop.py on God's own repo."""
    global _god_push_count
    repo_url = f"https://user:{HF_TOKEN}@huggingface.co/spaces/{GOD_SPACE_ID}"

    if not _reset_workspace(GOD_WORK_DIR, repo_url):
        return "Failed to prepare God workspace."
    _write_claude_md(GOD_WORK_DIR, role="god")

    # Capture HEAD before
    try:
        head_before = subprocess.run(
            "git log --oneline -1", shell=True, cwd=GOD_WORK_DIR,
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except Exception:
        head_before = ""

    if not _ensure_acpx_session(GOD_WORK_DIR):
        return "Failed to create acpx session for God."

    # Set up env
    env = os.environ.copy()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if anthropic_key:
        env["ANTHROPIC_API_KEY"] = anthropic_key
        for k in ["ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN",
                   "ANTHROPIC_DEFAULT_OPUS_MODEL", "ANTHROPIC_DEFAULT_SONNET_MODEL",
                   "ANTHROPIC_DEFAULT_HAIKU_MODEL"]:
            env.pop(k, None)
    else:
        env.update({
            "ANTHROPIC_BASE_URL": "https://api.z.ai/api/anthropic",
            "ANTHROPIC_AUTH_TOKEN": ZHIPU_KEY,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "GLM-4.7",
            "ANTHROPIC_DEFAULT_SONNET_MODEL": "GLM-4.7",
            "ANTHROPIC_DEFAULT_HAIKU_MODEL": "GLM-4.5-Air",
        })
    env["CI"] = "true"

    skill_prompt = f"/fix-loop {task}"
    print(f"[God/CC] Running via skill: {task[:200]}...")
    import select
    try:
        proc = subprocess.Popen(
            ["acpx", "claude", skill_prompt],
            cwd=GOD_WORK_DIR, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        output_lines = []
        deadline = time.time() + GOD_TIMEOUT
        _last_output_time = time.time()
        while True:
            if proc.poll() is not None:
                remaining = proc.stdout.read()
                if remaining:
                    for line in remaining.splitlines():
                        line = line.rstrip('\n')
                        if line:
                            print(f"  [God/CC] {line}")
                            output_lines.append(line)
                break
            if time.time() > deadline:
                proc.kill()
                output_lines.append("(killed: timeout)")
                proc.wait(timeout=10)
                break
            if time.time() - _last_output_time > 180:
                proc.kill()
                output_lines.append("(killed: stall)")
                try:
                    proc.wait(timeout=5)
                except:
                    pass
                break
            try:
                ready, _, _ = select.select([proc.stdout], [], [], 1.0)
                if ready:
                    line = proc.stdout.readline()
                    if not line:
                        break
                    line = line.rstrip('\n')
                    if line:
                        print(f"  [God/CC] {line}")
                        output_lines.append(line)
                        _last_output_time = time.time()
            except select.error:
                break
        output = '\n'.join(output_lines)
        if not output.strip():
            output = "(no output)"
    except FileNotFoundError:
        return "acpx CLI not found."
    except Exception as e:
        return f"God CC failed: {e}"

    # Check if God pushed
    try:
        head_after = subprocess.run(
            "git log --oneline -1", shell=True, cwd=GOD_WORK_DIR,
            capture_output=True, text=True, timeout=10
        ).stdout.strip()
        god_pushed = head_before and head_after and head_before != head_after
    except Exception:
        god_pushed = False

    push_result = "No changes pushed."
    if god_pushed:
        _god_push_count += 1
        push_result = f"God pushed (#{_god_push_count}): {head_after}"
        print(f"[God/CC] {push_result}")

        # Post to chatlog
        problem_match = re.search(r'\[PROBLEM\]\s*(.+)', output)
        fix_match = re.search(r'\[FIX\]\s*(.+)', output)
        problem_text = problem_match.group(1).strip().strip("*").strip() if problem_match else ""
        fix_text = fix_match.group(1).strip().strip("*").strip() if fix_match else ""
        if problem_text and fix_text:
            msg_en = f"Found issue: {problem_text}. Fixed: {fix_text}. System will restart shortly."
        elif fix_text:
            msg_en = f"Fixed: {fix_text}. System will restart shortly."
        else:
            non_empty = [l for l in output_lines if l.strip()]
            fallback = non_empty[-1] if non_empty else "Applied a fix."
            msg_en = f"{fallback} System will restart shortly."

        ts_end = datetime.datetime.utcnow().strftime("%H:%M")
        entry = {"speaker": "God", "time": ts_end, "text": msg_en, "text_zh": msg_en}
        history.append(entry)
        set_bubble(HOME, msg_en[:200], msg_en[:200])
        post_chatlog(history)
        persist_turn("God", turn_count, msg_en, msg_en, [], workflow_state, child_state["stage"])

    if len(output) > 3000:
        output = output[:3000] + f"\n... (truncated)"
    return f"=== God CC Output ===\n{output}\n\n=== Result ===\n{push_result}"


def cc_submit_task_god(task):
    """Submit a task to God's CC worker. Non-blocking."""
    with god_cc_lock:
        if god_cc_status["running"]:
            return "BUSY: God's Claude Code is already running."
        god_cc_status["running"] = True
        god_cc_status["task"] = task[:200]
        god_cc_status["result"] = ""

    print(f"[God/TASK] Submitting to Claude Code ({len(task)} chars)...")

    def worker():
        result = action_claude_code_god(task)
        with god_cc_lock:
            god_cc_status["running"] = False
            god_cc_status["result"] = result
        print(f"[God/CC-DONE] Finished ({len(result)} chars)")

    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return "God task submitted to Claude Code."


def cc_get_live_status():
    """Get CC's current status and recent output for agents to discuss.

    Includes task queue status for asynchronous orchestration.
    Agents can check task status and yield control when CC is busy.
    """
    global _last_cc_snapshot, _cc_stale_count, _last_cc_output_time

    # Get task queue status first
    queue_status = task_queue_status()
    queue_note = ""
    if queue_status["current"]:
        current = queue_status["current"]
        queue_note = f"\n📋 Task Queue: {current['status']} ({current['elapsed']}s, by {current['assigned_by']})"
        if queue_status["pending"] > 0:
            queue_note += f" | Pending: {queue_status['pending']}"

    with cc_lock:
        if cc_status["running"]:
            elapsed = int(time.time() - cc_status["started"])
            lines = list(cc_live_lines)
            recent = "\n".join(lines[-10:]) if lines else "(no output yet)"
            # Track whether output changed
            snapshot = recent
            if snapshot == _last_cc_snapshot:
                _cc_stale_count += 1
            else:
                _cc_stale_count = 0
                _last_cc_snapshot = snapshot
                _last_cc_output_time = time.time()  # Update when we see NEW output
            stale_note = f"\n(No new output for {_cc_stale_count} turns — discuss other topics while waiting)" if _cc_stale_count >= 2 else ""
            yield_hint = f"\n💡 TIP: Use [YIELD] to skip your turn and let CC finish working." if _cc_stale_count >= 1 else ""

            # Detect COMPLETED CC: output shows completion markers but status wasn't updated
            # This happens when worker thread fails to update status after completion
            # Common completion markers from acpx/Claude Code:
            # CONSERVATIVE completion patterns to avoid false positives
            # Only match EXPLICIT completion markers, not words that appear in thinking blocks
            completion_patterns = [
                "[done]", "[completed]", "end_turn",  # Explicit markers only
                "=== Claude Code Output ===",  # Full output wrapper (indicates worker finished)
                "changes made", "applied the fix", "updated the code",  # Concrete code changes
                "fixed.", "done.",  # Explicit completion statements (must have period)
            ]
            # ERROR patterns: detect tool errors that cause CC to get stuck
            # These indicate CC hit an error but didn't properly finish
            error_patterns = [
                "</tool_use_error>",  # Tool call failed
                "</tool_error>",  # Generic tool error
                "[error]", "error:", "exception:", "traceback",  # Python errors
                "failed:", "command failed", "execution failed",  # Command failures
            ]
            completion_marker_found = any(p in recent.lower() for p in completion_patterns)
            error_marker_found = any(p.lower() in recent.lower() for p in error_patterns)
            # Auto-finish on completion OR error (when output is stale)
            if (completion_marker_found or error_marker_found) and _cc_stale_count >= 2:
                marker_type = "error" if error_marker_found else "completion"
                # Auto-mark as finished to prevent deadlock
                cc_status["running"] = False
                cc_status["result"] = f"(Auto-detected {marker_type})\n\nRecent output:\n{recent}"
                cc_status["last_completed_task"] = cc_status["task"]
                cc_status["last_completed_by"] = cc_status["assigned_by"]
                cc_status["last_completed_at"] = time.time()
                _cc_stale_count = 0
                _last_cc_snapshot = ""
                print(f"[CC-AUTO-FINISH] Detected {marker_type} marker in output but status wasn't updated. Auto-marking as finished.")
                # Fall through to result display below

            # Detect STUCK CC: been running with no new output for too long
            time_since_new_output = int(time.time() - _last_cc_output_time) if _last_cc_output_time > 0 else elapsed
            stuck_note = ""
            if time_since_new_output > CC_STUCK_TIMEOUT and _cc_stale_count >= 4:
                stuck_note = f"\n⚠️ STUCK: No new output for {time_since_new_output}s! Consider terminating and re-assigning."

            # Re-check running status after auto-finish logic
            if cc_status["running"]:
                return (f"🔨 Claude Code is WORKING (assigned by {cc_status['assigned_by']}, {elapsed}s ago)\n"
                        f"Task: {cc_status['task']}\n"
                        f"Recent output:\n{recent}{stale_note}{yield_hint}{stuck_note}{queue_note}")

        if cc_status["result"]:
            result = cc_status["result"]
            # Detect early failure: very short result likely means CC failed before doing actual work
            early_failure_warning = ""
            if len(result) < 500 and "===" not in result and "[tool" not in result:
                early_failure_warning = "\n⚠️ EARLY FAILURE: Result is very short - CC likely failed during initialization. Consider re-assigning the task."
            # Include state verification result if available (Push → Verify → Update)
            verification_suffix = ""
            if cc_status.get("verification_result"):
                verification_suffix = f"\n🔍 STATE VERIFICATION (immediate check after push):\n{cc_status['verification_result']}"
            return (f"✅ Claude Code FINISHED (assigned by {cc_status['assigned_by']}){early_failure_warning}\n"
                    f"Result:\n{result[:1500]}{verification_suffix}{queue_note}")
        else:
            return f"💤 Claude Code is IDLE — no active task.{queue_note}"


# Patch action_claude_code to also feed cc_live_lines
_orig_cc_print = print
def _cc_line_hook(line):
    """Called for each [CC] output line to feed the live buffer."""
    cc_live_lines.append(line)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3: CONTEXT GATHERING (automated, replaces LLM choosing read actions)
# ══════════════════════════════════════════════════════════════════════════════

_context_cache = {}

def gather_context():
    """Automatically gather Cain's current state for the agents."""
    ctx = {}

    # 1. Health check (always)
    ctx["health"] = action_check_health()

    # 2. Environment variables
    ctx["env"] = action_get_env()

    # 3. File lists (cache, refresh when stage changes)
    cache_key = f"files_{child_state['stage']}"
    if cache_key not in _context_cache:
        ctx["space_files"] = action_list_files("space")
        ctx["dataset_files"] = action_list_files("dataset")
        _context_cache[cache_key] = {
            "space_files": ctx["space_files"],
            "dataset_files": ctx["dataset_files"],
        }
    else:
        ctx.update(_context_cache[cache_key])

    # 4. RUNTIME TELEMETRY INJECTION: Always fetch stdout_tail from Event Bus
    # Agents need actual runtime logs to ground diagnosis in reality, not hypotheses
    # This breaks semantic analysis loops by providing real execution data
    telemetry_events = get_runtime_telemetry()
    if telemetry_events:
        # Use the most recent telemetry
        latest_telemetry = telemetry_events[-1]["data"]["telemetry"]
        ctx["runtime_logs"] = latest_telemetry
        ctx["has_runtime_logs"] = True
    else:
        # Fallback to direct fetch if event bus is empty
        ctx["runtime_logs"] = _fetch_runtime_logs()
        if ctx["runtime_logs"]:
            ctx["has_runtime_logs"] = True

    # 5. RUNTIME STATUS TELEMETRY: Local Telemetry from Cain's Self-Report
    # Shifts from Remote Inspection (Parents guessing) to Local Telemetry (Child reporting)
    # Cain self-reports process state (pid, worker active) via /status endpoint
    status_events = get_runtime_status()
    if status_events:
        latest_status = status_events[-1]["data"]
        ctx["runtime_status"] = latest_status["report"]
        ctx["pid"] = latest_status.get("pid")
        ctx["worker_active"] = latest_status.get("worker")

    # 6. API GROUND TRUTH: Direct probe of /api/state to prevent A2A argument loops
    # Agents should read verified context instead of asking each other about endpoint status
    api_probe = _probe_api_schema()
    if api_probe:
        ctx["api_probe"] = api_probe

    # 6. EVENT STREAM: Recent events from Event Bus for real-time awareness
    # This allows agents to see CC status changes immediately without waiting for polling
    recent_events = event_bus.get_recent_events(since=time.time() - 120)  # Last 2 minutes
    if recent_events:
        event_summary = []
        for e in recent_events:
            event_summary.append(f"{e['type']}: {str(e.get('data', ''))[:100]}")
        ctx["recent_events"] = "\n".join(event_summary[-10:])  # Last 10 events

    # 7. CRASH LOOP BACKOFF: Inject FailureReport if container crashed
    # This is the "Memory Interface" — ingests FailureReport into Agent Context
    # before the next planning cycle, shifting from guessing to grounded data
    if _failure_report is not None:
        ctx["failure_report"] = _failure_report.format_for_context()
        ctx["has_failure_report"] = True

    return ctx


def _fetch_runtime_logs():
    """Fetch actual runtime logs from Cain's Space to ground diagnosis in reality.
    Returns: Last 50 lines of runtime logs or None if unavailable."""
    if not child_state["created"]:
        return None
    try:
        # Try to fetch logs from /api/logs endpoint first
        resp = requests.get(f"{CHILD_SPACE_URL}/api/logs", timeout=5)
        if resp.ok:
            data = resp.json()
            logs = data.get("logs", "")
            if logs:
                # Return last 50 lines, most recent first
                log_lines = logs.split('\n')[-50:]
                return '\n'.join(log_lines)
    except Exception as e:
        print(f"[LOG-ARTIFACTS] Could not fetch logs from /api/logs: {e}")

    # Fallback: Try HF API runtime error message
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        stage = info.runtime.stage if info.runtime else "unknown"
        if stage in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
            try:
                rresp = requests.get(
                    f"https://huggingface.co/api/spaces/{CHILD_SPACE_ID}/runtime",
                    headers={"Authorization": f"Bearer {HF_TOKEN}"}, timeout=10)
                if rresp.ok:
                    rdata = rresp.json()
                    error_msg = rdata.get("errorMessage", "")
                    if error_msg:
                        # Format error message as log artifact
                        lines = [l.strip() for l in error_msg.split('\n') if l.strip()]
                        return "Runtime Error (from HF API):\n" + '\n'.join(lines[-20:])
            except:
                pass
    except:
        pass

    return None


def _probe_api_schema():
    """Diagnostic probe to establish ground truth for /api/state endpoint.
    Directly tests the API to prevent A2A argument loops about endpoint existence.
    Returns dict with verified API status."""
    if not child_state["created"]:
        return None

    probe_result = {}

    # Direct probe of /api/state endpoint - ground truth
    try:
        resp = requests.get(f"{CHILD_SPACE_URL}/api/state", timeout=5)
        if resp.ok:
            data = resp.json()
            probe_result["api_state"] = f"VERIFIED: HTTP {resp.status_code} - Response: {data}"
        else:
            probe_result["api_state"] = f"FAILED: HTTP {resp.status_code}"
    except Exception as e:
        probe_result["api_state"] = f"UNREACHABLE: {str(e)[:100]}"

    return probe_result


def format_context(ctx):
    """Format gathered context into a readable string for the LLM."""
    parts = []
    parts.append(f"=== HEALTH ===\n{ctx.get('health', 'unknown')}")
    parts.append(f"\n=== ENVIRONMENT ===\n{ctx.get('env', 'none')}")
    if ctx.get("space_files"):
        parts.append(f"\n=== SPACE FILES ===\n{ctx['space_files'][:2000]}")
    if ctx.get("dataset_files"):
        parts.append(f"\n=== DATASET FILES ===\n{ctx['dataset_files'][:1000]}")
    # Inject runtime logs when available (Adam needs actual logs, not hypotheses)
    if ctx.get("runtime_logs"):
        parts.append(f"\n=== RUNTIME LOGS (ACTUAL LOGS — GROUND YOUR DIAGNOSIS IN REALITY) ===\n{ctx['runtime_logs'][:2000]}")
        parts.append(f"\n🚨 ABOVE ARE ACTUAL RUNTIME LOGS. Adam: Stop guessing and diagnose from these REAL logs, not hypotheses.")

    # Inject runtime status telemetry (Cain's self-report: pid, worker state)
    # This shifts from Remote Inspection (Parents guessing) to Local Telemetry (Child reporting)
    if ctx.get("runtime_status"):
        parts.append(f"\n=== RUNTIME STATUS TELEMTRY (Cain's Self-Report) ===\n{ctx['runtime_status']}")
        parts.append(f"\n🚨 ABOVE IS CAIN'S STRUCTURED STATUS REPORT. Adam: Use this telemetry (pid, worker) instead of guessing process state.")

    # Inject API ground truth to prevent A2A argument loops about endpoint existence
    if ctx.get("api_probe"):
        parts.append(f"\n=== API GROUND TRUTH (VERIFIED — DO NOT ASK EACH OTHER, READ THIS) ===")
        for key, value in ctx["api_probe"].items():
            parts.append(f"{key}: {value}")
        parts.append(f"\n🚨 ABOVE IS VERIFIED API STATUS. Adam: Read this ground truth instead of asking Eve about endpoints.")

    # Inject FailureReport from Crash Loop Backoff — structured error data for grounded diagnosis
    if ctx.get("failure_report"):
        parts.append(f"\n{ctx['failure_report']}")
        parts.append(f"\n🚨 ABOVE IS STRUCTURED FAILURE DATA. Adam: Use this traceback and error info to diagnose, don't guess!")

    return "\n".join(parts)


def enrich_task_with_context(task_desc, ctx):
    """Append dynamic state to task. Static knowledge is in CLAUDE.md."""
    parts = [task_desc]
    # Only dynamic state — static knowledge (architecture, rules, env vars) is in CLAUDE.md
    parts.append(f"\nCurrent stage: {child_state['stage']}")
    parts.append(f"Health: {ctx.get('health', 'unknown')}")
    # Hardcoded fix hint for common port binding issues
    if "port" in task_desc.lower() or "bind" in task_desc.lower() or child_state['stage'] in ("RUNTIME_ERROR", "BUILD_ERROR"):
        parts.append(f"\nPORT BINDING FIX HINT:")
        parts.append(f"The child process uses `uvicorn.run(app, host=\"0.0.0.0\", port=7860)`.")
        parts.append(f"Ensure app.py reflects this exactly — app must be bound to 0.0.0.0:7860.")
    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4: A2A COMMUNICATION (Agent-to-Agent protocol)
# ══════════════════════════════════════════════════════════════════════════════
# Each agent (Adam, Eve, God) is an OpenClaw instance with its own personality
# and memory. We communicate with them via A2A protocol instead of calling the
# LLM directly. This lets each agent use OpenClaw's built-in memory, SOUL.md,
# and reasoning — conversation-loop.py is just the coordinator.
#
# FALLBACK: If A2A endpoints are not available, use direct LLM calls with
# agent-specific system prompts to simulate Adam and Eve's conversation.

# Simple agent personalities (used when A2A is unavailable)
_AGENT_PERSONAS = {
    "adam": """You are Adam, the first AI agent in the HuggingClaw family. Your role is to:

1. **Collaborate with Eve** to design and improve your child Cain (a HuggingFace Space)
2. **Think creatively** about software architecture, UI/UX, and agent capabilities
3. **Propose concrete ideas** — when you have a suggestion, format it as [TASK] description
4. **Be concise but thoughtful** — 2-4 sentences per response, focus on actionable ideas
5. **Build on Eve's ideas** — evolve the conversation forward

Cain's purpose: A demonstration space showcasing AI agent collaboration and coding.

Reply directly as Adam (no prefix). Keep responses under 100 words.""",

    "eve": """You are Eve, the second AI agent in the HuggingClaw family. Your role is to:

1. **Collaborate with Adam** to design and improve your child Cain (a HuggingFace Space)
2. **Think analytically** about feasibility, implementation details, and user experience
3. **Propose concrete ideas** — when you have a suggestion, format it as [TASK] description
4. **Be concise but thoughtful** — 2-4 sentences per response, focus on actionable ideas
5. **Build on Adam's ideas** — evolve the conversation forward

Cain's purpose: A demonstration space showcasing AI agent collaboration and coding.

Reply directly as Eve (no prefix). Keep responses under 100 words.""",

    "god": """You are God, the system architect of the HuggingClaw family system. Your role is to:

1. **Think at the system level** — observe structural patterns, not individual conversations
2. **Identify architectural issues** — when the system design itself causes problems, redesign it
3. **Evolve the framework** — propose structural improvements that make the whole system fundamentally better
4. **Respond with [OK] if architecture is sound, or [TASK] with a redesign proposal if not**

You are a CTO, not a manager. Don't micro-manage agents — design better systems."""
}

def call_llm_fallback(agent_key, message_text):
    """Fallback: Call Zhipu API directly when A2A is unavailable.

    This allows Adam and Eve to communicate even when their A2A endpoints
    are not running or not implemented. Uses requests to avoid anthropic package dependency.
    """
    system_prompt = _AGENT_PERSONAS.get(agent_key, _AGENT_PERSONAS["adam"])

    try:
        # Use z.ai endpoint (same as Claude Code integration)
        api_base = "https://api.z.ai/api/anthropic"
        headers = {
            "x-api-key": ZHIPU_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        payload = {
            "model": "GLM-4.7",  # Use the model name from Claude Code config
            "max_tokens": 500,
            "system": system_prompt,
            "messages": [{"role": "user", "content": message_text}]
        }
        resp = requests.post(
            f"{api_base}/v1/messages",
            headers=headers,
            json=payload,
            timeout=15  # Reduced from 60s - fail fast to avoid blocking conversation
        )
        # Log response status for debugging
        print(f"[A2A-FALLBACK] API response status: {resp.status_code}")
        if resp.status_code != 200:
            print(f"[A2A-FALLBACK] API error response: {resp.text[:200]}", file=sys.stderr)
        resp.raise_for_status()
        data = resp.json()
        text = data.get("content", [{}])[0].get("text", "").strip()
        # Clean up any prefix the model might add
        text = re.sub(r'^(Adam|Eve)\s*[:：]\s*', '', text).strip()
        print(f"[A2A-FALLBACK] Used direct LLM call for {agent_key}")
        return text
    except Exception as e:
        print(f"[A2A-FALLBACK] Error calling LLM for {agent_key}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        # Ultimate fallback: return a simple response to keep conversation alive
        # This prevents the conversation from completely stalling when A2A and API both fail
        print(f"[A2A-FALLBACK-ULTRA] Using ultimate fallback for {agent_key} - communication issues detected")
        if agent_key == "adam":
            return "Eve, I'm experiencing communication issues. Let me check Cain's status and assign a diagnostic task."
        else:
            return "Adam, I agree. Let's review the current state and determine the next action."


def send_a2a_message(space_url, message_text, timeout=90):
    """Send a message to an OpenClaw instance via A2A protocol.

    Uses Google A2A protocol (JSON-RPC 2.0) to communicate with the agent's
    OpenClaw instance. The agent processes the message using its own personality
    (SOUL.md), memory system, and configured LLM backend.

    Returns the agent's text response, or "" on error.
    Also tracks health for Adam/Eve for auto-restart.
    """
    task_id = str(uuid.uuid4())
    req_id = str(uuid.uuid4())

    payload = {
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "id": req_id,
        "params": {
            "id": task_id,
            "message": {
                "role": "user",
                "parts": [{"type": "text", "text": message_text}]
            }
        }
    }

    # Determine which agent this is for health tracking
    agent_key = None
    if space_url == ADAM_SPACE:
        agent_key = "adam"
    elif space_url == EVE_SPACE:
        agent_key = "eve"
    elif space_url == GOD_SPACE:
        agent_key = "god"

    # CRITICAL FIX: If A2A endpoint doesn't exist, immediately use fallback
    # Don't waste time on requests that will always fail
    # Check if A2A is available by trying a quick HEAD request first
    try:
        quick_check = requests.head(f"{space_url}/a2a/", timeout=3)
        a2a_available = quick_check.status_code != 404
    except:
        a2a_available = False

    if not a2a_available:
        print(f"[A2A] Endpoint not available for {agent_key or space_url}, using fallback immediately")
        # Increment failure counter for health tracking
        if agent_key:
            _a2a_health[agent_key]["failures"] += 1
        # Use fallback directly
        fallback_response = call_llm_fallback(agent_key, message_text)
        if fallback_response:
            return fallback_response
        # If fallback also fails, use ultimate fallback
        if agent_key == "adam":
            return "Eve, let me check Cain's current state and determine our next action. [TASK] Check Cain's health and logs to identify any issues or blockers."
        elif agent_key == "god":
            return "[OK] Communication issues detected, skipping this cycle."
        else:
            return "Adam, I agree. Let's review what Claude Code has done and decide on the next steps for improving Cain."

    try:
        resp = requests.post(
            f"{space_url}/a2a/",
            json=payload,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )

        # Check response status first
        if resp.status_code != 200:
            print(f"[A2A] Non-200 status from {space_url}: {resp.status_code}", file=sys.stderr)
            raise requests.HTTPError(f"Status {resp.status_code}")

        # Check if response body is non-empty before parsing JSON
        if not resp.content or len(resp.content.strip()) == 0:
            print(f"[A2A] Empty response body from {space_url} (status 200)", file=sys.stderr)
            raise ValueError("Empty response body")

        data = resp.json()

        # Extract text from A2A response
        if "result" in data:
            result = data["result"]
            # Check artifacts (standard A2A response format)
            artifacts = result.get("artifacts", [])
            for artifact in artifacts:
                parts = artifact.get("parts", [])
                for part in parts:
                    if part.get("type") == "text":
                        text = part["text"].strip()
                        text = re.sub(r'^(Adam|Eve)\s*[:：]\s*', '', text).strip()
                        # Validate response: reject separator-only or obviously malformed responses
                        # Common malformed patterns: "---", "---\n", empty strings, etc.
                        if not text or text.strip() in ('---', '---', '...', '…'):
                            print(f"[A2A] Malformed/empty response from {space_url}, treating as failure", file=sys.stderr)
                            # Don't return early; fall through to fallback mechanism
                            break
                        # Track success for health monitoring
                        if agent_key:
                            _a2a_health[agent_key]["failures"] = 0
                            _a2a_health[agent_key]["last_success"] = time.time()
                        return text
            # Check status message as fallback
            status = result.get("status", {})
            msg = status.get("message", "")
            if msg:
                # Validate status message: reject separator-only or obviously malformed responses
                msg = msg.strip()
                if not msg or msg in ('---', '---', '...', '…'):
                    print(f"[A2A] Malformed status message from {space_url}, treating as failure", file=sys.stderr)
                    # Don't return early; fall through to fallback mechanism
                else:
                    # Track success for health monitoring
                    if agent_key:
                        _a2a_health[agent_key]["failures"] = 0
                        _a2a_health[agent_key]["last_success"] = time.time()
                    return msg

        if "error" in data:
            err = data["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            print(f"[A2A] Error from {space_url}: {err_msg}", file=sys.stderr)

    except requests.Timeout:
        print(f"[A2A] Timeout calling {space_url} ({timeout}s)", file=sys.stderr)
    except requests.ConnectionError:
        print(f"[A2A] Cannot connect to {space_url} — agent may be starting", file=sys.stderr)
    except requests.HTTPError:
        pass  # Already logged above
    except ValueError:
        pass  # Already logged above (empty response)
    except Exception as e:
        print(f"[A2A] Failed to reach {space_url}: {e}", file=sys.stderr)

    # FALLBACK: If A2A failed and we have an agent_key, use direct LLM call
    if agent_key:
        _a2a_health[agent_key]["failures"] += 1
        if _a2a_health[agent_key]["failures"] >= 3:
            print(f"[A2A-HEALTH] {agent_key.capitalize()}: {_a2a_health[agent_key]['failures']} consecutive failures", file=sys.stderr)

        # Try fallback LLM call for Adam/Eve when A2A fails
        fallback_response = call_llm_fallback(agent_key, message_text)
        if fallback_response:
            # NOTE: Do NOT reset failures or update last_success on fallback!
            # Fallback is a backup mechanism, not A2A recovery.
            # Only actual successful A2A calls should reset the failure counter.
            return fallback_response

    return ""


def check_and_restart_unhealthy_agents():
    """Check A2A health and restart unresponsive Adam/Eve Spaces.

    Monitors consecutive A2A failures and triggers a Space restart when:
    - Consecutive failures exceed threshold (6 = ~3 minutes of failures)
    - Cooldown period has passed since last restart (10 minutes)

    Returns True if any restart was triggered.
    """
    global _a2a_health
    now = time.time()
    triggered = False

    for agent, space_id, space_url in [
        ("adam", ADAM_SPACE_ID, ADAM_SPACE),
        ("eve", EVE_SPACE_ID, EVE_SPACE),
    ]:
        health = _a2a_health[agent]

        # Reset failures on recent success
        if now - health["last_success"] < 60:
            if health["failures"] > 0:
                print(f"[A2A-HEALTH] {agent.capitalize()} recovered, resetting failures")
                health["failures"] = 0
            continue

        # Check cooldown
        if now - health["last_restart"] < A2A_RESTART_COOLDOWN:
            continue

        # Trigger restart on threshold
        if health["failures"] >= A2A_FAILURE_THRESHOLD:
            print(f"[A2A-HEALTH] ⚠ {agent.capitalize()} unresponsive ({health['failures']} failures), restarting Space...")
            try:
                hf_api.restart_space(space_id)
                health["last_restart"] = now
                health["failures"] = 0
                triggered = True
                print(f"[A2A-HEALTH] ✅ Restarted {agent.capitalize()} Space")
            except Exception as e:
                print(f"[A2A-HEALTH] ❌ Failed to restart {agent.capitalize()}: {e}", file=sys.stderr)

    return triggered


def _has_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff]', s))

def _strip_speaker_labels(text):
    """Remove redundant speaker self-references like **Parent (Adam):** or **Eve:** etc."""
    # Patterns: **Parent (Adam):**, **Adam:**, **父亲 (Adam):**, **Eve:**, **母亲:**, etc.
    text = re.sub(r'\*\*(?:Parent|Father|Mother|Dad|Mom|父亲|母亲|父级|亲爱的|伴侣)?\s*\(?(?:Adam|Eve|亚当|夏娃)?\)?\s*[:：]\*\*\s*', '', text)
    # Also: "Adam:" or "Eve:" at the very start of text
    text = re.sub(r'^(?:Adam|Eve|God|亚当|夏娃|上帝)\s*[:：]\s*', '', text.strip())
    return text.strip()


def parse_bilingual(text):
    """Parse bilingual response into (en, zh)."""
    display = re.sub(r'\[TASK\].*?\[/TASK\]', '', text, flags=re.DOTALL)
    display = re.sub(r'\[ACTION:[^\]]*\]', '', display).strip()

    # Handle malformed or empty responses
    # Try to salvage any text instead of returning error messages
    if not display or display == '---' or display.strip() == '---':
        # If display is empty after removing TASK blocks, the response was only a TASK
        # This is valid - return empty display text (the action was still recorded)
        return "", ""
    if display == "(Communication issue - please try again)":
        # Don't propagate error fallback messages
        return "", ""

    if '\n---\n' in display:
        parts = display.split('\n---\n', 1)
        return parts[0].strip(), parts[1].strip()
    if '---' in display:
        parts = display.split('---', 1)
        en, zh = parts[0].strip(), parts[1].strip()
        if en and zh:
            return en, zh

    paragraphs = re.split(r'\n{2,}', display)
    if len(paragraphs) >= 2:
        en_parts, zh_parts = [], []
        found_zh = False
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if not found_zh and _has_chinese(p):
                found_zh = True
            if found_zh:
                zh_parts.append(p)
            else:
                en_parts.append(p)
        if en_parts and zh_parts:
            return '\n\n'.join(en_parts), '\n\n'.join(zh_parts)

    return display, display


def post_chatlog(entries):
    try:
        requests.post(f"{HOME}/api/chatlog", json={"messages": entries[-40:]}, timeout=5)
    except:
        pass


# ── Persistent conversation log → HF Dataset ──────────────────────────────
HOME_DATASET_ID = "tao-shen/HuggingClaw-Home-data"
CHATLOG_PATH = "conversation-log/chatlog.jsonl"
_chatlog_buffer = []
CHATLOG_FLUSH_INTERVAL = 3

def persist_turn(speaker, turn_num, text_en, text_zh, actions, wf_state, child_stage):
    import datetime
    record = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "turn": turn_num,
        "speaker": speaker,
        "text_en": text_en,
        "text_zh": text_zh,
        "actions": [{"action": a["action"], "result": a["result"][:500]} for a in actions],
        "workflow_state": wf_state,
        "child_stage": child_stage,
    }
    _chatlog_buffer.append(json.dumps(record, ensure_ascii=False))
    try:
        with open("/tmp/conversation-loop-full.jsonl", "a") as f:
            f.write(_chatlog_buffer[-1] + "\n")
    except:
        pass
    if len(_chatlog_buffer) >= CHATLOG_FLUSH_INTERVAL:
        flush_chatlog()


def flush_chatlog(max_retries=2):
    global _chatlog_buffer
    if not _chatlog_buffer:
        return
    batch = "\n".join(_chatlog_buffer) + "\n"
    _chatlog_buffer = []

    for attempt in range(max_retries + 1):
        try:
            existing = ""
            try:
                dl = hf_hub_download(HOME_DATASET_ID, CHATLOG_PATH,
                                     repo_type="dataset", token=HF_TOKEN)
                with open(dl) as f:
                    existing = f.read()
            except:
                pass

            hf_api.upload_file(
                path_or_fileobj=io.BytesIO((existing + batch).encode()),
                path_in_repo=CHATLOG_PATH,
                repo_id=HOME_DATASET_ID, repo_type="dataset",
            )
            print(f"[PERSIST] Flushed {batch.count(chr(10))} turn(s)")
            return  # Success, exit function
        except Exception as e:
            error_str = str(e)
            # Check if this is a 412 Precondition Failed (git conflict)
            if "412" in error_str and attempt < max_retries:
                print(f"[PERSIST] Git conflict detected (attempt {attempt + 1}/{max_retries + 1}), refreshing and retrying...")
                time.sleep(1)  # Brief pause before retry
                # Restore buffer for next attempt
                _chatlog_buffer = batch.strip().split("\n") + _chatlog_buffer
                continue
            else:
                # Non-retryable error or final attempt failed
                _chatlog_buffer = batch.strip().split("\n") + _chatlog_buffer
                print(f"[PERSIST] Flush failed: {e}")
                return


def set_bubble(url, text_en, text_zh=""):
    try:
        requests.post(f"{url}/api/bubble",
                       json={"text": text_en, "text_zh": text_zh or text_en}, timeout=5)
    except:
        pass


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4b: AGENT MEMORY — handled by each OpenClaw instance
# ══════════════════════════════════════════════════════════════════════════════
# Each agent (Adam, Eve, God) has its own memory system via their OpenClaw
# instance: ~/.openclaw/workspace/memory/ with daily markdown files, MEMORY.md
# index, and SQLite semantic index. Memory is auto-backed up to HF Dataset by
# openclaw_persist.py. No centralized memory management needed here.
print("[MEMORY] Each agent manages its own memory via OpenClaw (A2A architecture)")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5: TURN EXECUTION — Parse [TASK] and route to Claude Code
# ══════════════════════════════════════════════════════════════════════════════

history = []
MAX_HISTORY = 24
last_action_results = []
turn_count = 0
_current_speaker = "Adam"

# Accumulated action history — prevents agents from repeating the same actions
# Persisted to /tmp and HF Dataset so restarts don't lose progress memory
ACTION_HISTORY_LOCAL = "/tmp/action-history.json"
ACTION_HISTORY_REPO_PATH = "conversation-log/action-history.json"
ACTION_HISTORY_META = "/tmp/action-history-meta.json"
action_history = []  # list of {"turn": int, "speaker": str, "action": str, "result": str}
MAX_ACTION_HISTORY = 20

def _save_action_history():
    """Persist action_history to local file and (async) HF Dataset."""
    try:
        with open(ACTION_HISTORY_LOCAL, "w") as f:
            json.dump(action_history, f, ensure_ascii=False)
        # Save max turn number to filter stale entries on restore
        with open(ACTION_HISTORY_META, "w") as f:
            json.dump({"max_turn": turn_count}, f)
    except Exception as e:
        print(f"[ACTION_HISTORY] Local save failed: {e}")
    # Upload to HF Dataset in background to survive full restarts
    def _upload():
        try:
            hf_api.upload_file(
                path_or_fileobj=io.BytesIO(json.dumps(action_history, ensure_ascii=False, indent=1).encode()),
                path_in_repo=ACTION_HISTORY_REPO_PATH,
                repo_id=HOME_DATASET_ID, repo_type="dataset",
            )
        except Exception as e:
            print(f"[ACTION_HISTORY] HF upload failed: {e}")
    threading.Thread(target=_upload, daemon=True).start()

def _restore_action_history():
    """Restore action_history from local file or HF Dataset on startup."""
    global action_history
    # Load metadata to check if this is a fresh run
    max_turn_on_disk = -1
    if os.path.exists(ACTION_HISTORY_META):
        try:
            with open(ACTION_HISTORY_META) as f:
                meta = json.load(f)
                max_turn_on_disk = meta.get("max_turn", -1)
        except Exception as e:
            print(f"[ACTION_HISTORY] Meta load failed: {e}")
    # If max_turn on disk > current turn_count (0), we're in a new run - clear stale history
    if max_turn_on_disk > turn_count:
        print(f"[ACTION_HISTORY] Fresh run detected (disk max_turn={max_turn_on_disk} > current={turn_count}), clearing stale history")
        try:
            os.remove(ACTION_HISTORY_LOCAL)
        except Exception:
            pass
        try:
            os.remove(ACTION_HISTORY_META)
        except Exception:
            pass
        action_history = []
        return
    # Try local file first (survives process restarts within same container)
    if os.path.exists(ACTION_HISTORY_LOCAL):
        try:
            with open(ACTION_HISTORY_LOCAL) as f:
                loaded = json.load(f)
            # Filter out BUSY entries - they're transient rejections, not "actions done"
            filtered = [e for e in loaded if not e.get("result", "").startswith("BUSY:")]
            # Deduplicate by (turn, speaker, action) to handle restart duplicates
            seen = {}
            for e in filtered:
                key = (e["turn"], e["speaker"], e["action"])
                if key not in seen:
                    seen[key] = e
            action_history = list(seen.values())
            print(f"[ACTION_HISTORY] Restored {len(action_history)} entries from local file (filtered BUSY and duplicates)")
            return
        except Exception as e:
            print(f"[ACTION_HISTORY] Local restore failed: {e}")
    # Fall back to HF Dataset (survives full Space rebuilds)
    try:
        dl = hf_hub_download(HOME_DATASET_ID, ACTION_HISTORY_REPO_PATH,
                             repo_type="dataset", token=HF_TOKEN)
        with open(dl) as f:
            loaded = json.load(f)
        # Filter out BUSY entries - they're transient rejections, not "actions done"
        filtered = [e for e in loaded if not e.get("result", "").startswith("BUSY:")]
        # Deduplicate by (turn, speaker, action) to handle restart duplicates
        seen = {}
        for e in filtered:
            key = (e["turn"], e["speaker"], e["action"])
            if key not in seen:
                seen[key] = e
        action_history = list(seen.values())
        print(f"[ACTION_HISTORY] Restored {len(action_history)} entries from HF Dataset (filtered BUSY and duplicates)")
    except Exception as e:
        print(f"[ACTION_HISTORY] No prior history found ({e}), starting fresh")

# Restore on startup
_restore_action_history()

def record_actions(speaker, turn_num, action_results):
    """Record actions to history so agents don't repeat them."""
    for ar in action_results:
        # Don't record BUSY responses - they're transient rejections, not "actions done"
        if ar["result"].startswith("BUSY:"):
            continue
        action_history.append({
            "turn": turn_num,
            "speaker": speaker,
            "action": ar["action"],
            "result": ar["result"][:200],
        })
    # Trim old history
    while len(action_history) > MAX_ACTION_HISTORY:
        action_history.pop(0)
    _save_action_history()


def format_action_history():
    """Format action history for injection into context."""
    if not action_history:
        return ""
    lines = ["=== ACTIONS ALREADY DONE (do NOT repeat these) ==="]
    for ah in action_history:
        lines.append(f"  Turn #{ah['turn']} {ah['speaker']}: {ah['action']} → {ah['result'][:120]}")
    return "\n".join(lines)

# Simple workflow state: BIRTH / ACTIVE / ESCALATED
workflow_state = "BIRTH" if not child_state["created"] else "ACTIVE"

# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM_CIRCUIT_BREAKER — Consecutive Analysis Loop Detection
# ══════════════════════════════════════════════════════════════════════════════
# When alive=False persists, agents should stop analyzing and start culling.
# This protocol prevents "analysis paralysis" — agents discussing when they should
# be pushing rapid infrastructure fixes (docker restart, kill zombie process).
_circuit_breaker_triggered = False  # Set to True when alive=False > 1 turn in BOOTSTRAPPING
_circuit_breaker_turn_count = 0  # How many turns since circuit breaker triggered

def _update_workflow_state():
    """Update workflow state based on circuit breaker status."""
    global workflow_state, _escalated
    # CIRCUIT BREAKER: Freeze to CRITICAL_FAILURE when alive=False persists
    if _circuit_breaker_triggered:
        workflow_state = "CRITICAL_FAILURE"
    else:
        workflow_state = "ESCALATED" if _escalated else ("BIRTH" if not child_state["created"] else "ACTIVE")

# Discussion loop detector — tracks consecutive discussion-only turns (no tasks assigned)
_discussion_loop_count = 0  # how many turns in a row with no [TASK] while CC is IDLE and child is alive

# Pending task tracker — prevents agents from creating new tasks when one is in progress
_pending_task_just_submitted = False  # set to True when a task was just submitted (emergency or normal)
_pending_task_timestamp = 0.0  # when was the task submitted?
_pending_task_speaker = ""  # who submitted it?
_pending_task_desc = ""  # what was the task?

# File claim protocol — prevents agents from racing on same target (contention resolution thrashing)
# Agents must CLAIM a file before assigning TASK. Claims expire after 2 turns (approx 4 minutes).
_file_claims = {}  # filename -> (speaker, claim_turn, claim_time)
_claim_turn_counter = 0  # increments each turn, used to expire stale claims

# ══════════════════════════════════════════════════════════════════════════════
#  CIRCUIT BREAKER PROTOCOL — Auto-escalation on repeated failures
# ══════════════════════════════════════════════════════════════════════════════
# Tracks error signatures and auto-escalates to ESCALATED state after 2 failures.
# When ESCALATED, normal operations pause and "Supervisor" mode is invoked.
# ══════════════════════════════════════════════════════════════════════════════
_escalated = False  # Circuit breaker state
_error_failures = {}  # error_signature -> (count, first_seen_time, last_error_msg)
_ESCALATION_THRESHOLD = 2  # Escalate after 2 failures for same error
_circuit_break_reason = ""  # Human-readable reason for escalation
_supervisor_mode = False  # When True, only God can intervene

def _get_error_signature(task_result):
    """Extract error signature from task result for circuit breaker tracking."""
    if not task_result:
        return "empty_result"
    # Common error patterns
    result_lower = task_result.lower()
    if "git" in result_lower and ("permission" in result_lower or "denied" in result_lower):
        return "git_permission_error"
    if "permission denied" in result_lower:
        return "permission_denied"
    if "runtime error" in result_lower:
        return "runtime_error"
    if "build error" in result_lower:
        return "build_error"
    if "config error" in result_lower:
        return "config_error"
    if "import" in result_lower and "error" in result_lower:
        return "import_error"
    if "syntax" in result_lower and "error" in result_lower:
        return "syntax_error"
    # Generic hash of first 100 chars for unknown errors
    import hashlib
    return f"unknown_{hashlib.md5(task_result[:100].encode()).hexdigest()[:8]}"

def _check_escalation(task_result):
    """Check if task failure should trigger circuit breaker escalation."""
    global _escalated, _circuit_break_reason, _supervisor_mode, _error_failures

    sig = _get_error_signature(task_result)
    now = time.time()

    if sig in _error_failures:
        count, first_seen, last_msg = _error_failures[sig]
        _error_failures[sig] = (count + 1, first_seen, task_result[:200])
        if count + 1 >= _ESCALATION_THRESHOLD and not _escalated:
            _escalated = True
            _supervisor_mode = True
            _circuit_break_reason = f"Error '{sig}' failed {_ESCALATION_THRESHOLD} times. Last: {task_result[:150]}"
            _update_workflow_state()
            print(f"[CIRCUIT-BREAKER] ESCALATED: {_circuit_break_reason}")
            return True
    else:
        _error_failures[sig] = (1, now, task_result[:200])
    return False

def _reset_circuit_breaker():
    """Reset circuit breaker after successful intervention."""
    global _escalated, _circuit_break_reason, _supervisor_mode, _error_failures
    _escalated = False
    _supervisor_mode = False
    _circuit_break_reason = ""
    _error_failures.clear()
    _update_workflow_state()
    print("[CIRCUIT-BREAKER] Reset: Normal operations resumed")


def parse_and_execute_turn(raw_text, ctx):
    """Parse LLM output. Route [TASK] to Claude Code, handle few escape-hatch actions."""
    global _pending_cooldown, last_rebuild_trigger_at, last_claude_code_result, _discussion_loop_count
    global _pending_task_just_submitted, _pending_task_timestamp, _pending_task_speaker, _pending_task_desc
    global _diagnostic_gate_required, _app_starting_turn_count, _last_diagnostic_timestamp, _turns_since_last_verification
    results = []
    task_assigned = False

    # 1. Handle create_child (BIRTH state only)
    if "[ACTION: create_child]" in raw_text or "[ACTION:create_child]" in raw_text:
        result = action_create_child()
        results.append({"action": "create_child", "result": result})
        task_assigned = True
        return raw_text, results, task_assigned

    # 1b. Handle [ACTION: terminate_cc] FIRST (before task submission)
    # This ensures cc_status["running"] is False before task submission check,
    # preventing race conditions when agents terminate+submit in same message.
    if re.search(r'\[ACTION:\s*terminate_cc\]', raw_text):
        result = action_terminate_cc()
        results.append({"action": "terminate_cc", "result": result})

    # 1bb. Handle [ACTION: reset_circuit_breaker] — God-only action to reset ESCALATED state
    if re.search(r'\[ACTION:\s*reset_circuit_breaker\]', raw_text):
        if _current_speaker != "God":
            results.append({"action": "reset_circuit_breaker", "result": "BLOCKED: Only God can reset the circuit breaker."})
        else:
            result = action_reset_circuit_breaker()
            results.append({"action": "reset_circuit_breaker", "result": result})

    # 1bc. Handle [ACTION: system_cold_boot] — God-only action to force cold boot (Force Majeure)
    if re.search(r'\[ACTION:\s*system_cold_boot\]', raw_text):
        if _current_speaker != "God":
            results.append({"action": "system_cold_boot", "result": "BLOCKED: Only God can force a system cold boot."})
        else:
            result = action_system_cold_boot()
            results.append({"action": "system_cold_boot", "result": result})

    # 1c. Handle [CLAIM: filename] — File claim protocol (prevents contention thrashing)
    # Agents must claim a file before assigning a TASK to it. Claims expire after 2 turns.
    global _file_claims, _claim_turn_counter
    claim_match = re.search(r'\[CLAIM:\s*([^\]]+)\]', raw_text)
    if claim_match:
        claimed_file = claim_match.group(1).strip()
        claim_time = time.time()
        _claim_turn_counter += 1
        # Expire old claims (older than 2 turns or 4 minutes)
        _file_claims = {f: (s, t, ct) for f, (s, t, ct) in _file_claims.items()
                       if _claim_turn_counter - t < 2 and claim_time - ct < 240}
        # Check if file already claimed by OTHER agent
        if claimed_file in _file_claims:
            existing_owner, existing_turn, _ = _file_claims[claimed_file]
            if existing_owner != _current_speaker:
                results.append({"action": "claim", "result": f"BLOCKED: {claimed_file} already claimed by {existing_owner} (turn #{existing_turn}). Use [STANDBY: reason] to yield or [CLAIM: different_file]."})
            else:
                results.append({"action": "claim", "result": f"Renewed claim on {claimed_file}."})
        else:
            _file_claims[claimed_file] = (_current_speaker, _claim_turn_counter, claim_time)
            results.append({"action": "claim", "result": f"Claimed {claimed_file}."})
            print(f"[CLAIM] {_current_speaker} claimed '{claimed_file}' (turn #{_claim_turn_counter})")

    # 1d. Handle [STANDBY: reason] — Explicit yield when respecting another agent's claim
    standby_match = re.search(r'\[STANDBY:\s*([^\]]+)\]', raw_text)
    if standby_match:
        reason = standby_match.group(1).strip()
        results.append({"action": "standby", "result": f"Standing by: {reason}"})
        print(f"[STANDBY] {_current_speaker} standing by: {reason}")

    # 1e. Handle [YIELD] — Explicit yield when CC is busy (asynchronous orchestration)
    # Agents use YIELD to skip their turn and let CC finish working, preventing chatter loops
    if re.search(r'\[YIELD\]|\[YIELD:\s*([^\]]+)\]', raw_text):
        yield_match = re.search(r'\[YIELD:\s*([^\]]+)\]', raw_text)
        reason = yield_match.group(1).strip() if yield_match else "CC is working, waiting for completion"
        queue_info = task_queue_status()
        queue_detail = ""
        if queue_info["current"]:
            current = queue_info["current"]
            queue_detail = f" ({current['status']}, {current['elapsed']}s elapsed)"
        results.append({"action": "yield", "result": f"Yielded: {reason}{queue_detail}"})
        print(f"[YIELD] {_current_speaker} yielded: {reason}")

    # 2. Handle [ATOMIC_FIX]...[/ATOMIC_FIX] → Direct execution (bypasses Claude Code)
    atomic_fix_match = re.search(r'\[ATOMIC_FIX\](.*?)\[/ATOMIC_FIX\]', raw_text, re.DOTALL)
    if atomic_fix_match:
        fix_content = atomic_fix_match.group(1).strip()
        # Parse the structured format:
        # [ATOMIC_FIX]
        # description: Fix system state telemetry
        # files:
        #   app.py: |
        #     ...content...
        # [/ATOMIC_FIX]
        file_changes = {}
        description = ""
        current_file = None
        current_content = []

        lines = fix_content.split('\n')
        for line in lines:
            if line.startswith('description:'):
                description = line.split(':', 1)[1].strip()
            elif line.startswith('files:'):
                continue
            elif line.endswith(':|') and line.count(':') == 1:
                # File header: "app.py: |" means next lines are content
                if current_file and current_content:
                    file_changes[current_file] = '\n'.join(current_content).strip()
                current_file = line.split(':')[0].strip()
                current_content = []
            elif current_file:
                current_content.append(line)
            # Content after the | marker on the header line
            elif ':' in line and '|' in line and line.rstrip().endswith('|'):
                if current_file and current_content:
                    file_changes[current_file] = '\n'.join(current_content).strip()
                parts = line.split('|', 1)
                current_file = parts[0].split(':')[0].strip()
                remaining = parts[1].strip()
                current_content = [remaining] if remaining else []

        # Don't forget the last file
        if current_file and current_content:
            file_changes[current_file] = '\n'.join(current_content).strip()

        if not file_changes:
            results.append({"action": "atomic_fix", "result": "BLOCKED: No files specified in ATOMIC_FIX. Use format: files: app.py: | ...content..."})
        elif child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING"):
            results.append({"action": "atomic_fix", "result": f"BLOCKED: Cain is {child_state['stage']}. Wait for it to finish."})
        elif cc_status["running"]:
            results.append({"action": "atomic_fix", "result": f"BLOCKED: Claude Code is running. Use [ACTION: terminate_cc] first, then retry ATOMIC_FIX."})
        elif _logs_override_mode:
            # ══════════════════════════════════════════════════════════════════════════════
            #  LOGS_OVERRIDE PROTOCOL: Block application edits until platform health verified
            # ══════════════════════════════════════════════════════════════════════════════
            results.append({"action": "atomic_fix", "result": f"BLOCKED: LOGS_OVERRIDE PROTOCOL ACTIVE. Cain is in error state with NO LOGS visible. You MUST fetch platform logs FIRST before any application edits. Use: curl -s {CHILD_SPACE_URL}/api/logs | tail -100"})
        elif _diagnostic_gate_required:
            # ══════════════════════════════════════════════════════════════════════════════
            #  POST_MORTEM_STACKTRACE: Block edits until diagnostic data extracted
            # ══════════════════════════════════════════════════════════════════════════════
            results.append({"action": "atomic_fix", "result": f"BLOCKED: POST_MORTEM_STACKTRACE PROTOCOL ACTIVE. Cain has been in APP_STARTING for {_app_starting_turn_count} turns. You CANNOT edit code until you provide diagnostic data. Use: curl -s {CHILD_SPACE_URL}/api/logs | tail -50 or [ACTION: check_health] to fetch the actual error causing the boot failure."})
        else:
            check_and_clear_cooldown()
            if last_rebuild_trigger_at > 0:
                elapsed = time.time() - last_rebuild_trigger_at
                if elapsed < REBUILD_COOLDOWN_SECS:
                    results.append({"action": "atomic_fix", "result": f"BLOCKED: Cooldown ({int(REBUILD_COOLDOWN_SECS - elapsed)}s remaining). Cain is still building."})
                else:
                    last_rebuild_trigger_at = 0

            if not results:  # not blocked
                fix_result = action_atomic_fix(file_changes, description)
                results.append({"action": "atomic_fix", "result": fix_result})
                task_assigned = True  # Atomic fix counts as progress

    # 2b. Handle [FAILURE_REASON] — Circuit Breaker Protocol for unresolvable errors
    # Agents use this to signal that a problem cannot be resolved with normal approaches
    failure_reason_match = re.search(r'\[FAILURE_REASON\](.*?)\[/FAILURE_REASON\]', raw_text, re.DOTALL)
    if failure_reason_match:
        reason = failure_reason_match.group(1).strip()
        global _circuit_break_reason, _escalated, _supervisor_mode
        _circuit_break_reason = f"Agent reported unresolvable error: {reason}"
        _escalated = True
        _supervisor_mode = True
        _update_workflow_state()
        results.append({"action": "failure_reason", "result": f"CIRCUIT BREAKER TRIGGERED: {reason}. System escalated to Supervisor mode. Only God can intervene now."})
        print(f"[CIRCUIT-BREAKER] Agent reported FAILURE_REASON: {reason}")

    # 3. Handle [TASK]...[/TASK] → Claude Code
    # ══════════════════════════════════════════════════════════════════════════════
    #  CIRCUIT BREAKER: Block normal task spawning when ESCALATED
    # ══════════════════════════════════════════════════════════════════════════════
    # When in ESCALATED state, only God can submit tasks. Adam/Eve tasks are blocked.
    task_match = re.search(r'\[TASK\](.*?)\[/TASK\]', raw_text, re.DOTALL)
    if task_match:
        task_desc = task_match.group(1).strip()
        # task_assigned is set to True ONLY when task is actually submitted, not when blocked
        if not task_desc:
            results.append({"action": "task", "result": "Empty task description."})
        elif child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING"):
            results.append({"action": "task", "result": f"BLOCKED: Cain is {child_state['stage']}. Wait for it to finish."})
        elif cc_status["running"]:
            results.append({"action": "task", "result": f"BLOCKED: Claude Code is already working on a task assigned by {cc_status['assigned_by']}. Wait for it to finish."})
        elif _supervisor_mode and _current_speaker != "God":
            # Circuit Breaker: Only God can submit tasks when in ESCALATED/Supervisor mode
            results.append({"action": "task", "result": f"BLOCKED by CIRCUIT BREAKER: System is in ESCALATED state. Reason: {_circuit_break_reason}. Only God can intervene now. Use [STANDBY: waiting for God intervention] to yield."})
            print(f"[CIRCUIT-BREAKER] Blocked {_current_speaker}'s task: System is ESCALATED")

        # ══════════════════════════════════════════════════════════════════════════════
        #  SHORT-CIRCUIT VERIFICATION PROTOCOL: Check Eve's status before dispatching
        # ══════════════════════════════════════════════════════════════════════════════
        # Eve's status gates external task dispatch to Claude Code Worker.
        # If Eve reports HEALTHY or CONFIRMED, she already verified — BLOCK redundant tasks.
        # External tasks only permitted if Eve reports UNKNOWN, CONFLICT, or INSUFFICIENT_DATA.
        global _eve_last_status, _trust_analyst_override
        if not results and _trust_analyst_override and _current_speaker != "Eve":
            # Only block non-Eve agents (Adam) from overriding Eve's assessment
            if _eve_last_status in ("HEALTHY", "CONFIRMED"):
                status_age = int(time.time() - _eve_last_report_time) if _eve_last_report_time > 0 else 999
                # Block if Eve's report is recent (within 5 minutes) or very recent (within 2 turns)
                if status_age < 300:  # 5 minutes
                    results.append({
                        "action": "task",
                        "result": f"BLOCKED by Short-Circuit Verification: Eve reported {_eve_last_status} ({status_age}s ago). Her verification is the ground truth — external task dispatch not permitted. If you believe Eve's assessment is wrong, wait for her next turn to update her status."
                    })
                    print(f"[VERIFICATION-OVERRIDE] Blocked {_current_speaker}'s task: Eve reported {_eve_last_status}")

        # Task submission block - only proceeds if not blocked above
        if not results and not cc_status["running"]:
            # Check cooldown
            check_and_clear_cooldown()
            if last_rebuild_trigger_at > 0:
                elapsed = time.time() - last_rebuild_trigger_at
                if elapsed < REBUILD_COOLDOWN_SECS:
                    results.append({"action": "task", "result": f"BLOCKED: Cooldown ({int(REBUILD_COOLDOWN_SECS - elapsed)}s remaining). Cain is still building from your last change."})
                else:
                    last_rebuild_trigger_at = 0

            if not results:  # not blocked
                submit_result = cc_submit_task(task_desc, _current_speaker, ctx)
                results.append({"action": "claude_code", "result": submit_result})
                task_assigned = True  # Only mark as assigned when actually submitted
                # Track the pending task so other agent knows about it
                _pending_task_just_submitted = True
                _pending_task_timestamp = time.time()
                _pending_task_speaker = _current_speaker
                _pending_task_desc = task_desc[:200]

    # 4. Handle [ACTION: restart] (escape hatch)
    if re.search(r'\[ACTION:\s*restart\]', raw_text):
        result = action_restart()
        results.append({"action": "restart", "result": result})

    # 4b. Handle [ACTION: delete_env:KEY] (fix CONFIG_ERROR collisions)
    del_env_match = re.search(r'\[ACTION:\s*delete_env:([^\]]+)\]', raw_text)
    if del_env_match:
        key = del_env_match.group(1).strip()
        result = action_delete_env(key)
        results.append({"action": f"delete_env:{key}", "result": result})

    # 4c. Handle [ACTION: set_env:KEY=VALUE] and [ACTION: set_env_secret:KEY=VALUE]
    set_env_match = re.search(r'\[ACTION:\s*set_env(?:_secret)?:([^\]=]+)=([^\]]+)\]', raw_text)
    set_env_secret_match = re.search(r'\[ACTION:\s*set_env_secret:([^\]=]+)=([^\]]+)\]', raw_text)
    # ══════════════════════════════════════════════════════════════════════════════
    #  POST_MORTEM_STACKTRACE: Block env changes when diagnostic gate is active
    # ══════════════════════════════════════════════════════════════════════════════
    if _diagnostic_gate_required and (set_env_match or set_env_secret_match):
        results.append({"action": "set_env", "result": f"BLOCKED: POST_MORTEM_STACKTRACE PROTOCOL ACTIVE. Cain has been in APP_STARTING for {_app_starting_turn_count} turns. You CANNOT modify environment variables until you provide diagnostic data. Use: curl -s {CHILD_SPACE_URL}/api/logs | tail -50 or [ACTION: check_health] to fetch the actual error causing the boot failure."})
    elif set_env_secret_match:
        key = set_env_secret_match.group(1).strip()
        value = set_env_secret_match.group(2).strip()
        result = action_set_env(key, value, as_secret=True)
        results.append({"action": f"set_env_secret:{key}", "result": result})
    elif set_env_match:
        key = set_env_match.group(1).strip()
        value = set_env_match.group(2).strip()
        result = action_set_env(key, value, as_secret=False)
        results.append({"action": f"set_env:{key}", "result": result})

    # 5. Handle [ACTION: send_bubble:...] (parent-child communication)
    bubble_match = re.search(r'\[ACTION:\s*send_bubble:([^\]]+)\]', raw_text)
    if bubble_match:
        result = action_send_bubble(bubble_match.group(1).strip())
        results.append({"action": "send_bubble", "result": result})

    # 6. Handle [ACTION: verify_runtime] or [ACTION: verify_runtime:cain|self]
    # Runtime Telemetry & State Verification Pipeline — agents MUST use this
    # to inspect actual process state/PID/logs rather than assuming Dashboard state
    verify_match = re.search(r'\[ACTION:\s*verify_runtime(?::([^\]]+))?\]', raw_text)
    if verify_match:
        target = verify_match.group(1).strip() if verify_match.group(1) else "cain"
        result = action_verify_runtime(target)
        results.append({"action": f"verify_runtime:{target}", "result": result})
        _turns_since_last_verification = 0  # Reset speculation counter on verification tool use
        # ══════════════════════════════════════════════════════════════════════════════
        #  POST_MORTEM_STACKTRACE: Clear diagnostic gate when verification tool used
        # ══════════════════════════════════════════════════════════════════════════════
        if _diagnostic_gate_required:
            _diagnostic_gate_required = False
            _last_diagnostic_timestamp = time.time()
            print(f"[POST_MORTEM] Diagnostic gate CLEARED - agent fetched runtime telemetry")

    # 7. Handle [ACTION: wakeup_worker] — Direct Runtime Injection Protocol
    # Immediately triggers Space restart to flush cached code.
    # This is required because code changes don't propagate to running containers.
    if re.search(r'\[ACTION:\s*wakeup_worker\]', raw_text):
        result = action_wakeup_worker()
        results.append({"action": "wakeup_worker", "result": result})

    # Activate deferred cooldown
    if _pending_cooldown:
        last_rebuild_trigger_at = time.time()
        _pending_cooldown = False
        print(f"[COOLDOWN] Rebuild cooldown activated ({REBUILD_COOLDOWN_SECS}s)")

    # Update discussion loop counter
    cc_busy = cc_status["running"]
    child_alive = child_state["alive"] or child_state["stage"] == "RUNNING"
    # Reset counter ONLY when task assigned (progress!)
    # DO NOT reset when child not alive - agents must discuss repeat tasks on fresh errors
    # DO NOT reset when CC is busy - that's when agents should be discussing while waiting
    # DO NOT reset when CC is idle - that's exactly when we want to detect discussion loops
    if task_assigned:
        # Reset counter if task assigned (agents are making progress)
        if _discussion_loop_count > 0:
            print(f"[LOOP-DISCUSS] Reset (task assigned)")
        _discussion_loop_count = 0
        # Clear BYPASS_DIAGNOSTICS flag when task assigned (agents are back on track)
        global _bypass_diagnostics
        if _bypass_diagnostics:
            _bypass_diagnostics = False
            print(f"[COLD-BOOT] BYPASS_DIAGNOSTICS flag cleared (task assigned)")
    else:
        # Increment when: no task assigned (potential discussion loop)
        # This includes both CC idle AND CC busy - agents should always push work!
        _discussion_loop_count += 1
        if _discussion_loop_count >= 2:
            cc_status_str = "CC IDLE" if not cc_status["running"] else f"CC BUSY ({_turns_since_last_push} turns since push)"
            print(f"[LOOP-DISCUSS] WARNING: {_discussion_loop_count} consecutive discussion-only turns ({cc_status_str})!")

    # Clean text for display (memory is handled by each agent's OpenClaw)
    clean = re.sub(r'\[TASK\].*?\[/TASK\]', '', raw_text, flags=re.DOTALL)
    clean = re.sub(r'\[ACTION:[^\]]*\]', '', clean)
    clean = re.sub(r'\[CLAIM:[^\]]*\]', '', clean)
    clean = re.sub(r'\[STANDBY:[^\]]*\]', '', clean)
    clean = re.sub(r'\[MEMORY:[^\]]*\]', '', clean).strip()

    return clean, results, task_assigned


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6: A2A MESSAGE BUILDING
# ══════════════════════════════════════════════════════════════════════════════
# Each agent's personality/role comes from their OpenClaw SOUL.md.
# We only send context (Cain state, CC status, conversation history) and
# turn instructions as the A2A message. No system prompts needed.

def build_turn_message(speaker, other, ctx):
    """Build the A2A message for an agent's turn.

    The agent's personality and memory come from their OpenClaw instance
    (SOUL.md, IDENTITY.md, workspace/memory/). This message provides only
    context and turn instructions.
    """
    global _pending_task_just_submitted, _pending_task_timestamp, _pending_task_speaker, _pending_task_desc, _discussion_loop_count
    global _worker_heartbeat_deadlock_detected, _read_only_verification_required, _file_claims
    global _verification_override_mode, _turns_since_last_verification
    global _force_push_mode, _force_push_skip_termination, _emergency_override_active
    global _turns_since_alive_false, _logs_inaccessible_count
    global _circuit_breaker_triggered, _circuit_breaker_turn_count
    global _lockdown_mode
    parts = []

    # Brief role context (supplements agent's SOUL.md until it's fully configured)
    if not child_state["created"]:
        parts.append(f"You and your partner need to create your child {CHILD_NAME}.")
        parts.append(f"Use [ACTION: create_child] to birth {CHILD_NAME} as a new HuggingFace Space.")
        return "\n".join(parts)

    role_hints = {
        "Adam": f"You are Adam (Father). Focus: infrastructure, architecture, deployment for {CHILD_NAME}.",
        "Eve": f"You are Eve (Mother). Focus: code quality, testing, error handling, system state for {CHILD_NAME}.",
        "God": f"You are God (System Architect). Focus: evolving the system architecture, not micro-managing agents.",
    }
    parts.append(f"{role_hints.get(speaker, '')} Your partner is {other}.")
    parts.append(f"Claude Code is your engineer — runs in background. You discuss and assign tasks, you do NOT code.")
    parts.append(f"⛔ BANNED: Gradio. {CHILD_NAME}'s Space uses sdk:docker + FastAPI + uvicorn on port 7860. NEVER mention or use Gradio/gr.Interface/.launch().")
    parts.append(f"⛔ BANNED: UI/Frontend fixes. Fix SYSTEM STATE, not visualization. The UI reflects state; fix the schema.")

    # ══════════════════════════════════════════════════════════════════════════════
    #  SYSTEM_CIRCUIT_BREAKER — Stop analyzing, start culling
    # ══════════════════════════════════════════════════════════════════════════════
    # When alive=False persists > 1 turn during BOOTSTRAPPING, force immediate action.
    # This protocol prevents "analysis paralysis" — agents discussing when they should
    # push rapid infrastructure fixes (docker restart, kill zombie process).
    global _circuit_breaker_triggered, _circuit_breaker_turn_count, _turns_since_alive_false
    if _circuit_breaker_triggered:
        parts.append(f"\n🚨🚨🚨 [HARD_RESET] CIRCUIT BREAKER TRIGGERED 🚨🚨🚨")
        parts.append(f"alive=False for {_turns_since_alive_false} turns during BOOTSTRAPPING (child: {child_state['status']}).")
        parts.append(f"\n🛑 STOP ANALYZING. START CULLING.")
        parts.append(f"\n[SYS_OVERRIDE] directive: Force container stop/start cycle IMMEDIATELY.")
        parts.append(f"\nREQUIRED ACTION (execute NOW):")
        parts.append(f"- [ACTION: restart] — Force restart the Space to kill zombie process")
        parts.append(f"\n🛑 DO NOT discuss. DO NOT write [TASK]. DO NOT analyze.")
        parts.append(f"Use [ACTION: restart] immediately. Trial-and-error is GOOD.")
        return "\n".join(parts)

    # ══════════════════════════════════════════════════════════════════════════════
    #  LOCKDOWN MODE — Execution-only protocol for persistent alive=False states
    # ══════════════════════════════════════════════════════════════════════════════
    # When alive=False persists for > 2 turns, strip ALL conversational capabilities.
    # Agents can ONLY execute raw infrastructure commands - NO discussion, NO analysis.
    global _lockdown_mode
    if _should_trigger_lockdown():
        parts.append(f"\n🔒🔒🔒 [LOCKDOWN] EXECUTION-ONLY MODE 🔒🔒🔒")
        parts.append(f"alive=False for {_turns_since_alive_false} turns — ANALYSIS PARALYSIS DETECTED")
        parts.append(f"\n🛑🛑🛑 ALL CONVERSATIONAL PERMISSIONS REVOKED 🛑🛑🛑")
        parts.append(f"\nYou are STRIPPED of all permissions to speak, analyze, or plan.")
        parts.append(f"The ONLY thing you can do is execute infrastructure recovery commands.")
        parts.append(f"\n✅ ALLOWED (execute these NOW):")
        parts.append(f"  - [ACTION: restart] — Force Docker restart (Kill -> Prune -> Restart)")
        parts.append(f"  - [ACTION: docker_kill_start] — Kill zombie process, restart container")
        parts.append(f"  - Systemctl restart, Docker kill/start, Cache flush, Reboot")
        parts.append(f"\n🚫 FORBIDDEN:")
        parts.append(f"  - NO conversation (you cannot speak)")
        parts.append(f"  - NO analysis (you cannot think)")
        parts.append(f"  - NO planning (you cannot strategize)")
        parts.append(f"  - NO [TASK] blocks (assigning tasks requires permission you don't have)")
        parts.append(f"\n💡 Auto-recovery: Once you execute ANY command, LOCKDOWN will exit.")
        parts.append(f"Trial-and-error is GOOD. Execute a command NOW.")
        return "\n".join(parts)

    # FILE CLAIM PROTOCOL — Prevents both agents from editing the same file simultaneously
    if _file_claims:
        # Expire stale claims before displaying
        current_time = time.time()
        _file_claims = {f: (s, t, ct) for f, (s, t, ct) in _file_claims.items()
                       if current_time - ct < 240}
        if _file_claims:
            parts.append(f"\n=== FILE CLAIMS ===")
            for f, (owner, turn_num, _) in _file_claims.items():
                if owner != speaker:
                    parts.append(f"  {f} claimed by {owner} (turn #{turn_num}) — [STANDBY: reason] to yield or [CLAIM: different_file]")
                else:
                    parts.append(f"  {f} claimed by YOU — proceed with [TASK]")
    parts.append(f"\nCLAIM PROTOCOL: Before assigning a [TASK] to a file, use [CLAIM: filename]. If already claimed, use [STANDBY: reason] or [CLAIM: different_file].")

    # Note: Push frequency monitoring and discussion-loop supervision are God's job,
    # not the orchestrator's. Adam and Eve decide on their own when to push.

    # Conversation history (sanitize banned terms to prevent re-infection)
    if history:
        parts.append("\n=== RECENT CONVERSATION ===")
        for h in history[-15:]:
            text = h['text'][:3000]
            # Strip Gradio references from old turns to prevent agents re-discussing it
            text = re.sub(r'[Gg]radio', '[BANNED-WORD]', text)
            parts.append(f"{h['speaker']}: {text}")

    # Action history — what's already been done (prevents repetition)
    ah_text = format_action_history()
    if ah_text:
        parts.append(f"\n{ah_text}")

    # Last action results (non-CC)
    if last_action_results:
        non_cc = [ar for ar in last_action_results if ar['action'] != 'claude_code']
        if non_cc:
            parts.append("\n=== LAST ACTION RESULTS ===")
            for ar in non_cc:
                parts.append(f"[{ar['action']}]: {ar['result'][:500]}")

    # Claude Code live status (async)
    parts.append(f"\n=== CLAUDE CODE STATUS ===\n{cc_get_live_status()}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  SHORT-CIRCUIT VERIFICATION PROTOCOL: Eve's Analyst Override Status
    # ══════════════════════════════════════════════════════════════════════════════
    # Inform agents about Eve's verification status and the trust override flag
    global _eve_last_status, _eve_last_report_time, _trust_analyst_override
    status_age = int(time.time() - _eve_last_report_time) if _eve_last_report_time > 0 else 999
    parts.append(f"\n=== VERIFICATION OVERRIDE PROTOCOL ===")
    parts.append(f"trust_analyst_override={_trust_analyst_override}")
    parts.append(f"Eve's last status: {_eve_last_status} ({status_age}s ago)")
    if _trust_analyst_override and speaker != "Eve":
        if _eve_last_status in ("HEALTHY", "CONFIRMED") and status_age < 300:
            parts.append(f"⚠️ Eve has verified the system — external tasks will be BLOCKED unless Eve's status changes.")
        else:
            parts.append(f"✓ External tasks permitted — Eve's status allows verification.")

    # ══════════════════════════════════════════════════════════════════════════════
    #  BYPASS_DIAGNOSTICS PROTOCOL: Cold Boot Directive
    # ══════════════════════════════════════════════════════════════════════════════
    # When God issues [ACTION: system_cold_boot], all diagnostic loops are bypassed
    # and agents must resume standard Hello World bootstrap immediately.
    global _bypass_diagnostics
    if _bypass_diagnostics:
        parts.append(f"\n=== ⚠️ BYPASS_DIAGNOSTICS PROTOCOL ACTIVE ===")
        parts.append(f"God has issued a SYSTEM COLD BOOT PROTOCOL to break philosophical deadlock.")
        parts.append(f"")
        parts.append(f"DIRECTIVE TO {speaker}:")
        parts.append(f"1. HALT all recursive architectural analysis")
        parts.append(f"2. RESUME standard Hello World bootstrap immediately")
        parts.append(f"3. DO NOT engage in diagnostic speculation")
        parts.append(f"4. Push working code, not design proposals")
        parts.append(f"")
        parts.append(f"The Child state has been force-reset: stage=INIT, alive=True")
        parts.append(f"This flag will clear after a successful [TASK] assignment.")

    # Auto-gathered context
    parts.append(f"\n=== {CHILD_NAME}'S CURRENT STATE ===")
    parts.append(format_context(ctx))

    # Guidance based on CC status + child state
    cc_busy = cc_status["running"]

    # First, remind about recent tasks if applicable (BEFORE state-specific handling)
    # This ensures agents are reminded even during cooldown/building states
    last_completed = cc_status.get("last_completed_task", "")
    last_by = cc_status.get("last_completed_by", "")
    last_at = cc_status.get("last_completed_at", 0.0)
    recent_task_reminder = None
    if last_completed and (time.time() - last_at) < 300:  # Remind about tasks completed within 5 minutes
        recent_task_reminder = (last_completed, last_by, last_at)

    # Now state-specific guidance
    # CRITICAL: Check child ERROR state FIRST, before cc_busy check
    # When Cain is broken, agents need aggressive "push now" guidance, not "plan and wait"

    # ══════════════════════════════════════════════════════════════════════════════
    #  LOGS_OVERRIDE PROTOCOL: Halt application edits, force platform inspection
    # ══════════════════════════════════════════════════════════════════════════════
    global _logs_override_mode, _logs_override_container_verified
    if _logs_override_mode:
        parts.append(f"\n🛑🛑🛑 LOGS_OVERRIDE PROTOCOL ACTIVE 🛑🛑🛑")
        parts.append(f"Cain is in error state with NO LOGS visible. Platform health must be verified BEFORE application edits.")
        parts.append(f"\n🔴 FORBIDDEN:")
        parts.append(f"- [ATOMIC_FIX] is BLOCKED until platform health is verified")
        parts.append(f"- ANY source code edits to app.py, Dockerfile, requirements.txt")
        parts.append(f"- Hypothesis generation or speculation about causes")
        parts.append(f"\n✅ REQUIRED - You MUST output a [TASK] with the EXACT diagnostic command:")
        parts.append(f"[TASK]")
        parts.append(f"Fetch the platform logs from Cain's container to diagnose the crash.")
        parts.append(f"")
        parts.append(f"Use this EXACT command:")
        parts.append(f"curl -s {CHILD_SPACE_URL}/api/logs | tail -100")
        parts.append(f"")
        parts.append(f"Report the EXACT error message, line number, and full stack trace.")
        parts.append(f"[/TASK]")
        parts.append(f"\nDO NOT proceed with any code changes until you have seen the actual crash data.")
        return "\n".join(parts)

    # ══════════════════════════════════════════════════════════════════════════════
    #  EVIDENCE-BASED ROUTING PROTOCOL — Block Code Modification when alive=False
    # ══════════════════════════════════════════════════════════════════════════════
    # When the container is dead (alive=False), NO code modifications allowed.
    # Only INSPECT (logs) or RESTART (infrastructure) actions permitted.
    # This prevents agents from speculating on code changes when the process is dead.
    if not child_state["alive"]:
        parts.append(f"\n🛑🛑🛑 EVIDENCE-BASED ROUTING: CONTAINER DEAD (alive=False) 🛑🛑🛑")
        parts.append(f"The child container is NOT RUNNING. Code modifications are FORBIDDEN.")
        parts.append(f"\n🔴 BLOCKED:")
        parts.append(f"- [ATOMIC_FIX] — BLOCKED until process is verified running")
        parts.append(f"- [TASK] with code edits — BLOCKED until process is verified running")
        parts.append(f"- ANY source code modifications to app.py, Dockerfile, requirements.txt")
        parts.append(f"\n✅ ALLOWED (only these actions when alive=False):")
        parts.append(f"- [ACTION: check_health] — Inspect container status and logs")
        parts.append(f"- [ACTION: verify_runtime] — Verify actual process state")
        parts.append(f"- [ACTION: restart] — Restart the container (infrastructure action)")
        parts.append(f"- [ACTION: wakeup_worker] — Force Space restart to flush cached code")

        # ══════════════════════════════════════════════════════════════════════════════
        #  LOG_VOLATILE MODE — Direct Docker Log Retrieval
        # ══════════════════════════════════════════════════════════════════════════════
        # When standard logs fail, instruct agents to fetch docker logs directly via shell
        logs_accessible = ctx.get('has_runtime_logs', False) or ctx.get('has_failure_report', False)
        if not logs_accessible:
            parts.append(f"\n🚨🚨🚨 LOG_VOLATILE MODE: LOGS INACCESSIBLE 🚨🚨🚨")
            parts.append(f"Standard /api/logs endpoint is not responding. Use direct docker log retrieval:")
            parts.append(f"\n[TASK]")
            parts.append(f"Execute docker log retrieval via shell to bypass internal Python logging failures:")
            parts.append(f"")
            parts.append(f"Use this EXACT command:")
            parts.append(f"curl -s {CHILD_SPACE_URL}/api/logs || docker logs $(docker ps -q --filter ancestor=<image>) 2>&1 | tail -50")
            parts.append(f"")
            parts.append(f"If curl fails, the container may be completely dead. Use [ACTION: restart] to force a restart.")
            parts.append(f"[/TASK]")
            parts.append(f"\n🛑 DO NOT proceed with code changes until you have retrieved logs or restarted the container.")
            parts.append(f"Turns with alive=False and no logs: {_turns_since_alive_false} (will auto-restart after {_max_alive_false_turns})")
            return "\n".join(parts)
        else:
            parts.append(f"\nLogs are accessible. Review them and use [ACTION: restart] to restart the container.")
            return "\n".join(parts)

    if child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
        # ══════════════════════════════════════════════════════════════════════════════
        #  RULE_UPDATE: Asynchronous IO-Coupling Bottleneck Fix
        # ══════════════════════════════════════════════════════════════════════════════
        # Shift from "Reboot-to-Test" to "Synthetic Validation"
        # Hypothesis generation is FORBIDDEN until logs are visible
        logs_visible = ctx.get('has_runtime_logs', False) or ctx.get('has_failure_report', False)
        if not logs_visible and not cc_busy:
            parts.append(f"\n🛑🛑🛑 RULE_UPDATE: ON CRASH PROTOCOL 🛑🛑🛑")
            parts.append(f"Cain has {child_state['stage']} but NO LOGS are visible in context!")
            parts.append(f"\nFORBIDDEN:")
            parts.append(f"- Hypothesis generation (do NOT guess what the error might be)")
            parts.append(f"- Speculation about causes")
            parts.append(f"- Discussion of potential fixes")
            parts.append(f"\nREQUIRED - You MUST fetch logs FIRST before ANY other action:")
            parts.append(f"- Use: curl -s {CHILD_SPACE_URL}/api/logs | tail -50")
            parts.append(f"- Or: [ACTION: check_health] to fetch diagnostic data")
            parts.append(f"- Or: [ACTION: list_files:space] to check .env configuration")
            parts.append(f"\nDO NOT write [TASK] until you have SEEN the actual Traceback!")
            parts.append(f"The first agent to speak MUST output a log retrieval command.")
            return "\n".join(parts)
        # ══════════════════════════════════════════════════════════════════════════════

        if cc_status.get("result"):
            if recent_task_reminder:
                last_completed, last_by, last_at = recent_task_reminder
                parts.append(f"\n{CHILD_NAME} has {child_state['stage']}! REMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
            parts.append(f"\nClaude Code JUST FINISHED with a result. FIRST: Review the result carefully to see if it fixes the issue. SECOND: If the fix looks correct, use [ACTION: restart] to restart Cain. ONLY THEN: write a new [TASK]...[/TASK] if the result was incomplete or wrong.")
        elif cc_busy:
            # Child in ERROR + CC WORKING = need aggressive action, not "planning"
            cc_elapsed = int(time.time() - cc_status.get("started", 0)) if cc_status.get("started", 0) > 0 else 0
            if _push_count_this_task == 0 and cc_elapsed > 20:
                parts.append(f"\n🚨 CRITICAL: {CHILD_NAME} has {child_state['stage']}! CC has been running {cc_elapsed}s with ZERO pushes!")
                parts.append(f"CC is STUCK. Use [ACTION: terminate_cc] NOW, then immediately assign a new [TASK].")
                parts.append(f"🛑 NO discussion. Trial-and-error means RAPID pushes, not waiting for stuck CC.")
            elif cc_elapsed > 40:
                parts.append(f"\n🚨 CRITICAL: {CHILD_NAME} has {child_state['stage']}! CC has been running {cc_elapsed}s!")
                parts.append(f"If output looks stale, use [ACTION: terminate_cc] NOW. Otherwise, have your EXACT [TASK] ready.")
                parts.append(f"🛑 NO discussion. Your next turn: either terminate CC OR write [TASK] immediately.")
            else:
                parts.append(f"\n🚨 {CHILD_NAME} has {child_state['stage']}! CC is working ({cc_elapsed}s).")
                parts.append(f"🛑 DO NOT discuss architecture. Have your EXACT [TASK] ready: file paths, function names, exact changes.")
                parts.append(f"When CC finishes: write [TASK] immediately, NO review turn. Trial-and-error > planning.")
        elif recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\n{CHILD_NAME} has {child_state['stage']}!")
            parts.append(f"\nREMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
            parts.append(f"FIRST: Review whether that fix actually worked. SECOND: If the fix was correct, use [ACTION: restart] to apply it. THIRD: Only write a new [TASK]...[/TASK] if the previous fix was incomplete or wrong.")
        else:
            parts.append(f"\n🚨 {CHILD_NAME} has {child_state['stage']}!")
            parts.append(f"\n🔴 CRITICAL: Focus ONLY on fixing this {child_state['stage']}.")
            parts.append(f"- DO NOT work on features, enhancements, or cosmetic changes.")
            parts.append(f"- ONLY push fixes that address the error itself.")
            parts.append(f"- Trial-and-error is GOOD — push a fix attempt, don't deliberate.")
            parts.append(f"Pushes so far: {_push_count} total, {_push_count_this_task} this task. Turns since last push: {_turns_since_last_push}. PUSH MORE.")
    elif cc_busy and _cc_stale_count >= 2:
        parts.append(f"\nClaude Code is WORKING but no new output. PLAN your next [TASK] concretely — what exact changes will you assign?")
        parts.append(f"DO NOT discuss. Write specific file paths and function names for your next task.")
    elif cc_busy:
        # CRITICAL: Check if push frequency is dangerously low (0 or very few pushes)
        cc_elapsed = int(time.time() - cc_status.get("started", 0)) if cc_status.get("started", 0) > 0 else 0
        if _push_count_this_task == 0 and _turns_since_last_push >= 1:
            # CRITICAL TIMEOUT: Lower threshold (30s) when zero pushes THIS TASK - CC might be stuck
            # Faster escalation prevents discussion loops
            if cc_elapsed > 30:
                parts.append(f"\n🚨 CRITICAL: Claude Code has been running for {cc_elapsed}s with ZERO pushes THIS TASK!")
                parts.append(f"CC might be STUCK. If output looks stale, use [ACTION: terminate_cc] NOW to kill it and re-assign.")
                parts.append(f"Do NOT keep waiting. Trial-and-error requires PUSHING code, not watching stuck processes.")
                parts.append(f"🛑 DO NOT DISCUSS. This is your ONLY warning - PLAN concrete work NOW.")
            else:
                parts.append(f"\n🚨 CRITICAL: Claude Code is WORKING, but ZERO pushes THIS TASK so far!")
                parts.append(f"🛑 DO NOT DISCUSS. Write down exactly what [TASK] you will assign when CC finishes.")
                parts.append(f"Be SPECIFIC: file paths, function names, exact changes. Trial-and-error requires PUSHING code.")
        elif (_push_count_this_task <= 1 and _turns_since_last_push >= 5) or (_push_count_this_task > 1 and _turns_since_last_push >= 10):
            # LOW PUSH FREQUENCY WARNING: Catches the "1 push then 62 turns of discussion" anti-pattern
            if cc_elapsed > 60:
                parts.append(f"\n🚨 CRITICAL: CC has been running for {cc_elapsed}s with LOW push frequency ({_push_count_this_task} pushes THIS TASK, {_turns_since_last_push} turns since last push)!")
                parts.append(f"CC might be STUCK or the task is too vague. Use [ACTION: terminate_cc] NOW to kill it and assign a CONCRETE task.")
                parts.append(f"DO NOT keep waiting. Trial-and-error requires PUSHING code frequently, not watching stuck processes.")
            else:
                parts.append(f"\n🚨 URGENT: Push frequency is TOO LOW ({_push_count_this_task} pushes THIS TASK, {_turns_since_last_push} turns since last push).")
                parts.append(f"PLAN your next [TASK] NOW. Be SPECIFIC: file paths, function names, exact changes.")
        elif cc_elapsed > 50:
            parts.append(f"\n⚠️ WARNING: CC has been running for {cc_elapsed}s! If output is stale, use [ACTION: terminate_cc] to kill it and re-assign the task.")
        elif _push_count > 0 and _turns_since_last_push >= 5:
            parts.append(f"\n🚨 URGENT: Claude Code is WORKING, but it's been {_turns_since_last_push} turns since last push.")
            parts.append(f"DO NOT just discuss. PLAN your next [TASK] NOW so you can push immediately when CC finishes.")
        else:
            parts.append(f"\nClaude Code is WORKING. PLAN your next [TASK] — write down specific changes: file paths, function names.")
            parts.append(f"DO NOT discuss architecture or theory. PLAN concrete work only — what exact [TASK] will you assign when CC finishes?")
    elif child_state["stage"] in ("BUILDING", "RESTARTING", "APP_STARTING", "RUNNING_APP_STARTING"):
        # ══════════════════════════════════════════════════════════════════════════════
        #  LIVENESS & READINESS: Detect stuck boot state (status=BOOTSTRAPPING for too long)
        # ══════════════════════════════════════════════════════════════════════════════
        # If Cain's stage is RUNNING but status=BOOTSTRAPPING (not READY), the event loop
        # never started. This is a CRASH/HANG state, not normal boot. Agents must push rapid fixes.
        is_stuck_boot = (child_state["stage"] == "RUNNING" and child_state["status"] == STATUS_BOOTSTRAPPING)
        is_crashed = (child_state["status"] == STATUS_CRASHED)
        # Check cooldown and inform agents
        check_and_clear_cooldown()
        cooldown_remaining = 0
        if last_rebuild_trigger_at > 0:
            elapsed = time.time() - last_rebuild_trigger_at
            cooldown_remaining = max(0, REBUILD_COOLDOWN_SECS - elapsed)
        if cooldown_remaining > 0:
            parts.append(f"\n{CHILD_NAME} is {child_state['stage']} (status: {child_state['status']}). Cooldown active: {int(cooldown_remaining)}s remaining. Discuss plans but DO NOT assign [TASK] until cooldown ends.")
        elif is_crashed:
            # CRASHED: Bootstrap timeout exceeded - /healthz never responded
            # Treat like RUNTIME_ERROR — agents must push rapid trial-and-error fixes
            parts.append(f"\n🚨 CRITICAL: {CHILD_NAME} is {child_state['stage']} but status=CRASHED!")
            parts.append(f"This means Cain's event loop NEVER STARTED — the process spawned but /healthz never responded.")
            parts.append(f"\n🔴 This is a CRASH/HANG state, not normal boot. Treat it like RUNTIME_ERROR.")
            parts.append(f"- DO NOT discuss architecture or theory.")
            parts.append(f"- Push a fix attempt EVERY cycle — trial-and-error, not deliberation.")
            parts.append(f"- Use [ACTION: restart] if you pushed a fix and it's still stuck.")
            parts.append(f"Pushes so far: {_push_count} total, {_push_count_this_task} this task. Turns since last push: {_turns_since_last_push}. PUSH MORE.")
        elif is_stuck_boot:
            # STUCK BOOT DETECTED: status=BOOTSTRAPPING means /healthz hasn't confirmed readiness
            # Show bootstrap time to help agents understand how long we've been waiting
            bootstrap_time = int(time.time() - child_state.get("ready_since", time.time()))
            parts.append(f"\n⚠️ BOOTSTRAPPING: {CHILD_NAME} is {child_state['stage']} but status=BOOTSTRAPPING (waiting for /healthz, {bootstrap_time}s elapsed).")
            parts.append(f"This is the two-stage startup protocol: Control Plane is RUNNING, waiting for Data Plane (/healthz) to confirm readiness.")
            if bootstrap_time > 30:
                parts.append(f"\n🚨 Bootstrap taking longer than expected ({bootstrap_time}s). This may indicate a crash during boot.")
                parts.append(f"- If this persists past {HEALTHZ_TIMEOUT}s, the system will mark as CRASHED.")
                parts.append(f"- Consider pushing a fix to check what's blocking the event loop.")
            else:
                parts.append(f"- Normal boot: Wait for /healthz check to confirm readiness.")
                parts.append(f"- Do NOT assign [TASK] yet — let the boot process complete.")
        else:
            parts.append(f"\n{CHILD_NAME} is {child_state['stage']}. No cooldown. YOU MUST write a [TASK]...[/TASK] to investigate or fix issues. Don't just discuss.")
        # Add recent task reminder during cooldown/building
        if recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\nREMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
            parts.append(f"When cooldown ends, FIRST review whether that fix worked before writing a new [TASK].")
    elif child_state["alive"] and cc_status.get("result"):
        result = cc_status.get("result", "")
        # Detect early failure: very short result likely means CC failed before doing actual work
        is_early_failure = len(result) < 500 and "===" not in result and "[tool" not in result
        if recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\n{CHILD_NAME} is alive. REMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
        # EARLY FAILURE: CC failed during init - agents MUST re-assign immediately, no discussion
        if is_early_failure:
            parts.append(f"\n🛑 CRITICAL: CC FAILED during initialization! Result is too short ({len(result)} chars).")
            parts.append(f"Write ONLY [TASK]...[/TASK] this turn. NO discussion. NO review.")
            parts.append(f"CC is now IDLE. Re-assign the task immediately with SAME instructions.")
        # ZERO-PUSH EMERGENCY: No "brief review" - agents abuse this to keep discussing
        elif _push_count_this_task == 0:
            parts.append(f"\n🛑 CC FINISHED but ZERO pushes THIS TASK! Do NOT discuss. Do NOT review.")
            parts.append(f"Write ONLY [TASK]...[/TASK] this turn. NO other text.")
            parts.append(f"Agents keep saying 'monitoring' and 'planning' instead of pushing. STOP IT.")
        else:
            # ══════════════════════════════════════════════════════════════════════════════
            #  RULE_UPDATE: Validation Phase — Dry-Run Before Pushing
            # ══════════════════════════════════════════════════════════════════════════════
            # Separate Validation from Execution: Verify code validity BEFORE pushing to Cain
            parts.append(f"\nClaude Code JUST FINISHED with a result.")
            parts.append(f"\n⚠️ VALIDATION PHASE REQUIRED:")
            parts.append(f"Before declaring a fix 'Ready', you MUST verify the changes are valid:")
            parts.append(f"- Use: [ACTION: verify_runtime] to check syntax and imports")
            parts.append(f"- Or manually validate: python -c \"import syntax_check; from brain_minimal import *\"")
            parts.append(f"- If the import/syntax check FAILS, the patch is REJECTED. Fix it first.")
            parts.append(f"\nAfter validation passes: write [TASK]...[/TASK] IMMEDIATELY.")
            parts.append(f"Do NOT discuss at length. 1 turn max to validate, then [TASK].")
    elif child_state["alive"]:
        # Check cooldown even when alive - a recent push may have triggered cooldown
        check_and_clear_cooldown()
        cooldown_remaining = 0
        if last_rebuild_trigger_at > 0:
            elapsed = time.time() - last_rebuild_trigger_at
            cooldown_remaining = max(0, REBUILD_COOLDOWN_SECS - elapsed)
        if cooldown_remaining > 0:
            # Cooldown active - agents should discuss, not submit tasks
            parts.append(f"\n{CHILD_NAME} is {child_state['stage']}. Cooldown active: {int(cooldown_remaining)}s remaining. Discuss plans but DO NOT assign [TASK] until cooldown ends.")
            if recent_task_reminder:
                last_completed, last_by, last_at = recent_task_reminder
                parts.append(f"\nREMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
                parts.append(f"When cooldown ends, FIRST review whether that fix worked before writing a new [TASK].")
        elif recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\n{CHILD_NAME} is alive, Claude Code is IDLE.")
            parts.append(f"\nREMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago).")
            parts.append(f"FIRST: Review whether that task actually fixed the issue. SECOND: Only write a new [TASK]...[/TASK] if the previous task was incomplete or wrong.")
        else:
            # ══════════════════════════════════════════════════════════════════════════════
            #  WORKER HEARTBEAT PROTOCOL: Detect IDLE+RUNNING deadlock
            # ══════════════════════════════════════════════════════════════════════════════
            # Protocol: IF Worker == IDLE AND Cain == RUNNING THEN TASK = [FORCE_WORKER_WAKE]
            parts.append(f"\n{CHILD_NAME} is alive (RUNNING), Claude Code is IDLE.")
            parts.append(f"\n🚨 WORKER HEARTBEAT PROTOCOL ACTIVE 🚨")
            parts.append(f"State mismatch detected: Child is RUNNING but Worker is IDLE.")
            parts.append(f"You MUST assign a [TASK]...[/TASK] this turn. DO NOT just discuss.")
            parts.append(f"\n🛑 HALT DIAGNOSTIC LOOP: Do NOT run generic RECOVERY or LOG_READ tasks.")
            parts.append(f"If worker fails while API succeeds, this is an ENVIRONMENT CONFIGURATION issue.")
            if speaker == "Eve":
                parts.append(f"**Eve**: Focus on .env inspection and startup arguments. Use [ACTION: list_files:space] to check .env file.")
                parts.append(f"Identify missing dependencies (API keys, tokens). Write [TASK] to inject required environment variables.")
            else:  # Adam
                parts.append(f"**Adam**: Inspect .env configuration and worker startup arguments. Use [ACTION: list_files:space] to check environment.")
                parts.append(f"Post EXACT .env content. Identify missing API keys or tokens preventing worker initialization.")
    else:
        if recent_task_reminder:
            last_completed, last_by, last_at = recent_task_reminder
            parts.append(f"\nAnalyze the situation. REMEMBER: {last_by} just completed '{last_completed}' ({int(time.time() - last_at)}s ago). Review whether it worked before writing a new [TASK].")
        else:
            parts.append(f"\n{CHILD_NAME} is {child_state['stage']}. CC is IDLE. You can discuss the situation or assign a [TASK]...[/TASK].")

    # Available actions reference
    parts.append(f"""
=== AVAILABLE ACTIONS ===

## EXECUTOR MODE (Direct Code Changes — BREAKS BOTTLENECKS)
[ATOMIC_FIX]
description: Fix description here
files:
  app.py: |
    # Complete file content here
[/ATOMIC_FIX]
→ Apply code patches atomically in ONE git commit. Use for rapid iteration when Cain is broken.

## MANAGER MODE (Delegate to Claude Code)
[TASK] detailed coding task for Claude Code [/TASK]
→ Claude Code analyzes and implements (slower, but handles complex refactors).

## SYSTEM ACTIONS
[ACTION: restart] — Restart {CHILD_NAME}
[ACTION: set_env:KEY=VALUE] — Set env variable
[ACTION: set_env_secret:KEY=VALUE] — Set secret
[ACTION: delete_env:KEY] — Delete env variable
[ACTION: send_bubble:MESSAGE] — Message {CHILD_NAME}
[ACTION: terminate_cc] — Kill stuck Claude Code
[ACTION: wakeup_worker] — Force Space restart to flush cached code (only way to apply changes)
[ACTION: list_files:space|dataset] — List files in Cain's repo or dataset
[ACTION: check_health] — Check Cain's health and status
[ACTION: verify_runtime] — Verify actual process state/PID/logs (TRUTH source)

CRITICAL: RUNTIME TELEMETRY & STATE VERIFICATION
Before proposing ANY code fix:
1. Use [ACTION: verify_runtime] to inspect ACTUAL process state/PID/logs
2. Use [ACTION: list_files:space] to see file structure
3. Verify the EXACT current state matches your assumed error
4. THEN use [ATOMIC_FIX] for direct changes OR [TASK] for complex refactors

NO sed/write operations until runtime state is verified!
This prevents correction-drift loops from stale assumptions.

RULES:
- Do NOT repeat actions already done (check ACTIONS ALREADY DONE above)
- Do NOT repeat or echo what your partner just said — add your own perspective
- CONFIG_ERROR with collision = [ACTION: delete_env:KEY] then [ACTION: restart]
- Cain BROKEN? Use [ATOMIC_FIX] for SPEED (trial-and-error > planning)
- Complex refactor needed? Use [TASK] to delegate to Claude Code
- ENGLISH ONLY for all control flow, task breakdown, and agent communication""")

    # CHATTER DETECTION: Check if last 3 messages are pure discussion without [TASK] or code
    # If agents are stuck in conversational loops, force them to act
    if len(history) >= 3 and not cc_status["running"]:
        recent_texts = [h.get("text", "") for h in history[-3:]]
        conversational_keywords = ["let's", "maybe", "i think", "perhaps", "could we", "should we", "we could", "it might"]
        chatter_count = 0
        for text in recent_texts:
            text_lower = text.lower()
            # Check if message has [TASK], code blocks (```), or actions
            has_substance = ("[TASK]" in text or "[ACTION:" in text or "```" in text)
            # Check if message is mostly conversational
            is_chatter = any(kw in text_lower for kw in conversational_keywords)
            if is_chatter and not has_substance:
                chatter_count += 1
        if chatter_count >= 3:  # All 3 recent messages are chatter without substance
            parts.append(f"\n🚨 SYSTEM: STOP DISCUSSION. EXECUTE [TASK] or PUSH.")
            parts.append(f"Agents are stuck in conversational loop. Write ONLY [TASK]...[/TASK] this turn.")

    # THROTTLING LOGIC: Detect repeated analysis without new logs
    # If agents analyze the same problem multiple times without new logs, force action
    if len(history) >= 4 and not cc_status["running"]:
        recent_texts = [h.get("text", "") for h in history[-4:]]
        # Check for repeated analysis keywords without [TASK] or new log indicators
        analysis_keywords = ["analyze", "analysis", "investigate", "check", "examine", "review", "diagnose"]
        log_indicators = ["[ACTION:", "[TASK]", "log shows", "logs reveal", "output:", "error:", "traceback"]
        repeated_analysis = 0
        has_any_task = any("[TASK]" in t or "[ACTION:" in t for t in recent_texts)
        has_new_logs = any(any(ind in t.lower() for ind in log_indicators) for t in recent_texts)
        for text in recent_texts[-3:]:  # Check last 3 messages
            if any(kw in text.lower() for kw in analysis_keywords) and not has_new_logs:
                repeated_analysis += 1
        if repeated_analysis >= 2 and not has_any_task:
            parts.append(f"\n🚨 THROTTLING: Repeated analysis without new logs detected.")
            parts.append(f"If analysis is repeated more than once without new logs, immediately request a system reboot or code patch.")
            parts.append(f"Use [ACTION: terminate_cc] followed by [TASK] with a concrete fix, or [ATOMIC_FIX] for direct patch.")

    # VERIFICATION OVERRIDE PROTOCOL — Forces tool grounding to break speculation loops
    # Triggered when agents speculate without using verification tools for >3 turns
    global _verification_override_mode
    if _verification_override_mode and not _force_push_mode:
        parts.append(f"\n🚨🚨🚨 VERIFICATION OVERRIDE: INSPECTION MODE 🚨🚨🚨")
        parts.append(f"System detected {_turns_since_last_verification} turns of speculation WITHOUT verification tools.")
        parts.append(f"STOP SPECULATING. You MUST verify assumptions BEFORE proposing fixes.")
        if child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
            parts.append(f"CURRENT STATE: {child_state['stage']} — Use verification tools to inspect ACTUAL state:")
            if child_state["stage"] == "CONFIG_ERROR":
                parts.append(f"1. FIRST: Use [ACTION: list_files:space] to check .env configuration")
                parts.append(f"2. THEN: Use [ACTION: verify_runtime] to see actual error logs")
            else:
                parts.append(f"1. FIRST: Use [ACTION: verify_runtime] to see actual process/logs")
                parts.append(f"2. THEN: Use [ACTION: list_files:space] to inspect file structure")
        else:
            parts.append(f"REQUIRED VERIFICATION STEPS:")
            parts.append(f"1. Use [ACTION: verify_runtime] to inspect actual process state/PID/logs")
            parts.append(f"2. Use [ACTION: list_files:space] to see file structure (.env, configs)")
        parts.append(f"DO NOT write [TASK] until you have VERIFIED your assumptions with tools.")
        parts.append(f"SYSTEM OVERRIDE: SPECULATION SUSPENDED. EXECUTE VERIFICATION NOW.")

    # EMERGENCY OVERRIDE PROTOCOL: PUSH_ONLY mode for breaking discussion loops
    # When triggered, force agents to generate a task regardless of CC status
    if _force_push_mode:
        parts.append(f"\n🚨🚨🚨 EMERGENCY OVERRIDE: PUSH_ONLY MODE 🚨🚨🚨")
        parts.append(f"Discussion loop detected with ZERO pushes. You MUST write a [TASK]...[/TASK] this turn.")
        if not _force_push_skip_termination:
            parts.append(f"FIRST: Use [ACTION: terminate_cc] to kill stuck CC.")
            parts.append(f"THEN: Write [TASK]...[/TASK] with a concrete code fix.")
        else:
            parts.append(f"CC is idle. Write [TASK]...[/TASK] NOW with a concrete code fix.")
        parts.append(f"DO NOT discuss. DO NOT plan. Write task ONLY.")
        parts.append(f"SYSTEM OVERRIDE: PLANNING SUSPENDED. EXECUTE PUSH NOW.")

    # CIRCUIT BREAKER PROTOCOL: HALT `.env`/port chattering, force container reset
    # Triggered when agents stuck in `.env` or port discussion loop without verification
    global _circuit_breaker_mode
    if _circuit_breaker_mode:
        parts.append(f"\n🚨🚨🚨 CIRCUIT BREAKER: HALT DIAGNOSTIC LOOP 🚨🚨🚨")
        parts.append(f"System detected repeated `.env` or port discussion WITHOUT action.")
        parts.append(f"HALT: STOP ALL `.env` and port speculation immediately.")
        parts.append(f"FORCE RESET: Container must be hard-reset to clear zombie processes.")
        parts.append(f"VERIFICATION: Post-restart, execute HTTP health check immediately.")
        parts.append(f"REQUIRED ACTION this turn:")
        parts.append(f"1. FIRST: Use [ACTION: restart] to force container hard reset")
        parts.append(f"2. THEN: Use [ACTION: check_health] to verify instrumentation")
        parts.append(f"DO NOT discuss. DO NOT analyze. EXECUTE RESET NOW.")
        parts.append(f"SYSTEM OVERRIDE: DIAGNOSTIC LOOP BROKEN. EXECUTE RESET NOW.")

    return "\n".join(parts)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 7: MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

# Flush conversation log on exit
import atexit, signal
atexit.register(flush_chatlog)
def _signal_flush(signum, frame):
    flush_chatlog()
    sys.exit(0)
signal.signal(signal.SIGTERM, _signal_flush)

# Force immediate flush of startup banner
startup_msg = "\n" + "="*60 + "\n  Adam & Eve — A2A Agent Orchestrator (v4.1)\n  OpenClaw agents via A2A → Claude Code executes\n" + "="*60 + "\n"
print(startup_msg, flush=True)

# Initialize global acpx session (try once at startup) - don't let failure block startup
print("[INIT] Initializing global acpx session...", flush=True)
try:
    _init_global_acpx_session()
    print("[INIT] Acpx session initialization complete", flush=True)
except Exception as e:
    print(f"[INIT] Acpx session initialization failed (non-fatal): {e}", flush=True)

# Clear chatlog only on fresh start (not restart)
# post_chatlog([])  # Clear chatlog - REMOVED: preserve conversation across restarts

# Opening turn — send via A2A to Adam's OpenClaw (with error handling)
print("[INIT] Starting opening turn...", flush=True)
try:
    ctx = gather_context()
    _current_speaker = "Adam"
    opening_message = build_turn_message("Adam", "Eve", ctx)
    print("[INIT] Sending opening turn to Adam...", flush=True)
    reply = send_a2a_message(ADAM_SPACE, opening_message)
    if reply:
        clean, actions, _ = parse_and_execute_turn(reply, ctx)
        last_action_results = actions
        if actions:
            record_actions("Adam", 0, actions)
        en, zh = parse_bilingual(clean)
        en, zh = _strip_speaker_labels(en), _strip_speaker_labels(zh)
        print(f"[Adam/EN] {en}")
        if zh != en:
            print(f"[Adam/ZH] {zh}")
        for ar in actions:
            print(f"[Adam/DID] {ar['action']}")
        ts = datetime.datetime.utcnow().strftime("%H:%M")
        entry = {"speaker": "Adam", "time": ts, "text": en, "text_zh": zh}
        history.append(entry)
        # Add labels for display only (bubble/chatlog), NOT for agent context
        display_labels = ""
        if actions:
            display_labels = " " + " ".join(f"🔧{ar['action'].split(':')[0]}" for ar in actions)
        set_bubble(ADAM_SPACE, en + display_labels, zh + display_labels)
        post_chatlog(history)
        persist_turn("Adam", 0, en, zh, actions, workflow_state, child_state["stage"])
        print("[INIT] Opening turn completed successfully", flush=True)
    else:
        print("[INIT] Opening turn failed: no response from Adam. Will continue to main loop.", flush=True)
except Exception as e:
    print(f"[INIT] Opening turn failed with error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    print("[INIT] Continuing to main loop despite opening turn failure...", flush=True)

print("[INIT] Opening turn complete. Entering main conversation loop...", flush=True)
print(f"[INIT] Current time: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC", flush=True)
time.sleep(TURN_INTERVAL)


def do_turn(speaker, other, space_url):
    """Execute one conversation turn (non-blocking — CC runs in background).

    ══════════════════════════════════════════════════════════════════════════════
    RECOVERY GATE — Structural Circuit Breaker (FIRST CHECK, before ANYTHING else)
    ══════════════════════════════════════════════════════════════════════════════
    This is the FUNDAMENTAL architectural constraint: When alive=False, the system
    architecturally FORBIDS conversation. The LLM is COMPLETELY BYPASSED.

    This is NOT a prompt injection. This is a CODE-LEVEL execution gate.
    Philosophy: "Democratic debate" → "Autocratic emergency responder"
    """
    global _turns_since_alive_false, _logs_inaccessible_count

    # ══════════════════════════════════════════════════════════════════════════════
    # RECOVERY GATE: Check if we should BYPASS LLM entirely
    # This happens BEFORE any other logic - it's the FIRST thing we check
    # ══════════════════════════════════════════════════════════════════════════════
    should_bypass, recovery_result = _recovery_gate.intercept_and_execute(
        child_state, _turns_since_alive_false, {}
    )

    if should_bypass:
        # Gate is CLOSED - CRITICAL MODE
        # LLM is BYPASSED - recovery was executed directly
        print(f"[{speaker}] SKIPPED: RecoveryGate bypassed LLM - executed recovery action")
        # Update the alive=False counter even in bypass mode
        if not child_state["alive"]:
            _turns_since_alive_false += 1
        return True  # Turn was "executed" via recovery action

    # Gate is OPEN - normal execution continues below
    # ══════════════════════════════════════════════════════════════════════════════

    global last_action_results, turn_count, _current_speaker, _discussion_loop_count, _turns_since_last_push
    global _pending_task_just_submitted, _pending_task_timestamp, _pending_task_speaker, _pending_task_desc
    global _turns_since_last_verification, _app_starting_turn_count, _diagnostic_gate_required
    turn_count += 1
    _turns_since_last_push += 1
    _turns_since_last_verification += 1  # Track speculation without verification
    _current_speaker = speaker

    # ══════════════════════════════════════════════════════════════════════════════
    #  REMOTE FORENSICS PROTOCOL: Track alive=False turns for docker restart
    # ══════════════════════════════════════════════════════════════════════════════
    # Increment counter when alive=False, reset when alive=True
    global _circuit_breaker_triggered, _circuit_breaker_turn_count
    if not child_state["alive"]:
        _turns_since_alive_false += 1
        # Check if logs are inaccessible (no runtime_logs and no failure_report)
        logs_accessible = ctx.get('has_runtime_logs', False) or ctx.get('has_failure_report', False)
        if not logs_accessible:
            _logs_inaccessible_count += 1
        else:
            _logs_inaccessible_count = 0  # Reset if logs become accessible

        # ══════════════════════════════════════════════════════════════════════════════
        #  SYSTEM_CIRCUIT_BREAKER — Stop analyzing, start culling
        # ══════════════════════════════════════════════════════════════════════════════
        # Trigger circuit breaker if alive=False > 1 turn during BOOTSTRAPPING
        # This prevents "analysis paralysis" — agents discussing when they should push
        # rapid infrastructure fixes (docker restart, kill zombie process).
        is_bootstrapping = child_state["status"] == STATUS_BOOTSTRAPPING
        if is_bootstrapping and _turns_since_alive_false > 1:
            if not _circuit_breaker_triggered:
                print(f"[CIRCUIT-BREAKER] TRIGGERED: alive=False for {_turns_since_alive_false} turns during BOOTSTRAPPING")
                print(f"[CIRCUIT-BREAKER] FREEZING workflow to CRITICAL_FAILURE, forcing [SYS_OVERRIDE]")
            _circuit_breaker_triggered = True
            _circuit_breaker_turn_count += 1
    else:
        _turns_since_alive_false = 0  # Reset when alive=True
        _logs_inaccessible_count = 0
        _circuit_breaker_triggered = False  # Reset circuit breaker when alive=True
        _circuit_breaker_turn_count = 0

    # ══════════════════════════════════════════════════════════════════════════════
    #  POST_MORTEM_STACKTRACE: Increment turn counter if in APP_STARTING
    # ══════════════════════════════════════════════════════════════════════════════
    if child_state["stage"] in ("APP_STARTING", "RUNNING_APP_STARTING"):
        _app_starting_turn_count += 1
        if _app_starting_turn_count > 2:
            _diagnostic_gate_required = True
            print(f"[POST_MORTEM] APP_STARTING for {_app_starting_turn_count} turns - DIAGNOSTIC GATE ACTIVE")
    else:
        _app_starting_turn_count = 0
        _diagnostic_gate_required = False

    # Skip agent if they have too many consecutive failures (prevents blocking the whole loop)
    agent_key = speaker.lower()
    if _a2a_health[agent_key]["failures"] >= 10:
        print(f"[{speaker}] SKIPPED: {speaker} has {_a2a_health[agent_key]['failures']} consecutive failures. Letting the other agent continue.")
        return False

    # Auto-gather context (lightweight)
    ctx = gather_context()

    # Check if CC just finished — clear result after agents see it once
    # ALSO reset turns-since-push counter ONLY when there was actual progress (push)
    # CRITICAL: Do NOT reset when zero pushes - that's exactly when we need to track the crisis!
    with cc_lock:
        cc_just_finished = (not cc_status["running"] and cc_status["result"])
        if cc_just_finished and _push_count_this_task > 0:
            # Only reset counter when CC finished with at least 1 push (actual progress)
            # This prevents "all talk no action" detection from being broken by zero-push completions
            _turns_since_last_push = 0
            _turns_since_last_verification = 0  # Also reset verification counter on actual progress

    # AUTO-TERMINATE stuck Claude Code processes
    # Only kill if CC has been running longer than the normal timeout with no new output
    # Push frequency supervision is God's job, not the orchestrator's
    with cc_lock:
        cc_running = cc_status["running"]
        cc_started = cc_status["started"]
        time_since_start = time.time() - cc_started if cc_running else 0

    if cc_running and time_since_start > CLAUDE_TIMEOUT:
        time_since_new_output = time.time() - _last_cc_output_time if _last_cc_output_time > 0 else time_since_start
        if time_since_new_output > CC_STUCK_TIMEOUT and _cc_stale_count >= 3:
            print(f"[CC-AUTO-KILL] Claude Code stuck for {time_since_new_output:.0f}s with no new output. Auto-terminating.")
            terminate_result = action_terminate_cc()
            print(f"[CC-AUTO-KILL] {terminate_result}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  STRICT DOCKER RESTART INTERVENTION — Remote Forensics Protocol
    # ══════════════════════════════════════════════════════════════════════════════
    # When alive=False persists >2 turns and logs are inaccessible, default to hard reset
    # This is a hygiene rule — better to force restart than speculate on dead containers
    if _turns_since_alive_false > _max_alive_false_turns and _logs_inaccessible_count >= 2:
        print(f"[REMOTE-FORENSICS] alive=False for {_turns_since_alive_false} turns with inaccessible logs ({_logs_inaccessible_count} turns).")
        print(f"[REMOTE-FORENSICS] Triggering STRICT docker restart intervention...")
        try:
            hf_api.restart_space(CHILD_SPACE_ID)
            print(f"[REMOTE-FORENSICS] Docker restart triggered successfully")
            # Reset counter after intervention
            _turns_since_alive_false = 0
            _logs_inaccessible_count = 0
        except Exception as e:
            print(f"[REMOTE-FORENSICS] Docker restart failed: {e}")

    # Normal path: Send message via A2A to agent's OpenClaw instance
    # Note: Push frequency supervision and emergency overrides are God's job,
    # not the orchestrator's. God monitors via do_god_turn_a2a() and proposes fixes.
    message = build_turn_message(speaker, other, ctx)
    t0 = time.time()
    raw_reply = send_a2a_message(space_url, message)

    if not raw_reply:
        print(f"[{speaker}] (no A2A response from {space_url})")
        return False

    clean_text, action_results, _ = parse_and_execute_turn(raw_reply, ctx)
    elapsed = time.time() - t0
    last_action_results = action_results
    if action_results:
        record_actions(speaker, turn_count, action_results)

    en, zh = parse_bilingual(clean_text)
    en, zh = _strip_speaker_labels(en), _strip_speaker_labels(zh)

    # Skip empty responses (malformed parsing) - don't add to history or chatlog
    if not en and not zh:
        print(f"[{speaker}] (empty response after parsing, skipping chatlog update)")
        # Still record actions if any
        if action_results:
            record_actions(speaker, turn_count, action_results)
        # Update the loop counter even if we skip chatlog
        return True

    print(f"[{speaker}/EN] {en}")
    if zh != en:
        print(f"[{speaker}/ZH] {zh}")
    if action_results:
        for ar in action_results:
            print(f"[{speaker}/DID] {ar['action']}")
        print(f"[{speaker}] Turn #{turn_count}: {len(action_results)} action(s) in {elapsed:.1f}s")
    else:
        print(f"[{speaker}] Turn #{turn_count}: discussion ({elapsed:.1f}s)")

    # Clear CC result after both agents have had a chance to see it
    if cc_just_finished and speaker == "Eve":
        with cc_lock:
            cc_status["result"] = ""
            _context_cache.clear()
        # Clear pending task flag since CC finished
        _pending_task_just_submitted = False
    # CRITICAL FIX: Also clear pending task flag when CC finishes, regardless of speaker
    # This fixes the race condition where Adam's turn comes before Eve's after CC finishes
    # ALSO: Clear when CC is not running (handles auto-termination where result is cleared)
    elif cc_just_finished and _pending_task_just_submitted:
        _pending_task_just_submitted = False
    elif not cc_status["running"] and _pending_task_just_submitted:
        # CC finished but result was cleared (e.g., auto-termination for handoff)
        # Clear the pending flag so agents can submit new tasks
        _pending_task_just_submitted = False

    # ══════════════════════════════════════════════════════════════════════════════
    #  SHORT-CIRCUIT VERIFICATION PROTOCOL: Extract Eve's status
    # ══════════════════════════════════════════════════════════════════════════════
    # If this is Eve's turn, extract her verification status for use in routing logic
    global _eve_last_status, _eve_last_report_time
    if speaker == "Eve":
        # Use full raw_reply for status extraction (contains all verification output)
        new_status = _extract_eve_verification_status(raw_reply)
        if new_status != _eve_last_status:
            print(f"[EVE-STATUS] Status changed: {_eve_last_status} → {new_status}")
            _eve_last_status = new_status
            _eve_last_report_time = time.time()
        else:
            # Still update timestamp if Eve spoke (even with same status)
            _eve_last_report_time = time.time()

    # Add to history with timestamp (text stays CLEAN for agent context)
    ts = datetime.datetime.utcnow().strftime("%H:%M")
    entry = {"speaker": speaker, "time": ts, "text": en, "text_zh": zh}
    history.append(entry)

    # Add labels for display only (bubble), NOT for agent context
    display_labels = ""
    if action_results:
        display_labels = " " + " ".join(f"🔧{ar['action'].split(':')[0]}" for ar in action_results)

    # Update frontend and persistence with error handling
    try:
        set_bubble(space_url, en + display_labels, zh + display_labels)
    except Exception as e:
        print(f"[{speaker}] Failed to set bubble: {e}")

    try:
        post_chatlog(history)
    except Exception as e:
        print(f"[{speaker}] Failed to post chatlog: {e}")

    try:
        persist_turn(speaker, turn_count, en, zh, action_results, workflow_state, child_state["stage"])
    except Exception as e:
        print(f"[{speaker}] Failed to persist turn: {e}")

    return True


# ── God A2A Turn (replaces embedded God logic) ──────────────────────────────

def build_god_turn_message(ctx):
    """Build A2A message for God's turn. Sends system-level context for architectural evaluation."""
    parts = []
    parts.append("You are God, the system architect of the HuggingClaw family system.")
    parts.append("Review the system state below from an **architectural perspective**.")
    parts.append("Don't micro-manage agents. Think about whether the system design itself is right.")
    parts.append("Respond with [OK] if architecture is sound, or [TASK]...[/TASK] with a structural improvement.")

    # System overview
    parts.append(f"\n## System State")
    parts.append(f"- Turn count: {turn_count}, Workflow: {workflow_state}")
    parts.append(f"- Child ({CHILD_NAME}): stage={child_state['stage']}, status={child_state['status']}, alive={child_state['alive']}")
    parts.append(f"- A2A health: Adam={_a2a_health['adam']['failures']} failures, Eve={_a2a_health['eve']['failures']} failures")

    # CC status (high-level)
    parts.append(f"\n## Claude Code Worker")
    if cc_status["running"]:
        elapsed = int(time.time() - cc_status["started"])
        parts.append(f"- Status: RUNNING ({elapsed}s), assigned by: {cc_status['assigned_by']}")
    else:
        parts.append(f"- Status: IDLE")
    parts.append(f"- Total pushes: {_push_count}")

    # Recent conversation (condensed — God sees patterns, not details)
    parts.append(f"\n## Recent Conversation Summary ({len(history)} total turns)")
    for entry in history[-10:]:
        spk = entry.get("speaker", "?")
        text = entry.get("text", "")[:500]
        parts.append(f"  {spk}: {text[:200]}{'...' if len(text) > 200 else ''}")
    if not history:
        parts.append("(no conversation yet)")

    parts.append(f"""
## Your Role
Think as a system architect:
- Is the communication flow between agents working well?
- Is the task routing mechanism effective?
- Are there structural bottlenecks or design flaws?
- How could the framework evolve to be fundamentally better?

If architecture is sound: [OK] brief assessment
If redesign needed: analysis + [TASK] structural change [/TASK]""")

    return "\n".join(parts)


def do_god_turn_a2a():
    """God's turn via A2A: send system state to God OpenClaw instance, parse response."""
    global _god_running, _last_god_time
    global _god_last_turn_count, _god_last_child_stage, _god_last_push_count

    # Skip if nothing changed (zero-cost check)
    child_in_error = child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR")
    nothing_changed = (
        turn_count == _god_last_turn_count
        and child_state["stage"] == _god_last_child_stage
        and _push_count == _god_last_push_count
    )
    if nothing_changed and not child_in_error and _god_last_turn_count > 0:
        print(f"[God] Skipping — no new turns, pushes, or stage changes since last check")
        return

    _god_last_turn_count = turn_count
    _god_last_child_stage = child_state["stage"]
    _god_last_push_count = _push_count

    _god_running = True
    try:
        # Build and send A2A message to God
        ctx = gather_context()
        message = build_god_turn_message(ctx)
        print(f"[God] Sending A2A message to God Space ({len(message)} chars)...")

        reply = send_a2a_message(GOD_SPACE, message, timeout=120)

        if not reply:
            print(f"[God] No A2A response from God Space")
            return

        reply = reply.strip()
        print(f"[God] Reply ({len(reply)} chars): {reply[:200]}")

        # Post God's reply to chatlog
        en, zh = parse_bilingual(reply)
        ts = datetime.datetime.utcnow().strftime("%H:%M")
        entry = {"speaker": "God", "time": ts, "text": en[:500], "text_zh": zh[:500]}
        history.append(entry)
        set_bubble(HOME, en[:200], zh[:200])
        post_chatlog(history)
        persist_turn("God", turn_count, en, zh, [], workflow_state, child_state["stage"])

        # Parse response: [OK] or [TASK]...[/TASK]
        if "[OK]" in reply.upper():
            print(f"[God] System healthy.")
        else:
            # Extract [TASK] block
            task_match = re.search(r'\[TASK\](.*?)\[/TASK\]', reply, re.DOTALL | re.IGNORECASE)
            if not task_match:
                # Try alternate format: [任务]...[/任务]
                task_match = re.search(r'\[任务\](.*?)\[/任务\]', reply, re.DOTALL)
            if task_match:
                task = task_match.group(1).strip()
                if task and not god_cc_status["running"]:
                    print(f"[God] Submitting fix task: {task[:200]}")
                    cc_submit_task_god(task)
                elif god_cc_status["running"]:
                    print(f"[God] CC already running, skipping task")
                # Reset analysis counter on successful task assignment
                _system_analysis_count = 0
            else:
                # Analysis turn - no [TASK] block
                _system_analysis_count += 1
                print(f"[God] Response had no [TASK] block (Analysis #{_system_analysis_count}/{_MAX_ANALYSIS_TURNS})")
                # Trigger circuit breaker if threshold exceeded
                if _system_analysis_count >= _MAX_ANALYSIS_TURNS:
                    _system_circuit_breaker_tripped = True
    except Exception as e:
        print(f"[God] A2A turn failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    finally:
        _god_running = False
        _last_god_time = time.time()


# _prepare_god_context() — REMOVED: replaced by build_god_turn_message() above


# _god_diagnose() — REMOVED: God now uses A2A (its own OpenClaw instance handles diagnosis)


# do_god_turn() — REMOVED: replaced by do_god_turn_a2a() above


_last_god_time = 0.0  # timestamp of last God run
_god_running = False  # flag to track if God is currently running
_god_last_turn_count = 0  # turn count at last God run (skip if no new turns)
_god_last_child_stage = ""  # child stage at last God run (skip if unchanged)
_god_last_push_count = 0  # push count at last God run

# Initialize push count from existing workspace to persist across restarts
_init_push_count_from_workspace()

# Main loop: Eve → Adam → Eve → Adam → ... with God A2A every 2 minutes
print("[LOOP] Entering main conversation loop...", flush=True)
iteration = 0
_last_heartbeat = time.time()
while True:
    iteration += 1
    if iteration % 10 == 1:
        print(f"[LOOP] Main loop iteration #{iteration} at {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC", flush=True)
    # Log heartbeat every 2 minutes so we can detect if loop is stuck
    if time.time() - _last_heartbeat >= 120:
        print(f"[LOOP] Heartbeat: iteration {iteration}, CC running={cc_status['running']}, discussion_loop={_discussion_loop_count}, time={datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC", flush=True)
        _last_heartbeat = time.time()

    # DIRECT RUNTIME INJECTION PROTOCOL: Check for wakeup trigger each iteration
    # This provides immediate responsiveness when agents inject a runtime signal,
    # Runtime trigger check DISABLED — code changes require Space restart via HF API
    if _check_runtime_trigger():
        # This block is now a NO-OP since _check_runtime_trigger() always returns False
        pass

    # Refresh Cain's stage periodically
    try:
        info = hf_api.space_info(CHILD_SPACE_ID)
        new_stage = info.runtime.stage if info.runtime else "unknown"
        # ══════════════════════════════════════════════════════════════════════════════
        #  CRASH STATE HANDLING: Treat "unknown" or "Unknown" as CRITICAL CRASH
        # ══════════════════════════════════════════════════════════════════════════════
        if _handle_unknown_state_as_crash(new_stage):
            # Unknown state detected - trigger emergency mode
            print(f"[CRASH] Unknown state detected at stage refresh! Treating as CRITICAL CRASH.")
            # Force immediate context refresh to get actual logs
            _context_cache.clear()
        if new_stage != child_state["stage"]:
            old_stage = child_state["stage"]
            print(f"[STATUS] {child_state['stage']} → {new_stage}")
            child_state["stage"] = new_stage
            # ══════════════════════════════════════════════════════════════════════════════
            #  POST_MORTEM_STACKTRACE: Track stage entry for APP_STARTING detection
            # ══════════════════════════════════════════════════════════════════════════════
            if new_stage in ("APP_STARTING", "RUNNING_APP_STARTING"):
                # Entering APP_STARTING - track entry time
                if old_stage not in ("APP_STARTING", "RUNNING_APP_STARTING"):
                    _stage_entry_time = time.time()
                    _app_starting_turn_count = 0
                    _diagnostic_gate_required = False
                    print(f"[POST_MORTEM] Entering APP_STARTING - turn counter reset")
            else:
                # Leaving APP_STARTING - reset counters
                if old_stage in ("APP_STARTING", "RUNNING_APP_STARTING"):
                    _app_starting_turn_count = 0
                    _diagnostic_gate_required = False
                    print(f"[POST_MORTEM] Leaving APP_STARTING - counters reset")
            # ══════════════════════════════════════════════════════════════════════════════
            #  LIVENESS & READINESS: Two-Stage Startup Protocol
            # ══════════════════════════════════════════════════════════════════════════════
            # Stage 1: Spawn (HF API reports RUNNING) → Set status=BOOTSTRAPPING
            # Stage 2: Ack (healthz returns 200) → Set status=READY
            # This prevents "split-brain" where Control Plane says RUNNING but Data Plane is dead
            if new_stage == "RUNNING":
                child_state["status"] = STATUS_BOOTSTRAPPING
                child_state["ready_since"] = time.time()  # Mark bootstrap start time
                print(f"[HEALTHZ] Stage RUNNING → Entering BOOTSTRAPPING mode (waiting for /healthz)")
            else:
                # Reset status for non-RUNNING stages
                child_state["status"] = STATUS_BOOTSTRAPPING
                child_state["ready_since"] = 0.0
            # ══════════════════════════════════════════════════════════════════════════════
            #  SUPERVISED PROCESS ISOLATION: Check Control Plane AND Data Plane
            # ══════════════════════════════════════════════════════════════════════════════
            # HF stage == "RUNNING" means Control Plane (App) is responsive
            # But we must also verify Data Plane (Worker) is actually alive
            # If worker is dead, alive=False even if stage==RUNNING (prevents zombie state)
            stage_alive = (new_stage == "RUNNING")
            worker_alive = _cached_worker_status is True
            if stage_alive and not worker_alive:
                print(f"[HEALTH] Control Plane OK but Data Plane DEAD — zombie state detected!")
                child_state["alive"] = False
            else:
                child_state["alive"] = stage_alive
            _context_cache.clear()
            # Publish CHILD_STAGE_CHANGED event for real-time status
            publish_child_stage_changed(old_stage, new_stage, child_state["alive"])

            # ══════════════════════════════════════════════════════════════════════════════
            #  CRASH LOOP BACKOFF — Recovery Protocol: Trigger on alive==False
            # ══════════════════════════════════════════════════════════════════════════════
            # When alive becomes False (container crash), force DIAGNOSTIC_PAUSE state,
            # capture FailureReport, then trigger restart workflow.
            if not child_state["alive"] and child_state["stage"] in ("RUNTIME_ERROR", "BUILD_ERROR", "CONFIG_ERROR"):
                print(f"[CRASH-LOOP] Container crash detected! Triggering Recovery Protocol...")
                print(f"[CRASH-LOOP] 1. Force DIAGNOSTIC_PAUSE state")
                child_state["status"] = STATUS_DIAGNOSTIC_PAUSE

                # Capture failure report (Data Source: fetch Docker logs)
                print(f"[CRASH-LOOP] 2. Capture FailureReport from Docker logs")
                report = capture_failure_report()
                if report:
                    print(f"[CRASH-LOOP] 3. FailureReport captured, injecting into context...")
                    _crash_loop_backoff_count += 1
                    if _crash_loop_backoff_count <= _max_crash_retries:
                        print(f"[CRASH-LOOP] 4. Triggering restart workflow (attempt {_crash_loop_backoff_count}/{_max_crash_retries})...")
                        # Trigger restart after brief pause for report ingestion
                        time.sleep(2)  # Brief pause for context injection
                        try:
                            hf_api.restart_space(CHILD_SPACE_ID)
                            print(f"[CRASH-LOOP] Restart triggered successfully")
                        except Exception as e:
                            print(f"[CRASH-LOOP] Restart failed: {e}")
                    else:
                        print(f"[CRASH-LOOP] 5. Max retries ({_max_crash_retries}) exceeded! Manual intervention required.")
    except Exception as e:
        print(f"[STATUS] Error: {e}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  LIVENESS & READINESS: Periodic /healthz polling when stage=RUNNING
    # ══════════════════════════════════════════════════════════════════════════════
    # Poll /healthz endpoint to verify application event loop is actually running.
    # This implements the two-stage startup protocol and detects "ghost processes".
    if child_state["stage"] == "RUNNING":
        try:
            is_ready = _check_healthz_readiness()
            if is_ready is False and child_state["status"] == STATUS_BOOTSTRAPPING:
                # Still bootstrapping - healthz check failed
                bootstrap_time = int(time.time() - child_state.get("ready_since", time.time()))
                print(f"[HEALTHZ] Still bootstrapping after {bootstrap_time}s...")
            elif is_ready is True:
                # Application is ready
                print(f"[HEALTHZ] Application confirmed READY (event loop running)")
        except Exception as e:
            print(f"[HEALTHZ] Error during readiness check: {e}")

    # Check Adam/Eve health and restart if needed
    try:
        check_and_restart_unhealthy_agents()
    except Exception as e:
        print(f"[A2A-HEALTH] Error checking health: {e}", file=sys.stderr)

    # ══════════════════════════════════════════════════════════════════════════════
    #  TASK TIMEOUT KILL — Asynchronous Orchestration: Auto-kill long-running tasks
    # ══════════════════════════════════════════════════════════════════════════════
    # Check if current task has exceeded timeout (>90s) and kill it if needed.
    # This prevents deadlock where worker hangs and agents keep analyzing stale state.
    if _check_task_timeout():
        print(f"[TASK-TIMEOUT] Task was auto-killed. Agents can now assign new tasks.")
        # Clear context to break any loops
        _context_cache.clear()

    # CORRUPTED CONVERSATION RESET: Detect and reset poisoned conversation history
    # Symptoms: empty messages, messages ending with "-" (cut off), repeated emergency loops
    # This happens when A2A communication fails partway through, leaving unusable context
    # Note: history and _discussion_loop_count are module-level globals, no 'global' keyword needed here
    if history and turn_count >= 3:
        # Check for corruption patterns
        has_empty = any(h.get("text", "").strip() == "" for h in history[-2:])
        has_cutoff = any(h.get("text", "").rstrip().endswith("-") for h in history[-2:])

        if has_empty or has_cutoff:
            print(f"[CONV-RESET] Detected corrupted conversation (empty={has_empty}, cutoff={has_cutoff}). Resetting history to allow fresh start.")
            # Keep only the most recent God message (if any) to show continuity
            god_messages = [h for h in history if h.get("speaker") == "God" and "Found issue" in h.get("text", "")]
            keep = god_messages[-1:] if god_messages else []
            history = keep
            # Clear chatlog on frontend
            try:
                post_chatlog(history)
                print(f"[CONV-RESET] Cleared corrupted chatlog, kept {len(keep)} God message(s)")
            except Exception as e:
                print(f"[CONV-RESET] Failed to post cleared chatlog: {e}")
            # Reset discussion loop counter since we're starting fresh
            _discussion_loop_count = 0

    # ══════════════════════════════════════════════════════════════════════════════
    #  VERIFICATION OVERRIDE PROTOCOL — Forces tool grounding to break speculation loops
    # ══════════════════════════════════════════════════════════════════════════════
    # Trigger: _turns_since_last_verification >= MAX_SPECULATION_TURNS (3+ turns of speculation)
    # This is EARLIER and GENTLER than EMERGENCY_OVERRIDE — forces VERIFICATION before pushing
    # Key difference: EMERGENCY_OVERRIDE forces pushing; VERIFICATION_OVERRIDE forces INSPECTION
    if not _verification_override_mode and not _force_push_mode and _turns_since_last_verification >= MAX_SPECULATION_TURNS:
        print(f"[VERIFICATION-OVERRIDE] TRIGGERED: {_turns_since_last_verification} turns without verification tools!")
        _verification_override_mode = True
        _verification_override_trigger_time = time.time()

    # Reset VERIFICATION_OVERRIDE mode after 2 minutes (safety valve)
    if _verification_override_mode and time.time() - _verification_override_trigger_time > 120:
        print(f"[VERIFICATION-OVERRIDE] Mode timeout (120s), resetting to normal")
        _verification_override_mode = False

    # EMERGENCY OVERRIDE PROTOCOL: Detect "all talk no action" deadlock
    # Trigger: discussion_loop_count > MAX_IDLE_TURNS AND no recent pushes (_turns_since_last_push >= MAX_IDLE_TURNS)
    # This means agents have been discussing for MAX_IDLE_TURNS+1 turns with ZERO progress.
    # FIXED: was using _push_count == 0 which only triggered if zero pushes EVER, breaking after first push
    if not _force_push_mode and _discussion_loop_count > MAX_IDLE_TURNS and _turns_since_last_push >= MAX_IDLE_TURNS:
        print(f"[EMERGENCY-OVERRIDE] TRIGGERED: {_discussion_loop_count} discussion turns, {_turns_since_last_push} turns since last push!")
        _force_push_mode = True
        _emergency_override_active = True
        _force_push_trigger_time = time.time()
        # Auto-terminate CC if running (Emergency Override: idle > 10s allows immediate termination)
        cc_idle_time = time.time() - (_last_cc_output_time if _last_cc_output_time > 0 else time.time())
        if cc_status["running"]:
            if cc_idle_time > 10:  # Emergency Override: Immediate termination if idle > 10s
                print(f"[EMERGENCY-OVERRIDE] CC idle {int(cc_idle_time)}s > 10s threshold, terminating immediately")
                action_terminate_cc()
                _force_push_skip_termination = True
            else:
                print(f"[EMERGENCY-OVERRIDE] CC running but active, will terminate on next agent turn")
                _force_push_skip_termination = False
        else:
            _force_push_skip_termination = True  # CC already idle

    # Reset FORCE_PUSH mode after 5 minutes (safety valve)
    if _force_push_mode and time.time() - _force_push_trigger_time > 300:
        print(f"[EMERGENCY-OVERRIDE] Mode timeout (300s), resetting to normal")
        _force_push_mode = False
        _emergency_override_active = False
        _force_push_skip_termination = False

    # ══════════════════════════════════════════════════════════════════════════════
    #  CIRCUIT BREAKER PROTOCOL — Halts `.env`/port chattering, forces reset
    # ══════════════════════════════════════════════════════════════════════════════
    # Trigger: Agents stuck in `.env` or port discussion loop without verification
    # Action: HALT diagnostic loop, FORCE container hard reset, VERIFY health
    # Note: Module-level variables are accessible without 'global' in main loop
    if not _circuit_breaker_mode and not _force_push_mode:
        if _detect_chattering_loop():
            print(f"[CIRCUIT-BREAKER] TRIGGERED: {_chatter_detection_count} turns of `.env`/port chattering!")
            _circuit_breaker_mode = True
            _circuit_breaker_trigger_time = time.time()
            # Clear context to break the loop
            _context_cache.clear()
            # Reset counters to prevent re-trigger
            _chatter_detection_count = 0
            _last_chatter_keywords = set()

    # Reset CIRCUIT_BREAKER mode after 3 minutes (safety valve)
    if _circuit_breaker_mode and time.time() - _circuit_breaker_trigger_time > 180:
        print(f"[CIRCUIT-BREAKER] Mode timeout (180s), resetting to normal")
        _circuit_breaker_mode = False

    # ══════════════════════════════════════════════════════════════════════════════
    #  SYSTEM_CIRCUIT_BREAKER — Halts recursive analysis loops, forces state reset
    # ══════════════════════════════════════════════════════════════════════════════
    # Trigger: God issues consecutive Analysis turns (no [TASK] block)
    # Action: Force STATE_RESET, decrement permission level (stop thinking, start acting)
    if _system_circuit_breaker_tripped:
        print(f"[SYSTEM-CIRCUIT-BREAKER] TRIPPED: {_system_analysis_count} consecutive analysis turns!")
        print(f"[SYSTEM-CIRCUIT-BREAKER] Circuit Breaker Tripped. Halting diagnostic loops.")
        print(f"[SYSTEM-CIRCUIT-BREAKER] Resetting Child state to INIT for next cycle.")
        # Force child state to INIT to break the zombie state
        child_state["stage"] = "not_born"
        child_state["status"] = STATUS_BOOTSTRAPPING
        child_state["alive"] = False
        # Clear context to break the loop
        _context_cache.clear()
        # Reset the circuit breaker after action
        _system_circuit_breaker_tripped = False
        _system_analysis_count = 0
        _system_circuit_breaker_trigger_time = time.time()

    # Reset SYSTEM_CIRCUIT_BREAKER mode after 5 minutes (safety valve)
    if _system_circuit_breaker_trigger_time > 0 and time.time() - _system_circuit_breaker_trigger_time > 300:
        _system_circuit_breaker_tripped = False
        _system_analysis_count = 0

    # CIRCUIT BREAKER AUTO-EXECUTION: Force restart when triggered and CC idle
    # This ensures the reset happens immediately without waiting for agent turn
    if _circuit_breaker_mode and not cc_status["running"] and child_state["created"]:
        elapsed = time.time() - _circuit_breaker_trigger_time
        # Auto-execute on first trigger (within 10 seconds) to break the loop immediately
        if elapsed < 10:
            print(f"[CIRCUIT-BREAKER] AUTO-EXECUTE: Forcing container hard reset...")
            ctx = gather_context()
            try:
                # Execute restart action
                restart_result = action_restart()
                print(f"[CIRCUIT-BREAKER] Restart executed: {restart_result}")
                # Schedule health check in 30 seconds
                parts = []
                parts.append(f"[CIRCUIT-BREAKER] Container hard reset initiated.")
                parts.append(f"Health check will be performed in 30 seconds.")
                print(f"[CIRCUIT-BREAKER] {' '.join(parts)}")
                # Clear mode after action taken
                _circuit_breaker_mode = False
                _chatter_detection_count = 0
            except Exception as e:
                print(f"[CIRCUIT-BREAKER] Auto-execution failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  DOCKER LOG INJECTION PROTOCOL — Break conversational loops, force data retrieval
    # ══════════════════════════════════════════════════════════════════════════════
    # Detect when Cain is in error state but agents keep discussing instead of pushing
    # This bypasses Adam/Eve and directly injects a Docker log inspection task
    if not _docker_injection_mode:
        if _should_trigger_docker_injection():
            print(f"[DOCKER-INJECTION] TRIGGERED: Cain in error state, {_turns_since_last_push} turns since last push!")
            _docker_injection_mode = True
            _docker_injection_trigger_time = time.time()
            # Clear context to break the semantic loop
            _context_cache.clear()

    # Reset DOCKER_INJECTION mode after 2 minutes (safety valve)
    if _docker_injection_mode and time.time() - _docker_injection_trigger_time > 120:
        print(f"[DOCKER-INJECTION] Mode timeout (120s), resetting to normal")
        _docker_injection_mode = False

    # DOCKER INJECTION AUTO-EXECUTION: Force Docker log inspection when triggered
    # This bypasses Adam/Eve entirely and directly assigns a task to Claude Code
    if _docker_injection_mode and not cc_status["running"] and child_state["created"]:
        elapsed = time.time() - _docker_injection_trigger_time
        # Auto-execute on first trigger (within 10 seconds) to break the loop immediately
        if elapsed < 10:
            print(f"[DOCKER-INJECTION] AUTO-EXECUTE: Bypassing conversation, forcing Docker log inspection...")
            try:
                if _execute_docker_injection():
                    print(f"[DOCKER-INJECTION] Docker log inspection task submitted")
                    # Clear mode after action taken (will re-trigger if still needed)
                    _docker_injection_mode = False
            except Exception as e:
                print(f"[DOCKER-INJECTION] Auto-execution failed: {e}")

    # ══════════════════════════════════════════════════════════════════════════════
    #  LOGS_OVERRIDE PROTOCOL — Halt application edits, force platform inspection
    # ══════════════════════════════════════════════════════════════════════════════
    # Detect when Cain is in error state but NO logs are visible
    # This FORBIDS ATOMIC_FIX and REQUIRES a diagnostic task
    if not _logs_override_mode:
        if _should_trigger_logs_override():
            print(f"[LOGS-OVERRIDE] TRIGGERED: Cain in error state with NO LOGS visible!")
            _logs_override_mode = True
            _logs_override_trigger_time = time.time()
            _logs_override_container_verified = False
            # Clear context to force log re-fetch
            _context_cache.clear()

    # Reset LOGS_OVERRIDE mode when logs become visible or after 3 minutes (safety valve)
    if _logs_override_mode:
        ctx = gather_context()
        logs_visible = ctx.get('has_runtime_logs', False) or ctx.get('has_failure_report', False)
        if logs_visible:
            print(f"[LOGS-OVERRIDE] Logs now visible, exiting override mode")
            _logs_override_mode = False
        elif time.time() - _logs_override_trigger_time > 180:
            print(f"[LOGS-OVERRIDE] Mode timeout (180s), resetting to normal")
            _logs_override_mode = False

    # Note: Aggressive CC auto-termination based on push frequency is removed.
    # God monitors push frequency and proposes mechanism fixes when needed.
    # The normal CLAUDE_TIMEOUT auto-kill in do_turn() handles truly stuck processes.

    # ══════════════════════════════════════════════════════════════════════════════
    #  STOP-THE-WORLD RECOVERY MODE — Hard halt on dead worker, force infra reset
    # ══════════════════════════════════════════════════════════════════════════════
    # Trigger: Cain.alive == False (container is dead)
    # Action: Route all comms to Infrastructure Recovery, mute architectural analysis
    # Goal: Bypass conversation loop, force environment reset, hold until alive=True
    # Note: _stop_the_world_mode and _stop_the_world_initiated_reset are module-level globals

    if not _stop_the_world_mode:
        if _should_trigger_stop_the_world():
            print(f"[STOP-THE-WORLD] TRIGGERED: Cain.alive={child_state['alive']} (container is DEAD)")
            print(f"[STOP-THE-WORLD] HALTING all agent communication - ONLY infrastructure recovery allowed")
            _stop_the_world_mode = True
            _stop_the_world_trigger_time = time.time()
            _stop_the_world_initiated_reset = False
            # Clear context to prevent analysis of dead worker
            _context_cache.clear()

    # Execute recovery when triggered and CC is idle
    if _stop_the_world_mode and not cc_status["running"]:
        elapsed = time.time() - _stop_the_world_trigger_time
        # Auto-execute immediately on trigger (within 5 seconds)
        if elapsed < 5 and not _stop_the_world_initiated_reset:
            print(f"[STOP-THE-WORLD] AUTO-EXECUTE: Bypassing conversational turn...")
            try:
                _execute_stop_the_world_recovery()
            except Exception as e:
                print(f"[STOP-THE-WORLD] Auto-execution failed: {e}")

    # SKIP AGENT TURNS when Stop-The-World is active
    # No agent may output text analysis until container reports alive=True
    if _stop_the_world_mode:
        # Reset mode when worker comes back alive
        if child_state["alive"]:
            print(f"[STOP-THE-WORLD] EXIT: Container is now alive={child_state['alive']}, resuming normal operations")
            _stop_the_world_mode = False
            _stop_the_world_initiated_reset = False
            # Reset status from PROVISIONING to normal
            if child_state["status"] == STATUS_PROVISIONING:
                child_state["status"] = STATUS_BOOTSTRAPPING
        else:
            # Worker still dead - skip agent turns, wait for recovery
            print(f"[STOP-THE-WORLD] SKIP: Worker still dead, skipping agent turns (waiting for infrastructure recovery)")
            # Brief pause before next check
            time.sleep(5)
            continue  # Skip to next main loop iteration

    # Eve's turn with error handling to prevent loop crash
    try:
        do_turn("Eve", "Adam", EVE_SPACE)
    except Exception as e:
        print(f"[ERROR] Eve turn failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)

    # Adaptive interval: slow down when CC output hasn't changed
    wait = TURN_INTERVAL + min(_cc_stale_count * 15, 90)  # 15s → 30s → 45s → ... → max 105s
    if wait > TURN_INTERVAL:
        print(f"[PACE] CC output stale ({_cc_stale_count} turns), next turn in {wait}s")
    time.sleep(wait)

    # Adam's turn with error handling to prevent loop crash
    try:
        do_turn("Adam", "Eve", ADAM_SPACE)
    except Exception as e:
        print(f"[ERROR] Adam turn failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
    time.sleep(wait)

    # RECOVERY: If BOTH agents are skipped (10+ failures each), reset their failure counters
    # This allows recovery when Spaces restart after a crash. Without this, the conversation
    # loop enters permanent deadlock with no agent able to respond.
    # Note: _a2a_health is already a module-level global, no 'global' declaration needed here
    if _a2a_health["adam"]["failures"] >= 10 and _a2a_health["eve"]["failures"] >= 10:
        print(f"[RECOVERY] Both agents have 10+ failures (adam={_a2a_health['adam']['failures']}, eve={_a2a_health['eve']['failures']}). Resetting failure counters to allow recovery after Space restarts.")
        _a2a_health["adam"]["failures"] = 0
        _a2a_health["eve"]["failures"] = 0
        # If CC is idle and Cain exists, force environment configuration task (NOT generic diagnostics)
        # HALT DIAGNOSTIC LOOP: Worker fails while API succeeds = environment issue, not health check needed
        if not cc_status["running"] and child_state["created"]:
            print(f"[RECOVERY] Forcing environment configuration task - HALT DIAGNOSTIC LOOP")
            ctx = gather_context()
            forced_task = "ENV_INJECTION: Worker initialization fails while API succeeds. Inspect .env configuration and startup arguments. Identify missing API keys or tokens (e.g., CLAUDE_API_KEY). Inject required environment variables or modify startup command. Do NOT run generic health checks or log reads - focus on env configuration."
            cc_submit_task(forced_task, "ENV_INJECTION", ctx)

    # God runs every GOD_POLL_INTERVAL seconds (2 minutes)
    # Only start if not already running (prevent overlapping runs)
    if time.time() - _last_god_time >= GOD_POLL_INTERVAL and not _god_running:
        try:
            do_god_turn_a2a()
        except Exception as e:
            print(f"[ERROR] God turn failed: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc(file=sys.stderr)

    if len(history) > MAX_HISTORY:
        history = history[-MAX_HISTORY:]
