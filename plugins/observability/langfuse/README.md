# Langfuse Observability Plugin

This plugin ships bundled with Hermes but is **opt-in** — it only loads when
you explicitly enable it. It answers operational questions such as:

- Which agent turn, gateway message, cron job, tool, or subagent ran?
- Which provider/model was used, how long did it take, and did it fail?
- What token usage and estimated cost were reported by Hermes?
- Which session/job IDs correlate a trace back to local logs?

## Enable

Pick one:

```bash
# Interactive: walks you through credentials + SDK install + enable
hermes tools  # → Langfuse Observability

# Manual
pip install langfuse
hermes plugins enable observability/langfuse
```

## Required credentials

Set these in `~/.hermes/.env` (or via `hermes tools`):

```bash
HERMES_LANGFUSE_PUBLIC_KEY=pk-lf-...
HERMES_LANGFUSE_SECRET_KEY=sk-lf-...
HERMES_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK or credentials the hooks no-op silently — the plugin fails
open.

By default, prompt/response/tool content is summarized with length and SHA-256
metadata instead of being sent raw. Set `HERMES_LANGFUSE_CAPTURE_CONTENT=true`
only in trusted environments where sending raw content to Langfuse is approved.

## Verify

```bash
hermes plugins list                 # observability/langfuse should show "enabled"
hermes chat -q "hello"              # then check Langfuse for a "Hermes turn" trace
```

## Optional tuning

```bash
HERMES_LANGFUSE_ENV=production       # environment tag
HERMES_LANGFUSE_RELEASE=v1.0.0       # release tag
HERMES_LANGFUSE_SAMPLE_RATE=0.5      # sample 50% of traces
HERMES_LANGFUSE_MAX_CHARS=12000      # max chars per field (default: 12000)
HERMES_LANGFUSE_FLUSH_TIMEOUT_SECONDS=2
HERMES_LANGFUSE_SERVICE=hermes-agent
HERMES_LANGFUSE_CAPTURE_CONTENT=false
HERMES_LANGFUSE_DEBUG=true           # verbose plugin logging
```

## What is traced

The plugin consumes Hermes observer hooks and emits stable Langfuse traces and
spans for:

- agent turns and provider API attempts (`pre_api_request`, `post_api_request`,
  `api_request_error`)
- tool calls (`pre_tool_call`, `post_tool_call`)
- gateway agent runs (`gateway_agent_run_start`, `gateway_agent_run_finish`)
- cron jobs (`cron_job_start`, `cron_job_finish`)
- delegated subagents (`subagent_start`, `subagent_stop`)

Trace metadata includes service, component, environment, session/task IDs,
job IDs/names, provider/model/API mode, status/error class, duration, token
usage, and estimated cost where Hermes already has that data.

## OpenClaw/Hermes VM rollout

1. Install the optional SDK in the Hermes virtualenv: `pip install langfuse`.
2. Put the required keys and URL in the runtime environment or `~/.hermes/.env`:
   `HERMES_LANGFUSE_PUBLIC_KEY`, `HERMES_LANGFUSE_SECRET_KEY`,
   `HERMES_LANGFUSE_BASE_URL=https://langfuse.mmishchenko.dev`,
   `HERMES_LANGFUSE_ENV=default`, and
   `HERMES_LANGFUSE_SERVICE=openclaw-hermes`.
3. Enable the bundled plugin: `hermes plugins enable observability/langfuse`.
4. Restart the Hermes/OpenClaw gateway or supervisor-managed service so the
   plugin and environment are loaded.
5. Run one harmless gateway message or cron dry run, then query Langfuse for
   `service=openclaw-hermes` and components such as `agent_turn`, `llm_request`,
   `tool_call`, `gateway_message`, and `cron_job`.

Do not expose private Langfuse ports publicly; use the existing public HTTPS
endpoint or local loopback from the VM.

## Disable

```bash
hermes plugins disable observability/langfuse
```
