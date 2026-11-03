# Agents

For detailed information about Meta Agents Research Environments Agents API, please refer to the comprehensive documentation:

[docs/api_reference/agents.rst](../../docs/api_reference/agents.rst)


### Request-boundary checkpoints

`BaseAgent.pending_step` exposes the prompt and format retry count while an LLM request is pending. An external runtime may capture this dictionary together with the agent logs, iteration counters, task, attachments, prompts, and custom state **inside the engine call, before inference starts**. It must coordinate this snapshot with any external tool/environment state and preserve the native message/attachment types.

Restore the captured agent state and tool registry, then call `execute_agent_loop()`. A non-null pending step skips its already-consumed pre-step notifications and reuses the saved prompt/retry count. It does not execute any earlier action again. Successful generation clears the pending step; recoverable request errors clear it before rebuilding native history. Capturing state after an interrupted loop has unwound is insufficient because its iteration-finalization code may already have run.

A successful terminal reply or wait requires an observation following its tool-call log. Tool failures remain native error-recovery steps and do not end the turn.
