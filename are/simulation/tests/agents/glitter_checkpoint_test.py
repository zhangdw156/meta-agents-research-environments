"""Request-boundary recovery and successful terminal-action regressions."""

import copy
from types import SimpleNamespace

import pytest

from are.simulation.agents.are_simulation_agent_config import (
    ARESimulationReactBaseAgentConfig,
)
from are.simulation.agents.default_agent.agent_factory import (
    are_simulation_react_json_agent,
)
from are.simulation.tools import Tool


class Interrupted(BaseException):
    pass


class Reply(Tool):
    name = "AgentUserInterface__send_message_to_user"
    description = "Reply to the user"
    inputs = {"content": {"type": "string", "description": "Message"}}

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, content):
        self.calls.append(content)
        if content == "fail":
            raise ValueError("Delivery failed")
        return "delivered"


def make(engine, tool):
    agent = are_simulation_react_json_agent(engine, ARESimulationReactBaseAgentConfig(use_custom_logger=False))
    agent.tools = {tool.name: tool}
    agent.notification_system = SimpleNamespace(message_queue=SimpleNamespace(
        has_environment_stop_message=lambda: False, get_by_timestamp=lambda **kwargs: []))
    return agent


def action(content):
    return 'Thought: reply\nAction: {"action":"AgentUserInterface__send_message_to_user","action_input":{"content":"' + content + '"}}'


def test_failed_terminal_tool_does_not_end_turn():
    replies = iter([action("fail"), action("success")])
    tool = Reply()
    agent = make(lambda *args, **kwargs: next(replies), tool)
    agent.run("reply")
    assert tool.calls == ["fail", "success"]
    assert agent.iterations == 2


def test_restore_pending_format_retry_keeps_prompt_and_retry_count():
    requests = []
    saved = {}
    tool = Reply()

    def engine(prompt, **kwargs):
        requests.append(copy.deepcopy(prompt))
        if len(requests) == 1:
            return "invalid"
        saved.update({key: copy.deepcopy(getattr(agent, key)) for key in (
            "logs", "pending_step", "iterations", "planning_counter", "task", "attachments", "custom_state")})
        raise Interrupted()

    agent = make(engine, tool)
    with pytest.raises(Interrupted):
        agent.run("reply")
    assert saved["pending_step"]["format_try_count"] == 1
    assert tool.calls == []
    restored_requests = []

    def resumed(prompt, **kwargs):
        restored_requests.append(copy.deepcopy(prompt))
        return action("success")

    other = make(resumed, tool)
    other.initialize()
    for key, value in saved.items():
        setattr(other, key, value)
    other.execute_agent_loop()
    assert restored_requests == [requests[1]]
    assert tool.calls == ["success"]
    assert other.pending_step is None
    assert other.iterations == saved["iterations"] + 1
