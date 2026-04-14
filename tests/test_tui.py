from unittest.mock import patch

import pytest
from click.testing import CliRunner

pytest.importorskip("textual")

from textual.app import App, ComposeResult
from textual.widgets import Static

from coagent.cli import cli
from coagent.events import (
    AdvisorCallEvent,
    PolicyCheckEvent,
    RunCompleteEvent,
    RunStartEvent,
    TurnCompleteEvent,
)
from coagent.schemas import CoagentConfig, ModelConfig
from coagent.tui import CoagentApp, LoopEventMessage, MessageLog, StatusBar

pytestmark = pytest.mark.anyio


class StatusBarTestApp(App):
    def compose(self) -> ComposeResult:
        yield StatusBar()


async def test_status_bar_default_render():
    async with StatusBarTestApp().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        rendered = bar.render()
        rendered_str = rendered.plain
        assert "Turn 0/0" in rendered_str
        assert "Tokens: 0" in rendered_str
        assert "Cost: $0.0000" in rendered_str
        assert "Idle" in rendered_str


async def test_status_bar_updates():
    async with StatusBarTestApp().run_test() as pilot:
        bar = pilot.app.query_one(StatusBar)
        bar.model_name = "ollama/llama3"
        bar.turn_info = "2/20"
        bar.total_tokens = 1234
        bar.cost = 0.0012
        bar.status = "running"
        rendered = bar.render()
        rendered_str = rendered.plain if hasattr(rendered, "plain") else str(rendered)
        assert "ollama/llama3" in rendered_str
        assert "2/20" in rendered_str
        assert "1234" in rendered_str
        assert "0.0012" in rendered_str
        assert "Running" in rendered_str


class MessageLogTestApp(App):
    def compose(self) -> ComposeResult:
        yield MessageLog()


async def test_message_log_add_task():
    async with MessageLogTestApp().run_test() as pilot:
        log = pilot.app.query_one(MessageLog)
        log.add_event(
            RunStartEvent(
                task="Solve this problem",
                executor_model="test/model",
                advisor_model="test/advisor",
                max_turns=20,
            )
        )
        await pilot.pause()
        widgets = log.query(Static)
        all_text = " ".join(str(w.content) for w in widgets)
        assert "Solve this problem" in all_text


async def test_message_log_add_turn():
    async with MessageLogTestApp().run_test() as pilot:
        log = pilot.app.query_one(MessageLog)
        log.add_event(
            TurnCompleteEvent(
                turn=1,
                max_turns=20,
                content="Here is my analysis of the problem...",
                tool_calls=[],
                prompt_tokens=10,
                completion_tokens=5,
                cumulative_cost=0.001,
                status="running",
            )
        )
        await pilot.pause()
        widgets = log.query(Static)
        all_text = " ".join(str(w.content) for w in widgets)
        assert "Turn 1" in all_text
        assert "analysis" in all_text


async def test_message_log_add_advisor():
    async with MessageLogTestApp().run_test() as pilot:
        log = pilot.app.query_one(MessageLog)
        log.add_event(
            AdvisorCallEvent(
                turn=2,
                reason="low_confidence",
                advisor_status="continue",
                diagnosis="You are on the right track.",
                confidence=0.8,
            )
        )
        await pilot.pause()
        widgets = log.query(Static)
        all_text = " ".join(str(w.content) for w in widgets)
        assert "Advisor" in all_text
        assert "right track" in all_text


async def test_message_log_policy_check_consult():
    async with MessageLogTestApp().run_test() as pilot:
        log = pilot.app.query_one(MessageLog)
        log.add_event(
            PolicyCheckEvent(
                turn=1,
                should_consult=True,
                reason="low_confidence",
            )
        )
        await pilot.pause()
        widgets = log.query(Static)
        all_text = " ".join(str(w.content) for w in widgets)
        assert "low_confidence" in all_text


async def test_message_log_run_complete():
    async with MessageLogTestApp().run_test() as pilot:
        log = pilot.app.query_one(MessageLog)
        log.add_event(
            RunCompleteEvent(
                turns=3,
                advisor_calls=1,
                status="completed",
                final_answer="Done.",
                usage={"total": {"cost_usd": 0.005}},
            )
        )
        await pilot.pause()
        widgets = log.query(Static)
        all_text = " ".join(str(w.content) for w in widgets)
        assert "completed" in all_text
        assert "3 turns" in all_text


async def test_app_has_message_log_and_status_bar():
    from unittest.mock import patch

    config = CoagentConfig(
        executor=ModelConfig(model="test/exec"),
        advisor=ModelConfig(model="test/adv"),
    )
    app = CoagentApp(config=config, task="test task")
    with patch.object(app, "_execute"):
        async with app.run_test() as pilot:
            assert pilot.app.query_one(MessageLog) is not None
            assert pilot.app.query_one(StatusBar) is not None


async def test_app_shows_input_when_no_task():
    from textual.widgets import Input

    config = CoagentConfig(
        executor=ModelConfig(model="test/exec"),
        advisor=ModelConfig(model="test/adv"),
    )
    app = CoagentApp(config=config, task=None)
    async with app.run_test() as pilot:
        inputs = pilot.app.query(Input)
        assert len(inputs) >= 1


async def test_app_processes_run_start_event():
    from unittest.mock import patch

    config = CoagentConfig(
        executor=ModelConfig(model="test/exec"),
        advisor=ModelConfig(model="test/adv"),
    )
    app = CoagentApp(config=config, task="test")
    with patch.object(app, "_execute"):
        async with app.run_test() as pilot:
            start = RunStartEvent(
                task="test",
                executor_model="test/exec",
                advisor_model="test/adv",
                max_turns=20,
            )
            pilot.app.post_message(LoopEventMessage(start))
            await pilot.pause()
            bar = pilot.app.query_one(StatusBar)
            assert bar.model_name == "test/exec"
            assert bar.status == "running"


async def test_app_processes_run_complete_event():
    from unittest.mock import patch

    config = CoagentConfig(
        executor=ModelConfig(model="test/exec"),
        advisor=ModelConfig(model="test/adv"),
    )
    app = CoagentApp(config=config, task="test")
    with patch.object(CoagentApp, "_execute"):
        async with app.run_test() as pilot:
            complete = RunCompleteEvent(
                turns=2,
                advisor_calls=0,
                status="completed",
                final_answer="Done.",
                usage={"total": {"cost_usd": 0.001}},
            )
            pilot.app.post_message(LoopEventMessage(complete))
            await pilot.pause()
            bar = pilot.app.query_one(StatusBar)
            assert bar.status == "completed"


def test_coagent_bare_shows_help_with_tui_default():
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0


def test_coagent_launches_tui_as_default():
    with patch("coagent.cli.CoagentApp") as MockApp:
        mock_instance = MockApp.return_value
        mock_instance.run.return_value = None
        CliRunner().invoke(cli, ["--executor", "test/model", "Do a task"])
        MockApp.assert_called_once()


async def test_app_displays_error():
    from unittest.mock import patch

    config = CoagentConfig(
        executor=ModelConfig(model="test/exec"),
        advisor=ModelConfig(model="test/adv"),
    )
    from coagent.events import ErrorEvent

    app = CoagentApp(config=config, task="test")
    with patch.object(app, "_execute"):
        async with app.run_test() as pilot:
            error = ErrorEvent(message="Connection refused", traceback="Traceback ...")
            pilot.app.post_message(LoopEventMessage(error))
            await pilot.pause()
            log = pilot.app.query_one(MessageLog)
            widgets = log.query(Static)
            all_text = " ".join(str(w.content) for w in widgets)
            assert "Connection refused" in all_text
