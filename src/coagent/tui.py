from __future__ import annotations

import logging

from rich.text import Text
from textual.containers import VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widgets import Static

from coagent.events import LoopEvent

logger = logging.getLogger(__name__)

_CONTENT_PREVIEW_CHARS = 500


class StatusBar(Static):
    """Footer status bar showing model info, tokens, cost, and run status."""

    DEFAULT_CSS = """
    StatusBar {
        dock: bottom;
        height: 2;
        background: $surface;
        color: $text-muted;
        padding: 0 2;
    }
    """

    model_name: reactive[str] = reactive("")
    turn_info: reactive[str] = reactive("0/0")
    total_tokens: reactive[int] = reactive(0)
    cost: reactive[float] = reactive(0.0)
    status: reactive[str] = reactive("idle")

    def render(self) -> Text:
        line1 = Text()
        if self.model_name:
            line1.append(f" {self.model_name}", style="bold")
            line1.append(" │ ", style="dim")
        line1.append(f"Turn {self.turn_info}")
        line1.append(" │ ", style="dim")
        line1.append(f"Tokens: {self.total_tokens}")

        line2 = Text()
        line2.append(f" Cost: ${self.cost:.4f}")
        line2.append(" │ ", style="dim")

        status_style = {
            "idle": "dim",
            "running": "bold yellow",
            "completed": "bold green",
            "failed": "bold red",
        }.get(self.status, "")
        line2.append(self.status.capitalize(), style=status_style)

        result = Text()
        result.append_text(line1)
        result.append("\n")
        result.append_text(line2)
        return result


class MessageLog(VerticalScroll):
    """Scrollable container for displaying run events as messages."""

    class EventReceived(Message):
        """Posted from any thread to deliver an event to the message log."""

        def __init__(self, event: LoopEvent) -> None:
            super().__init__()
            self.loop_event = event

    DEFAULT_CSS = """
    MessageLog {
        height: 1fr;
        padding: 1 2;
    }
    MessageLog .task-header {
        color: $accent;
        margin: 0 0 1 0;
    }
    MessageLog .turn-header {
        color: $text;
        margin: 1 0 0 0;
    }
    MessageLog .turn-content {
        color: $text-muted;
        margin: 0 0 0 4;
    }
    MessageLog .policy-line {
        color: $text-disabled;
        margin: 0 0 0 2;
    }
    MessageLog .advisor-block {
        color: $warning;
        margin: 1 0 0 2;
    }
    MessageLog .completion {
        color: $success;
        margin: 1 0 0 0;
    }
    MessageLog .failure {
        color: $error;
        margin: 1 0 0 0;
    }
    """

    def add_event(self, event: LoopEvent) -> None:
        """Thread-safe: post event to the main event loop for rendering."""
        self.post_message(self.EventReceived(event))

    def on_message_log_event_received(self, message: EventReceived) -> None:
        """Render the event on the main thread."""
        self._render_event(message.loop_event)

    def _render_event(self, event: LoopEvent) -> None:
        """Mount widgets for the given event. Must be called on the main thread."""
        if event.kind == "run_start":
            text = Text()
            text.append(f"> Task: {event.task}", style="bold")
            self.mount(Static(text, classes="task-header"))
        elif event.kind == "turn_complete":
            header = Text()
            header.append("⎿ ", style="dim")
            header.append(f"[Executor] Turn {event.turn}", style="bold")
            header.append(
                f"  ({event.prompt_tokens + event.completion_tokens} tokens)",
                style="dim",
            )
            self.mount(Static(header, classes="turn-header"))
            display_content = event.content[:_CONTENT_PREVIEW_CHARS]
            if len(event.content) > _CONTENT_PREVIEW_CHARS:
                display_content += "..."
            self.mount(Static(display_content, classes="turn-content"))
        elif event.kind == "policy_check":
            if event.should_consult:
                line = Text()
                line.append("→ ", style="bold yellow")
                line.append(f"Consulting advisor: {event.reason}", style="yellow")
                self.mount(Static(line, classes="policy-line"))
            else:
                line = Text()
                line.append("· ", style="dim")
                line.append(f"Policy: {event.reason}", style="dim")
                self.mount(Static(line, classes="policy-line"))
        elif event.kind == "advisor_call":
            block = Text()
            block.append("⎿ ", style="dim")
            block.append("[Advisor] ", style="bold yellow")
            block.append(f"{event.advisor_status}", style="yellow")
            block.append(f" (confidence: {event.confidence:.1f})\n", style="dim")
            block.append(f"  Diagnosis: {event.diagnosis}")
            if event.next_step:
                block.append(f"\n  Next step: {event.next_step}")
            if event.recommended_plan:
                block.append(f"\n  Plan: {event.recommended_plan}")
            self.mount(Static(block, classes="advisor-block"))
        elif event.kind == "run_complete":
            css_class = "completion" if event.status == "completed" else "failure"
            icon = "✓" if event.status == "completed" else "✗"
            line = Text()
            line.append(f"{icon} ", style="bold")
            line.append(f"Run {event.status}", style="bold")
            line.append(f" — {event.turns} turns, {event.advisor_calls} advisor calls")
            cost = event.usage.get("total", {}).get("cost_usd", 0)
            line.append(f", ${cost:.4f}", style="dim")
            self.mount(Static(line, classes=css_class))
        else:
            logger.warning("MessageLog: unhandled event kind %r", event.kind)

        self.scroll_end(animate=False)
