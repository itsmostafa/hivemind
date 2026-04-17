import os
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from click.testing import CliRunner

from hivemind._templates import DEFAULT_CONFIG_YAML
from hivemind.cli import _auto_trace_dir, _timestamped_trace_path, cli

FIXED_DT = datetime(2026, 4, 11, 15, 30, 45, tzinfo=timezone.utc)


def test_file_path_inserts_timestamp_before_extension():
    with patch("hivemind.cli.datetime") as mock_dt:
        mock_dt.now.return_value = FIXED_DT
        result = _timestamped_trace_path("traces/run.jsonl")
    assert result == "traces/run_20260411_153045.jsonl"


def test_directory_path_generates_timestamped_filename():
    with patch("hivemind.cli.datetime") as mock_dt:
        mock_dt.now.return_value = FIXED_DT
        result = _timestamped_trace_path("traces/")
    assert result == "traces/20260411_153045.jsonl"


def test_directory_without_trailing_slash():
    with (
        patch("hivemind.cli.datetime") as mock_dt,
        patch("hivemind.cli.os.path.isdir", return_value=True),
    ):
        mock_dt.now.return_value = FIXED_DT
        result = _timestamped_trace_path("traces")
    assert result == "traces/20260411_153045.jsonl"


def test_no_extension_appends_jsonl_with_timestamp():
    with patch("hivemind.cli.datetime") as mock_dt:
        mock_dt.now.return_value = FIXED_DT
        result = _timestamped_trace_path("traces/run")
    assert result == "traces/run_20260411_153045.jsonl"


def test_auto_trace_dir_strips_home_prefix(tmp_path):
    fake_home = str(tmp_path)
    fake_cwd = str(tmp_path / "Projects" / "tools" / "hivemind")
    with (
        patch("hivemind.cli.os.getcwd", return_value=fake_cwd),
        patch("hivemind.cli.os.path.expanduser", return_value=fake_home),
    ):
        result = _auto_trace_dir()
    expected = (
        os.path.join(fake_home, ".hivemind", "logs", "Projects-tools-hivemind") + os.sep
    )
    assert result == expected


def test_auto_trace_dir_non_home_path(tmp_path):
    fake_home = str(tmp_path / "home" / "user")
    fake_cwd = "/opt/myproject/src"
    with (
        patch("hivemind.cli.os.getcwd", return_value=fake_cwd),
        patch("hivemind.cli.os.path.expanduser", return_value=fake_home),
    ):
        result = _auto_trace_dir()
    expected = (
        os.path.join(fake_home, ".hivemind", "logs", "opt-myproject-src") + os.sep
    )
    assert result == expected


def test_auto_trace_dir_produces_correct_timestamped_path(tmp_path):
    fake_home = str(tmp_path)
    fake_cwd = str(tmp_path / "Projects" / "tools" / "hivemind")
    with (
        patch("hivemind.cli.os.getcwd", return_value=fake_cwd),
        patch("hivemind.cli.os.path.expanduser", return_value=fake_home),
        patch("hivemind.cli.datetime") as mock_dt,
    ):
        mock_dt.now.return_value = FIXED_DT
        result = _timestamped_trace_path(_auto_trace_dir())
    expected = os.path.join(
        fake_home,
        ".hivemind",
        "logs",
        "Projects-tools-hivemind",
        "20260411_153045.jsonl",
    )
    assert result == expected


def test_init_creates_user_config(home_tmp):
    runner = CliRunner()
    config_path = home_tmp / ".hivemind" / "config.yml"
    assert not config_path.exists()
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0
    assert config_path.exists()
    assert config_path.read_text() == DEFAULT_CONFIG_YAML
    assert str(config_path) in result.output


def test_init_refuses_to_overwrite_without_force(home_tmp):
    runner = CliRunner()
    config_path = home_tmp / ".hivemind" / "config.yml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("original content")
    result = runner.invoke(cli, ["init"])
    assert result.exit_code != 0
    assert config_path.read_text() == "original content"


def test_init_overwrites_with_force(home_tmp):
    runner = CliRunner()
    config_path = home_tmp / ".hivemind" / "config.yml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("original content")
    result = runner.invoke(cli, ["init", "--force"])
    assert result.exit_code == 0
    assert config_path.read_text() == DEFAULT_CONFIG_YAML


def _make_chat_mocks():
    mock_config = MagicMock()
    mock_config.logging.trace_file = None
    mock_config.logging.level = "INFO"

    mock_result = MagicMock()
    mock_result.final_answer = "answer"
    mock_result.usage_summary = {}
    mock_result.state = MagicMock()

    return mock_config, mock_result


def test_chat_quits_immediately_on_q():
    mock_config, _ = _make_chat_mocks()
    runner = CliRunner()
    with (
        patch("hivemind.cli.load_config", return_value=MagicMock()),
        patch("hivemind.cli.merge_cli_overrides", return_value=mock_config),
        patch("hivemind.cli.TraceLogger"),
        patch("hivemind.cli.ModelClient"),
        patch("hivemind.cli.Advisor"),
        patch("hivemind.cli.DecisionPolicy"),
        patch("hivemind.cli.ExecutorLoop") as mock_loop_cls,
    ):
        result = runner.invoke(cli, ["chat"], input="q\n")

    assert result.exit_code == 0
    mock_loop_cls.return_value.run.assert_not_called()


def test_chat_help_prints_and_continues():
    mock_config, _ = _make_chat_mocks()
    runner = CliRunner()
    with (
        patch("hivemind.cli.load_config", return_value=MagicMock()),
        patch("hivemind.cli.merge_cli_overrides", return_value=mock_config),
        patch("hivemind.cli.TraceLogger"),
        patch("hivemind.cli.ModelClient"),
        patch("hivemind.cli.Advisor"),
        patch("hivemind.cli.DecisionPolicy"),
        patch("hivemind.cli.ExecutorLoop") as mock_loop_cls,
    ):
        result = runner.invoke(cli, ["chat"], input="h\nq\n")

    assert "quit" in result.output
    mock_loop_cls.return_value.run.assert_not_called()


def test_chat_carries_state_across_turns():
    mock_config, _ = _make_chat_mocks()
    sentinel_state = MagicMock()
    mock_result1 = MagicMock()
    mock_result1.final_answer = "first answer"
    mock_result1.usage_summary = {}
    mock_result1.state = sentinel_state
    mock_result2 = MagicMock()
    mock_result2.final_answer = "second answer"
    mock_result2.usage_summary = {}
    mock_result2.state = MagicMock()

    runner = CliRunner()
    with (
        patch("hivemind.cli.load_config", return_value=MagicMock()),
        patch("hivemind.cli.merge_cli_overrides", return_value=mock_config),
        patch("hivemind.cli.TraceLogger"),
        patch("hivemind.cli.ModelClient"),
        patch("hivemind.cli.Advisor"),
        patch("hivemind.cli.DecisionPolicy"),
        patch("hivemind.cli.ExecutorLoop") as mock_loop_cls,
    ):
        mock_loop_cls.return_value.run.side_effect = [mock_result1, mock_result2]
        runner.invoke(cli, ["chat"], input="first\nsecond\nq\n")

    calls = mock_loop_cls.return_value.run.call_args_list
    assert len(calls) == 2
    _, kwargs = calls[1]
    assert kwargs["state"] is sentinel_state


def test_chat_reset_clears_state():
    mock_config, _ = _make_chat_mocks()
    mock_result1 = MagicMock()
    mock_result1.final_answer = "hello answer"
    mock_result1.usage_summary = {}
    mock_result1.state = MagicMock()
    mock_result2 = MagicMock()
    mock_result2.final_answer = "follow answer"
    mock_result2.usage_summary = {}
    mock_result2.state = MagicMock()

    runner = CliRunner()
    with (
        patch("hivemind.cli.load_config", return_value=MagicMock()),
        patch("hivemind.cli.merge_cli_overrides", return_value=mock_config),
        patch("hivemind.cli.TraceLogger"),
        patch("hivemind.cli.ModelClient"),
        patch("hivemind.cli.Advisor"),
        patch("hivemind.cli.DecisionPolicy"),
        patch("hivemind.cli.ExecutorLoop") as mock_loop_cls,
    ):
        mock_loop_cls.return_value.run.side_effect = [mock_result1, mock_result2]
        result = runner.invoke(cli, ["chat"], input="hello\nr\nfollow\nq\n")

    assert "[session reset]" in result.output
    calls = mock_loop_cls.return_value.run.call_args_list
    assert len(calls) == 2
    _, kwargs = calls[1]
    assert kwargs["state"] is None


def test_chat_search_flag_passes_through():
    mock_config, _ = _make_chat_mocks()
    runner = CliRunner()
    with (
        patch("hivemind.cli.load_config", return_value=MagicMock()),
        patch(
            "hivemind.cli.merge_cli_overrides", return_value=mock_config
        ) as mock_merge,
        patch("hivemind.cli.TraceLogger"),
        patch("hivemind.cli.ModelClient"),
        patch("hivemind.cli.Advisor"),
        patch("hivemind.cli.DecisionPolicy"),
        patch("hivemind.cli.ExecutorLoop"),
    ):
        runner.invoke(cli, ["chat", "--search"], input="q\n")

    _, kwargs = mock_merge.call_args
    assert kwargs["search_enabled"] is True


def test_force_consult_flag_passes_through_to_merge_cli_overrides():
    """--force-consult should pass force_consult=True to merge_cli_overrides."""
    mock_config = MagicMock()
    mock_config.logging.trace_file = None
    mock_config.logging.level = "INFO"

    mock_result = MagicMock()
    mock_result.state.status = "completed"
    mock_result.final_answer = "done"
    mock_result.usage_summary = {}
    mock_result.advisor_history = []

    runner = CliRunner()
    with (
        patch("hivemind.cli.load_config", return_value=MagicMock()),
        patch(
            "hivemind.cli.merge_cli_overrides", return_value=mock_config
        ) as mock_merge,
        patch("hivemind.cli.TraceLogger"),
        patch("hivemind.cli.ExecutorLoop") as mock_loop_cls,
    ):
        mock_loop_cls.return_value.run.return_value = mock_result
        runner.invoke(cli, ["run", "--force-consult", "test task"])

    _, kwargs = mock_merge.call_args
    assert kwargs["force_consult"] is True
