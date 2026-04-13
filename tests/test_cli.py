from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from click.testing import CliRunner

from coagent.cli import _timestamped_trace_path, cli

FIXED_DT = datetime(2026, 4, 11, 15, 30, 45, tzinfo=timezone.utc)


def test_file_path_inserts_timestamp_before_extension():
    with patch("coagent.cli.datetime") as mock_dt:
        mock_dt.now.return_value = FIXED_DT
        result = _timestamped_trace_path("traces/run.jsonl")
    assert result == "traces/run_20260411_153045.jsonl"


def test_directory_path_generates_timestamped_filename():
    with patch("coagent.cli.datetime") as mock_dt:
        mock_dt.now.return_value = FIXED_DT
        result = _timestamped_trace_path("traces/")
    assert result == "traces/20260411_153045.jsonl"


def test_directory_without_trailing_slash():
    with (
        patch("coagent.cli.datetime") as mock_dt,
        patch("coagent.cli.os.path.isdir", return_value=True),
    ):
        mock_dt.now.return_value = FIXED_DT
        result = _timestamped_trace_path("traces")
    assert result == "traces/20260411_153045.jsonl"


def test_no_extension_appends_jsonl_with_timestamp():
    with patch("coagent.cli.datetime") as mock_dt:
        mock_dt.now.return_value = FIXED_DT
        result = _timestamped_trace_path("traces/run")
    assert result == "traces/run_20260411_153045.jsonl"


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
        patch("coagent.cli.load_config", return_value=MagicMock()),
        patch(
            "coagent.cli.merge_cli_overrides", return_value=mock_config
        ) as mock_merge,
        patch("coagent.cli.ExecutorLoop") as mock_loop_cls,
    ):
        mock_loop_cls.return_value.run.return_value = mock_result
        runner.invoke(cli, ["run", "--force-consult", "test task"])

    _, kwargs = mock_merge.call_args
    assert kwargs["force_consult"] is True
