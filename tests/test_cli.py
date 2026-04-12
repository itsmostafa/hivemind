from unittest.mock import patch
from datetime import datetime, timezone

from coagent.cli import _timestamped_trace_path

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
