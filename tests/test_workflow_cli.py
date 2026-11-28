"""Exercise orchestration, failure handling, and explicit review boundaries."""

from pathlib import Path
from unittest.mock import Mock

from pydantic import SecretStr

from clipforge import workflow_cli


def _args(monkeypatch, *args: str) -> None:
    monkeypatch.setattr("sys.argv", ["workflow", *args])


def _keys(monkeypatch) -> None:
    monkeypatch.setattr(
        workflow_cli, "TranscriptionSettings", lambda: Mock(deepgram_api_key=SecretStr("test"))
    )
    monkeypatch.setattr(
        workflow_cli, "BoundarySettings", lambda: Mock(groq_api_key=SecretStr("test"))
    )


def test_prepare_orders_steps_and_defaults_to_one_window(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    job = tmp_path / "job"
    _keys(monkeypatch)
    run = Mock(return_value=0)
    monkeypatch.setattr(workflow_cli, "_run", run)
    _args(monkeypatch, "prepare", str(video), str(job))
    assert workflow_cli.main() == 0
    assert job.is_dir()
    assert [call.args[0] for call in run.call_args_list] == [
        "clipforge.cli",
        "clipforge.transcribe_cli",
        "clipforge.semantic_candidates_cli",
    ]
    assert run.call_args.args[-2:] == ("--max-windows", "1")


def test_failed_or_partial_step_stops_and_keeps_job(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    _keys(monkeypatch)
    run = Mock(side_effect=[0, 0, 2])
    monkeypatch.setattr(workflow_cli, "_run", run)
    _args(monkeypatch, "prepare", str(video), str(tmp_path / "job"))
    assert workflow_cli.main() == 2
    assert (tmp_path / "job").is_dir()
    assert run.call_count == 3


def test_existing_job_is_never_overwritten(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    job = tmp_path / "job"
    job.mkdir()
    marker = job / "audio.wav"
    marker.write_bytes(b"original")
    _keys(monkeypatch)
    run = Mock()
    monkeypatch.setattr(workflow_cli, "_run", run)
    _args(monkeypatch, "prepare", str(video), str(job))
    assert workflow_cli.main() == 1
    assert marker.read_bytes() == b"original"
    run.assert_not_called()


def test_missing_key_stops_before_creating_job(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    monkeypatch.setattr(
        workflow_cli, "TranscriptionSettings", lambda: Mock(deepgram_api_key=SecretStr(""))
    )
    run = Mock()
    monkeypatch.setattr(workflow_cli, "_run", run)
    _args(monkeypatch, "prepare", str(video), str(tmp_path / "job"))
    assert workflow_cli.main() == 1
    assert not (tmp_path / "job").exists()
    run.assert_not_called()


def test_export_requires_explicit_range_and_forwards_tail_option(
    tmp_path: Path, monkeypatch
) -> None:
    job = tmp_path / "job"
    job.mkdir()
    run = Mock(return_value=0)
    monkeypatch.setattr(workflow_cli, "_run", run)
    _args(
        monkeypatch,
        "export",
        "source.mp4",
        str(job),
        "clip.mp4",
        "100",
        "200",
        "--allow-video-tail-hold",
    )
    assert workflow_cli.main() == 0
    assert run.call_args.args == (
        "clipforge.video_export_cli",
        "source.mp4",
        str(job / "audio.wav"),
        "clip.mp4",
        "100",
        "200",
        "--allow-video-tail-hold",
    )


def test_resume_does_not_retranscribe(tmp_path: Path, monkeypatch) -> None:
    job = tmp_path / "job"
    job.mkdir()
    run = Mock(return_value=2)
    monkeypatch.setattr(workflow_cli, "_run", run)
    _args(monkeypatch, "resume", str(job), str(job / "more.json"), "--start-window", "3")
    assert workflow_cli.main() == 2
    assert run.call_count == 1
    assert run.call_args.args[0] == "clipforge.semantic_candidates_cli"
    assert run.call_args.args[-4:] == ("--start-window", "3", "--max-windows", "1")


def test_resume_rejects_invalid_windows(tmp_path: Path, monkeypatch) -> None:
    run = Mock()
    monkeypatch.setattr(workflow_cli, "_run", run)
    _args(monkeypatch, "resume", str(tmp_path), str(tmp_path / "more.json"), "--start-window", "-1")
    assert workflow_cli.main() == 1
    run.assert_not_called()


def test_extraction_failure_never_calls_providers(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    _keys(monkeypatch)
    run = Mock(return_value=1)
    monkeypatch.setattr(workflow_cli, "_run", run)
    _args(monkeypatch, "prepare", str(video), str(tmp_path / "job"))
    assert workflow_cli.main() == 1
    assert run.call_count == 1
    assert run.call_args.args[0] == "clipforge.cli"


def test_transcription_failure_never_calls_boundary_provider(tmp_path: Path, monkeypatch) -> None:
    video = tmp_path / "source.mp4"
    video.touch()
    _keys(monkeypatch)
    run = Mock(side_effect=[0, 1])
    monkeypatch.setattr(workflow_cli, "_run", run)
    _args(monkeypatch, "prepare", str(video), str(tmp_path / "job"))
    assert workflow_cli.main() == 1
    assert run.call_count == 2
