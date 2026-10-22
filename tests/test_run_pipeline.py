"""Kontrollera stegens ordning, miljö och att fel stoppar körningen."""
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from src import run_pipeline as pipeline


@pytest.fixture
def project(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    for name in pipeline.STEPS:
        (tmp_path / "src" / name).touch()
    (tmp_path / "data/raw").mkdir(parents=True)
    (tmp_path / "data/raw/yelp_labelled.txt").touch()
    monkeypatch.setattr(pipeline, "PROJECT_ROOT", tmp_path)
    return tmp_path


@pytest.mark.parametrize("download", [False, True])
def test_order_environment_and_no_final_evaluation(project, monkeypatch, download):
    run = Mock()
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    pipeline.run_pipeline(download=download)
    commands = [call.args[0] for call in run.call_args_list]
    assert [Path(command[1]).name for command in commands] == [
        "load_data.py", "prepare_data.py", "split_data.py", "train_baseline.py", "train_model.py"
    ]
    assert all(command[0] == sys.executable for command in commands)
    assert ("--download" in commands[0]) == download
    assert all("--download" not in command for command in commands[1:])
    assert all(call.kwargs == {"cwd": project, "check": True} for call in run.call_args_list)


def test_missing_data_explains_download(project, monkeypatch):
    (project / "data/raw/yelp_labelled.txt").unlink()
    run = Mock()
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with pytest.raises(FileNotFoundError, match="--download"):
        pipeline.run_pipeline()
    run.assert_not_called()
    pipeline.run_pipeline(download=True)
    assert run.call_count == 5


def test_failed_step_prevents_training(project, monkeypatch):
    run = Mock(side_effect=[None, subprocess.CalledProcessError(1, [sys.executable, "prepare_data.py"])])
    monkeypatch.setattr(pipeline.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        pipeline.run_pipeline()
    assert run.call_count == 2


def test_cli_failure_returns_nonzero(project, monkeypatch, capsys):
    (project / "data/raw/yelp_labelled.txt").unlink()
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py"])
    assert pipeline.main() == 1
    assert "--download" in capsys.readouterr().err
