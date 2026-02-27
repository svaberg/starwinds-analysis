from __future__ import annotations

import json
import logging

from starwinds_analysis.pipelines.dummy_pipeline import process_plt_file
from starwinds_analysis.pipelines.sw_pipe import SwPipeResults, discover_plt_files, main, run_sw_pipe


def test_discover_plt_files_finds_only_current_directory(tmp_path):
    (tmp_path / "a.plt").write_text("")
    (tmp_path / "b.PLT").write_text("")
    (tmp_path / "ignore.txt").write_text("")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.plt").write_text("")

    files = discover_plt_files(tmp_path)
    assert [path.name for path in files] == ["a.plt", "b.PLT"]


def test_dummy_pipeline_process_without_sink_does_not_fail(tmp_path, caplog):
    file_path = tmp_path / "gamma.plt"
    file_path.write_text("")
    with caplog.at_level(logging.INFO, logger="starwinds_analysis.pipelines.dummy_pipeline"):
        process_plt_file(file_path)
    assert [record.getMessage() for record in caplog.records] == [
        "gamma.plt",
    ]


def test_run_sw_pipe_logs_placeholder_file_names_only(tmp_path, caplog):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "beta.plt").write_text("")

    with caplog.at_level(logging.DEBUG):
        results = run_sw_pipe(tmp_path)

    assert isinstance(results, SwPipeResults)
    assert [path.name for path in results.discovered_files] == ["alpha.plt", "beta.plt"]
    assert [path.name for path in results.processed_files] == ["alpha.plt", "beta.plt"]
    assert results.skipped_files == []
    assert results.computed_results == {}
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "starwinds_analysis.pipelines.dummy_pipeline"
        and record.levelno == logging.INFO
    ]
    assert messages == [
        "alpha.plt",
        "beta.plt",
    ]


def test_run_sw_pipe_noclobber_skips_already_processed_files(tmp_path):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "beta.plt").write_text("")

    first = run_sw_pipe(tmp_path)
    second = run_sw_pipe(tmp_path, noclobber=True)

    assert [path.name for path in first.processed_files] == ["alpha.plt", "beta.plt"]
    assert second.processed_files == []
    assert [path.name for path in second.skipped_files] == ["alpha.plt", "beta.plt"]


def test_run_sw_pipe_default_clobber_reprocesses_files(tmp_path):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "beta.plt").write_text("")

    first = run_sw_pipe(tmp_path)
    second = run_sw_pipe(tmp_path)

    assert [path.name for path in first.processed_files] == ["alpha.plt", "beta.plt"]
    assert [path.name for path in second.processed_files] == ["alpha.plt", "beta.plt"]
    assert second.skipped_files == []


def test_run_sw_pipe_writes_processed_state_file(tmp_path):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.plt").write_text("")

    run_sw_pipe(tmp_path, recursive=True)
    state_file = tmp_path / ".sw-pipe.processed.json"
    assert state_file.exists()
    payload = json.loads(state_file.read_text())
    assert payload["processed_files"] == ["alpha.plt", "nested/beta.plt"]
    assert payload["computed_results"] == {}


def test_sw_pipe_main_scans_current_directory(tmp_path, monkeypatch, capsys):
    (tmp_path / "one.plt").write_text("")
    (tmp_path / "two.PLT").write_text("")
    monkeypatch.chdir(tmp_path)

    code = main([])
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.err.splitlines() if line.strip()]

    assert code == 0
    assert lines == [
        "[info] dummy_pipeline one.plt",
        "[info] dummy_pipeline two.PLT",
    ]


def test_sw_pipe_main_emit_logger_level_is_independent(tmp_path, monkeypatch, capsys):
    (tmp_path / "one.plt").write_text("")
    monkeypatch.chdir(tmp_path)

    code = main(["--log-level", "WARNING", "--emit-log-level", "DEBUG"])
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.err.splitlines() if line.strip()]

    assert code == 0
    assert lines == []
