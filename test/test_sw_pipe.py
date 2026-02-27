from __future__ import annotations

import logging

from starwinds_analysis.pipelines.sw_pipe import SwPipeResults, discover_plt_files, main, run_sw_pipe


def test_discover_plt_files_finds_only_current_directory(tmp_path):
    (tmp_path / "a.plt").write_text("")
    (tmp_path / "b.PLT").write_text("")
    (tmp_path / "ignore.txt").write_text("")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.plt").write_text("")

    files = discover_plt_files(tmp_path)
    assert [path.name for path in files] == ["a.plt", "b.PLT"]


def test_run_sw_pipe_logs_placeholder_file_names_only(tmp_path, caplog):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "beta.plt").write_text("")

    with caplog.at_level(logging.INFO, logger="starwinds_analysis.pipelines.sw_pipe"):
        results = run_sw_pipe(tmp_path)

    assert isinstance(results, SwPipeResults)
    assert [path.name for path in results.discovered_files] == ["alpha.plt", "beta.plt"]
    assert [path.name for path in results.processed_files] == ["alpha.plt", "beta.plt"]
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "starwinds_analysis.pipelines.sw_pipe" and record.levelno == logging.INFO
    ]
    assert messages == ["alpha.plt", "beta.plt"]


def test_sw_pipe_main_scans_current_directory(tmp_path, monkeypatch, capsys):
    (tmp_path / "one.plt").write_text("")
    (tmp_path / "two.PLT").write_text("")
    monkeypatch.chdir(tmp_path)

    code = main([])
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.err.splitlines() if line.strip()]

    assert code == 0
    assert lines == ["one.plt", "two.PLT"]
