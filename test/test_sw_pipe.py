from __future__ import annotations

import json
import logging
import re

from starwinds_analysis.pipelines.dummy_pipeline import name_letter_counts, name_profile_payload, process_plt_file
from starwinds_analysis.pipelines.sw_pipe import SwPipeResults, discover_plt_files, main, run_sw_pipe


def test_discover_plt_files_finds_only_current_directory(tmp_path):
    (tmp_path / "a.plt").write_text("")
    (tmp_path / "b.PLT").write_text("")
    (tmp_path / "ignore.txt").write_text("")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "c.plt").write_text("")

    files = discover_plt_files(tmp_path)
    assert [path.name for path in files] == ["a.plt", "b.PLT"]


def test_name_letter_counts_counts_alpha_only():
    assert name_letter_counts("a1-b2") == (1, 1)


def test_name_profile_payload_outputs_float_string_array():
    value = name_profile_payload("alpha")
    assert value == (0.4, "consonant-rich", [5, 2, 3])


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
    alpha = results.computed_results["alpha.plt"]
    beta = results.computed_results["beta.plt"]
    assert alpha["letter_counts"] == {"vowels": 2, "consonants": 3}
    assert alpha["name_vowel_fraction"] == 0.4
    assert alpha["name_dominance"] == "consonant-rich"
    assert alpha["name_shape"] == [5, 2, 3]
    assert beta["letter_counts"] == {"vowels": 2, "consonants": 2}
    assert beta["name_vowel_fraction"] == 0.5
    assert beta["name_dominance"] == "vowel-rich"
    assert beta["name_shape"] == [4, 2, 2]
    alpha_trace = alpha["trace"]
    beta_trace = beta["trace"]
    assert alpha_trace["module"] == "starwinds_analysis.pipelines.emit.dummy_pipeline"
    assert beta_trace["module"] == "starwinds_analysis.pipelines.emit.dummy_pipeline"
    assert set(alpha_trace["sources"]) == {"letter_counts", "name_vowel_fraction", "name_dominance", "name_shape"}
    assert set(beta_trace["sources"]) == {"letter_counts", "name_vowel_fraction", "name_dominance", "name_shape"}
    assert re.match(r"^name_letter_counts:\d+$", alpha_trace["sources"]["letter_counts"])
    assert re.match(r"^name_profile_payload:\d+$", alpha_trace["sources"]["name_vowel_fraction"])
    assert re.match(r"^name_profile_payload:\d+$", alpha_trace["sources"]["name_dominance"])
    assert re.match(r"^name_profile_payload:\d+$", alpha_trace["sources"]["name_shape"])
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
    alpha = payload["computed_results"]["alpha.plt"]
    beta = payload["computed_results"]["nested/beta.plt"]
    assert alpha["letter_counts"] == {"vowels": 2, "consonants": 3}
    assert alpha["name_vowel_fraction"] == 0.4
    assert alpha["name_dominance"] == "consonant-rich"
    assert alpha["name_shape"] == [5, 2, 3]
    assert beta["letter_counts"] == {"vowels": 2, "consonants": 2}
    assert beta["name_vowel_fraction"] == 0.5
    assert beta["name_dominance"] == "vowel-rich"
    assert beta["name_shape"] == [4, 2, 2]
    assert alpha["trace"]["module"] == "starwinds_analysis.pipelines.emit.dummy_pipeline"
    assert beta["trace"]["module"] == "starwinds_analysis.pipelines.emit.dummy_pipeline"


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
    expected_patterns = [
        r"^\[debug\] starwinds_analysis\.pipelines\.emit\.dummy_pipeline\.name_letter_counts:\d+ letter_counts .+",
        r"^\[debug\] starwinds_analysis\.pipelines\.emit\.dummy_pipeline\.name_profile_payload:\d+ name_vowel_fraction .+",
        r"^\[debug\] starwinds_analysis\.pipelines\.emit\.dummy_pipeline\.name_profile_payload:\d+ name_dominance .+",
        r"^\[debug\] starwinds_analysis\.pipelines\.emit\.dummy_pipeline\.name_profile_payload:\d+ name_shape .+",
    ]
    assert len(lines) == len(expected_patterns)
    assert all(re.match(pattern, line) for line, pattern in zip(lines, expected_patterns))
