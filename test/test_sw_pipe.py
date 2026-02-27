from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
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
    assert alpha["letter_counts"]["value"] == {"vowels": 2, "consonants": 3}
    assert alpha["name_vowel_fraction"]["value"] == 0.4
    assert alpha["name_dominance"]["value"] == "consonant-rich"
    assert alpha["name_shape"]["value"] == [5, 2, 3]
    assert beta["letter_counts"]["value"] == {"vowels": 2, "consonants": 2}
    assert beta["name_vowel_fraction"]["value"] == 0.5
    assert beta["name_dominance"]["value"] == "vowel-rich"
    assert beta["name_shape"]["value"] == [4, 2, 2]
    alpha_meta = alpha["meta"]
    beta_meta = beta["meta"]
    assert Path(alpha_meta["input_file"]).is_absolute()
    assert Path(beta_meta["input_file"]).is_absolute()
    assert Path(alpha_meta["input_file"]).name == "alpha.plt"
    assert Path(beta_meta["input_file"]).name == "beta.plt"
    assert "file_hash_sha256" not in alpha_meta
    assert "file_hash_sha256" not in beta_meta
    alpha_start = datetime.fromisoformat(alpha_meta["start_time_utc"].replace("Z", "+00:00"))
    alpha_end = datetime.fromisoformat(alpha_meta["end_time_utc"].replace("Z", "+00:00"))
    beta_start = datetime.fromisoformat(beta_meta["start_time_utc"].replace("Z", "+00:00"))
    beta_end = datetime.fromisoformat(beta_meta["end_time_utc"].replace("Z", "+00:00"))
    assert alpha_end >= alpha_start
    assert beta_end >= beta_start
    for entry in (
        alpha["letter_counts"],
        alpha["name_vowel_fraction"],
        alpha["name_dominance"],
        alpha["name_shape"],
        beta["letter_counts"],
        beta["name_vowel_fraction"],
        beta["name_dominance"],
        beta["name_shape"],
    ):
        source = entry["source"]
        assert source["module"] == "starwinds_analysis.pipelines.dummy_pipeline"
        assert source["function"] in {"name_letter_counts", "name_profile_payload"}
        assert isinstance(source["line"], int)
        assert source["line"] > 0
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
    assert alpha["letter_counts"]["value"] == {"vowels": 2, "consonants": 3}
    assert alpha["name_vowel_fraction"]["value"] == 0.4
    assert alpha["name_dominance"]["value"] == "consonant-rich"
    assert alpha["name_shape"]["value"] == [5, 2, 3]
    assert beta["letter_counts"]["value"] == {"vowels": 2, "consonants": 2}
    assert beta["name_vowel_fraction"]["value"] == 0.5
    assert beta["name_dominance"]["value"] == "vowel-rich"
    assert beta["name_shape"]["value"] == [4, 2, 2]
    assert Path(alpha["meta"]["input_file"]).name == "alpha.plt"
    assert Path(beta["meta"]["input_file"]).name == "beta.plt"
    assert "start_time_utc" in alpha["meta"]
    assert "end_time_utc" in alpha["meta"]
    assert alpha["letter_counts"]["source"]["module"] == "starwinds_analysis.pipelines.dummy_pipeline"
    assert beta["letter_counts"]["source"]["module"] == "starwinds_analysis.pipelines.dummy_pipeline"


def test_run_sw_pipe_can_record_file_hash(tmp_path):
    file_path = tmp_path / "alpha.plt"
    file_path.write_text("abc")
    results = run_sw_pipe(tmp_path, include_file_hash=True)
    expected_hash = hashlib.sha256(b"abc").hexdigest()
    alpha = results.computed_results["alpha.plt"]
    assert alpha["meta"]["file_hash_sha256"] == expected_hash


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
