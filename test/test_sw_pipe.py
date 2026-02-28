from __future__ import annotations

from datetime import datetime
import hashlib
import json
import logging
import numpy as np
from pathlib import Path
import re

from starwinds_analysis.pipelines.dummy_pipeline import (
    name_codepoints_payload,
    name_letter_counts,
    name_profile_payload,
    name_waveform_payload,
    process_plt_file,
)
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


def test_name_codepoints_payload_outputs_numpy_array():
    value = name_codepoints_payload("ab")
    assert np.array_equal(value, np.array([97, 98], dtype=np.int32))


def test_name_waveform_payload_outputs_large_numpy_array():
    value = name_waveform_payload("ab")
    assert isinstance(value, np.ndarray)
    assert value.dtype == np.float64
    assert value.shape == (131072,)


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
    assert alpha["name_codepoints"]["value"] == [97, 108, 112, 104, 97]
    assert sorted(alpha["name_waveform"]["value"].keys()) == ["path"]
    alpha_wave_path = tmp_path / alpha["name_waveform"]["value"]["path"]
    assert alpha_wave_path.exists()
    assert beta["letter_counts"]["value"] == {"vowels": 2, "consonants": 2}
    assert beta["name_vowel_fraction"]["value"] == 0.5
    assert beta["name_dominance"]["value"] == "vowel-rich"
    assert beta["name_shape"]["value"] == [4, 2, 2]
    assert beta["name_codepoints"]["value"] == [98, 101, 116, 97]
    assert sorted(beta["name_waveform"]["value"].keys()) == ["path"]
    beta_wave_path = tmp_path / beta["name_waveform"]["value"]["path"]
    assert beta_wave_path.exists()
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
        alpha["name_codepoints"],
        alpha["name_waveform"],
        beta["letter_counts"],
        beta["name_vowel_fraction"],
        beta["name_dominance"],
        beta["name_shape"],
        beta["name_codepoints"],
        beta["name_waveform"],
    ):
        source = entry["source"]
        assert source["module"] == "starwinds_analysis.pipelines.dummy_pipeline"
        assert source["function"] in {
            "name_letter_counts",
            "name_profile_payload",
            "name_codepoints_payload",
            "name_waveform_payload",
        }
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
    state_file = tmp_path / "sw-pipe.processed.json"
    assert state_file.exists()
    payload = json.loads(state_file.read_text())
    assert payload["processed_files"] == ["alpha.plt", "nested/beta.plt"]
    alpha = payload["computed_results"]["alpha.plt"]
    beta = payload["computed_results"]["nested/beta.plt"]
    assert alpha["letter_counts"]["value"] == {"vowels": 2, "consonants": 3}
    assert alpha["name_vowel_fraction"]["value"] == 0.4
    assert alpha["name_dominance"]["value"] == "consonant-rich"
    assert alpha["name_shape"]["value"] == [5, 2, 3]
    assert alpha["name_codepoints"]["value"] == [97, 108, 112, 104, 97]
    assert sorted(alpha["name_waveform"]["value"].keys()) == ["path"]
    assert beta["letter_counts"]["value"] == {"vowels": 2, "consonants": 2}
    assert beta["name_vowel_fraction"]["value"] == 0.5
    assert beta["name_dominance"]["value"] == "vowel-rich"
    assert beta["name_shape"]["value"] == [4, 2, 2]
    assert beta["name_codepoints"]["value"] == [98, 101, 116, 97]
    assert sorted(beta["name_waveform"]["value"].keys()) == ["path"]
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


def test_run_sw_pipe_offloads_large_numpy_array_to_artifact(tmp_path):
    (tmp_path / "alpha.plt").write_text("")

    def process_file(_path):
        recorder_log = logging.getLogger("recorder.test_pipeline")
        recorder_log.setLevel(logging.DEBUG)
        recorder_log.debug("big_array %r", np.arange(200_000, dtype=np.float64))

    results = run_sw_pipe(tmp_path, process_file=process_file, array_offload_min_bytes=1_000_000)
    entry = results.computed_results["alpha.plt"]["big_array"]
    value = entry["value"]
    assert sorted(value.keys()) == ["path"]
    artifact_path = tmp_path / value["path"]
    assert artifact_path.exists()
    loaded = np.load(artifact_path)
    assert loaded.shape == (200000,)
    assert float(loaded[0]) == 0.0
    assert float(loaded[-1]) == 199999.0
    assert entry["source"]["module"] == "test_pipeline"


def test_run_sw_pipe_warns_when_state_json_is_large(tmp_path, caplog):
    (tmp_path / "alpha.plt").write_text("")
    with caplog.at_level(logging.WARNING, logger="starwinds_analysis.pipelines.sw_pipe"):
        run_sw_pipe(tmp_path, json_warn_bytes=1)
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("state file is large" in message for message in warnings)


def test_run_sw_pipe_supports_slice_pipeline(tmp_path, monkeypatch):
    (tmp_path / "alpha.plt").write_text("")
    called: list[tuple[str, bool]] = []

    def fake_quicklook_process(path, *, force_3d=False):
        called.append((Path(path).name, bool(force_3d)))
        recorder = logging.getLogger("recorder.starwinds_analysis.pipelines.slice")
        recorder.debug("quicklook_marker %r", "ok")

    import starwinds_analysis.pipelines.slice as slice_pipeline

    monkeypatch.setattr(slice_pipeline, "process_plt_file", fake_quicklook_process)
    results = run_sw_pipe(tmp_path, pipeline="slice")

    assert called == [("alpha.plt", False)]
    payload = results.computed_results["alpha.plt"]
    assert payload["meta"]["pipeline"] == "slice"
    assert payload["quicklook_marker"]["value"] == "ok"
    assert payload["quicklook_marker"]["source"]["module"] == "starwinds_analysis.pipelines.slice"


def test_run_sw_pipe_supports_slice_force_3d_flag(tmp_path, monkeypatch):
    (tmp_path / "alpha.plt").write_text("")
    called: list[tuple[str, bool]] = []

    def fake_quicklook_process(path, *, force_3d=False):
        called.append((Path(path).name, bool(force_3d)))

    import starwinds_analysis.pipelines.slice as slice_pipeline

    monkeypatch.setattr(slice_pipeline, "process_plt_file", fake_quicklook_process)
    run_sw_pipe(tmp_path, pipeline="slice", force_slice_3d=True)

    assert called == [("alpha.plt", True)]


def test_volume_process_skips_non_3d_input(tmp_path, monkeypatch):
    file_path = tmp_path / "alpha.plt"
    file_path.write_text("")
    calls = []

    class Fake2DDataset:
        corners = np.zeros((1, 4), dtype=int)

    class FakeSmartDs:
        @classmethod
        def from_file(cls, _path):
            return Fake2DDataset()

    def fake_mass_loss_vs_radius(*_args, **_kwargs):
        calls.append(object())
        return {}

    import starwinds_analysis.pipelines.volume as volume_pipeline

    monkeypatch.setattr(volume_pipeline, "SmartDs", FakeSmartDs)
    monkeypatch.setattr(volume_pipeline, "mass_loss_vs_radius", fake_mass_loss_vs_radius)
    volume_pipeline.process_plt_file(file_path)

    assert calls == []


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


def test_sw_pipe_main_record_logger_level_is_independent(tmp_path, monkeypatch, capsys):
    (tmp_path / "one.plt").write_text("")
    monkeypatch.chdir(tmp_path)

    code = main(["--log-level", "WARNING", "--record-log-level", "DEBUG"])
    captured = capsys.readouterr()
    lines = [line.strip() for line in captured.err.splitlines() if line.strip()]

    assert code == 0
    expected_patterns = [
        r"^\[debug\] recorder\.starwinds_analysis\.pipelines\.dummy_pipeline\.name_letter_counts:\d+ letter_counts .+",
        r"^\[debug\] recorder\.starwinds_analysis\.pipelines\.dummy_pipeline\.name_profile_payload:\d+ name_vowel_fraction .+",
        r"^\[debug\] recorder\.starwinds_analysis\.pipelines\.dummy_pipeline\.name_profile_payload:\d+ name_dominance .+",
        r"^\[debug\] recorder\.starwinds_analysis\.pipelines\.dummy_pipeline\.name_profile_payload:\d+ name_shape .+",
        r"^\[debug\] recorder\.starwinds_analysis\.pipelines\.dummy_pipeline\.name_codepoints_payload:\d+ name_codepoints .+",
        r"^\[debug\] recorder\.starwinds_analysis\.pipelines\.dummy_pipeline\.name_waveform_payload:\d+ name_waveform .+",
    ]
    assert len(lines) >= len(expected_patterns)
    assert all(any(re.match(pattern, line) for line in lines) for pattern in expected_patterns)
