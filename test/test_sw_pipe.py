from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from starwinds_analysis.pipelines.dummy_pipeline import name_codepoints_payload
from starwinds_analysis.pipelines.dummy_pipeline import name_letter_counts
from starwinds_analysis.pipelines.dummy_pipeline import name_profile_payload
from starwinds_analysis.pipelines.dummy_pipeline import name_waveform_payload
from starwinds_analysis.pipelines.dummy_pipeline import process_plt_file
from starwinds_analysis.pipelines.sw_pipe import discover_plt_files
from starwinds_analysis.pipelines.sw_pipe import main
from starwinds_analysis.pipelines.sw_pipe import run_sw_pipe


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


@pytest.mark.design_lockin
def test_run_sw_pipe_writes_processed_state_file(tmp_path):
    (tmp_path / "alpha.plt").write_text("")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.plt").write_text("")

    run_sw_pipe(tmp_path, recursive=True)
    state_file = tmp_path / "sw-pipe.dummy.processed.json"
    assert state_file.exists()
    payload = json.loads(state_file.read_text())
    assert payload["processed_files"] == ["alpha.plt", "nested/beta.plt"]
    assert sorted(payload["computed_results"].keys()) == ["alpha.plt", "nested/beta.plt"]
    assert "meta" in payload["computed_results"]["alpha.plt"]
    assert "letter_counts" in payload["computed_results"]["alpha.plt"]


@pytest.mark.design_lockin
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
    assert (tmp_path / "sw-pipe.custom.processed.json").exists()
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


@pytest.mark.design_lockin
def test_run_sw_pipe_warns_when_state_json_is_large(tmp_path, caplog):
    (tmp_path / "alpha.plt").write_text("")
    with caplog.at_level(logging.WARNING, logger="starwinds_analysis.pipelines.sw_pipe"):
        run_sw_pipe(tmp_path, json_warn_bytes=1)
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("state file is large" in message for message in warnings)


def test_sw_pipe_main_accepts_builtin_pipeline_names_on_empty_directory(tmp_path):
    for pipeline_name in ("dummy", "slice", "volume", "shell"):
        code = main([str(tmp_path), "--pipeline", pipeline_name, "--log-level", "ERROR", "--record-log-level", "ERROR"])
        state_file = tmp_path / f"sw-pipe.{pipeline_name}.processed.json"
        payload = json.loads(state_file.read_text())
        assert code == 0
        assert payload["processed_files"] == []
