from __future__ import annotations

import json
import pytest

from starwinds_analysis.pipelines.sw_pipe_results import main


def _write_state(tmp_path):
    payload = {
        "processed_files": ["alpha.plt", "beta.plt"],
        "computed_results": {
            "alpha.plt": {
                "meta": {
                    "input_file": "/tmp/alpha.plt",
                    "start_time_utc": "2026-01-01T00:00:00Z",
                    "end_time_utc": "2026-01-01T00:00:01Z",
                },
                "letter_counts": {
                    "value": {"vowels": 2, "consonants": 3},
                    "source": {
                        "module": "starwinds_analysis.pipelines.dummy_pipeline",
                        "function": "name_letter_counts",
                        "line": 20,
                    },
                },
                "name_dominance": {
                    "value": "consonant-rich",
                    "source": {
                        "module": "starwinds_analysis.pipelines.dummy_pipeline",
                        "function": "name_profile_payload",
                        "line": 37,
                    },
                },
            },
            "beta.plt": {
                "meta": {
                    "input_file": "/tmp/beta.plt",
                    "start_time_utc": "2026-01-01T00:00:02Z",
                    "end_time_utc": "2026-01-01T00:00:03Z",
                },
                "letter_counts": {
                    "value": {"vowels": 2, "consonants": 2},
                    "source": {
                        "module": "starwinds_analysis.pipelines.dummy_pipeline",
                        "function": "name_letter_counts",
                        "line": 20,
                    },
                },
            },
        },
    }
    path = tmp_path / "sw-pipe.dummy.processed.json"
    path.write_text(json.dumps(payload, indent=2))
    return path


def test_sw_pipe_results_lists_fields(tmp_path, capsys):
    path = _write_state(tmp_path)
    code = main(["--state", str(path), "--list-fields"])
    output = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert code == 0
    assert output == ["letter_counts", "name_dominance"]


def test_sw_pipe_results_lists_files(tmp_path, capsys):
    path = _write_state(tmp_path)
    code = main(["--state", str(path), "--list-files"])
    output = [line.strip() for line in capsys.readouterr().out.splitlines() if line.strip()]
    assert code == 0
    assert output == ["alpha.plt", "beta.plt"]


def test_sw_pipe_results_field_query_value_only(tmp_path, capsys):
    path = _write_state(tmp_path)
    code = main(["--state", str(path), "--field", "letter_count"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data == {
        "alpha.plt": {"vowels": 2, "consonants": 3},
        "beta.plt": {"vowels": 2, "consonants": 2},
    }


@pytest.mark.design_lockin
def test_sw_pipe_results_field_query_with_source(tmp_path, capsys):
    path = _write_state(tmp_path)
    code = main(["--state", str(path), "--field", "letter_counts", "--file", "alpha.plt", "--with-source"])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert data["value"] == {"vowels": 2, "consonants": 3}
    assert data["source"]["function"] == "name_letter_counts"


def test_sw_pipe_results_default_dump_all(tmp_path, capsys):
    path = _write_state(tmp_path)
    code = main(["--state", str(path)])
    data = json.loads(capsys.readouterr().out)
    assert code == 0
    assert "computed_results" in data
    assert "processed_files" in data
