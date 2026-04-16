"""Tests for the generic corpus importer.

Mocks the MemoryLayer so we don't load sbert — focused on the reader
and field-mapping logic, which is the actual surface area."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.import_corpus import (
    Record,
    _iter_csv,
    _iter_json,
    _iter_jsonl,
    _letta_to_records,
    _map_record,
    _mem0_to_records,
    _zep_to_records,
)


def test_iter_jsonl_yields_one_dict_per_line(tmp_path: Path) -> None:
    p = tmp_path / "x.jsonl"
    p.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    rows = list(_iter_jsonl(p))
    assert rows == [{"a": 1}, {"b": 2}]


def test_iter_json_unwraps_common_wrapper_keys(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"items": [{"k": 1}, {"k": 2}]}), encoding="utf-8")
    assert list(_iter_json(p)) == [{"k": 1}, {"k": 2}]

    p.write_text(json.dumps({"memories": [{"m": 1}]}), encoding="utf-8")
    assert list(_iter_json(p)) == [{"m": 1}]


def test_iter_json_handles_bare_list(tmp_path: Path) -> None:
    p = tmp_path / "x.json"
    p.write_text(json.dumps([{"a": 1}, {"a": 2}]), encoding="utf-8")
    assert list(_iter_json(p)) == [{"a": 1}, {"a": 2}]


def test_iter_csv_returns_dict_per_row(tmp_path: Path) -> None:
    p = tmp_path / "x.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "author"])
        w.writerow(["hi", "alex"])
        w.writerow(["bye", "bobbi"])
    rows = list(_iter_csv(p))
    assert rows == [
        {"text": "hi", "author": "alex"},
        {"text": "bye", "author": "bobbi"},
    ]


def test_map_record_uses_explicit_metadata_fields() -> None:
    row: dict[str, Any] = {
        "text": "  hello  ", "tag": "greet", "extra": "dropped"
    }
    rec = _map_record(row, text_field="text", metadata_fields=["tag"])
    assert rec == Record(text="hello", metadata={"tag": "greet"})


def test_map_record_defaults_all_non_text_fields_to_metadata() -> None:
    row = {"text": "t", "a": 1, "b": 2}
    rec = _map_record(row, text_field="text", metadata_fields=[])
    assert rec is not None
    assert rec.metadata == {"a": 1, "b": 2}


def test_map_record_skips_empty_or_missing_text() -> None:
    assert _map_record({"text": ""}, text_field="text", metadata_fields=[]) is None
    assert _map_record({"text": "   "}, text_field="text", metadata_fields=[]) is None
    assert _map_record({"other": "x"}, text_field="text", metadata_fields=[]) is None


def test_mem0_preset_maps_memory_and_user_id() -> None:
    rows = [
        {"memory": "Alex likes tea", "user_id": "alex", "created_at": "2026-01-01"},
        {"text": "Bobbi prefers coffee", "user_id": "bobbi"},
        {"memory": "", "user_id": "carl"},  # skipped
    ]
    recs = list(_mem0_to_records(iter(rows)))
    assert len(recs) == 2
    assert recs[0].text == "Alex likes tea"
    assert recs[0].metadata == {
        "user_id": "alex",
        "created_at": "2026-01-01",
        "source": "mem0",
    }
    assert recs[1].metadata["source"] == "mem0"


def test_letta_preset_merges_metadata_block() -> None:
    rows = [
        {
            "text": "archival entry",
            "metadata": {"topic": "work"},
            "created_at": "2026-01-02",
            "tags": ["note"],
        },
    ]
    recs = list(_letta_to_records(iter(rows)))
    assert len(recs) == 1
    assert recs[0].metadata == {
        "topic": "work",
        "created_at": "2026-01-02",
        "tags": ["note"],
        "source": "letta",
    }


def test_zep_preset_maps_role_and_content() -> None:
    rows = [
        {"role": "user", "content": "hi there", "created_at": "2026-01-03"},
        {"role": "assistant", "content": "hello!"},
        {"role": "system", "content": ""},  # skipped
    ]
    recs = list(_zep_to_records(iter(rows)))
    assert [r.text for r in recs] == ["hi there", "hello!"]
    assert recs[0].metadata["role"] == "user"
    assert recs[0].metadata["source"] == "zep"


def test_iter_json_rejects_unsupported_root(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text(json.dumps("a string"), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported JSON root"):
        list(_iter_json(p))
