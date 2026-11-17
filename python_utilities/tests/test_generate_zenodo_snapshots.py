"""Tests for the TSML Zenodo metadata snapshot generator."""

import json
from unittest import mock

import pytest
from jsonschema import Draft202012Validator, FormatChecker

from python_utilities import generate_zenodo_snapshots as snapshots


def make_file(key="Example.ts", size=10):
    """Create the subset of a Zenodo file used by the utility."""
    return {
        "key": key,
        "size": size,
        "checksum": "md5:0123456789abcdef0123456789abcdef",
        "links": {"self": f"https://zenodo.org/api/files/{key}"},
    }


def make_record(record_id, concept_id, created, title="Example", files=None):
    """Create the subset of a Zenodo record used by the utility."""
    if files is None:
        files = [make_file(f"{title}.ts")]
    return {
        "id": record_id,
        "conceptrecid": concept_id,
        "created": created,
        "metadata": {"title": title},
        "links": {"self_html": f"https://zenodo.org/records/{record_id}"},
        "files": files,
    }


def test_builds_and_validates_version_file_and_summary_snapshots():
    """Build normalized snapshots with complete version and file coverage."""
    records = [
        make_record(1, "100", "2024-01-01", "Versioned"),
        make_record(2, "100", "2025-01-01", "Versioned"),
        make_record(
            3,
            "200",
            "2026-01-01",
            "Single",
            [make_file("Single_TEST.ts"), make_file("Single_TRAIN.ts")],
        ),
    ]
    current, _ = snapshots.record_lists.split_record_versions(records)

    versions = snapshots.build_version_rows(records, current)
    files = snapshots.build_file_rows(records)
    retrieved_at = "2026-08-11T15:30:00Z"
    snapshot = snapshots.build_snapshot(versions, files, retrieved_at)

    snapshots.validate_snapshot(versions, files, snapshot)
    assert [row["record_id"] for row in versions] == [1, 2, 3]
    assert [row["record_id"] for row in versions if row["is_current"]] == [2, 3]
    assert snapshot == {
        "community": "tsml",
        "retrieved_at": retrieved_at,
        "version_count": 3,
        "current_record_count": 2,
        "previous_version_count": 1,
        "file_count": 4,
    }


def test_committed_snapshots_match_their_json_schemas():
    """Validate every committed snapshot row and its cross-file integrity."""
    versions = [
        json.loads(line)
        for line in snapshots.VERSIONS_SNAPSHOT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    files = [
        json.loads(line)
        for line in snapshots.FILES_SNAPSHOT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    snapshot_rows = [
        json.loads(line)
        for line in snapshots.SNAPSHOT_PATH.read_text(
            encoding="utf-8"
        ).splitlines()
    ]
    assert snapshot_rows
    schema_values = [
        (snapshots.VERSION_SCHEMA_PATH, versions),
        (snapshots.FILE_SCHEMA_PATH, files),
        (snapshots.SNAPSHOT_SCHEMA_PATH, snapshot_rows),
    ]

    for schema_path, values in schema_values:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        validator = Draft202012Validator(
            schema,
            format_checker=FormatChecker(),
        )
        for value in values:
            validator.validate(value)

    snapshots.validate_snapshot(versions, files, snapshot_rows[-1])


@pytest.mark.parametrize(
    ("file_row", "message"),
    [
        (
            {
                "concept_id": 100,
                "record_id": 999,
                "key": "orphan.ts",
            },
            "unknown record ID",
        ),
        (
            {
                "concept_id": 100,
                "record_id": 1,
                "key": "Example.ts",
            },
            "Duplicate file",
        ),
    ],
)
def test_rejects_invalid_relationships_between_snapshots(file_row, message):
    """Reject orphaned and duplicate file references across snapshots."""
    records = [make_record(1, "100", "2026-01-01")]
    versions = snapshots.build_version_rows(records, records)
    files = snapshots.build_file_rows(records)
    files.append(file_row)
    snapshot = snapshots.build_snapshot(
        versions,
        files,
        "2026-08-11T15:30:00Z",
    )

    with pytest.raises(ValueError, match=message):
        snapshots.validate_snapshot(versions, files, snapshot)


def test_writes_deterministic_jsonl_without_leaving_a_temporary_file(tmp_path):
    """Write stable JSONL formatting and atomically replace the destination."""
    jsonl_path = tmp_path / "rows.jsonl"

    snapshots.write_jsonl(jsonl_path, [{"second": 2, "first": 1}])

    assert jsonl_path.read_text(encoding="utf-8") == (
        '{"first": 1, "second": 2}\n'
    )
    assert not (tmp_path / ".rows.jsonl.tmp").exists()


def test_snapshot_history_compares_only_the_latest_counts(tmp_path):
    """Ignore metadata changes unless the latest set of counts has changed."""
    path = tmp_path / "snapshot.jsonl"
    records = [make_record(1, "100", "2026-01-01")]
    versions = snapshots.build_version_rows(records, records)
    files = snapshots.build_file_rows(records)
    first = snapshots.build_snapshot(
        versions,
        files,
        "2026-08-11T15:30:00Z",
    )
    repeated = snapshots.build_snapshot(
        versions,
        files,
        "2026-08-12T15:30:00Z",
    )
    snapshots.write_jsonl(path, [first])

    assert snapshots.snapshot_counts_have_changed(path, repeated) is False
    repeated["file_count"] += 1
    assert snapshots.snapshot_counts_have_changed(path, repeated) is True


def test_main_uses_only_anonymous_requests(monkeypatch):
    """Generate snapshots without resolving a production credential."""
    record = make_record(1, "100", "2026-01-01")
    opener = mock.Mock()
    load_token = mock.Mock()
    create_opener = mock.Mock(return_value=opener)
    fetch_records = mock.Mock(return_value=[record])
    append_jsonl = mock.Mock()
    monkeypatch.setattr(snapshots, "confirm_run", mock.Mock(return_value=True))
    monkeypatch.setattr(snapshots.record_lists, "load_token", load_token)
    monkeypatch.setattr(snapshots.record_lists, "create_opener", create_opener)
    monkeypatch.setattr(snapshots.record_lists, "fetch_records", fetch_records)
    write_jsonl = mock.Mock()
    monkeypatch.setattr(snapshots, "write_jsonl", write_jsonl)
    monkeypatch.setattr(snapshots, "append_jsonl", append_jsonl)
    monkeypatch.setattr(
        snapshots,
        "snapshot_counts_have_changed",
        mock.Mock(return_value=True),
    )
    monkeypatch.setattr(
        snapshots,
        "utc_now",
        mock.Mock(return_value="2026-08-11T15:30:00Z"),
    )

    result = snapshots.main()

    assert result == 0
    load_token.assert_not_called()
    create_opener.assert_called_once_with(None)
    fetch_records.assert_called_once_with(opener, page_size=25)
    assert write_jsonl.call_count == 2
    append_jsonl.assert_called_once()


def test_main_does_not_append_unchanged_snapshot_history(tmp_path, monkeypatch):
    """Keep history unchanged when regenerated JSONL content is identical."""
    record = make_record(1, "100", "2026-01-01")
    versions = snapshots.build_version_rows([record], [record])
    files = snapshots.build_file_rows([record])
    history = snapshots.build_snapshot(
        versions,
        files,
        "2026-08-10T15:30:00Z",
    )
    versions_path = tmp_path / "zenodo_versions.jsonl"
    files_path = tmp_path / "zenodo_files.jsonl"
    history_path = tmp_path / "snapshot.jsonl"
    snapshots.write_jsonl(versions_path, versions)
    snapshots.write_jsonl(files_path, files)
    snapshots.write_jsonl(history_path, [history])
    monkeypatch.setattr(snapshots, "MANIFEST_DIR", tmp_path)
    monkeypatch.setattr(snapshots.record_lists, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(snapshots, "VERSIONS_SNAPSHOT_PATH", versions_path)
    monkeypatch.setattr(snapshots, "FILES_SNAPSHOT_PATH", files_path)
    monkeypatch.setattr(snapshots, "SNAPSHOT_PATH", history_path)
    monkeypatch.setattr(snapshots, "confirm_run", mock.Mock(return_value=True))
    monkeypatch.setattr(snapshots.record_lists, "create_opener", mock.Mock())
    monkeypatch.setattr(
        snapshots.record_lists,
        "fetch_records",
        mock.Mock(return_value=[record]),
    )

    result = snapshots.main()

    assert result == 0
    assert history_path.read_text(encoding="utf-8") == snapshots.jsonl_contents(
        [history]
    )
