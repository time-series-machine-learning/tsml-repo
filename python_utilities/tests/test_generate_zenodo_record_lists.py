"""Tests for the TSML Zenodo record-list generator."""

import json
from io import BytesIO
from unittest import mock
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pytest

from python_utilities import generate_zenodo_record_lists as tsml_records


def make_record(record_id, concept_id, created, title="Example"):
    """Create the subset of a Zenodo record used by the utility."""
    return {
        "id": record_id,
        "conceptrecid": concept_id,
        "created": created,
        "metadata": {"title": title},
    }


def test_confirmation_summarizes_actions_and_accepts_enter():
    """Continue only after describing production reads and local replacements."""
    input_fn = mock.Mock(return_value="")

    confirmed = tsml_records.confirm_run(input_fn)

    assert confirmed is True
    input_fn.assert_called_once_with(
        "This reads production Zenodo and replaces the local record lists. "
        "Press Enter to continue: "
    )


@pytest.mark.parametrize("side_effect", ["cancel", EOFError()])
def test_confirmation_cancels_safely(side_effect):
    """Reject non-empty confirmation and non-interactive input."""
    input_fn = mock.Mock(
        side_effect=side_effect if isinstance(side_effect, BaseException) else None,
        return_value=side_effect if isinstance(side_effect, str) else None,
    )

    confirmed = tsml_records.confirm_run(input_fn)

    assert confirmed is False


def test_main_cancels_before_loading_credentials(monkeypatch):
    """Do not inspect credentials or make requests before confirmation."""
    confirmation = mock.Mock(return_value=False)
    load_token = mock.Mock()
    monkeypatch.setattr(tsml_records, "confirm_run", confirmation)
    monkeypatch.setattr(tsml_records, "load_token", load_token)

    result = tsml_records.main()

    assert result == 1
    confirmation.assert_called_once_with(None)
    load_token.assert_not_called()


def test_loads_token_from_repository_root(tmp_path, monkeypatch, capsys):
    """Verify a root token is loaded without exposing its value."""
    token_path = tmp_path / ".zenodo_token"
    token_path.write_text("test-token\n", encoding="utf-8")
    monkeypatch.setattr(tsml_records, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(tsml_records, "TOKEN_PATH", token_path)

    token = tsml_records.load_token()

    assert token == "test-token"
    assert "Zenodo token loaded from .zenodo_token." in capsys.readouterr().out


@pytest.mark.parametrize(
    ("contents", "expected_message"),
    [
        (None, "using unauthenticated requests"),
        ("", "token file is empty"),
    ],
)
def test_missing_or_empty_token_is_optional(
    tmp_path, monkeypatch, capsys, contents, expected_message
):
    """Verify missing or empty tokens use anonymous requests."""
    token_path = tmp_path / ".zenodo_token"
    if contents is not None:
        token_path.write_text(contents, encoding="utf-8")
    monkeypatch.setattr(tsml_records, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(tsml_records, "TOKEN_PATH", token_path)

    token = tsml_records.load_token()

    assert token is None
    assert expected_message in capsys.readouterr().out


def test_opener_has_user_agent_and_optional_authorization():
    """Verify the opener identifies itself and adds optional authentication."""
    authenticated_headers = dict(tsml_records.create_opener("test-token").addheaders)
    anonymous_headers = dict(tsml_records.create_opener(None).addheaders)

    assert authenticated_headers["User-Agent"] == tsml_records.USER_AGENT
    assert authenticated_headers["Authorization"] == "Bearer test-token"
    assert "Authorization" not in anonymous_headers


def test_get_page_builds_query_and_retries_transient_error(monkeypatch):
    """Verify page queries and retry handling for transient failures."""
    payload = {"hits": {"hits": []}}
    transient_error = HTTPError(
        tsml_records.API_URL,
        503,
        "Service unavailable",
        {"Retry-After": "0"},
        None,
    )
    opener = mock.Mock()
    opener.open.side_effect = [
        transient_error,
        BytesIO(json.dumps(payload).encode("utf-8")),
    ]
    sleep = mock.Mock()
    monkeypatch.setattr(tsml_records.time, "sleep", sleep)

    result = tsml_records.get_page(opener, page=2, page_size=25)

    assert result == payload
    sleep.assert_called_once_with(0)
    requested_url = opener.open.call_args_list[-1].args[0]
    query = parse_qs(urlparse(requested_url).query)
    assert query == {
        "all_versions": ["true"],
        "communities": ["tsml"],
        "page": ["2"],
        "size": ["25"],
        "sort": ["newest"],
    }


def test_fetch_records_stops_at_reported_total(monkeypatch):
    """Verify pagination stops after collecting the reported total."""
    first_record = make_record(1, "100", "2025-01-01")
    second_record = make_record(2, "200", "2026-01-01")
    pages = [
        {"hits": {"hits": [first_record], "total": 2}},
        {"hits": {"hits": [second_record], "total": 2}},
    ]
    opener = mock.Mock()
    get_page = mock.Mock(side_effect=pages)
    monkeypatch.setattr(tsml_records, "get_page", get_page)

    records = tsml_records.fetch_records(opener, page_size=25)

    assert records == [first_record, second_record]
    assert get_page.call_args_list == [
        mock.call(opener, 1, 25),
        mock.call(opener, 2, 25),
    ]


def test_fetch_records_rejects_incorrect_total(monkeypatch):
    """Verify incomplete pagination fails the reported-total check."""
    record = make_record(1, "100", "2025-01-01")
    pages = [
        {"hits": {"hits": [record], "total": 2}},
        {"hits": {"hits": [], "total": 2}},
    ]
    get_page = mock.Mock(side_effect=pages)
    monkeypatch.setattr(tsml_records, "get_page", get_page)

    with pytest.raises(ValueError, match="reported 2 records but returned 1"):
        tsml_records.fetch_records(mock.Mock(), page_size=25)


def test_splits_multiple_previous_versions():
    """Verify one current version is separated from every older version."""
    records = [
        make_record(1, "100", "2024-01-01", "Versioned"),
        make_record(4, "200", "2025-06-01", "Single"),
        make_record(3, "100", "2026-01-01", "Versioned"),
        make_record(2, "100", "2025-01-01", "Versioned"),
    ]

    current, previous = tsml_records.split_record_versions(records)

    assert [record["id"] for record in current] == [3, 4]
    assert [record["id"] for record in previous] == [2, 1]


@pytest.mark.parametrize(
    "record",
    [
        {"conceptrecid": "100", "metadata": {"title": "No ID"}},
        {"id": 1, "conceptrecid": "100", "metadata": {}},
    ],
)
def test_rejects_record_without_id_or_title(record):
    """Verify records without an ID or title are rejected."""
    with pytest.raises(ValueError, match="without an ID or title"):
        tsml_records.split_record_versions([record])


def test_writes_expected_record_list_without_leaving_temporary_file(tmp_path):
    """Verify record lists are written atomically in the expected format."""
    records = [
        make_record(2, "200", "2026-01-01", "Second"),
        make_record(1, "100", "2025-01-01", "First"),
    ]
    output_path = tmp_path / "current-versions"

    tsml_records.write_record_list(output_path, records)

    assert output_path.read_text(encoding="utf-8") == "Second: 2\nFirst: 1\n"
    assert not (tmp_path / ".current-versions.tmp").exists()


def test_main_rejects_duplicate_titles_before_writing(monkeypatch):
    """Verify duplicate current titles fail before any list is written."""
    records = [
        make_record(1, "100", "2024-01-01", "Duplicate"),
        make_record(2, "200", "2025-01-01", "Duplicate"),
    ]
    write_record_list = mock.Mock()
    monkeypatch.setattr(tsml_records, "confirm_run", mock.Mock(return_value=True))
    monkeypatch.setattr(tsml_records, "load_token", mock.Mock(return_value=None))
    monkeypatch.setattr(tsml_records, "create_opener", mock.Mock())
    monkeypatch.setattr(tsml_records, "fetch_records", mock.Mock(return_value=records))
    monkeypatch.setattr(tsml_records, "write_record_list", write_record_list)

    with pytest.raises(ValueError, match="Duplicate current-version titles"):
        tsml_records.main()

    write_record_list.assert_not_called()
