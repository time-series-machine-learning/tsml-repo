"""Tests for the shared Zenodo maintenance helpers."""

from unittest import mock
from urllib.request import HTTPRedirectHandler, Request

import pytest

from python_utilities import zenodo_maintenance as zm


def make_deposition(
    tags=("time series",),
    state="done",
    submitted=True,
    title="Example",
):
    """Create a synthetic Zenodo deposition payload matching real state fields."""
    return {
        "metadata": {
            "title": title,
            "creators": [{"name": "TSML Archive Maintainers"}],
            "keywords": list(tags),
        },
        "state": state,
        "submitted": submitted,
    }


def make_change(old_tags=("time series",), new_tags=("rehab", "time series")):
    """Create a planned tag update for ``apply_record_change`` tests."""

    def mutate(draft_metadata):
        draft_metadata["keywords"] = list(new_tags)

    return zm.PlannedChange(
        record_id="123",
        title="Example",
        old_value=old_tags,
        new_value=new_tags,
        mutate_metadata=mutate,
        extract_value=zm.extract_record_tags,
    )


def test_resolves_file_and_explicit_records_in_order(tmp_path):
    """Support title-only, ``title: ID``, numeric IDs, comments, and ``--record``."""
    index_path = tmp_path / "current-versions"
    index_path.write_text("Alpha: 1\nBeta: 2\n", encoding="utf-8")
    record_file = tmp_path / "targets"
    record_file.write_text(
        "# selected\nAlpha\nDataset: with colon: 42\n7\n",
        encoding="utf-8",
    )

    refs = zm.resolve_record_references(
        record_file=record_file,
        explicit_records=["Beta"],
        index_path=index_path,
    )

    assert refs == [
        zm.RecordReference("1", "Alpha"),
        zm.RecordReference("42", "Dataset: with colon"),
        zm.RecordReference("7"),
        zm.RecordReference("2", "Beta"),
    ]


def test_rejects_missing_or_duplicate_targets(tmp_path):
    """Reject missing index entries, empty invocations, and duplicate IDs/titles."""
    index_path = tmp_path / "current-versions"
    index_path.write_text("Alpha: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Specify a record_file"):
        zm.resolve_record_references(index_path=index_path)

    with pytest.raises(ValueError, match="Missing"):
        zm.resolve_record_references(
            explicit_records=["Missing"],
            index_path=index_path,
        )

    with pytest.raises(ValueError, match="repeats record ID 1"):
        zm.resolve_record_references(
            explicit_records=["Alpha", "1"],
            index_path=index_path,
        )


def test_enforces_allowed_api_roots_and_credential_separation(tmp_path, monkeypatch):
    """Prevent HTTP URLs, unknown hosts, or mixing production/sandbox tokens."""
    prod_token = tmp_path / ".zenodo_token"
    sandbox_token = tmp_path / ".zenodo_sandbox_token"
    monkeypatch.setattr(zm, "PRODUCTION_TOKEN_PATH", prod_token)
    monkeypatch.setattr(zm, "SANDBOX_TOKEN_PATH", sandbox_token)

    with pytest.raises(zm.ZenodoMaintenanceError, match="Unsupported Zenodo API root"):
        zm.normalize_api_root("http://zenodo.org/api")
    with pytest.raises(zm.ZenodoMaintenanceError, match="Unsupported Zenodo API root"):
        zm.normalize_api_root("https://example.invalid/api")

    assert zm.resolve_token_path(zm.PRODUCTION_API_ROOT) == prod_token
    assert zm.resolve_token_path(zm.SANDBOX_API_ROOT) == sandbox_token

    with pytest.raises(zm.ZenodoMaintenanceError, match="production .zenodo_token"):
        zm.resolve_token_path(zm.SANDBOX_API_ROOT, prod_token)
    with pytest.raises(zm.ZenodoMaintenanceError, match=".zenodo_sandbox_token"):
        zm.resolve_token_path(zm.PRODUCTION_API_ROOT, sandbox_token)


def test_authenticated_client_blocks_redirects():
    """Never follow HTTP redirects when carrying a bearer token."""
    request = Request(f"{zm.PRODUCTION_API_ROOT}/records/123")
    destination = "https://example.invalid/records/123"

    anon_client = zm.ZenodoClient(None, zm.PRODUCTION_API_ROOT)
    auth_client = zm.ZenodoClient("dummy-token", zm.PRODUCTION_API_ROOT)

    anon_handler = next(
        h for h in anon_client.opener.handlers if isinstance(h, HTTPRedirectHandler)
    )
    auth_handler = next(
        h for h in auth_client.opener.handlers if isinstance(h, HTTPRedirectHandler)
    )

    assert (
        anon_handler.redirect_request(request, None, 302, "Found", {}, destination)
        is not None
    )
    assert (
        auth_handler.redirect_request(request, None, 302, "Found", {}, destination)
        is None
    )


def test_preview_never_loads_credentials_or_opens_edit_sessions(monkeypatch, capsys):
    """Default preview uses only unauthenticated public GET requests."""
    read_client = mock.Mock()
    read_client.get_public_record_metadata.return_value = (
        "Example",
        {"title": "Example", "keywords": ["time series"]},
    )
    client_factory = mock.Mock(return_value=read_client)
    load_token = mock.Mock()
    monkeypatch.setattr(zm, "load_required_token", load_token)

    result = zm.run_metadata_update(
        [zm.RecordReference("123", "Example")],
        lambda rid, title, md: make_change(),
        apply=False,
        client_factory=client_factory,
        noun="tag",
    )

    assert result == 0
    load_token.assert_not_called()
    client_factory.assert_called_once_with(None, zm.PRODUCTION_API_ROOT)
    read_client.request_json.assert_not_called()
    output = capsys.readouterr().out
    assert "CHANGE\t123\tExample" in output
    assert "No edit sessions were opened" in output


def test_apply_cancels_before_loading_token_when_not_confirmed(
    tmp_path, monkeypatch, capsys
):
    """Cancelling at the APPLY prompt must not read the token file or mutate."""
    read_client = mock.Mock()
    read_client.get_public_record_metadata.return_value = (
        "Example",
        {"title": "Example", "keywords": ["time series"]},
    )
    load_token = mock.Mock()
    monkeypatch.setattr(zm, "load_required_token", load_token)

    result = zm.run_metadata_update(
        [zm.RecordReference("123", "Example")],
        lambda rid, title, md: make_change(),
        apply=True,
        token_path=tmp_path / "custom-token",
        client_factory=mock.Mock(return_value=read_client),
        input_fn=lambda _: "no",
    )

    assert result == 1
    load_token.assert_not_called()
    assert "Cancelled" in capsys.readouterr().out


def test_apply_skips_token_and_mutations_when_nothing_changed(monkeypatch, capsys):
    """Exit cleanly without unlocking records when all targets already match."""
    read_client = mock.Mock()
    read_client.get_public_record_metadata.return_value = (
        "Example",
        {"title": "Example", "keywords": ["rehab", "time series"]},
    )
    load_token = mock.Mock()
    monkeypatch.setattr(zm, "load_required_token", load_token)

    unchanged = make_change(
        old_tags=("rehab", "time series"),
        new_tags=("rehab", "time series"),
    )
    result = zm.run_metadata_update(
        [zm.RecordReference("123", "Example")],
        lambda rid, title, md: unchanged,
        apply=True,
        client_factory=mock.Mock(return_value=read_client),
    )

    assert result == 0
    load_token.assert_not_called()
    assert "Nothing to apply" in capsys.readouterr().out


def test_discards_open_inprogress_edit_when_update_or_edit_response_fails():
    """Discard ``state='inprogress'`` drafts (including HTTP 204 empty responses)."""
    client = mock.Mock()
    client.request_json.side_effect = [
        make_deposition(["time series"], state="done", submitted=True),
        RuntimeError("edit response lost"),
        make_deposition(["time series"], state="inprogress", submitted=True),
        None,  # Zenodo-RDM POST /actions/discard returns 204 No Content
        make_deposition(["time series"], state="done", submitted=True),
    ]

    with pytest.raises(RuntimeError, match="edit response lost"):
        zm.apply_record_change(client, make_change())

    assert client.request_json.call_args_list[-3:] == [
        mock.call("GET", "deposit/depositions/123"),
        mock.call("POST", "deposit/depositions/123/actions/discard"),
        mock.call("GET", "deposit/depositions/123"),
    ]


def test_discard_existing_edit_recovers_unedited_session_and_refuses_edited_draft():
    """Reset an unedited ``inprogress`` session only when ``--discard-existing-edit`` is set."""
    client = mock.Mock()
    client.request_json.return_value = make_deposition(
        ["time series"], state="inprogress", submitted=True
    )
    with pytest.raises(zm.ZenodoMaintenanceError, match="--discard-existing-edit"):
        zm.apply_record_change(client, make_change(), discard_existing_edit=False)

    # Refuse even with --discard-existing-edit if the draft has modified metadata
    client.request_json.return_value = make_deposition(
        ["unsaved-edit"], state="inprogress", submitted=True
    )
    with pytest.raises(zm.ZenodoMaintenanceError, match="modified metadata"):
        zm.apply_record_change(client, make_change(), discard_existing_edit=True)

    # Recover cleanly when draft matches published pre-image and discard returns 204 None
    client = mock.Mock()
    inprogress = make_deposition(["time series"], state="inprogress", submitted=True)
    locked = make_deposition(["time series"], state="done", submitted=True)
    updated_draft = make_deposition(
        ["rehab", "time series"], state="inprogress", submitted=True
    )
    published = make_deposition(
        ["rehab", "time series"], state="done", submitted=True
    )
    client.request_json.side_effect = [
        inprogress,  # initial GET sees open edit
        inprogress,  # discard_open_edit GET
        None,  # discard_open_edit POST discard -> 204 No Content
        locked,  # discard_open_edit verification GET -> done
        locked,  # re-fetched locked GET
        inprogress,  # POST actions/edit
        inprogress,  # GET draft
        updated_draft,  # PUT metadata
        published,  # POST actions/publish -> done
    ]
    zm.apply_record_change(client, make_change(), discard_existing_edit=True)
    assert client.request_json.call_args_list[2] == mock.call(
        "POST", "deposit/depositions/123/actions/discard"
    )
    assert client.request_json.call_args_list[-1] == mock.call(
        "POST", "deposit/depositions/123/actions/publish"
    )


@pytest.mark.parametrize(
    ("deposition", "message"),
    [
        (make_deposition(["time series"], state="unsubmitted", submitted=False), "not a locked published record"),
        (make_deposition(["newer"], state="done", submitted=True), "changed after the preview"),
    ],
)
def test_refuses_unsubmitted_draft_or_stale_preview(deposition, message):
    """Refuse to touch unsubmitted drafts or metadata modified since preview."""
    client = mock.Mock()
    client.request_json.return_value = deposition

    with pytest.raises(zm.ZenodoMaintenanceError, match=message):
        zm.apply_record_change(client, make_change())

    client.request_json.assert_called_once_with("GET", "deposit/depositions/123")
