"""Shared pytest fixtures enforcing offline and credential-free test execution."""

import socket

import pytest

from python_utilities import generate_zenodo_record_lists, zenodo_maintenance


@pytest.fixture(autouse=True)
def isolate_credentials_and_network(tmp_path, monkeypatch):
    """Redirect default token resolution and block live network connections."""
    monkeypatch.setattr(
        generate_zenodo_record_lists,
        "TOKEN_PATH",
        tmp_path / ".missing_zenodo_token",
    )
    monkeypatch.setattr(
        zenodo_maintenance,
        "PRODUCTION_TOKEN_PATH",
        tmp_path / ".missing_zenodo_token",
    )
    monkeypatch.setattr(
        zenodo_maintenance,
        "SANDBOX_TOKEN_PATH",
        tmp_path / ".missing_zenodo_sandbox_token",
    )

    def blocked_network(*args, **kwargs):
        raise AssertionError("Unit and consistency tests must not make network calls.")

    monkeypatch.setattr(socket, "create_connection", blocked_network)
    monkeypatch.setattr(socket.socket, "connect", blocked_network)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked_network)
