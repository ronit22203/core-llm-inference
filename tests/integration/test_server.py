"""Integration test stub for a live server (skipped when server unavailable)."""

import pytest
from src.core.client import check_server, list_models

SERVER_URL = "http://localhost:30000"


@pytest.fixture(scope="module")
def live_server():
    if not check_server(SERVER_URL, timeout=2.0):
        pytest.skip("SGLang server not running — skipping integration tests")
    return SERVER_URL


def test_server_health(live_server):
    assert check_server(live_server)


def test_list_models(live_server):
    models = list_models(live_server)
    assert isinstance(models, list)
    assert len(models) > 0
