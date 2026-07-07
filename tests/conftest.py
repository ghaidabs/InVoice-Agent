import pytest
import importlib


@pytest.fixture(autouse=True)
def reset_mock_db():
    """
    mock/db.py loads JSON into module-level dicts once at import time and
    mutates them in place (create_invoice, add_line_item, confirm_invoice_draft
    all do this). Without a reset, tests that create/modify invoices will leak
    state into later tests. Reload the module before each test so every test
    starts from the on-disk JSON fixtures.
    """
    import mock.db
    importlib.reload(mock.db)
    yield