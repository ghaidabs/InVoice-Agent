import pytest
from agent.normalizer import normalize_invoice_number


def test_clean_invoice_number_roundtrips():
    assert normalize_invoice_number("ACME_F202606-001", "ACME") == "ACME_F202606-001"


def test_sloppy_formatting_still_normalizes():
    # spaces/dashes are noise, digits are all present and in order
    assert normalize_invoice_number("acme f 202606 001", "ACME") == "ACME_F202606-001"


def test_missing_digits_returns_none():
    # only 7 digits present, not 9 — must not guess
    assert normalize_invoice_number("ACME_F2026-001", "ACME") is None


def test_none_input_returns_none():
    assert normalize_invoice_number(None, "ACME") is None


def test_empty_slug_returns_none():
    assert normalize_invoice_number("ACME_F202606-001", "") is None


def test_slug_not_present_falls_back_to_whole_string_digits():
    # current behavior: if slug isn't found, idx == -1, so it scans the
    # entire string for digits. Document/lock in this behavior explicitly
    # so a future change to it is a deliberate decision, not an accident.
    result = normalize_invoice_number("NOVA_F202606-001", "ACME")
    assert result == "ACME_F202606-001"  # slug in output is ACME's, not NOVA's — worth flagging


def test_wrong_company_slug_produces_wrong_but_valid_looking_number():
    # This is a real risk: passing NOVA's invoice text with ACME's slug
    # silently produces a plausible-looking ACME number instead of failing.
    # This test exists to make you decide if that's acceptable.
    result = normalize_invoice_number("NOVA_F202606-001", "ACME")
    assert result is not None  # currently succeeds — flag if you want this to instead return None