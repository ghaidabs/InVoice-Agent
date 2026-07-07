import pytest
from mock.db import (
    get_client_by_id, search_clients_by_name,
    get_product_by_id, search_products_by_name,
    get_invoice_by_number, get_invoices_by_client_name,
    add_line_item,
)

# Known fixture data from mock/clients.json, mock/products.json, mock/invoices.json:
# company_id=1 (ACME): clients 1,2 (Ahmed Salem, Sarra Trabelsi); products 1,2
# company_id=2 (NOVA):  clients 3,4 (Karim Mzoughi, Nour Gharbi); products 3,4,5
# invoice ACME_F202605-001 belongs to company_id=1
# invoice NOVA_F202606-001 belongs to company_id=2


def test_get_client_by_id_wrong_company_returns_none():
    # client 3 (Karim) belongs to company 2 — requesting under company 1 must fail
    assert get_client_by_id(3, company_id=1) is None


def test_get_client_by_id_correct_company_returns_client():
    assert get_client_by_id(1, company_id=1) is not None


def test_search_clients_by_name_does_not_cross_tenant():
    # searching for a NOVA client's name while scoped to ACME must return nothing
    results = search_clients_by_name("Karim", company_id=1)
    assert results == []


def test_search_clients_by_name_partial_match_stays_scoped():
    results = search_clients_by_name("a", company_id=1)  # matches many first/last names
    assert all(c["company_id"] == 1 for c in results)


def test_get_product_by_id_wrong_company_returns_none():
    assert get_product_by_id(3, company_id=1) is None  # product 3 is NOVA's


def test_search_products_by_name_scoped():
    results = search_products_by_name("Audit", company_id=1)  # ACME has no "Audit" product
    assert results == []
    results_nova = search_products_by_name("Audit", company_id=2)
    assert len(results_nova) >= 1
    assert all(p["company_id"] == 2 for p in results_nova)


def test_get_invoice_by_number_wrong_company_returns_none():
    # this invoice exists, but belongs to company_id=1, not 2
    assert get_invoice_by_number("ACME_F202605-001", company_id=2) is None


def test_get_invoice_by_number_correct_company_returns_invoice():
    inv = get_invoice_by_number("ACME_F202605-001", company_id=1)
    assert inv is not None
    assert inv["company_id"] == 1


def test_get_invoices_by_client_name_scoped():
    # "Nour" is a NOVA client — must not show up when scoped to ACME
    results = get_invoices_by_client_name("Nour", company_id=1)
    assert results == []


def test_get_invoices_by_client_name_empty_string_returns_empty():
    assert get_invoices_by_client_name("", company_id=1) == []
    assert get_invoices_by_client_name("   ", company_id=1) == []


def test_add_line_item_wrong_company_fails_silently_no_mutation():
    # Attempt to add a line item to a NOVA invoice while scoped as ACME (company_id=1)
    before = get_invoice_by_number("NOVA_F202606-001", company_id=2)
    before_len = len(before["articles"])

    result = add_line_item(
        invoice_number="NOVA_F202606-001",
        company_id=1,  # wrong tenant
        designation="Should not be added",
        quantity=1,
        unit_price=100.0,
        unit="piece",
    )
    assert result is None

    after = get_invoice_by_number("NOVA_F202606-001", company_id=2)
    assert len(after["articles"]) == before_len  # confirm no mutation happened despite the wrong-tenant call


def test_add_line_item_nonexistent_invoice_returns_none():
    result = add_line_item(
        invoice_number="ACME_F209912-999",
        company_id=1,
        designation="Ghost item",
        quantity=1,
        unit_price=10.0,
        unit="piece",
    )
    assert result is None