import pytest
from schemas.intents import (
    IntentType, CreateInvoiceIntent, LookupClientIntent, CheckInvoiceIntent,
    AddLineItemIntent, CheckPaymentIntent, ConfirmIntent, CancelIntent,
    UnknownIntent, LineItemInput,
)
from agent.session import SessionState, UserContext
from agent.dispatcher import dispatch
from api.mock_saas_client import MockSaasClient


@pytest.fixture
def acme_session():
    ctx = UserContext(user_id=1, company_id=1, role="ADMIN", company_slug="ACME")
    return SessionState(user=ctx)


@pytest.fixture
def acme_saas():
    return MockSaasClient(company_id=1, user_id=1, company_slug="ACME")


@pytest.fixture
def nova_saas():
    return MockSaasClient(company_id=2, user_id=2, company_slug="NOVA")


# ---------- confidence gating ----------

@pytest.mark.asyncio
async def test_low_confidence_intent_short_circuits_regardless_of_type(acme_saas, acme_session):
    intent = ConfirmIntent(intent=IntentType.CONFIRM, confidence=0.3, raw_transcript="confirme?")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False
    assert resp.clarification_question is not None
    assert acme_session.confirmation_pending is False


@pytest.mark.asyncio
async def test_low_confidence_create_invoice_does_not_create_draft(acme_saas, acme_session):
    intent = CreateInvoiceIntent(
        intent=IntentType.CREATE_INVOICE,
        confidence=0.4, raw_transcript="...",
        client_name="Ahmed Salem",
        articles=[LineItemInput(product_name="Consultation IT", quantity=1, unit_price=500.0)],
    )
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False
    assert acme_session.pending_draft is None


# ---------- confirm / cancel flow ----------

@pytest.mark.asyncio
async def test_confirm_with_nothing_pending_returns_error_not_crash(acme_saas, acme_session):
    intent = ConfirmIntent(intent=IntentType.CONFIRM, confidence=0.95, raw_transcript="confirme")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False
    assert resp.error == "no_pending_confirmation"


@pytest.mark.asyncio
async def test_cancel_with_nothing_pending_returns_error_not_crash(acme_saas, acme_session):
    intent = CancelIntent(intent=IntentType.CANCEL, confidence=0.95, raw_transcript="annule")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False
    assert resp.error == "no_pending_confirmation"


@pytest.mark.asyncio
async def test_create_invoice_then_confirm_full_flow(acme_saas, acme_session):
    create_intent = CreateInvoiceIntent(
        intent=IntentType.CREATE_INVOICE,
        confidence=0.95, raw_transcript="facture Ahmed Salem 2 jours consultation IT 500 dinars",
        client_name="Ahmed Salem",
        articles=[LineItemInput(product_name="Consultation IT", quantity=2, unit_price=500.0)],
    )
    draft_resp = await dispatch(create_intent, acme_saas, acme_session)

    assert draft_resp.requires_confirmation is True
    assert draft_resp.draft_payload is not None
    assert acme_session.pending_draft is not None
    assert acme_session.confirmation_pending is True

    confirm_intent = ConfirmIntent(intent=IntentType.CONFIRM, confidence=0.95, raw_transcript="confirme")
    final_resp = await dispatch(confirm_intent, acme_saas, acme_session)

    assert final_resp.success is True
    assert final_resp.display_data["status"] == "validated"
    assert final_resp.display_data["invoice_number"] is not None
    assert acme_session.pending_draft is None
    assert acme_session.confirmation_pending is False


@pytest.mark.asyncio
async def test_create_invoice_then_cancel_discards_draft(acme_saas, acme_session):
    create_intent = CreateInvoiceIntent(
        intent=IntentType.CREATE_INVOICE,
        confidence=0.95, raw_transcript="...",
        client_name="Ahmed Salem",
        articles=[LineItemInput(product_name="Consultation IT", quantity=1, unit_price=500.0)],
    )
    await dispatch(create_intent, acme_saas, acme_session)
    assert acme_session.pending_draft is not None

    cancel_intent = CancelIntent(intent=IntentType.CANCEL, confidence=0.95, raw_transcript="annule")
    resp = await dispatch(cancel_intent, acme_saas, acme_session)

    assert resp.success is True
    assert acme_session.pending_draft is None
    assert acme_session.confirmation_pending is False


# ---------- missing unit_price clarification ----------

@pytest.mark.asyncio
async def test_create_invoice_missing_price_asks_clarification(acme_saas, acme_session):
    intent = CreateInvoiceIntent(
        intent=IntentType.CREATE_INVOICE,
        confidence=0.95, raw_transcript="facture Ahmed Salem pour consultation IT",
        client_name="Ahmed Salem",
        articles=[LineItemInput(product_name="Consultation IT", quantity=1, unit_price=None)],
    )
    resp = await dispatch(intent, acme_saas, acme_session)

    assert resp.success is False
    assert resp.clarification_question is not None
    assert acme_session.awaiting_field == "unit_price"
    assert acme_session.pending_transcript == intent.raw_transcript
    assert acme_session.pending_draft is None


@pytest.mark.asyncio
async def test_add_line_item_missing_price_asks_clarification(acme_saas, acme_session):
    intent = AddLineItemIntent(
        intent=IntentType.ADD_LINE_ITEM,
        confidence=0.95, raw_transcript="ajoute 2 jours de dev web",
        invoice_number="ACME_F202605-001",
        product_name="Développement web", quantity=2, unit_price=None,
    )
    resp = await dispatch(intent, acme_saas, acme_session)

    assert resp.success is False
    assert acme_session.awaiting_field == "unit_price"


# ---------- client ambiguity ----------

@pytest.mark.asyncio
async def test_lookup_client_no_match(acme_saas, acme_session):
    intent = LookupClientIntent(intent=IntentType.LOOKUP_CLIENT, confidence=0.95, raw_transcript="...", client_name="Nonexistent Person")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False
    assert resp.error == "client_not_found"


@pytest.mark.asyncio
async def test_lookup_client_exact_single_match(acme_saas, acme_session):
    intent = LookupClientIntent(intent=IntentType.LOOKUP_CLIENT, confidence=0.95, raw_transcript="...", client_name="Ahmed Salem")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is True
    assert resp.display_data["first_name"] == "Ahmed"


# ---------- invoice / payment lookups ----------

@pytest.mark.asyncio
async def test_check_invoice_no_target_asks_clarification(acme_saas, acme_session):
    intent = CheckInvoiceIntent(intent=IntentType.CHECK_INVOICE, confidence=0.95, raw_transcript="check my invoice")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False
    assert resp.clarification_question is not None


@pytest.mark.asyncio
async def test_check_invoice_by_number_found(acme_saas, acme_session):
    intent = CheckInvoiceIntent(intent=IntentType.CHECK_INVOICE, confidence=0.95, raw_transcript="...", invoice_number="ACME_F202605-001")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is True
    assert resp.display_data[0]["invoice_number"] == "ACME_F202605-001"


@pytest.mark.asyncio
async def test_check_invoice_cross_tenant_not_found(acme_saas, acme_session):
    intent = CheckInvoiceIntent(intent=IntentType.CHECK_INVOICE, confidence=0.95, raw_transcript="...", invoice_number="NOVA_F202606-001")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False
    assert resp.error == "invoice_not_found"


@pytest.mark.asyncio
async def test_check_invoice_cross_tenant_via_nova_client(nova_saas):
    ctx = UserContext(user_id=2, company_id=2, role="ADMIN", company_slug="NOVA")
    session = SessionState(user=ctx)
    intent = CheckInvoiceIntent(intent=IntentType.CHECK_INVOICE, confidence=0.95, raw_transcript="...", invoice_number="ACME_F202605-001")
    resp = await dispatch(intent, nova_saas, session)
    assert resp.success is False
    assert resp.error == "invoice_not_found"


@pytest.mark.asyncio
async def test_check_payment_no_target_asks_clarification(acme_saas, acme_session):
    intent = CheckPaymentIntent(intent=IntentType.CHECK_PAYMENT, confidence=0.95, raw_transcript="is it paid")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False


@pytest.mark.asyncio
async def test_check_payment_found(acme_saas, acme_session):
    intent = CheckPaymentIntent(intent=IntentType.CHECK_PAYMENT, confidence=0.95, raw_transcript="...", invoice_number="ACME_F202605-001")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is True
    assert "paid" in resp.spoken_text.lower()


# ---------- add line item on nonexistent invoice ----------

@pytest.mark.asyncio
async def test_add_line_item_nonexistent_invoice(acme_saas, acme_session):
    intent = AddLineItemIntent(
        intent=IntentType.ADD_LINE_ITEM,
        confidence=0.95, raw_transcript="...",
        invoice_number="ACME_F209912-999",
        product_name="Consultation IT", quantity=1, unit_price=100.0,
    )
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False
    assert resp.error == "invoice_not_found"


# ---------- unknown intent ----------

@pytest.mark.asyncio
async def test_unknown_intent_returns_graceful_error(acme_saas, acme_session):
    intent = UnknownIntent(intent=IntentType.UNKNOWN, confidence=0.95, raw_transcript="asdkjfh", raw_text="asdkjfh")
    resp = await dispatch(intent, acme_saas, acme_session)
    assert resp.success is False
    assert resp.error == "unknown_intent"