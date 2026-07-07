from schemas.intents import (
    IntentType,
    CreateInvoiceIntent,
    LookupClientIntent,
    CheckInvoiceIntent,
    AddLineItemIntent,
    CheckPaymentIntent,
    ConfirmIntent,
    CancelIntent,
    UnknownIntent,
)
from schemas.responses import VoiceResponse
from agent.session import SessionState
from api.mock_saas_client import MockSaasClient
from templates.invoice import (
    build_invoice_draft_spoken,
    build_invoice_details_spoken,
    build_invoice_list_spoken,
    build_line_item_added_spoken,
)
from templates.client_lookup import build_client_spoken
from templates.payment_status import build_payment_status_spoken
from templates.languages import t

CONFIDENCE_THRESHOLD = 0.7


async def dispatch(intent, saas: MockSaasClient, session: SessionState) -> VoiceResponse:
    # Used for the branches below that don't otherwise have an intent-carried
    # language to localize with (confirm/cancel/unknown/unhandled fallback).
    language = getattr(intent, "language", "en")

    if intent.confidence < CONFIDENCE_THRESHOLD:
        prompt = _clarification_prompt(intent)
        return VoiceResponse(
            success=False,
            spoken_text=prompt,
            clarification_question=prompt,
        )

    match intent:
        case ConfirmIntent():
            return await _handle_confirm(session, saas, language)
        case CancelIntent():
            return _handle_cancel(session, language)
        case LookupClientIntent():
            return await _lookup_client(intent, saas)
        case CheckPaymentIntent():
            return await _check_payment(intent, saas)
        case CheckInvoiceIntent():
            return await _check_invoice(intent, saas)
        case CreateInvoiceIntent():
            return await _create_invoice_draft(intent, saas, session)
        case AddLineItemIntent():
            return await _add_line_item(intent, saas, session)
        case UnknownIntent():
            return VoiceResponse(
                success=False,
                spoken_text=t("unknown_intent", language),
                error="unknown_intent",
            )
        case _:
            return VoiceResponse(
                success=False,
                spoken_text=t("unhandled_intent", language),
                error="unhandled_intent_type",
            )


async def _resolve_single_client(client_name: str, saas: MockSaasClient, language: str):
    matches = await saas.lookup_client(client_name)
    if len(matches) == 0:
        return VoiceResponse(
            success=False,
            spoken_text=t("client_not_found", language, client_name=client_name),
            error="client_not_found",
        )
    if len(matches) == 1 and "_fuzzy_score" not in matches[0]:
        return matches[0]

    names = ", ".join(f"{m['first_name']} {m['last_name']}" for m in matches)
    question = (
        t("client_confirm_single", language, names=names)
        if len(matches) == 1
        else t("client_confirm_multiple", language, names=names)
    )
    return VoiceResponse(success=False, spoken_text=question, clarification_question=question, display_data=matches)


async def _lookup_client(intent: LookupClientIntent, saas: MockSaasClient) -> VoiceResponse:
    result = await _resolve_single_client(intent.client_name, saas, intent.language)
    if isinstance(result, VoiceResponse):
        return result
    client = result
    return VoiceResponse(
        success=True,
        spoken_text=build_client_spoken(client, language=intent.language),
        display_data=client,
    )


def _has_target(intent) -> bool:
    # True if the intent gives us something to look an invoice/client up by
    invoice_number = getattr(intent, "invoice_number", None)
    client_name = getattr(intent, "client_name", None)
    return bool(invoice_number) or bool(client_name and client_name.strip())


async def _check_payment(intent: CheckPaymentIntent, saas: MockSaasClient) -> VoiceResponse:
    if not _has_target(intent):
        question = t("invoice_or_client_missing", intent.language)
        return VoiceResponse(
            success=False,
            spoken_text=question,
            clarification_question=question,
        )

    if intent.invoice_number:
        inv = await saas.get_invoice(intent.invoice_number)
        invoices = [inv] if inv else []
    else:
        invoices = await saas.get_invoices_by_client(intent.client_name)

    if not invoices:
        return VoiceResponse(
            success=False,
            spoken_text=t("invoice_not_found", intent.language),
            error="invoice_not_found",
        )

    return VoiceResponse(
        success=True,
        spoken_text=build_payment_status_spoken(invoices, language=intent.language),
        display_data=invoices,
    )


async def _check_invoice(intent: CheckInvoiceIntent, saas: MockSaasClient) -> VoiceResponse:
    if not _has_target(intent):
        question = t("invoice_or_client_missing", intent.language)
        return VoiceResponse(
            success=False,
            spoken_text=question,
            clarification_question=question,
        )

    if intent.invoice_number:
        inv = await saas.get_invoice(intent.invoice_number)
        invoices = [inv] if inv else []
    else:
        invoices = await saas.get_invoices_by_client(intent.client_name)

    if not invoices:
        return VoiceResponse(
            success=False,
            spoken_text=t("invoice_not_found", intent.language),
            error="invoice_not_found",
        )

    spoken = (
        build_invoice_details_spoken(invoices[0], language=intent.language)
        if len(invoices) == 1
        else build_invoice_list_spoken(invoices, language=intent.language)
    )

    return VoiceResponse(
        success=True,
        spoken_text=spoken,
        display_data=invoices,
    )


async def _create_invoice_draft(
    intent: CreateInvoiceIntent, saas: MockSaasClient, session: SessionState
) -> VoiceResponse:
    result = await _resolve_single_client(intent.client_name, saas, intent.language)
    if isinstance(result, VoiceResponse):
        return result
    client = result

    missing_price_items = [item.product_name for item in intent.articles if item.unit_price is None]
    if missing_price_items:
        names = ", ".join(missing_price_items)
        question = t("missing_unit_price", intent.language, names=names)
        session.awaiting_field = "unit_price"
        session.pending_transcript = intent.raw_transcript
        return VoiceResponse(
            success=False,
            spoken_text=question,
            clarification_question=question,
        )

    line_items = []
    for item in intent.articles:
        matches = await saas.lookup_product(item.product_name)
        designation = matches[0]["designation"] if len(matches) == 1 else item.product_name
        line_items.append({
            "name": designation,
            "designation": designation,
            "description": "",
            "quantity": item.quantity,
            "unit_price": item.unit_price,
            "total_ht": round(item.quantity * item.unit_price, 3),
            "unit": item.unit or "piece",
        })
    draft = await saas.build_invoice_draft(
        client=client,
        line_items=line_items,
        currency=intent.currency,
        is_tva_applicable=intent.is_tva_applicable,
        is_stamp_applicable=intent.is_stamp_applicable,
        payment_method=intent.payment_method,
        terms_and_conditions=intent.terms_and_conditions,
        due_date=intent.due_date,
    )

    # Hold the draft in session state until the user explicitly confirms or cancels
    session.pending_draft = draft
    session.confirmation_pending = True
    session.last_intent = IntentType.CREATE_INVOICE.value

    session.awaiting_field = None
    session.pending_transcript = None

    return VoiceResponse(
        success=True,
        spoken_text=build_invoice_draft_spoken(draft, language=intent.language),
        requires_confirmation=True,
        draft_payload=draft,
    )


async def _add_line_item(intent: AddLineItemIntent, saas: MockSaasClient, session: SessionState) -> VoiceResponse:
    if intent.unit_price is None:
        question = t("missing_unit_price", intent.language, names=intent.product_name)
        session.awaiting_field = "unit_price"
        session.pending_transcript = intent.raw_transcript
        return VoiceResponse(
            success=False,
            spoken_text=question,
            clarification_question=question,
        )

    matches = await saas.lookup_product(intent.product_name)
    designation = matches[0]["designation"] if len(matches) == 1 else intent.product_name

    updated_invoice = await saas.add_line_item(
        invoice_number=intent.invoice_number,
        designation=designation,
        quantity=intent.quantity,
        unit_price=intent.unit_price,
        unit=intent.unit or "piece",
    )

    if not updated_invoice:
        return VoiceResponse(
            success=False,
            spoken_text=t("invoice_not_found_number", intent.language, invoice_number=intent.invoice_number),
            error="invoice_not_found",
        )

    session.awaiting_field = None
    session.pending_transcript = None
    return VoiceResponse(
        success=True,
        spoken_text=build_line_item_added_spoken(
            intent.invoice_number, intent.product_name, intent.quantity, language=intent.language
        ),
        display_data=updated_invoice,
    )


async def _handle_confirm(session: SessionState, saas: MockSaasClient, language: str) -> VoiceResponse:
    if not session.confirmation_pending or not session.pending_draft:
        return VoiceResponse(
            success=False,
            spoken_text=t("no_pending_confirmation", language),
            error="no_pending_confirmation",
        )

    invoice = await saas.confirm_invoice_draft(session.pending_draft)
    session.pending_draft = None
    session.confirmation_pending = False
    session.last_intent = None

    return VoiceResponse(
        success=True,
        spoken_text=t(
            "confirm_invoice_created",
            language,
            invoice_number=invoice["invoice_number"],
            client_name=invoice["client_name"],
            total_ttc=invoice["total_ttc"],
            currency=invoice["currency"],
        ),
        display_data=invoice,
    )


def _handle_cancel(session: SessionState, language: str) -> VoiceResponse:
    session.awaiting_field = None
    session.pending_transcript = None
    if not session.confirmation_pending:
        return VoiceResponse(
            success=False,
            spoken_text=t("no_pending_cancellation", language),
            error="no_pending_confirmation",
        )

    session.pending_draft = None
    session.confirmation_pending = False
    session.last_intent = None

    return VoiceResponse(
        success=True,
        spoken_text=t("draft_discarded", language),
    )


def _clarification_prompt(intent) -> str:
    language = getattr(intent, "language", "en")

    if isinstance(intent, (ConfirmIntent, CancelIntent)):
        return t("clarify_confirm_or_cancel", language)

    fields = getattr(intent, "low_confidence_fields", [])

    if "invoice_number" in fields:
        inv = getattr(intent, "invoice_number", None)
        return (
            t("clarify_invoice_number_confirm", language, invoice_number=inv)
            if inv
            else t("clarify_invoice_number", language)
        )
    if "client_name" in fields:
        return t("clarify_client_name", language)
    if "unit_price" in fields:
        return t("clarify_unit_price", language)
    if "articles" in fields:
        return t("clarify_articles", language)

    return t("clarify_generic", language)