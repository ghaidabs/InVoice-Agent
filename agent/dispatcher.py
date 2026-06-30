from schemas.intents import (
    IntentType,
    CreateInvoiceIntent,
    LookupClientIntent,
    CheckInvoiceIntent,
    AddLineItemIntent,
    CheckPaymentIntent,
    UnknownIntent,
)
from schemas.responses import VoiceResponse
from api.mock_saas_client import MockSaasClient
from templates.invoice import build_invoice_draft_spoken
from templates.client_lookup import build_client_spoken
from templates.payment_status import build_payment_status_spoken

CONFIDENCE_THRESHOLD = 0.7


async def dispatch(intent, saas: MockSaasClient) -> VoiceResponse:
    if intent.confidence < CONFIDENCE_THRESHOLD:
        return VoiceResponse(
            success=False,
            spoken_text=_clarification_prompt(intent),
            clarification_question=_clarification_prompt(intent),
        )

    match intent:
        case LookupClientIntent():
            return await _lookup_client(intent, saas)
        case CheckPaymentIntent():
            return await _check_payment(intent, saas)
        case CheckInvoiceIntent():
            return await _check_invoice(intent, saas)
        case CreateInvoiceIntent():
            return await _create_invoice_draft(intent, saas)
        case AddLineItemIntent():
            return await _add_line_item(intent, saas)
        case UnknownIntent():
            return VoiceResponse(
                success=False,
                spoken_text="I didn't understand that. Could you rephrase?",
                error="unknown_intent",
            )
        case _:
            return VoiceResponse(
                success=False,
                spoken_text="I didn't understand that. Could you rephrase?",
                error="unhandled_intent_type",
            )


async def _resolve_single_client(client_name: str, saas: MockSaasClient):
    # Returns either a single client dict, or a VoiceResponse if 0 or 2+ matches
    matches = await saas.lookup_client(client_name)
    if len(matches) == 0:
        return VoiceResponse(
            success=False,
            spoken_text=f"No client found matching '{client_name}'.",
            error="client_not_found",
        )
    if len(matches) > 1:
        names = ", ".join(f"{m['first_name']} {m['last_name']}" for m in matches)
        question = f"I found several clients matching that name: {names}. Which one did you mean?"
        return VoiceResponse(
            success=False,
            spoken_text=question,
            clarification_question=question,
            display_data=matches,
        )
    return matches[0]


async def _lookup_client(intent: LookupClientIntent, saas: MockSaasClient) -> VoiceResponse:
    result = await _resolve_single_client(intent.client_name, saas)
    if isinstance(result, VoiceResponse):
        return result
    client = result
    return VoiceResponse(
        success=True,
        spoken_text=build_client_spoken(client),
        display_data=client,
    )


async def _check_payment(intent: CheckPaymentIntent, saas: MockSaasClient) -> VoiceResponse:
    if intent.invoice_number:
        inv = await saas.get_invoice(intent.invoice_number)
        invoices = [inv] if inv else []
    else:
        invoices = await saas.get_invoices_by_client(intent.client_name or "")

    if not invoices:
        return VoiceResponse(
            success=False,
            spoken_text="I couldn't find any matching invoices.",
            error="invoice_not_found",
        )

    return VoiceResponse(
        success=True,
        spoken_text=build_payment_status_spoken(invoices),
        display_data=invoices,
    )


async def _check_invoice(intent: CheckInvoiceIntent, saas: MockSaasClient) -> VoiceResponse:
    if intent.invoice_number:
        inv = await saas.get_invoice(intent.invoice_number)
        invoices = [inv] if inv else []
    else:
        invoices = await saas.get_invoices_by_client(intent.client_name or "")

    if not invoices:
        return VoiceResponse(
            success=False,
            spoken_text="I couldn't find any matching invoices.",
            error="invoice_not_found",
        )

    return VoiceResponse(
        success=True,
        spoken_text=build_payment_status_spoken(invoices),  # reuse until a dedicated template exists
        display_data=invoices,
    )


async def _create_invoice_draft(intent: CreateInvoiceIntent, saas: MockSaasClient) -> VoiceResponse:
    result = await _resolve_single_client(intent.client_name, saas)
    if isinstance(result, VoiceResponse):
        return result
    client = result

    line_items = [
        {
            "name": item.product_name,
            "designation": item.product_name,
            "description": "",
            "quantity": item.quantity,
            "unit_price": item.unit_price or 0.0,
            "total_ht": item.quantity * (item.unit_price or 0.0),
            "unit": item.unit or "piece",
        }
        for item in intent.articles
    ]

    draft = await saas.build_invoice_draft(
        client=client,
        line_items=line_items,
        currency=intent.currency,
        is_tva_applicable=intent.is_tva_applicable,
        is_stamp_applicable=intent.is_stamp_applicable,
        payment_method=intent.payment_method,
        terms_and_conditions=intent.terms_and_conditions,
    )

    return VoiceResponse(
        success=True,
        spoken_text=build_invoice_draft_spoken(draft),
        requires_confirmation=True,
        draft_payload=draft,   # nothing written to db yet
    )


async def _add_line_item(intent: AddLineItemIntent, saas: MockSaasClient) -> VoiceResponse:
    updated_invoice = await saas.add_line_item(
        invoice_number=intent.invoice_number,
        designation=intent.product_name,
        quantity=intent.quantity,
        unit_price=intent.unit_price or 0.0,
        unit=intent.unit or "piece",
    )

    if not updated_invoice:
        return VoiceResponse(
            success=False,
            spoken_text=f"I couldn't find invoice {intent.invoice_number}.",
            error="invoice_not_found",
        )

    return VoiceResponse(
        success=True,
        spoken_text=f"Added {intent.quantity} of {intent.product_name} to invoice {intent.invoice_number}.",
        requires_confirmation=True,
        draft_payload=updated_invoice,
    )


def _clarification_prompt(intent) -> str:
    fields = getattr(intent, "low_confidence_fields", [])

    if "invoice_number" in fields:
        inv = getattr(intent, "invoice_number", None)
        return f"Did you mean invoice {inv}? Please confirm." if inv else "Which invoice number did you mean?"
    if "client_name" in fields:
        return f"I wasn't sure about the client name. Could you repeat it?"
    if "unit_price" in fields:
        return "I wasn't sure about the price. Could you confirm the amount?"
    if "articles" in fields:
        return "I wasn't sure about one of the items or amounts. Could you repeat that part?"

    return "I wasn't sure I caught that correctly. Could you repeat?"