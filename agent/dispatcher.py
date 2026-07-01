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

CONFIDENCE_THRESHOLD = 0.7


async def dispatch(intent, saas: MockSaasClient, session: SessionState) -> VoiceResponse:
    if intent.confidence < CONFIDENCE_THRESHOLD:
        return VoiceResponse(
            success=False,
            spoken_text=_clarification_prompt(intent),
            clarification_question=_clarification_prompt(intent),
        )

    match intent:
        case ConfirmIntent():
            return await _handle_confirm(session, saas)
        case CancelIntent():
            return _handle_cancel(session)
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
                spoken_text="I didn't understand that. Could you rephrase?",
                error="unknown_intent",
            )
        case _:
            return VoiceResponse(
                success=False,
                spoken_text="Sorry, an unsupported intent was received.",
                error="unhandled_intent_type",
            )


async def _resolve_single_client(client_name: str, saas: MockSaasClient):
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


def _has_target(intent) -> bool:
    # True if the intent gives us something to look an invoice/client up by
    invoice_number = getattr(intent, "invoice_number", None)
    client_name = getattr(intent, "client_name", None)
    return bool(invoice_number) or bool(client_name and client_name.strip())


async def _check_payment(intent: CheckPaymentIntent, saas: MockSaasClient) -> VoiceResponse:
    if not _has_target(intent):
        question = "Which invoice number or client did you mean?"
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
            spoken_text="I couldn't find any matching invoices.",
            error="invoice_not_found",
        )

    return VoiceResponse(
        success=True,
        spoken_text=build_payment_status_spoken(invoices),
        display_data=invoices,
    )


async def _check_invoice(intent: CheckInvoiceIntent, saas: MockSaasClient) -> VoiceResponse:
    if not _has_target(intent):
        question = "Which invoice number or client did you mean?"
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
            spoken_text="I couldn't find any matching invoices.",
            error="invoice_not_found",
        )

    spoken = (
        build_invoice_details_spoken(invoices[0])
        if len(invoices) == 1
        else build_invoice_list_spoken(invoices)
    )

    return VoiceResponse(
        success=True,
        spoken_text=spoken,
        display_data=invoices,
    )


async def _create_invoice_draft(
    intent: CreateInvoiceIntent, saas: MockSaasClient, session: SessionState
    ) -> VoiceResponse:
        result = await _resolve_single_client(intent.client_name, saas)
        if isinstance(result, VoiceResponse):
            return result
        client = result

        missing_price_items = [item.product_name for item in intent.articles if item.unit_price is None]
        if missing_price_items:
            names = ", ".join(missing_price_items)
            question = f"I didn't catch the price for {names}. What's the unit price?"
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
            spoken_text=build_invoice_draft_spoken(draft),
            requires_confirmation=True,
            draft_payload=draft,
        )


async def _add_line_item(intent: AddLineItemIntent, saas: MockSaasClient, session: SessionState) -> VoiceResponse:
    if intent.unit_price is None:
        question = f"I didn't catch the price for {intent.product_name}. What's the unit price?"
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
            spoken_text=f"I couldn't find invoice {intent.invoice_number}.",
            error="invoice_not_found",
        )

    session.awaiting_field = None
    session.pending_transcript = None
    return VoiceResponse(
        success=True,
        spoken_text=build_line_item_added_spoken(
            intent.invoice_number, intent.product_name, intent.quantity
        ),
        display_data=updated_invoice,
    )


async def _handle_confirm(session: SessionState, saas: MockSaasClient) -> VoiceResponse:
    if not session.confirmation_pending or not session.pending_draft:
        return VoiceResponse(
            success=False,
            spoken_text="There's nothing waiting to be confirmed right now.",
            error="no_pending_confirmation",
        )

    invoice = await saas.confirm_invoice_draft(session.pending_draft)
    session.pending_draft = None
    session.confirmation_pending = False
    session.last_intent = None

    return VoiceResponse(
        success=True,
        spoken_text=(
            f"Invoice {invoice['invoice_number']} has been created for "
            f"{invoice['client_name']}, total {invoice['total_ttc']} {invoice['currency']}."
        ),
        display_data=invoice,
    )


def _handle_cancel(session: SessionState) -> VoiceResponse:
    session.awaiting_field = None
    session.pending_transcript = None
    if not session.confirmation_pending:
        return VoiceResponse(
            success=False,
            spoken_text="There's nothing waiting to be cancelled right now.",
            error="no_pending_confirmation",
        )

    session.pending_draft = None
    session.confirmation_pending = False
    session.last_intent = None

    return VoiceResponse(
        success=True,
        spoken_text="Okay, I discarded the draft invoice.",
    )


def _clarification_prompt(intent) -> str:
    if isinstance(intent, (ConfirmIntent, CancelIntent)):
        return "Sorry, were you confirming or cancelling that? Could you repeat?"

    fields = getattr(intent, "low_confidence_fields", [])

    if "invoice_number" in fields:
        inv = getattr(intent, "invoice_number", None)
        return f"Did you mean invoice {inv}? Please confirm." if inv else "Which invoice number did you mean?"
    if "client_name" in fields:
        return "I wasn't sure about the client name. Could you repeat it?"
    if "unit_price" in fields:
        return "I wasn't sure about the price. Could you confirm the amount?"
    if "articles" in fields:
        return "I wasn't sure about one of the items or amounts. Could you repeat that part?"

    return "I wasn't sure I caught that correctly. Could you repeat?"