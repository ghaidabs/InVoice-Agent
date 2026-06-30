def build_invoice_draft_spoken(draft: dict) -> str:
    item_count = len(draft.get("articles", []))
    total_ttc = draft.get("total_ttc", 0.0)
    currency = draft.get("currency", "TND")
    client_name = draft.get("client_name", "the client")
    due_date = draft.get("due_date")

    item_phrase = "1 item" if item_count == 1 else f"{item_count} items"
    due_phrase = f" Due by {due_date}." if due_date else ""

    return (
        f"Draft invoice for {client_name} with {item_phrase}, "
        f"total {total_ttc} {currency}.{due_phrase} "
        f"Say 'confirm' to create it, or 'cancel' to discard."
    )


def build_invoice_details_spoken(invoice: dict) -> str:
    return (
        f"Invoice {invoice.get('invoice_number')} for {invoice.get('client_name')}, "
        f"status: {invoice.get('status')}, "
        f"total: {invoice.get('total_ttc')} {invoice.get('currency')}."
    )


def build_invoice_list_spoken(invoices: list[dict]) -> str:
    if len(invoices) == 1:
        return build_invoice_details_spoken(invoices[0])
    numbers = ", ".join(i.get("invoice_number", "unknown") for i in invoices)
    return f"I found {len(invoices)} invoices: {numbers}. Which one would you like details on?"


def build_line_item_added_spoken(invoice_number: str, designation: str, quantity: float) -> str:
    qty_str = str(int(quantity)) if quantity == int(quantity) else str(quantity)
    return f"Added {qty_str} of {designation} to invoice {invoice_number}."