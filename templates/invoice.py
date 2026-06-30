# templates/invoice.py

def build_invoice_draft_spoken(draft: dict) -> str:
    item_count = len(draft.get("articles", []))
    total_ttc = draft.get("total_ttc", 0.0)
    currency = draft.get("currency", "TND")
    client_name = draft.get("client_name", "the client")

    item_phrase = "1 item" if item_count == 1 else f"{item_count} items"

    return (
        f"Draft invoice for {client_name} with {item_phrase}, "
        f"total {total_ttc} {currency}. "
        f"Say 'confirm' to create it, or 'cancel' to discard."
    )


def build_invoice_details_spoken(invoice: dict) -> str:
    return (
        f"Invoice {invoice.get('invoice_number')} for {invoice.get('client_name')}, "
        f"status: {invoice.get('status')}, "
        f"total: {invoice.get('total_ttc')} {invoice.get('currency')}."
    )


def build_line_item_added_spoken(invoice_number: str, designation: str, quantity: float) -> str:
    qty_str = str(int(quantity)) if quantity == int(quantity) else str(quantity)
    return f"Added {qty_str} of {designation} to invoice {invoice_number}."