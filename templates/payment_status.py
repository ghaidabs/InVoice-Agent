def build_payment_status_spoken(invoices: list[dict]) -> str:
    if len(invoices) == 1:
        inv = invoices[0]
        status = inv.get("status", "unknown")
        if status == "paid":
            return f"Invoice {inv['invoice_number']} has been paid."
        elif status == "draft":
            return f"Invoice {inv['invoice_number']} is still a draft and hasn't been sent."
        elif status in ("cancelled", "canceled", "void"):
            return f"Invoice {inv['invoice_number']} was cancelled."
        else:
            return f"Invoice {inv['invoice_number']} is {status}, not yet paid."

    unpaid = [i for i in invoices if i.get("status") not in ("paid",)]
    paid = [i for i in invoices if i.get("status") == "paid"]
    return (
        f"Found {len(invoices)} invoices: {len(paid)} paid, "
        f"{len(unpaid)} not yet paid."
    )