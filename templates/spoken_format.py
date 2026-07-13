_CURRENCY_NAMES = {
    "TND": {"fr": "dinars tunisiens", "en": "Tunisian dinars"},
    "EUR": {"fr": "euros", "en": "euros"},
    "USD": {"fr": "dollars", "en": "US dollars"},
}

_STATUS_PHRASES = {
    "paid": {
        "fr": "a été payée",
        "en": "has been paid",
    },
    "draft": {
        "fr": "est encore un brouillon et n'a pas été envoyée",
        "en": "is still a draft and hasn't been sent",
    },
    "validated": {
        "fr": "a été validée",
        "en": "has been validated",
    },
    "cancelled": {
        "fr": "a été annulée",
        "en": "was cancelled",
    },
    "canceled": {
        "fr": "a été annulée",
        "en": "was cancelled",
    },
    "void": {
        "fr": "a été annulée",
        "en": "was cancelled",
    },
}


def spoken_currency(code: str | None, language: str = "fr") -> str:
    if not code:
        return ""
    entry = _CURRENCY_NAMES.get(code.upper())
    if not entry:
        return code
    return entry.get(language, entry["fr"])


def spoken_amount(value: float | None, currency: str | None, language: str = "fr") -> str:
    value = value or 0.0
    if value == int(value):
        amount_str = f"{int(value):,}".replace(",", " ")
    else:
        amount_str = f"{value:,.2f}".replace(",", " ")
    currency_str = spoken_currency(currency, language)
    return f"{amount_str} {currency_str}".strip()


def status_phrase(status: str | None, language: str = "fr") -> str:
    entry = _STATUS_PHRASES.get((status or "").lower())
    if not entry:
        # Unknown status
        if language == "en":
            return f"is currently {status}"
        return f"est actuellement au statut {status}"
    return entry.get(language, entry["fr"])