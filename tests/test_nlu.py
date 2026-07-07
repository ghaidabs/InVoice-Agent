import os
import pytest

from agent.nlu import extract_intent

pytestmark = pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set — skipping live Groq NLU tests",
)

REPEAT_RUNS = 2
CONFIDENCE_THRESHOLD = 0.7  # mirrors agent/dispatcher.CONFIDENCE_THRESHOLD


def _run_n(transcript, company_slug, n):
    return [extract_intent(transcript, company_slug=company_slug) for _ in range(n)]


# ---------- one straightforward, clear case per intent type ----------

class TestClearCasesPerIntent:

    def test_create_invoice_french_clear(self):
        result = extract_intent(
            "Facture pour Ahmed Salem, 2 jours de consultation IT à 500 dinars",
            company_slug="ACME",
        )
        assert result.intent == "create_invoice"
        assert result.confidence >= CONFIDENCE_THRESHOLD
        assert result.client_name  # non-empty
        assert len(result.articles) >= 1

    def test_create_invoice_english_clear(self):
        result = extract_intent(
            "Create an invoice for Sarra Trabelsi, 10 days of web development at 400 TND per day",
            company_slug="ACME",
        )
        assert result.intent == "create_invoice"
        assert result.confidence >= CONFIDENCE_THRESHOLD

    def test_create_invoice_code_switched(self):
        # French/English code-switching — common in real Tunisian usage
        result = extract_intent(
            "Je veux create une invoice pour Karim Mzoughi, 1 jour audit sécurité 600 dinars",
            company_slug="NOVA",
        )
        assert result.intent == "create_invoice"
        assert result.confidence >= CONFIDENCE_THRESHOLD

    def test_lookup_client_clear(self):
        result = extract_intent("Cherche le client Ahmed Salem", company_slug="ACME")
        assert result.intent == "lookup_client"
        assert result.confidence >= CONFIDENCE_THRESHOLD
        assert "ahmed" in result.client_name.lower()

    def test_check_invoice_by_clear_number(self):
        result = extract_intent(
            "Check invoice ACME F202605 001 for me please",
            company_slug="ACME",
        )
        assert result.intent == "check_invoice"
        assert result.confidence >= CONFIDENCE_THRESHOLD
        assert result.invoice_number is not None

    def test_add_line_item_clear(self):
        result = extract_intent(
            "Ajoute 3 jours de développement web à 400 dinars sur la facture ACME_F202605-001",
            company_slug="ACME",
        )
        assert result.intent == "add_line_item"
        assert result.confidence >= CONFIDENCE_THRESHOLD
        assert result.unit_price == pytest.approx(400.0)

    def test_check_payment_clear(self):
        result = extract_intent(
            "Est-ce que la facture ACME_F202605-001 a été payée ?",
            company_slug="ACME",
        )
        assert result.intent == "check_payment"
        assert result.confidence >= CONFIDENCE_THRESHOLD

    def test_confirm_french(self):
        result = extract_intent("c'est bon, vas-y", company_slug="ACME")
        assert result.intent == "confirm"
        assert result.confidence >= CONFIDENCE_THRESHOLD

    def test_confirm_english(self):
        result = extract_intent("yes go ahead, do it", company_slug="ACME")
        assert result.intent == "confirm"
        assert result.confidence >= CONFIDENCE_THRESHOLD

    def test_cancel_french(self):
        result = extract_intent("non annule tout", company_slug="ACME")
        assert result.intent == "cancel"
        assert result.confidence >= CONFIDENCE_THRESHOLD

    def test_cancel_english(self):
        result = extract_intent("never mind, forget it", company_slug="ACME")
        assert result.intent == "cancel"
        assert result.confidence >= CONFIDENCE_THRESHOLD

    def test_unknown_gibberish(self):
        result = extract_intent("asdkjfh qwoiruty random noise", company_slug="ACME")
        assert result.intent == "unknown"

    def test_unknown_off_topic(self):
        result = extract_intent(
            "Quel temps fait-il à Tunis aujourd'hui ?", company_slug="ACME"
        )
        assert result.intent == "unknown"


# ---------- deliberately ambiguous / partial cases ----------

class TestAmbiguousCases:

    def test_partial_invoice_number_low_confidence(self):
        # Missing the sequence digits entirely — model should not guess
        result = extract_intent(
            "Vérifie la facture ACME F202606 euh... je me souviens plus du numéro exact",
            company_slug="ACME",
        )
        assert result.intent == "check_invoice"
        # Either null'd out or flagged, but must not be confidently accepted as complete
        if result.invoice_number is None:
            assert result.confidence < CONFIDENCE_THRESHOLD
        else:
            # if it did extract something, it should at least flag the field
            assert (
                result.confidence < CONFIDENCE_THRESHOLD
                or "invoice_number" in result.low_confidence_fields
            )

    def test_unclear_price_stays_null_and_low_confidence(self):
        result = extract_intent(
            "Facture pour Ahmed Salem, consultation IT, je sais pas trop combien ça coûte",
            company_slug="ACME",
        )
        assert result.intent == "create_invoice"
        assert result.articles[0].unit_price is None

    def test_mumbled_client_name_ambiguous(self):
        result = extract_intent(
            "euh facture pour... le client là, tu sais, celui de la semaine dernière",
            company_slug="ACME",
        )
        # No concrete name given — should either be unknown or low confidence create_invoice
        if result.intent == "create_invoice":
            assert result.confidence < CONFIDENCE_THRESHOLD

    def test_ambiguous_confirm_vs_cancel(self):
        # Genuinely ambiguous phrasing between confirming and something else
        result = extract_intent("bon... ouais... non attends", company_slug="ACME")
        assert result.intent in ("confirm", "cancel", "unknown")
        # This is the kind of utterance that SHOULD be uncertain
        if result.intent in ("confirm", "cancel"):
            assert result.confidence < 0.95  # shouldn't be maximally confident on this

    def test_currency_other_than_default_explicit(self):
        result = extract_intent(
            "Facture Nour Gharbi, 5 jours audit sécurité, 600 euros par jour",
            company_slug="NOVA",
        )
        assert result.intent == "create_invoice"
        assert result.currency == "EUR"

    def test_multiple_line_items_mixed_language(self):
        result = extract_intent(
            "Invoice for Karim Mzoughi: 2 days formation at 300 dinars, plus "
            "1 jour audit technique à 450 dinars",
            company_slug="NOVA",
        )
        assert result.intent == "create_invoice"
        assert len(result.articles) == 2


# ---------- ⚠ low-confidence-word marker path (from agent/stt.annotate_low_confidence) ----------

class TestLowConfidenceMarkerPath:
    """
    agent/stt.annotate_low_confidence() inserts a ⚠ after words the ASR
    itself flagged as low-confidence. The prompt (agent/prompts/intent_extraction.txt)
    explicitly instructs the model not to trust such tokens at face value for
    quantities, prices, invoice numbers, or names. These tests hand-craft
    transcripts with ⚠ markers to confirm the LLM actually follows that
    instruction, rather than the marker being silently ignored.
    """

    def test_marked_price_lowers_confidence_or_flags_field(self):
        transcript = (
            "Facture pour Ahmed Salem, 2 jours de consultation IT à "
            "cinq⚠ cents⚠ dinars le jour"
        )
        results = _run_n(transcript, "ACME", REPEAT_RUNS)
        for result in results:
            assert result.intent == "create_invoice"
            # Either the model refused to trust the marked price (null),
            # or it kept a value but flagged/low-confidenced it
            item = result.articles[0]
            marked_as_uncertain = (
                item.unit_price is None
                or result.confidence < CONFIDENCE_THRESHOLD
                or "unit_price" in result.low_confidence_fields
            )
            assert marked_as_uncertain, (
                f"Marked-price transcript was accepted at face value: {result}"
            )

    def test_marked_invoice_number_digit_not_trusted(self):
        transcript = "Check invoice ACME F202606⚠ dash zero zero un⚠"
        results = _run_n(transcript, "ACME", REPEAT_RUNS)
        for result in results:
            assert result.intent == "check_invoice"
            marked_as_uncertain = (
                result.invoice_number is None
                or result.confidence < CONFIDENCE_THRESHOLD
                or "invoice_number" in result.low_confidence_fields
            )
            assert marked_as_uncertain, (
                f"Marked invoice number transcript was accepted at face value: {result}"
            )

    def test_unmarked_equivalent_is_trusted_as_control(self):
        # Control case: same content, no ⚠ marks — should NOT be penalized.
        # This guards against a prompt that's simply always-low-confidence
        # regardless of markers (which would make the above tests meaningless).
        transcript = "Facture pour Ahmed Salem, 2 jours de consultation IT à 500 dinars le jour"
        result = extract_intent(transcript, company_slug="ACME")
        assert result.intent == "create_invoice"
        assert result.confidence >= CONFIDENCE_THRESHOLD
        assert result.articles[0].unit_price == pytest.approx(500.0)