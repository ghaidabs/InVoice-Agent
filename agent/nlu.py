import json
from pathlib import Path
from groq import Groq
from agent.validator import redact_pii 
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
    LineItemInput,
)

PROMPT_PATH = Path(__file__).parent / "prompts" / "intent_extraction.txt"
SYSTEM_PROMPT = PROMPT_PATH.read_text()

client = Groq()  

_INTENT_MODEL_MAP = {
    IntentType.CREATE_INVOICE: CreateInvoiceIntent,
    IntentType.LOOKUP_CLIENT: LookupClientIntent,
    IntentType.CHECK_INVOICE: CheckInvoiceIntent,
    IntentType.ADD_LINE_ITEM: AddLineItemIntent,
    IntentType.CHECK_PAYMENT: CheckPaymentIntent,
    IntentType.CONFIRM: ConfirmIntent,
    IntentType.CANCEL: CancelIntent,
    IntentType.UNKNOWN: UnknownIntent,
}


def extract_intent(transcript: str):
    safe_transcript = redact_pii(transcript)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": safe_transcript},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )

    raw_text = response.choices[0].message.content

    try:
        raw = json.loads(raw_text) # Json string to dictionnary
    except json.JSONDecodeError:
        return UnknownIntent(
            intent=IntentType.UNKNOWN,
            raw_text=transcript,
            confidence=0.0,
            raw_transcript=transcript,
        )

    intent_type_str = raw.get("intent")
    try:
        intent_type = IntentType(intent_type_str) # converts to enum
    except ValueError:
        return UnknownIntent(
            intent=IntentType.UNKNOWN,
            raw_text=transcript,
            confidence=0.0,
            raw_transcript=transcript,
        )

    model_cls = _INTENT_MODEL_MAP[intent_type] # extract the model class based on the intent type
    raw["raw_transcript"] = transcript
    raw.setdefault("confidence", 0.0) # if confidence is not present, set it to 0.0

    # rebuild line_items into LineItemInput objects if present
    if "articles" in raw and raw["articles"] is not None:
        raw["articles"] = [LineItemInput(**item) for item in raw["articles"]]

    try:
        return model_cls(**raw) # validated pydantic object
    except Exception:
        return UnknownIntent(
            intent=IntentType.UNKNOWN,
            raw_text=transcript,
            confidence=0.0,
            raw_transcript=transcript,
        )

def extract_intent_with_context(transcript: str, session: "SessionState"):
    if session.awaiting_field and session.pending_transcript:
        combined = (
            f"{session.pending_transcript}. "
            f"The user is now answering the question \"What is the {session.awaiting_field}?\" "
            f"with: \"{transcript}\". Use this value directly as the {session.awaiting_field} — "
        )
        return extract_intent(combined)
    return extract_intent(transcript)