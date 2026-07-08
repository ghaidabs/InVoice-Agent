import logging
import re
from langdetect import detect
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "fr"
SUPPORTED_LANGUAGES = ["en", "fr"]
PII_ENTITIES = ["PHONE_NUMBER", "EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE"]
_PII_TRIGGER_RE = re.compile(
    r"\s*\b("
    r"email|e-?mail|adresse (e-?)?mail|courriel|"
    r"phone( number)?|téléphone|telephone|numéro de téléphone|numero de telephone|"
    r"matricule( fiscale)?|tax id|tax number|"
    r"iban|credit card|carte (bancaire|de crédit)"
    r")\b.*$",
    re.IGNORECASE,
)
_MATRICULE_FISCALE_RE = re.compile(r"\b\d{7}[A-Za-z]/[A-Za-z]/[A-Za-z]/\d{3}\b") # Tunisian matricule fiscale format

_nlp_engine = NlpEngineProvider(
    nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_lg"},
            {"lang_code": "fr", "model_name": "fr_core_news_lg"},
        ],
    }
).create_engine()

_analyzer = AnalyzerEngine(nlp_engine=_nlp_engine, supported_languages=SUPPORTED_LANGUAGES)
_anonymizer = AnonymizerEngine()


def _detect_language(text: str) -> str:
    try:
        lang = detect(text)
    except Exception:
        return DEFAULT_LANGUAGE
    return lang if lang in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


def _strip_after_trigger(text: str) -> str:
    return _PII_TRIGGER_RE.sub("", text).strip()


def redact_pii(text: str) -> str:
    if not text or not text.strip():
        return text

    redacted = _strip_after_trigger(text)   

    try:
        language = _detect_language(redacted)
        results = _analyzer.analyze(text=redacted, language=language, entities=PII_ENTITIES)
        if results:
            redacted = _anonymizer.anonymize(text=redacted, analyzer_results=results).text
    except Exception:
        logger.exception("PII analysis failed; using regex-only redaction")

    redacted = _MATRICULE_FISCALE_RE.sub("<TAX_ID>", redacted)
    return redacted