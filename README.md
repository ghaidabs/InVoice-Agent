# InVoice Agent

A **bilingual (French / English) voice assistant** for invoicing SaaS platforms.  
Users speak natural-language commands — *"Crée une facture pour Ahmed, 3 claviers à 45 dinars"* — and the agent extracts intents, resolves entities, and drives the invoicing API through a structured pipeline.

> **Status — MVP / active development.**  
> Core pipeline (STT → NLU → Dispatcher → Response) is functional.  
> TTS, LiveKit integration, FastAPI server, and production SaaS client are scaffolded but not yet implemented.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Project Structure](#project-structure)
3. [Pipeline Layers](#pipeline-layers)
   - [Speech-to-Text (STT)](#1-speech-to-text-stt)
   - [Validation & PII Redaction](#2-validation--pii-redaction)
   - [Natural Language Understanding (NLU)](#3-natural-language-understanding-nlu)
   - [Normalizer & Fuzzy Matching](#4-normalizer--fuzzy-matching)
   - [Retrieval-Augmented Generation (RAG)](#5-retrieval-augmented-generation-rag)
   - [Dispatcher](#6-dispatcher)
   - [Templates (Response Generation)](#7-templates-response-generation)
   - [Text-to-Speech (TTS)](#8-text-to-speech-tts)
4. [Schemas](#schemas)
5. [Session Management](#session-management)
6. [API Layer & Mock System](#api-layer--mock-system)
7. [Server (WIP)](#server-wip)
8. [Testing](#testing)
9. [Getting Started](#getting-started)
10. [Environment Variables](#environment-variables)

---

## Architecture Overview

```
┌──────────┐    PCM audio     ┌──────────┐   transcript    ┌──────────────┐
│  User /  │ ───────────────► │   STT    │ ──────────────► │  Validator   │
│  Mic     │                  │(Deepgram)│                 │ (PII redact) │
└──────────┘                  └──────────┘                 └──────┬───────┘
                                                                  │
                                                          safe transcript
                                                                  │
                                                                  ▼
┌──────────────┐   VoiceResponse   ┌────────────┐   Intent    ┌───────┐
│  Templates   │ ◄──────────────── │ Dispatcher │ ◄────────── │  NLU  │
│ (fr/en text) │                   │            │             │(Groq) │
└──────┬───────┘                   └─────┬──────┘             └───┬───┘
       │                                 │                        │
       ▼                                 ▼                        │
┌──────────┐                    ┌──────────────┐          ┌───────┴───────┐
│   TTS    │                    │  SaaS Client │          │  Normalizer   │
│  (TODO)  │                    │  (mock / real)│         │  Fuzzy / RAG  │
└──────────┘                    └──────────────┘          └───────────────┘
```

---

## Project Structure

```
voice assistant agent/
├── agent/                  # Core agent logic
│   ├── main.py             # CLI REPL entry point (text-based testing)
│   ├── stt.py              # Deepgram streaming STT session
│   ├── stt_console.py      # Microphone → STT console debugger
│   ├── nlu.py              # LLM-powered intent extraction (Groq)
│   ├── dispatcher.py       # Intent → action routing & business logic
│   ├── validator.py        # PII redaction (Presidio + regex)
│   ├── normalizer.py       # Invoice-number normalization
│   ├── fuzzy.py            # Fuzzy entity matching (rapidfuzz)
│   ├── session.py          # Session state & user context dataclasses
│   ├── tts.py              # Text-to-Speech (placeholder)
│   └── prompts/
│       └── intent_extraction.txt   # System prompt for NLU
│
├── api/                    # SaaS API client layer
│   ├── mock_saas_client.py # Mock client backed by local JSON DB
│   └── saas_client.py      # Real HTTP client (placeholder)
│
├── rag/                    # Retrieval-Augmented Generation
│   ├── embeddings.py       # Sentence-transformer embeddings
│   ├── indexer.py          # Qdrant vector indexing
│   └── retriever.py        # Similarity search against Qdrant
│
├── schemas/                # Pydantic data models
│   ├── intents.py          # Intent types & entity schemas
│   └── responses.py        # VoiceResponse model
│
├── templates/              # Spoken-text response builders (fr/en)
│   ├── clarification.py    # Bilingual string table & t() helper
│   ├── invoice.py          # Invoice draft / details / list responses
│   ├── client_lookup.py    # Client lookup responses
│   └── payment_status.py   # Payment status responses
│
├── mock/                   # Local mock database
│   ├── db.py               # In-memory DB with JSON seed data
│   ├── clients.json        # Sample client records
│   ├── invoices.json       # Sample invoice records
│   ├── products.json       # Sample product catalog
│   ├── companies.json      # Multi-tenant company records
│   └── users.json          # Mock user accounts
│
├── server/                 # FastAPI server (WIP)
│   ├── main.py             # Server entry point (placeholder)
│   ├── auth.py             # Auth middleware (placeholder)
│   └── routes/
│       ├── audio_proxy.py      # Audio WebSocket proxy (placeholder)
│       └── livekit_token.py    # LiveKit token endpoint (placeholder)
│
├── frontend/               # Web frontend (empty — planned)
│
├── tests/                  # Pytest test suite
│   ├── conftest.py
│   ├── test_dispatcher.py
│   ├── test_nlu.py
│   ├── test_normalizer.py
│   ├── test_rag.py
│   ├── test_responses_schema.py
│   └── test_mock_db_tenant_isolation.py
│
├── .env.example            # Required environment variable template
├── .gitignore
├── pytest.ini
└── requirements.txt
```

---

## Pipeline Layers

### 1. Speech-to-Text (STT)

| File | Description |
|------|-------------|
| `agent/stt.py` | Production `STTSession` class wrapping **Deepgram Nova-3** streaming WebSocket |
| `agent/stt_console.py` | Dev/debug tool — streams from system microphone, prints per-word confidence with color coding |

**Key details:**

- **Model:** `nova-3` with `language: "multi"` (auto-detects French/English).
- **Features enabled:** smart formatting, punctuation, numeral conversion, VAD events, interim results.
- Outputs a `TranscriptResult` dataclass containing the full transcript, per-word confidence scores, and detected language.
- **Confidence thresholds:** words below `0.55` are flagged; overall transcript below `0.6` is marked low-confidence.
- Low-confidence words are annotated with a `⚠` marker before being sent to the NLU, so the LLM knows which tokens are uncertain.
- Includes a `build_keyterms_for_company()` helper that pulls client names and product names from the DB to boost Deepgram's recognition accuracy via the `keyterm` feature.
- `LiveKitAudioSource` adapter is stubbed — will bridge LiveKit's `AudioStream` to `STTSession` once integrated.

---

### 2. Validation & PII Redaction

| File | Description |
|------|-------------|
| `agent/validator.py` | Redacts sensitive data from transcripts before they reach the LLM |

**What gets redacted:**

- Phone numbers, email addresses, credit card numbers, IBAN codes — via **Microsoft Presidio** with spaCy NLP backends (`en_core_web_lg`, `fr_core_news_lg`).
- Tunisian *matricule fiscale* (tax ID, format `1234567X/X/X/000`) — via custom regex.
- Auto-detects language with `langdetect` to pick the right spaCy model.

---

### 3. Natural Language Understanding (NLU)

| File | Description |
|------|-------------|
| `agent/nlu.py` | Sends redacted transcript to an LLM and parses structured intent JSON |
| `agent/prompts/intent_extraction.txt` | System prompt defining all intent schemas and extraction rules |

**How it works:**

1. The redacted transcript is sent to **Groq** (model: `openai/gpt-oss-120b`) with `temperature=0.1` and `response_format: json_object`.
2. The LLM returns a JSON object matching one of the defined intent schemas.
3. The raw JSON is validated into the corresponding **Pydantic** intent model.
4. Post-processing:
   - Invoice numbers are normalized via `normalizer.py`. If normalization fails (not enough digits), confidence is lowered to ≤ 0.4 and the field is added to `low_confidence_fields`.
   - Line-item arrays are parsed into `LineItemInput` Pydantic models.
5. `extract_intent_with_context()` handles **multi-turn follow-ups** — if the session is awaiting a specific field (e.g., a missing unit price), the pending transcript is prepended for context.

**Supported intents:**

| Intent | Entities extracted |
|--------|-------------------|
| `create_invoice` | `client_name`, `articles[]` (product, qty, price, unit), `currency`, `due_date`, TVA/stamp flags, payment method |
| `lookup_client` | `client_name` |
| `check_invoice` | `invoice_number` or `client_name` |
| `add_line_item` | `invoice_number`, `product_name`, `quantity`, `unit_price`, `unit` |
| `check_payment` | `invoice_number` or `client_name` |
| `confirm` | *(none — confirms a pending draft)* |
| `cancel` | *(none — cancels a pending draft)* |
| `unknown` | `raw_text` |

---

### 4. Normalizer & Fuzzy Matching

| File | Description |
|------|-------------|
| `agent/normalizer.py` | Converts spoken/partial invoice numbers into the canonical format `SLUG_FYYYYMM-SEQ` |
| `agent/fuzzy.py` | RapidFuzz-based entity resolution for client names and product names |

- **Normalizer:** extracts the company slug and 9+ digits, reassembles into `SLUG_FYYYYMM-SEQ`. Returns `None` if fewer than 9 digits are found (triggers a clarification flow).
- **Fuzzy matching:** uses `WRatio` scorer with a floor of 75%. Returns top-K matches with their `_fuzzy_score` attached, so the dispatcher can decide whether to auto-resolve or ask for confirmation.

---

### 5. Retrieval-Augmented Generation (RAG)

| File | Description |
|------|-------------|
| `rag/embeddings.py` | Produces dense vectors using `sentence-transformers/all-MiniLM-L6-v2` |
| `rag/indexer.py` | Creates and upserts into per-company Qdrant collections |
| `rag/retriever.py` | Cosine similarity search with a floor of `0.55` |

**How it's used:**

The RAG layer is a **fallback** for entity resolution. When a user says a client or product name that doesn't match via exact substring or fuzzy matching, the system:

1. Indexes all clients/products for the company into Qdrant (lazy, once).
2. Embeds the spoken query and runs a similarity search.
3. Returns candidates above the similarity floor.

Collections are named `{entity_type}s_{company_id}` (e.g., `clients_1`, `products_2`).

**Entity resolution cascade** (in `MockSaasClient`):

```
Exact substring match → Fuzzy match (rapidfuzz) → Semantic match (RAG)
```

---

### 6. Dispatcher

| File | Description |
|------|-------------|
| `agent/dispatcher.py` | Routes intents to handlers, orchestrates business logic, returns `VoiceResponse` |

**Key behaviors:**

- **Confidence gate:** intents below `0.7` confidence trigger a clarification prompt (field-specific when possible).
- **Pattern matching:** uses Python `match/case` on intent dataclass types.
- **Draft-first flow for `create_invoice`:**
  1. Resolve client (exact → fuzzy → RAG; ambiguous = ask user).
  2. Check for missing unit prices → ask if needed.
  3. Resolve product names against the catalog.
  4. Build a draft and store it in `session.pending_draft`.
  5. Return the draft summary with `requires_confirmation=True`.
  6. User says "confirm" → `ConfirmIntent` → draft is finalized and saved.
  7. User says "cancel" → `CancelIntent` → draft is discarded.
- **Add line item:** resolves product name, normalizes invoice number, appends to invoice and recalculates totals.
- **Check invoice / payment:** supports lookup by invoice number or client name; returns details or lists.

---

### 7. Templates (Response Generation)

| File | Description |
|------|-------------|
| `templates/clarification.py` | Bilingual string table with `t(key, language, **kwargs)` interpolation |
| `templates/invoice.py` | Spoken responses for drafts, invoice details, invoice lists, added line items |
| `templates/client_lookup.py` | Spoken responses for client lookups |
| `templates/payment_status.py` | Spoken responses for payment status queries |

All templates produce **natural language text** in either French or English, determined by the detected language of the user's utterance.

---

### 8. Text-to-Speech (TTS)

| File | Description |
|------|-------------|
| `agent/tts.py` | **Placeholder** — not yet implemented |

Will convert the `VoiceResponse.spoken_text` back to audio for the user. Planned integration via LiveKit plugins.

---

## Schemas

All data models live in `schemas/` and use **Pydantic v2**.

- **`intents.py`** — `IntentType` enum + one Pydantic model per intent type, all inheriting from `BaseIntent` (which carries `confidence`, `low_confidence_fields`, `raw_transcript`, and `language`).
- **`responses.py`** — `VoiceResponse` with built-in consistency validators (e.g., can't have `success=True` with an error, can't have `draft_payload` without `requires_confirmation`).

---

## Session Management

Defined in `agent/session.py`:

- **`UserContext`** — immutable per-session: `user_id`, `company_id`, `role`, `company_slug`.
- **`SessionState`** — mutable conversation state:
  - `pending_draft` — holds an invoice draft until confirmed/cancelled.
  - `confirmation_pending` — flags that a confirm/cancel is expected.
  - `awaiting_field` / `pending_transcript` — enables multi-turn clarification (e.g., *"What is the unit price?"* → user answers → combined transcript re-parsed by NLU).

---

## API Layer & Mock System

| File | Description |
|------|-------------|
| `api/mock_saas_client.py` | `MockSaasClient` — async façade over the local JSON DB, with entity resolution cascade |
| `api/saas_client.py` | **Placeholder** for the real HTTP client (`httpx`) that will hit the production SaaS API |
| `mock/db.py` | In-memory database backed by JSON seed files, with CRUD operations, pagination, and invoice number sequencing |

**Multi-tenancy:** All queries are scoped by `company_id`. The mock DB enforces tenant isolation — a user from ACME cannot see NOVA's clients or invoices.

**Mock data files:**

| File | Content |
|------|---------|
| `companies.json` | 2 companies (ACME, NOVA) |
| `users.json` | 3 users across roles (ADMIN, OWNER) |
| `clients.json` | Sample client records per company |
| `products.json` | Sample product catalog per company |
| `invoices.json` | Sample invoices in various statuses |

---

## Server (WIP)

The `server/` directory is scaffolded for a **FastAPI** application that will expose:

- **WebSocket audio proxy** — stream browser audio to the STT pipeline.
- **LiveKit token endpoint** — issue LiveKit room tokens for real-time voice sessions.
- **JWT auth middleware** — validate tokens from the SaaS platform.

All files are currently placeholders.

---

## Testing

Tests use **pytest** + **pytest-asyncio** and are located in `tests/`.

| Test file | What it covers |
|-----------|----------------|
| `test_dispatcher.py` | Intent routing, confidence gating, draft confirm/cancel flows, clarification logic |
| `test_nlu.py` | Intent extraction from sample transcripts (French/English) |
| `test_normalizer.py` | Invoice number normalization edge cases |
| `test_rag.py` | Embedding + Qdrant indexing + similarity retrieval |
| `test_responses_schema.py` | `VoiceResponse` Pydantic validators |
| `test_mock_db_tenant_isolation.py` | Multi-tenant data isolation in the mock DB |

Run all tests:

```bash
pytest
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- A Deepgram API key (for STT)
- A Groq API key (for NLU)
- *(Optional)* Qdrant instance (for RAG — can be local or cloud)
- *(Optional)* `sounddevice` + `numpy` for microphone testing

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd "voice assistant agent"

# 2. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download spaCy models (required for PII redaction)
python -m spacy download en_core_web_lg
python -m spacy download fr_core_news_lg

# 5. Configure environment
cp .env.example .env
# Edit .env and fill in your API keys
```

### Running

```bash
# Text-based REPL (no mic needed — type transcripts manually)
python -m agent.main

# Live microphone → STT console (requires sounddevice)
pip install sounddevice numpy
python -m agent.stt_console

# Run tests
pytest
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPGRAM_API_KEY` | Yes (for STT) | Deepgram API key for Nova-3 streaming |
| `GROQ_API_KEY` | Yes (for NLU) | Groq API key for LLM intent extraction |
| `QDRANT_ENDPOINT` | For RAG | Qdrant server URL |
| `QDRANT_API_KEY` | For RAG | Qdrant API key (if cloud) |
| `LIVEKIT_URL` | Future | LiveKit server URL |
| `LIVEKIT_API_KEY` | Future | LiveKit API key |
| `LIVEKIT_API_SECRET` | Future | LiveKit API secret |
| `SAAS_API_BASE_URL` | Future | Production SaaS API base URL |
| `SAAS_API_TEST_TOKEN` | Future | Test token for the SaaS API |
| `ENV` | Optional | Environment flag (e.g., `dev`, `prod`) |
