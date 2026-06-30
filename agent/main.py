"""
agent/main.py

Manual text-mode harness for the voice-invoicing pipeline, used while STT/TTS and
real per-call session storage aren't wired up yet.

Flow exercised here: transcript (typed) -> NLU (Groq) -> dispatcher -> VoiceResponse,
with a single in-memory SessionState held for the life of the process.

Requires GROQ_API_KEY to be set (loaded from .env). For testing dispatcher logic
without hitting the network or depending on LLM output, use
scripts/test_dispatcher_scenarios.py instead.
"""

from dotenv import load_dotenv
load_dotenv()  # must happen before importing agent.nlu, which builds the Groq client at import time

import asyncio

from agent.session import SessionState, UserContext
from agent.nlu import extract_intent
from agent.dispatcher import dispatch
from api.mock_saas_client import MockSaasClient
from mock.db import get_user_by_id, get_company_by_id

MOCK_USERS = {
    "1": 1,  # admin@acme.tn   / ADMIN / ACME
    "2": 2,  # admin@nova.tn   / ADMIN / NOVA
    "3": 3,  # owner@platform.tn / OWNER / ACME
}


def _build_session(user_id: int) -> SessionState:
    user_row = get_user_by_id(user_id)
    if not user_row:
        raise ValueError(f"No mock user with id {user_id}")
    company = get_company_by_id(user_row["company_id"])
    if not company:
        raise ValueError(f"No mock company with id {user_row['company_id']}")

    ctx = UserContext(
        user_id=user_row["id"],
        company_id=user_row["company_id"],
        role=user_row["role"],
        company_slug=company["slug"],
    )
    return SessionState(user=ctx)


def _select_user() -> int:
    print("Select a mock user to test as:")
    print("  1) admin@acme.tn     (ADMIN, ACME)")
    print("  2) admin@nova.tn     (ADMIN, NOVA)")
    print("  3) owner@platform.tn (OWNER, ACME)")
    choice = input("> ").strip()
    return MOCK_USERS.get(choice, 1)


def _print_response(resp) -> None:
    print(f"\n🗣  {resp.spoken_text}")
    if resp.clarification_question:
        print(f"   [clarification needed: {resp.clarification_question}]")
    if resp.requires_confirmation:
        print("   [awaiting confirm/cancel]")
    if resp.error:
        print(f"   [error: {resp.error}]")
    if resp.display_data is not None:
        print(f"   [display_data: {resp.display_data}]")
    print()


async def repl() -> None:
    user_id = _select_user()
    session = _build_session(user_id)
    saas = MockSaasClient(company_id=session.user.company_id, user_id=session.user.user_id)

    print(
        f"\nSession started — user_id={session.user.user_id} "
        f"company={session.user.company_slug} role={session.user.role}"
    )
    print("Type a transcript and hit enter. Type 'switch' to change user, 'exit' to quit.\n")

    while True:
        try:
            transcript = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return

        if not transcript:
            continue
        if transcript.lower() in ("exit", "quit"):
            print("bye")
            return
        if transcript.lower() == "switch":
            user_id = _select_user()
            session = _build_session(user_id)
            saas = MockSaasClient(company_id=session.user.company_id, user_id=session.user.user_id)
            print(f"Switched to user_id={session.user.user_id} company={session.user.company_slug}\n")
            continue

        intent = extract_intent(transcript)
        print(f"   [intent: {intent.intent}  confidence={intent.confidence}]")

        resp = await dispatch(intent, saas, session)
        _print_response(resp)


if __name__ == "__main__":
    asyncio.run(repl())