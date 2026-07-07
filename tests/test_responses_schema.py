import pytest
from pydantic import ValidationError
from schemas.responses import VoiceResponse


def test_success_with_error_raises():
    with pytest.raises(ValidationError):
        VoiceResponse(success=True, spoken_text="ok", error="something_wrong")


def test_draft_payload_without_confirmation_flag_raises():
    with pytest.raises(ValidationError):
        VoiceResponse(
            spoken_text="draft ready",
            draft_payload={"foo": "bar"},
            requires_confirmation=False,
        )


def test_draft_payload_with_confirmation_flag_ok():
    resp = VoiceResponse(
        spoken_text="draft ready",
        draft_payload={"foo": "bar"},
        requires_confirmation=True,
    )
    assert resp.requires_confirmation is True


def test_plain_success_response_ok():
    resp = VoiceResponse(spoken_text="all good")
    assert resp.success is True
    assert resp.error is None


def test_failure_with_error_ok():
    resp = VoiceResponse(success=False, spoken_text="nope", error="not_found")
    assert resp.success is False