"""The deterministic suite must never invoke a real model provider."""
import pytest
from openai import OpenAI
from typesafe_sdk import TypeSafeClient


@pytest.fixture(autouse=True)
def block_live_model_calls(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Live model call in an offline test: inject a fake client or actor")
    monkeypatch.setattr(OpenAI, "__init__", forbidden)
    monkeypatch.setattr(TypeSafeClient, "system_one", forbidden)
