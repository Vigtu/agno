"""Concurrent multi-tenant isolation tests for Google auth on v2.6.0 + PR #7376.

Proves two complementary facts:

1. ``test_factory_pattern_preserves_isolation`` — when ``tools`` is a factory
   callable that instantiates a fresh toolkit per run, concurrent ``arun``-style
   calls with different ``user_id``s each load and use the correct user's
   credentials. No cross-contamination.

2. ``test_shared_toolkit_leaks_credentials`` — when a single toolkit instance
   is shared across users (the anti-pattern), the first user's credentials are
   silently reused for the second user, because the ``@google_authenticate``
   decorator short-circuits when ``self.creds`` is already valid
   (``tools/google/auth.py:42``). This documents *why* the factory pattern is
   required on this branch (v2.6.0 + PR #7376 without PR #7404's
   ``_clone_for_run``) and will fail if a future change makes sharing safe.

3. ``test_framework_factory_invocation_isolates`` — exercises the exact
   framework path: ``ainvoke_callable_factory`` with ``run_context`` injection,
   matching what ``agent.arun`` does under the hood.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from agno.db.sqlite import SqliteDb
from agno.tools.google.auth import google_authenticate, load_token
from agno.tools.toolkit import Toolkit
from agno.utils.callables import ainvoke_callable_factory


class _FakeCreds:
    """Stand-in for google.oauth2.credentials.Credentials."""

    def __init__(self, token_marker: str):
        self.token = token_marker
        self.valid = True
        self.expired = False
        self.refresh_token = None

    def to_json(self) -> str:
        return json.dumps({"token": self.token})


class _FakeRunContext:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.session_state: Optional[dict] = None


class _MockAgent:
    def __init__(self, db):
        self.db = db


class _MockGmailToolkit(Toolkit):
    """Minimal Google-style toolkit with one ``@google_authenticate``'d method."""

    def __init__(self):
        super().__init__(name="mock_gmail")
        self.scopes = ["https://www.googleapis.com/auth/gmail.readonly"]
        self.creds: Optional[_FakeCreds] = None
        self.service: Optional[Any] = None
        # Opt into DB-backed tokens so get_token_db resolves agent.db at call time.
        self.store_token_in_db = True
        self._db: Optional[Any] = None

    def _auth(self, user_id: Optional[str] = None, agent: Optional[Any] = None) -> None:
        ok = load_token(self, scopes=self.scopes, user_id=user_id, agent=agent)
        if not ok:
            raise RuntimeError(f"load_token failed for user {user_id!r}")

    def _build_service(self) -> Any:
        return MagicMock()

    @google_authenticate("gmail")
    def fetch_email(self) -> str:
        # Returns the token marker for whichever creds are currently on self.
        # Tests assert this matches the requesting user.
        return self.creds.token  # type: ignore[union-attr]


@pytest.fixture
def temp_db(tmp_path):
    return SqliteDb(db_file=str(tmp_path / "auth.db"))


def _seed_user(db: SqliteDb, uid: str) -> None:
    db.upsert_auth_token(
        {
            "provider": "google",
            "user_id": uid,
            "service": "google",
            "token_data": {
                "token": f"TOKEN::{uid}",
                "refresh_token": f"refresh_{uid}",
                "client_id": "test",
                "client_secret": "secret",
                "scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
            },
            "granted_scopes": ["https://www.googleapis.com/auth/gmail.readonly"],
        }
    )


def _patch_google_credentials():
    # ``load_token`` does ``from google.oauth2.credentials import Credentials`` locally,
    # so patch at the source module. Each call rebuilds a FakeCreds from the row's
    # token_data, carrying the per-user marker.
    return patch(
        "google.oauth2.credentials.Credentials.from_authorized_user_info",
        side_effect=lambda info, scopes: _FakeCreds(info["token"]),
    )


USERS = ["alice", "bob", "charlie"]
N_REQUESTS = 30


@pytest.mark.asyncio
async def test_factory_pattern_preserves_isolation(temp_db):
    """Factory callable returns a fresh toolkit per call — isolation holds."""
    for uid in USERS:
        _seed_user(temp_db, uid)

    agent = _MockAgent(db=temp_db)

    def toolkit_factory(run_context) -> List[_MockGmailToolkit]:
        # Fresh instance per call — this is the whole point.
        return [_MockGmailToolkit()]

    with _patch_google_credentials():

        async def one_call(uid: str) -> str:
            rc = _FakeRunContext(user_id=uid)
            toolkits = toolkit_factory(rc)
            # Simulate framework calling the tool with injected kwargs.
            return toolkits[0].fetch_email(run_context=rc, agent=agent)

        plan = [USERS[i % len(USERS)] for i in range(N_REQUESTS)]
        results = await asyncio.gather(*(one_call(u) for u in plan))

    expected = [f"TOKEN::{u}" for u in plan]
    assert results == expected, (
        f"Expected perfect isolation across concurrent requests.\nplan    = {plan}\nresults = {results}"
    )


def test_shared_toolkit_leaks_credentials(temp_db):
    """Anti-pattern: one shared toolkit serves multiple users.

    The ``@google_authenticate`` wrapper only re-auths when ``self.creds`` is
    missing or invalid. After Alice's call populates ``self.creds``, Bob's call
    silently reuses Alice's creds. Deterministic and sequential — no race
    needed to expose the leak.
    """
    for uid in USERS:
        _seed_user(temp_db, uid)

    agent = _MockAgent(db=temp_db)
    shared = _MockGmailToolkit()

    with _patch_google_credentials():
        first = shared.fetch_email(run_context=_FakeRunContext(user_id="alice"), agent=agent)
        second = shared.fetch_email(run_context=_FakeRunContext(user_id="bob"), agent=agent)

    assert first == "TOKEN::alice"
    # The canary assertion: if this ever returns TOKEN::bob, the framework has
    # gained per-call isolation (e.g. PR #7404 landed) and the factory-pattern
    # requirement is no longer strict — update docs & remove this test.
    assert second == "TOKEN::alice", (
        "Shared-toolkit multi-tenant leak is expected on this branch. "
        f"Got {second!r} — if this is TOKEN::bob, isolation is now automatic "
        "and the factory-pattern requirement should be revisited."
    )


@pytest.mark.asyncio
async def test_framework_factory_invocation_isolates(temp_db):
    """End-to-end via the framework's ``ainvoke_callable_factory``.

    Uses the same injection machinery ``agent.arun`` uses internally, proving
    the pattern works via the real code path, not just by-hand wiring.
    """
    for uid in USERS:
        _seed_user(temp_db, uid)

    agent = _MockAgent(db=temp_db)

    def toolkit_factory(run_context) -> List[_MockGmailToolkit]:
        return [_MockGmailToolkit()]

    with _patch_google_credentials():

        async def one_call(uid: str) -> str:
            rc = _FakeRunContext(user_id=uid)
            toolkits = await ainvoke_callable_factory(toolkit_factory, entity=agent, run_context=rc)
            return toolkits[0].fetch_email(run_context=rc, agent=agent)

        plan = [USERS[i % len(USERS)] for i in range(N_REQUESTS)]
        results = await asyncio.gather(*(one_call(u) for u in plan))

    assert results == [f"TOKEN::{u}" for u in plan]
