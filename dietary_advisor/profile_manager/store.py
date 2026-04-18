"""SQLite-backed persistence for `UserProfile`.

We store the full profile as a JSON blob because (a) we always need the whole
thing at once and (b) the schema is governed by Pydantic which already gives
us validated round-trip serialization.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dietary_advisor.config import get_settings
from dietary_advisor.schemas.profile import UserProfile


class ProfileStore:
    """Single-table key-value store keyed by `user_id`."""

    def __init__(self, db_path: Path | None = None) -> None:
        settings = get_settings()
        self._path = db_path or settings.profile_db
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS profiles (
                    user_id TEXT PRIMARY KEY,
                    payload TEXT NOT NULL
                )
                """,
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self._path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def upsert(self, profile: UserProfile) -> None:
        payload = profile.model_dump_json()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO profiles (user_id, payload) VALUES (?, ?)",
                (profile.user_id, payload),
            )

    def get(self, user_id: str) -> UserProfile | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM profiles WHERE user_id = ?", (user_id,)).fetchone()
        if row is None:
            return None
        return UserProfile.model_validate_json(row[0])

    def delete(self, user_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM profiles WHERE user_id = ?", (user_id,))
            return cur.rowcount > 0

    def list_all(self) -> list[UserProfile]:
        with self._connect() as conn:
            rows = conn.execute("SELECT payload FROM profiles ORDER BY user_id").fetchall()
        return [UserProfile.model_validate_json(r[0]) for r in rows]

    def import_json(self, json_payload: str) -> UserProfile:
        """Validate + upsert a JSON blob produced by `export_json`."""
        profile = UserProfile.model_validate_json(json_payload)
        self.upsert(profile)
        return profile

    def export_json(self, user_id: str) -> str | None:
        profile = self.get(user_id)
        if profile is None:
            return None
        return profile.model_dump_json(indent=2)

    def import_dir(self, dir_path: Path) -> list[UserProfile]:
        """Bulk-import every `*.json` file in a directory (used for test fixtures)."""
        loaded: list[UserProfile] = []
        for p in sorted(dir_path.glob("*.json")):
            data = p.read_text(encoding="utf-8")
            # Allow both bare profile JSON and {"profile": {...}, ...} envelopes.
            obj = json.loads(data)
            if isinstance(obj, dict) and "profile" in obj and isinstance(obj["profile"], dict):
                obj = obj["profile"]
            profile = UserProfile.model_validate(obj)
            self.upsert(profile)
            loaded.append(profile)
        return loaded
