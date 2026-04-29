"""SQLite project registry — the global list of known projects."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from neme_extractor.storage.project import Project


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    slug            TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    folder          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    last_opened_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS projects_last_opened ON projects(last_opened_at DESC);
"""


@dataclass(frozen=True)
class RegistryEntry:
    slug: str
    name: str
    folder: str
    created_at: str
    last_opened_at: str


class ProjectRegistry:
    """SQLite-backed list of known projects.

    The registry holds only POINTERS — slug, display name, folder path,
    timestamps. The authoritative project state lives in `<folder>/project.json`,
    so deleting a project from the registry does not delete its files.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def register(self, project: Project) -> RegistryEntry:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO projects (slug, name, folder, created_at, last_opened_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(slug) DO UPDATE SET "
                "name = excluded.name, folder = excluded.folder, "
                "last_opened_at = excluded.last_opened_at",
                (project.slug, project.name, str(project.root.resolve()),
                 project.created_at.isoformat(), now),
            )
        got = self.get(project.slug)
        assert got is not None
        return got

    def unregister(self, slug: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM projects WHERE slug = ?", (slug,))

    def get(self, slug: str) -> RegistryEntry | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT slug, name, folder, created_at, last_opened_at "
                "FROM projects WHERE slug = ?", (slug,),
            ).fetchone()
        if row is None:
            return None
        return RegistryEntry(**dict(row))

    def list(self) -> list[RegistryEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT slug, name, folder, created_at, last_opened_at "
                "FROM projects ORDER BY last_opened_at DESC"
            ).fetchall()
        return [RegistryEntry(**dict(r)) for r in rows]

    def touch(self, slug: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE projects SET last_opened_at = ? WHERE slug = ?",
                (datetime.now(timezone.utc).isoformat(), slug),
            )
