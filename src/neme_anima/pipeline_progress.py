"""Stage-progress callback interface for the pipeline.

The pipeline runs on a worker thread inside the server, but is also invoked
directly from the CLI. Both paths import :class:`PipelineProgress` and call its
methods at stage boundaries; the no-op default lets the CLI run without any
plumbing while the server installs a real reporter that publishes events to
the WebSocket broadcaster.
"""

from __future__ import annotations


class PipelineProgress:
    """No-op base class. Override the methods you care about."""

    def stage_start(self, key: str, label: str, *, total: int = 0, message: str = "") -> None:
        ...

    def stage_advance(self, key: str, n: int = 1) -> None:
        ...

    def stage_message(self, key: str, message: str) -> None:
        ...

    def stage_done(self, key: str, *, message: str = "") -> None:
        ...

    def stage_fail(self, key: str, error: str) -> None:
        ...

    def finish(self, summary: dict | None = None) -> None:
        ...

    def wait_for_resume(self, *, message: str = "") -> None:
        """Block the runner thread until something calls resume(). Default is
        a no-op so non-server callers (CLI, tests) never pause.
        """
        ...


NULL_PROGRESS: PipelineProgress = PipelineProgress()


# Canonical stage definitions. The server pre-publishes these in PENDING state
# so the UI can render the full pipeline from the moment a job starts running.
EXTRACT_STAGES: list[tuple[str, str]] = [
    ("setup", "Setup"),
    ("scenes", "Scene detection"),
    ("detect", "Person detection"),
    ("track", "Tracking"),
    ("identify", "Identify · select · save"),
    ("dedup", "Dedup"),
    ("tag", "Tagging"),
]

RERUN_STAGES: list[tuple[str, str]] = [
    ("setup", "Setup"),
    ("identify", "Identify · select · save"),
    ("dedup", "Dedup"),
    ("tag", "Tagging"),
]
