"""FastAPI app factory + lifespan.

Wires the registry, broadcaster, and queue into `app.state` so route handlers
can reach them via `request.app.state`. The default runner (passed to JobQueue)
delegates to the project-centric `pipeline.run_extract` / `run_rerun`.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI

from neme_extractor.server.events import Broadcaster, Event
from neme_extractor.server.queue import JobQueue
from neme_extractor.server.registry import ProjectRegistry
from neme_extractor.storage.project import Project


def default_state_dir() -> Path:
    return Path.home() / ".neme-extractor"


async def _pipeline_runner(
    job_id: str,
    payload: dict,
    broadcaster: Broadcaster,
    cancel_token: asyncio.Event,
) -> None:
    """Run a pipeline job (extract or rerun) in a worker thread.

    The pipeline is synchronous + GPU-bound. Run it via asyncio.to_thread so
    the event loop stays responsive for WebSocket fan-out.
    """
    from neme_extractor.pipeline import run_extract, run_rerun

    kind = payload["kind"]  # "extract" | "rerun"
    project_folder = Path(payload["project_folder"])
    project = Project.load(project_folder)

    await broadcaster.publish(Event(
        type="job.progress",
        payload={"job_id": job_id, "project": project.slug,
                 "stage": "starting", "pct": 0},
    ))

    def _do_work() -> None:
        if kind == "extract":
            run_extract(project=project, source_idx=int(payload["source_idx"]))
        elif kind == "rerun":
            run_rerun(project=project, video_stem=str(payload["video_stem"]))
        else:
            raise ValueError(f"unknown job kind: {kind!r}")

    await asyncio.to_thread(_do_work)

    await broadcaster.publish(Event(
        type="job.done",
        payload={"job_id": job_id, "project": project.slug},
    ))


def create_app(*, state_dir: Path | None = None) -> FastAPI:
    state_dir = (state_dir or default_state_dir())
    state_dir.mkdir(parents=True, exist_ok=True)

    registry = ProjectRegistry(state_dir / "db.sqlite")
    broadcaster = Broadcaster()
    queue = JobQueue(runner=_pipeline_runner, broadcaster=broadcaster)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await queue.start()
        try:
            yield
        finally:
            await queue.stop()

    app = FastAPI(title="neme-extractor", lifespan=lifespan)
    app.state.registry = registry
    app.state.broadcaster = broadcaster
    app.state.queue = queue
    app.state.state_dir = state_dir

    @app.get("/api/health")
    async def health() -> dict:
        return {"ok": True}

    # Routers added later (Tasks 6-10) — currently stubs.
    from neme_extractor.server.api import projects, sources, refs, frames
    from neme_extractor.server.api import queue as queue_routes
    from neme_extractor.server.api import ws as ws_routes
    app.include_router(projects.router)
    app.include_router(sources.router)
    app.include_router(refs.router)
    app.include_router(frames.router)
    app.include_router(queue_routes.router)
    app.include_router(ws_routes.router)

    # Static SPA fallback — must be added LAST so /api/* routes win.
    static_dir = Path(__file__).parent / "static"
    if static_dir.exists() and (static_dir / "index.html").exists():
        from starlette.responses import FileResponse
        from starlette.staticfiles import StaticFiles

        # Mount asset files under /assets/* so the SPA's hashed bundle URLs work.
        if (static_dir / "assets").exists():
            app.mount("/assets", StaticFiles(directory=static_dir / "assets"), name="assets")

        @app.get("/", include_in_schema=False)
        @app.get("/{full_path:path}", include_in_schema=False)
        async def spa_fallback(full_path: str = "") -> FileResponse:
            # Don't intercept API requests — return 404 for those.
            if full_path.startswith("api/"):
                from fastapi import HTTPException
                raise HTTPException(status_code=404)
            return FileResponse(static_dir / "index.html")

    return app
