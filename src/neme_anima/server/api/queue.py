"""REST routes for /api/queue."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request, Response

router = APIRouter(prefix="/api/queue", tags=["queue"])


@router.get("")
async def list_queue(request: Request) -> list[dict]:
    snap = request.app.state.queue.snapshot()
    return [
        {"job_id": j.job_id, "status": j.status.value,
         "payload": j.payload, "error": j.error}
        for j in snap
    ]


@router.delete("/{job_id}", status_code=204)
async def cancel(request: Request, job_id: str) -> Response:
    ok = await request.app.state.queue.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"unknown job_id: {job_id}")
    return Response(status_code=204)


@router.post("/{job_id}/resume", status_code=204)
async def resume(request: Request, job_id: str) -> Response:
    """Release a job that's parked at ``progress.wait_for_resume()``."""
    progresses = request.app.state.active_progresses
    progress = progresses.get(job_id)
    if progress is None:
        raise HTTPException(
            status_code=404,
            detail=f"no running job with id {job_id} (already finished?)",
        )
    if not progress.is_paused:
        raise HTTPException(status_code=409, detail="job is not paused")
    progress.resume()
    return Response(status_code=204)
