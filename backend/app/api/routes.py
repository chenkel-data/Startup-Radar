from fastapi import APIRouter, Body, HTTPException, Query, Request

from app.models.extraction import (
    ClaimReviewIn,
    FeedbackIn,
    GraphResponse,
    IngestRequest,
    SearchResult,
    TaskStatus,
)
from app.observability import log_extraction_feedback
from app.graph.graph_store import GraphStore
from app.services.ingestion import IngestionService
from app.services.tasks import TaskManager

router = APIRouter()


def _graph(request: Request) -> GraphStore:
    return request.app.state.graph


def _ingestion(request: Request) -> IngestionService:
    return request.app.state.ingestion


def _tasks(request: Request) -> TaskManager:
    return request.app.state.tasks


@router.get("/health")
async def health(request: Request) -> dict:
    try:
        await request.app.state.neo4j.verify()
        database = "connected"
    except Exception as exc:
        database = f"disconnected: {exc}"
    return {
        "status": "ok",
        "database": database,
        "app": request.app.state.settings.app_name,
    }


@router.post("/schema/apply")
async def apply_schema(request: Request) -> dict:
    await _graph(request).apply_schema()
    return {"status": "applied"}


@router.post("/ingest", response_model=TaskStatus)
async def start_ingest(
    request: Request,
    payload: IngestRequest = Body(default_factory=IngestRequest),
) -> TaskStatus:
    task = _tasks(request).start(
        "ingest",
        lambda task_id: _ingestion(request).ingest(payload, ingest_run_id=task_id),
    )
    return task


@router.get("/ingest/{task_id}", response_model=TaskStatus)
async def ingest_status(request: Request, task_id: str) -> TaskStatus:
    status = _tasks(request).get(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("/search", response_model=list[SearchResult])
async def search(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=15, ge=1, le=50),
) -> list[SearchResult]:
    return await _graph(request).search(q, limit)


@router.get("/startup/{name:path}")
async def startup(request: Request, name: str) -> dict:
    profile = await _graph(request).entity_profile("Startup", name)
    if not profile:
        raise HTTPException(status_code=404, detail="Startup not found")
    return profile


@router.get("/investor/{name:path}")
async def investor(request: Request, name: str) -> dict:
    graph = _graph(request)
    for label in ("Investor", "Company", "Person"):
        profile = await graph.entity_profile(label, name)
        if not profile:
            continue
        if label == "Investor" or any(
            item.get("relationship") == "INVESTED_IN" for item in profile.get("related", [])
        ):
            return profile
    raise HTTPException(status_code=404, detail="Investor not found")


@router.delete("/graph")
async def clear_graph(request: Request) -> dict:
    """Delete every node and relationship in Neo4j for a clean re-ingest run."""
    deleted = await _graph(request).clear_all()
    return {"status": "cleared", "deleted_nodes": deleted}


@router.get("/graph", response_model=GraphResponse)
async def graph(
    request: Request,
    entity: str | None = Query(default=None),
    limit: int = Query(default=120, ge=10, le=500),
    view: str = Query(default="landscape", pattern="^(landscape|feed)$"),
) -> GraphResponse:
    return await _graph(request).graph(entity=entity, limit=limit, view=view)


@router.get("/nodes/{node_id}/claims")
async def node_claims(request: Request, node_id: str) -> dict:
    claims = await _graph(request).node_claims(node_id)
    if not claims:
        raise HTTPException(status_code=404, detail="Node not found")
    return claims


@router.post("/claims/review")
async def review_claim(request: Request, body: ClaimReviewIn) -> dict:
    reviewed = await _graph(request).review_claim(
        source_id=body.source_id,
        relationship=body.relationship,
        target_id=body.target_id,
        decision=body.decision,
        comment=body.comment,
        reviewer=body.reviewer,
    )
    if not reviewed:
        raise HTTPException(status_code=404, detail="Claim not found")
    return {"status": "ok", "decision": body.decision}


@router.get("/insights/trending-startups")
async def trending_startups(
    request: Request,
    days: int = Query(default=30, ge=1, le=365),
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict]:
    return await _graph(request).trending_startups(days=days, limit=limit)


@router.get("/insights/top-investors")
async def top_investors(
    request: Request,
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict]:
    return await _graph(request).top_investors(limit=limit)


@router.get("/insights/co-investments")
async def co_investments(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    return await _graph(request).co_investments(limit=limit)


@router.get("/insights/topic-clusters")
async def topic_clusters(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    return await _graph(request).topic_clusters(limit=limit)


@router.post("/traces/{trace_id}/feedback")
async def trace_feedback(trace_id: str, body: FeedbackIn) -> dict:
    """Attach a human assessment to one MLflow trace.

    Shows up in the trace's **Assessments** tab in the MLflow UI. Used by
    the frontend's "Flag this trace" affordance. Non-fatal if MLflow is
    disabled — returns ``{status: "skipped"}`` so the caller can still
    show a friendly UI message.
    """
    if not trace_id.strip():
        raise HTTPException(status_code=400, detail="trace_id is required")
    ok = log_extraction_feedback(
        trace_id=trace_id,
        label=body.label,
        target=body.target,
        comment=body.comment,
        reviewer=body.reviewer,
    )
    return {
        "status": "ok" if ok else "skipped",
        "trace_id": trace_id,
        "label": body.label,
        "target": body.target,
    }
