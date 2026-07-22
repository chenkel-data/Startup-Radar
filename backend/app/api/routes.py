from fastapi import APIRouter, Body, HTTPException, Query, Request

from app.models.extraction import (
    ClaimReviewIn,
    EntityAliasIn,
    EntityAliasResult,
    EntityDescriptionReviewIn,
    EntityDescriptionReviewResult,
    FeedbackIn,
    GraphResponse,
    IngestRequest,
    SearchResult,
    TaskStatus,
)
from app.observability import log_extraction_feedback
from app.graph.admin_store import AdminStore
from app.graph.claim_store import ClaimStore
from app.graph.entity_profile_store import EntityProfileStore
from app.graph.entity_resolution_store import AmbiguousAliasError, EntityResolutionStore
from app.graph.graph_read_store import GraphReadStore
from app.graph.insight_store import InsightStore
from app.services.entity_curation import profile_embedding_input_hash, profile_embedding_text
from app.services.ingestion import IngestionService
from app.services.tasks import TaskManager

router = APIRouter()


def _admin(request: Request) -> AdminStore:
    return request.app.state.admin_store


def _claims(request: Request) -> ClaimStore:
    return request.app.state.claim_store


def _entity_resolution(request: Request) -> EntityResolutionStore:
    return request.app.state.entity_resolution_store


def _entity_profiles(request: Request) -> EntityProfileStore:
    return request.app.state.entity_profile_store


def _graph_read(request: Request) -> GraphReadStore:
    return request.app.state.graph_read_store


def _insights(request: Request) -> InsightStore:
    return request.app.state.insight_store


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
    await _admin(request).apply_schema()
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
    if not status or status.name != "ingest":
        raise HTTPException(status_code=404, detail="Task not found")
    return status


@router.get("/search", response_model=list[SearchResult])
async def search(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(default=15, ge=1, le=50),
) -> list[SearchResult]:
    return await _graph_read(request).search(q, limit)


@router.get("/startup/{name:path}")
async def startup(request: Request, name: str) -> dict:
    profile = await _graph_read(request).entity_profile("Startup", name)
    if not profile:
        raise HTTPException(status_code=404, detail="Startup not found")
    return profile


@router.get("/investor/{name:path}")
async def investor(request: Request, name: str) -> dict:
    graph = _graph_read(request)
    for label in ("Investor", "Company", "Person"):
        profile = await graph.entity_profile(label, name)
        if not profile:
            continue
        if "investor" in profile.get("roles", []):
            return profile
    raise HTTPException(status_code=404, detail="Investor not found")


@router.delete("/graph")
async def clear_graph(request: Request) -> dict:
    """Delete every node and relationship in Neo4j for a clean re-ingest run."""
    deleted = await _admin(request).clear_all()
    return {"status": "cleared", "deleted_nodes": deleted}


@router.get("/graph", response_model=GraphResponse)
async def graph(
    request: Request,
    entity: str | None = Query(default=None),
    limit: int = Query(default=120, ge=10, le=500),
    view: str = Query(default="landscape", pattern="^(landscape|feed)$"),
) -> GraphResponse:
    return await _graph_read(request).graph(entity=entity, limit=limit, view=view)


@router.get("/entities/counts")
async def entity_counts(request: Request) -> dict[str, int]:
    return await _graph_read(request).entity_counts()


@router.get("/nodes/{node_id}/claims")
async def node_claims(request: Request, node_id: str) -> dict:
    claims = await _claims(request).node_claims(node_id)
    if not claims:
        raise HTTPException(status_code=404, detail="Node not found")
    return claims


@router.post("/nodes/{node_id}/aliases", response_model=EntityAliasResult)
async def add_entity_alias(
    request: Request,
    node_id: str,
    body: EntityAliasIn,
) -> EntityAliasResult:
    try:
        result = await _entity_resolution(request).add_alias(node_id, body.alias)
    except AmbiguousAliasError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return EntityAliasResult.model_validate(result)


@router.delete("/nodes/{node_id}/aliases", response_model=EntityAliasResult)
async def remove_entity_alias(
    request: Request,
    node_id: str,
    alias: str = Query(min_length=1, max_length=200),
) -> EntityAliasResult:
    try:
        result = await _entity_resolution(request).remove_alias(node_id, alias)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return EntityAliasResult.model_validate(result)


@router.post(
    "/nodes/{node_id}/description/review",
    response_model=EntityDescriptionReviewResult,
)
async def review_entity_description(
    request: Request,
    node_id: str,
    body: EntityDescriptionReviewIn,
) -> EntityDescriptionReviewResult:
    store = _entity_profiles(request)
    entity = await store.description_review_input(node_id)
    if entity is None:
        raise HTTPException(status_code=404, detail="Node not found")
    if body.description is None and not str(entity.get("current_description") or "").strip():
        raise HTTPException(status_code=400, detail="Entity has no description to review")

    embedding = None
    embedding_input_hash = None
    embedding_model = None
    embedding_service = request.app.state.embedding
    if body.description is not None and embedding_service is not None:
        embedding_text = profile_embedding_text(entity, body.description)
        embedding = await embedding_service.embed_one(embedding_text)
        embedding_input_hash = profile_embedding_input_hash(entity, body.description)
        settings = request.app.state.settings
        embedding_model = (
            settings.embedding_st_model
            if settings.embedding_provider == "sentence-transformers"
            else settings.embedding_model
        )

    result = await store.review_description(
        entity_id=node_id,
        decision=body.decision,
        description=body.description,
        comment=body.comment,
        reviewer=body.reviewer,
        embedding=embedding,
        embedding_input_hash=embedding_input_hash,
        embedding_model=embedding_model,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return EntityDescriptionReviewResult.model_validate(result)


@router.post("/claims/review")
async def review_claim(request: Request, body: ClaimReviewIn) -> dict:
    reviewed = await _claims(request).review_claim(
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
    return await _insights(request).trending_startups(days=days, limit=limit)


@router.get("/insights/top-investors")
async def top_investors(
    request: Request,
    limit: int = Query(default=10, ge=1, le=50),
) -> list[dict]:
    return await _insights(request).top_investors(limit=limit)


@router.get("/insights/co-investments")
async def co_investments(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    return await _insights(request).co_investments(limit=limit)


@router.get("/insights/topic-clusters")
async def topic_clusters(
    request: Request,
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    return await _insights(request).topic_clusters(limit=limit)


@router.post("/traces/{trace_id}/feedback")
async def trace_feedback(trace_id: str, body: FeedbackIn) -> dict:
    """Attach a human assessment to one MLflow trace.

    The assessment appears in the trace's **Assessments** tab in MLflow.
    When MLflow is disabled, the endpoint returns ``{status: "skipped"}``.
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
