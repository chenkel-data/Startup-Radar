from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from neo4j.exceptions import ServiceUnavailable as Neo4jServiceUnavailable
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.db.neo4j import Neo4jClient
from app.observability import init_mlflow
from app.graph.graph_store import GraphStore
from app.services.embedding import build_embedding_service
from app.services.ingestion import IngestionService
from app.services.llm import LLMExtractionService
from app.services.scraper import ArticleScraper
from app.services.tasks import TaskManager

settings = get_settings()
setup_logging(settings)
init_mlflow(settings)
logger = get_logger("app")

REQUEST_COUNT = Counter(
    "startup_radar_http_requests_total",
    "Total HTTP requests",
    ["method", "path", "status"],
)
REQUEST_LATENCY = Histogram(
    "startup_radar_http_request_duration_seconds",
    "HTTP request duration",
    ["method", "path"],
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    neo4j = Neo4jClient(settings)
    graph = GraphStore(neo4j)
    llm = LLMExtractionService(settings)
    embedding = build_embedding_service(settings, llm._client)
    scraper = ArticleScraper(settings)

    app.state.settings = settings
    app.state.neo4j = neo4j
    app.state.graph = graph
    app.state.llm = llm
    app.state.scraper = scraper
    app.state.ingestion = IngestionService(
        settings=settings,
        scraper=scraper,
        llm=llm,
        graph=graph,
        embedding=embedding,
    )
    app.state.tasks = TaskManager()

    try:
        await neo4j.verify()
        logger.info("neo4j_connected", extra={"event": "startup", "workflow_step": "database"})
        if settings.apply_schema_on_startup:
            await graph.apply_schema(settings.embedding_provider)
    except Exception as exc:
        logger.warning(
            "neo4j_unavailable",
            extra={"event": "startup", "workflow_step": "database", "error": str(exc)},
        )

    yield
    await neo4j.close()


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_observability(request: Request, call_next):
    started = perf_counter()
    response = await call_next(request)
    duration = perf_counter() - started
    path = request.url.path
    REQUEST_COUNT.labels(request.method, path, str(response.status_code)).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(duration)
    log_method = (
        logger.debug if _is_status_poll(request.method, path, response.status_code) else logger.info
    )
    log_method(
        "http_request",
        extra={
            "event": "http",
            "workflow_step": "request",
            "duration_ms": round(duration * 1000, 2),
            "detail": f"{request.method} {path} {response.status_code}",
        },
    )
    return response


def _is_status_poll(method: str, path: str, status_code: int) -> bool:
    return method in {"GET", "OPTIONS"} and path.startswith("/ingest/") and status_code < 400


@app.get("/metrics")
async def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


app.include_router(router)


@app.exception_handler(Neo4jServiceUnavailable)
async def neo4j_unavailable_handler(request: Request, exc: Neo4jServiceUnavailable) -> JSONResponse:
    logger.error(
        "neo4j_request_unavailable",
        extra={
            "event": "http",
            "workflow_step": "request",
            "detail": f"{request.method} {request.url.path}",
            "error": str(exc),
        },
    )
    return JSONResponse(
        status_code=503,
        content={
            "detail": "Database is unavailable. Ensure Neo4j is running and NEO4J_URI is correct.",
            "database": "unavailable",
        },
    )
