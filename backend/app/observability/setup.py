"""MLflow client startup hook.

Called once from FastAPI's `lifespan` after `setup_logging`. Pins the
tracking URI, ensures the target experiment exists, enables OpenAI
auto-instrumentation, and syncs the local prompt templates into the
MLflow prompt registry so the registry is always in sync with the code
without any manual registration step.

Failures are swallowed and downgrade the module to a no-op so a flapping
MLflow server never blocks ingestion — the kill switch is
`MLFLOW_ENABLED=false`.
"""

from __future__ import annotations

import json

import mlflow
import mlflow.openai

from app.core.config import Settings
from app.core.logging import get_logger

_logger = get_logger("observability.setup")
_enabled: bool = False


def init_mlflow(settings: Settings) -> None:
    """Configure the MLflow client for this process."""
    global _enabled

    if not settings.mlflow_enabled:
        _enabled = False
        _logger.info(
            "mlflow_disabled",
            extra={
                "event": "startup",
                "workflow_step": "observability",
                "detail": "MLFLOW_ENABLED=false; tracing and tracking are no-ops",
            },
        )
        return

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)
        if settings.mlflow_openai_autolog:
            mlflow.openai.autolog()
        _enabled = True
        _logger.info(
            "mlflow_ready",
            extra={
                "event": "startup",
                "workflow_step": "observability",
                "detail": (
                    f"uri={settings.mlflow_tracking_uri}; "
                    f"experiment={settings.mlflow_experiment_name}; "
                    f"openai_autolog={settings.mlflow_openai_autolog}"
                ),
            },
        )
        if settings.mlflow_use_prompt_registry:
            _sync_prompts(settings)
    except Exception as exc:
        _enabled = False
        _logger.warning(
            "mlflow_init_failed",
            extra={
                "event": "startup",
                "workflow_step": "observability",
                "error": str(exc),
                "detail": "tracing and tracking will be no-ops for this process",
            },
        )


def _sync_prompts(settings: Settings) -> None:
    """Register current local prompt templates and sync the configured alias.

    Called automatically on startup so the registry is always in sync with
    the code — no manual register_mlflow_prompts.py step needed.

    Only registers a new version when the template content has actually changed,
    so repeated restarts do not accumulate identical orphan versions.
    """
    from app.prompts.extraction import (
        build_extraction_prompt_registry_template,
        build_gleaning_prompt_registry_template,
    )

    entries = [
        (settings.mlflow_prompt_extraction_uri, build_extraction_prompt_registry_template),
        (settings.mlflow_prompt_gleaning_uri, build_gleaning_prompt_registry_template),
    ]
    for uri, build_fn in entries:
        name, alias = _parse_prompt_uri(uri)
        local_template = build_fn()
        try:
            existing = mlflow.genai.load_prompt(uri)
            if _template_eq(existing.template, local_template):
                _logger.debug(
                    "mlflow_prompt_up_to_date",
                    extra={
                        "event": "startup",
                        "workflow_step": "prompt_registry",
                        "detail": f"name={name}; alias={alias}; no change",
                    },
                )
                continue
        except Exception as exc:
            _logger.debug(
                "mlflow_prompt_not_yet_registered",
                extra={
                    "event": "startup",
                    "workflow_step": "prompt_registry",
                    "detail": f"name={name}; registering for the first time; reason={exc}",
                },
            )

        try:
            prompt = mlflow.genai.register_prompt(
                name=name,
                template=local_template,
                commit_message="auto-sync from local template on startup",
                tags={"app": "startup-radar"},
            )
            mlflow.genai.set_prompt_alias(name, alias, int(prompt.version))
            _logger.info(
                "mlflow_prompt_synced",
                extra={
                    "event": "startup",
                    "workflow_step": "prompt_registry",
                    "detail": f"name={name}; version={prompt.version}; alias={alias}",
                },
            )
        except Exception as exc:
            _logger.warning(
                "mlflow_prompt_sync_failed",
                extra={
                    "event": "startup",
                    "workflow_step": "prompt_registry",
                    "error": str(exc),
                    "detail": f"name={name}; uri={uri}",
                },
            )


def _template_eq(registered: object, local: list[dict]) -> bool:
    """True when the registered template content matches the local template."""
    try:
        canonical = json.dumps(registered, sort_keys=True)
        return canonical == json.dumps(local, sort_keys=True)
    except Exception as exc:
        _logger.debug(
            "mlflow_template_eq_failed",
            extra={
                "event": "startup",
                "workflow_step": "prompt_registry",
                "detail": f"could not compare templates; treating as changed; reason={exc}",
            },
        )
        return False


def _parse_prompt_uri(uri: str) -> tuple[str, str]:
    """Extract (name, alias) from e.g. 'prompts:/article_extraction@production'."""
    path = uri.removeprefix("prompts:/")
    name, _, alias = path.partition("@")
    return name, alias or "champion"


def is_enabled() -> bool:
    """True iff `init_mlflow` succeeded; safe to call before init."""
    return _enabled
