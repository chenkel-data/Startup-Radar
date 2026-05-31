import asyncio
import importlib
from functools import partial
from typing import Protocol, runtime_checkable

from openai import AsyncOpenAI

from app.core.config import Settings
from app.core.logging import get_logger

_BATCH_SIZE = 100


@runtime_checkable
class EmbeddingService(Protocol):
    async def embed(self, texts: list[str]) -> list[list[float]]: ...
    async def embed_one(self, text: str) -> list[float]: ...


class OpenAIEmbeddingService:
    def __init__(self, settings: Settings, client: AsyncOpenAI):
        self.settings = settings
        self._client = client
        self.logger = get_logger("embedding.openai")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), _BATCH_SIZE):
            batch = texts[i : i + _BATCH_SIZE]
            response = await self._create_untraced(batch)
            results.extend(item.embedding for item in response.data)
        return results

    async def embed_one(self, text: str) -> list[float]:
        vecs = await self.embed([text])
        return vecs[0]

    async def _create_untraced(self, batch: list[str]):
        create = self._client.embeddings.create
        original_create = getattr(create, "__wrapped__", None)
        if original_create is not None:
            # mlflow.openai.autolog wraps create() and logs full embedding vectors.
            create = partial(original_create, self._client.embeddings)
        return await create(model=self.settings.embedding_model, input=batch)


class SentenceTransformerEmbeddingService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.logger = get_logger("embedding.sentence_transformers")
        self._model = None  # lazy load on first call

    def _get_model(self):
        if self._model is None:
            try:
                sentence_transformers = importlib.import_module("sentence_transformers")
            except ModuleNotFoundError as exc:
                raise RuntimeError(
                    "EMBEDDING_PROVIDER=sentence-transformers requires the optional "
                    'dependency group. Install it with: pip install -e ".[embeddings]"'
                ) from exc

            self.logger.info("loading_st_model", extra={"model": self.settings.embedding_st_model})
            self._model = sentence_transformers.SentenceTransformer(
                self.settings.embedding_st_model
            )
        return self._model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        loop = asyncio.get_event_loop()
        model = self._get_model()
        # encode() is CPU-bound; run in executor to avoid blocking the event loop
        vecs = await loop.run_in_executor(
            None, partial(model.encode, texts, show_progress_bar=False)
        )
        return [v.tolist() for v in vecs]

    async def embed_one(self, text: str) -> list[float]:
        vecs = await self.embed([text])
        return vecs[0]


def build_embedding_service(
    settings: Settings, openai_client: AsyncOpenAI | None
) -> EmbeddingService | None:
    if settings.embedding_provider == "sentence-transformers":
        return SentenceTransformerEmbeddingService(settings)
    # openai (default)
    if openai_client is None:
        return None
    return OpenAIEmbeddingService(settings, openai_client)
