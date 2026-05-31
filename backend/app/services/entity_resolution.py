from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from rapidfuzz import distance as rf_distance
from rapidfuzz import fuzz as rf_fuzz

from app.core.config import Settings
from app.core.logging import get_logger
from app.models.extraction import (
    EntityType,
    ExtractedEntity,
    NormalizedEntity,
    strongest_evidence_status,
)

if TYPE_CHECKING:
    from app.services.embedding import EmbeddingService

ResolutionMethod = Literal["exact", "fuzzy", "embedding", "new"]

_MAX_DESCRIPTIONS = 5


@dataclass
class ResolutionOutcome:
    entity: NormalizedEntity
    method: ResolutionMethod
    candidate_name: str
    merged_into: str | None = None
    similarity_min: float | None = None
    registry_entry: RegistryEntry | None = (
        None  # set for "new" entities, used for embedding backfill
    )

    def to_dict(self) -> dict:
        return {
            "candidate": self.candidate_name,
            "method": self.method,
            "canonical": self.entity.canonical_name,
            "entity_id": self.entity.id,
            **({"merged_into": self.merged_into} if self.merged_into else {}),
            **(
                {"similarity_min": round(self.similarity_min, 3)}
                if self.similarity_min is not None
                else {}
            ),
        }


# ---------------------------------------------------------------------------
# NameNormalizer -- type-aware cleaning
# ---------------------------------------------------------------------------

_LEGAL_SUFFIXES: frozenset[str] = frozenset(
    {"gmbh", "ag", "ug", "se", "inc", "ltd", "limited", "llc", "holding", "holdings"}
)
_PERSON_TITLES: frozenset[str] = frozenset(
    {"dr", "prof", "dipl", "mba", "hr", "frau", "herr", "mr", "ms", "mrs", "sir"}
)
_PERSON_ROLES: frozenset[str] = frozenset(
    {
        "ceo",
        "cto",
        "cfo",
        "coo",
        "cpo",
        "cmo",
        "founder",
        "co-founder",
        "cofounder",
        "managing director",
        "general partner",
        "partner",
        "director",
        "president",
        "chairman",
        "chairwoman",
    }
)
_STOPWORDS: frozenset[str] = frozenset({"of", "the", "and", "in", "for", "by", "at", "to", "a"})

_ARTIFACT_RE = re.compile(
    r"<[^>]+>"  # HTML tags
    r"|[\[\]]"  # stray brackets
    r"|\s*[|\u2022\u00b7]\s*",  # pipe / bullet separators
    flags=re.UNICODE,
)
_LOCATION_PREFIX_RE = re.compile(
    r"^(?:[\w\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df&.+#\u2018\u2019.-]+"
    r"(?:\s+[\w\u00c4\u00d6\u00dc\u00e4\u00f6\u00fc\u00df&.+#\u2018\u2019.-]+){0,4})"
    r"[-\s]+based\s+",
    flags=re.IGNORECASE | re.UNICODE,
)
_NOISE_PREFIX_RE = re.compile(
    r"^(?:startup|scaleup|jungunternehmen|vc|investor|geldgeber)\s+",
    flags=re.IGNORECASE,
)
_PERSON_SUFFIX_RE = re.compile(
    r",\s*(?:"
    + "|".join(re.escape(r) for r in sorted(_PERSON_ROLES, key=len, reverse=True))
    + r")\s*$",
    flags=re.IGNORECASE,
)


class NameNormalizer:
    """Cleans and normalises entity names for display and key-based matching."""

    @staticmethod
    def display(raw: str) -> str:
        """Return a human-readable display name stripped of HTML and noise."""
        value = _ARTIFACT_RE.sub(" ", raw)
        value = re.sub(r"\s+", " ", value).strip(" -\u2013\u2014:.,;()[]")
        value = _LOCATION_PREFIX_RE.sub("", value).strip()
        value = _NOISE_PREFIX_RE.sub("", value).strip()
        return value

    @classmethod
    def key(cls, raw: str, entity_type: EntityType) -> str:
        """Return a fully normalised matching key for raw given entity_type."""
        value = unicodedata.normalize("NFKC", raw)
        value = _ARTIFACT_RE.sub(" ", value)
        value = value.replace("&", " and ")
        value = value.casefold().strip()
        value = _LOCATION_PREFIX_RE.sub("", value)
        value = _NOISE_PREFIX_RE.sub("", value)
        value = re.sub(r"[^\w\s+#.-]", " ", value, flags=re.UNICODE)
        value = re.sub(r"\b(?:startup|startups)\b", " ", value)
        value = re.sub(r"\s+", " ", value).strip(" .-")

        if entity_type == "Person":
            value = _PERSON_SUFFIX_RE.sub("", value).strip()
            tokens = value.split()
            tokens = [t.rstrip(".") for t in tokens if t.rstrip(".") not in _PERSON_TITLES]
            tokens = [t for t in tokens if t not in _PERSON_ROLES]
        else:
            tokens = value.split()
            if entity_type in ("Startup", "Investor"):
                tokens = [t for t in tokens if t not in _LEGAL_SUFFIXES]
            # Topic: keep all tokens, just normalised

        return " ".join(tokens).strip()

    @classmethod
    def token_set(cls, raw: str, entity_type: EntityType) -> frozenset[str]:
        """Return a frozenset of meaningful tokens from the normalised key."""
        k = cls.key(raw, entity_type)
        return frozenset(t for t in k.split() if len(t) >= 2 and t not in _STOPWORDS)

    @staticmethod
    def slug(raw: str) -> str:
        """Return a URL-safe ASCII slug for use as an entity ID."""
        value = unicodedata.normalize("NFKC", raw)
        value = re.sub(r"[^\w\s]", " ", value.casefold(), flags=re.UNICODE)
        value = re.sub(r"\s+", " ", value).strip()
        ascii_value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
        slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
        return slug or "unknown"


# ---------------------------------------------------------------------------
# SimilarityResult + MultiMetricScorer
# ---------------------------------------------------------------------------

_SHORT_NAME_LENGTH = 4
_SHORT_NAME_CAP = 0.95


@dataclass
class SimilarityResult:
    levenshtein: float
    jaro_winkler: float
    token_sort_ratio: float
    token_set_ratio: float
    jaccard: float

    @property
    def composite(self) -> float:
        return max(
            self.levenshtein,
            self.jaro_winkler,
            self.token_sort_ratio,
            self.token_set_ratio,
            self.jaccard,
        )

    @property
    def minimum(self) -> float:
        return min(
            self.levenshtein,
            self.jaro_winkler,
            self.token_sort_ratio,
            self.token_set_ratio,
            self.jaccard,
        )


class MultiMetricScorer:
    """Computes a 5-metric similarity ensemble between two normalised entity keys."""

    @staticmethod
    def score(
        key_a: str,
        tokens_a: frozenset[str],
        key_b: str,
        tokens_b: frozenset[str],
    ) -> SimilarityResult:
        lev = rf_distance.Levenshtein.normalized_similarity(key_a, key_b)
        jw = rf_distance.JaroWinkler.normalized_similarity(key_a, key_b)
        tsr = rf_fuzz.token_sort_ratio(key_a, key_b) / 100.0
        tset = rf_fuzz.token_set_ratio(key_a, key_b) / 100.0
        union = tokens_a | tokens_b
        jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0

        result = SimilarityResult(
            levenshtein=lev,
            jaro_winkler=jw,
            token_sort_ratio=tsr,
            token_set_ratio=tset,
            jaccard=jaccard,
        )

        # Short-name guard
        if min(len(key_a), len(key_b)) < _SHORT_NAME_LENGTH:
            cap = _SHORT_NAME_CAP
            result = SimilarityResult(
                levenshtein=min(lev, cap),
                jaro_winkler=min(jw, cap),
                token_sort_ratio=min(tsr, cap),
                token_set_ratio=min(tset, cap),
                jaccard=min(jaccard, cap),
            )

        return result


# ---------------------------------------------------------------------------
# Merge thresholds
# ---------------------------------------------------------------------------

_AUTO_MERGE_MIN_SCORE = 0.95


# ---------------------------------------------------------------------------
# RegistryEntry
# ---------------------------------------------------------------------------


@dataclass
class RegistryEntry:
    entity: NormalizedEntity
    keys: set[str] = field(default_factory=set)
    token_sets: list[frozenset[str]] = field(default_factory=list)
    descriptions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# EntityResolver -- exact, heuristic, and embedding matching
# ---------------------------------------------------------------------------


class EntityResolver:
    def __init__(
        self,
        settings: Settings,
        embedding: EmbeddingService | None = None,
        graph=None,
    ):
        self.settings = settings
        self._embedding_service = embedding
        self._graph = graph
        self.logger = get_logger("resolver")
        self._entries_by_type: dict[EntityType, list[RegistryEntry]] = {
            "Startup": [],
            "Investor": [],
            "Person": [],
            "Topic": [],
            "Company": [],
        }
        self._alias_index: dict[tuple[EntityType, str], RegistryEntry] = {}

    async def load_from_graph(self, graph_repository) -> None:
        for label in self._entries_by_type:
            rows = await graph_repository.list_entities(label)
            for row in rows:
                entity = NormalizedEntity(
                    id=row["id"],
                    label=label,
                    canonical_name=row.get("canonical_name") or row.get("name"),
                    name=row.get("name") or row.get("canonical_name"),
                    aliases=row.get("aliases") or [],
                    evidence_status=row.get("evidence_status") or "unsure",
                    description=row.get("description"),
                )
                entry = self._add_entry(entity)
                if row.get("descriptions"):
                    entry.descriptions = list(row["descriptions"])
        self.logger.info(
            "entity_registry_loaded",
            extra={
                "event": "resolution",
                "workflow_step": "registry_load",
                "count": sum(len(v) for v in self._entries_by_type.values()),
            },
        )

    async def resolve(
        self,
        entity_type: EntityType,
        entity: ExtractedEntity,
        candidate_evidence: str | None = None,  # kept for call-site compatibility
        precomputed_embedding: list[float] | None = None,
        blocked_entity_ids: set[str] | None = None,
        blocked_canonical_keys: set[str] | None = None,
    ) -> ResolutionOutcome:
        blocked_entity_ids = blocked_entity_ids or set()
        blocked_canonical_keys = blocked_canonical_keys or set()
        display = NameNormalizer.display(entity.name)
        all_names = [display, *(a for a in entity.aliases if a)]
        candidate_keys = [NameNormalizer.key(n, entity_type) for n in all_names]
        candidate_token_sets = [NameNormalizer.token_set(n, entity_type) for n in all_names]
        candidate_pairs = [(k, ts) for k, ts in zip(candidate_keys, candidate_token_sets) if k]
        if not candidate_pairs:
            candidate_pairs = [(NameNormalizer.slug(display), frozenset())]

        # Stage 1a: Exact key lookup
        for ckey, _ in candidate_pairs:
            exact = self._alias_index.get((entity_type, ckey))
            if exact:
                if _entry_is_blocked(exact, blocked_entity_ids, blocked_canonical_keys):
                    self.logger.info(
                        "entity_merge_blocked_by_relationship",
                        extra={
                            "event": "resolution",
                            "workflow_step": "entity_resolution",
                            "entity_type": entity_type,
                            "detail": (
                                f'blocked merge "{entity.name}" -> '
                                f'"{exact.entity.canonical_name}" because both entities '
                                "appear as endpoints of an article relationship"
                            ),
                        },
                    )
                    continue
                if await self._graph_relationship_blocks_merge(
                    exact,
                    entity_type=entity_type,
                    candidate_name=entity.name,
                    candidate_names=all_names,
                    method="exact",
                ):
                    continue
                self.logger.info(
                    "entity_merged_exact",
                    extra={
                        "event": "resolution",
                        "workflow_step": "entity_resolution",
                        "entity_type": entity_type,
                        "detail": f'merged "{entity.name}" -> "{exact.entity.canonical_name}"',
                    },
                )
                result = self._merge_into_entry(exact, entity, display, candidate_pairs)
                return ResolutionOutcome(
                    entity=result,
                    method="exact",
                    candidate_name=display,
                    merged_into=result.canonical_name,
                )

        # Stage 1b: Multi-metric heuristic matching
        best_entry, best_result = self._multi_metric_match(
            entity_type,
            candidate_pairs,
            blocked_entity_ids=blocked_entity_ids,
            blocked_canonical_keys=blocked_canonical_keys,
        )

        if best_entry and best_result.minimum >= _AUTO_MERGE_MIN_SCORE:
            if not await self._graph_relationship_blocks_merge(
                best_entry,
                entity_type=entity_type,
                candidate_name=entity.name,
                candidate_names=all_names,
                method="fuzzy",
                similarity=best_result.minimum,
            ):
                self.logger.info(
                    "entity_merged_heuristic",
                    extra={
                        "event": "resolution",
                        "workflow_step": "entity_resolution",
                        "entity_type": entity_type,
                        "detail": (
                            f'merged "{entity.name}" -> "{best_entry.entity.canonical_name}" '
                            f"min={best_result.minimum:.3f} "
                            f"lev={best_result.levenshtein:.3f} "
                            f"jw={best_result.jaro_winkler:.3f} "
                            f"tsr={best_result.token_sort_ratio:.3f} "
                            f"tset={best_result.token_set_ratio:.3f} "
                            f"jac={best_result.jaccard:.3f}"
                        ),
                    },
                )
                result = self._merge_into_entry(best_entry, entity, display, candidate_pairs)
                return ResolutionOutcome(
                    entity=result,
                    method="fuzzy",
                    candidate_name=display,
                    merged_into=result.canonical_name,
                    similarity_min=best_result.minimum,
                )

        # Stage 3: Embedding vector search via Neo4j index
        if (
            self.settings.enable_embedding_resolution
            and self._graph
            and (precomputed_embedding is not None or self._embedding_service)
        ):
            if precomputed_embedding is not None:
                candidate_vec = precomputed_embedding
            else:
                candidate_text = f"{display}. {entity.description or ''}".strip()
                candidate_vec = await self._embedding_service.embed_one(candidate_text)
            matched_id, cos_score = await self._graph.find_nearest_entity(
                label=entity_type,
                embedding=candidate_vec,
                min_score=self.settings.embedding_similarity_threshold,
                exclude_ids=blocked_entity_ids,
            )
            if matched_id:
                # Look up the in-memory entry by id so we can merge into it
                best_cos_entry = self._entry_by_id(entity_type, matched_id)
                if best_cos_entry and _entry_is_blocked(
                    best_cos_entry,
                    blocked_entity_ids,
                    blocked_canonical_keys,
                ):
                    best_cos_entry = None
            else:
                best_cos_entry = None

            if best_cos_entry:
                if not await self._graph_relationship_blocks_merge(
                    best_cos_entry,
                    entity_type=entity_type,
                    candidate_name=entity.name,
                    candidate_names=all_names,
                    method="embedding",
                    similarity=cos_score,
                ):
                    self.logger.info(
                        "entity_merged_embedding",
                        extra={
                            "event": "resolution",
                            "workflow_step": "entity_resolution",
                            "entity_type": entity_type,
                            "detail": (
                                f'merged "{entity.name}" -> '
                                f'"{best_cos_entry.entity.canonical_name}" '
                                f"cosine={cos_score:.3f}"
                            ),
                        },
                    )
                    result = self._merge_into_entry(
                        best_cos_entry, entity, display, candidate_pairs
                    )
                    return ResolutionOutcome(
                        entity=result,
                        method="embedding",
                        candidate_name=display,
                        merged_into=result.canonical_name,
                        similarity_min=cos_score,
                    )
            self.logger.info(
                "entity_embedding_no_match",
                extra={
                    "event": "resolution",
                    "workflow_step": "entity_resolution",
                    "entity_type": entity_type,
                    "detail": (
                        f'no embedding match for "{entity.name}" (best_cosine={cos_score:.3f})'
                    ),
                },
            )

        # New entity
        normalized = NormalizedEntity(
            id=f"{entity_type.lower()}:{NameNormalizer.slug(display)}",
            label=entity_type,
            canonical_name=display,
            name=display,
            aliases=_unique(all_names, entity_type),
            evidence_status=entity.evidence_status,
            description=entity.description,
        )
        new_entry = self._add_entry(normalized)
        # Seed the description pool with the initial description
        if entity.description and entity.description.strip():
            new_entry.descriptions = [entity.description.strip()]
        self.logger.info(
            "entity_created",
            extra={
                "event": "resolution",
                "workflow_step": "entity_resolution",
                "entity_type": entity_type,
                "detail": display,
            },
        )
        return ResolutionOutcome(
            entity=normalized,
            method="new",
            candidate_name=display,
            registry_entry=new_entry,
        )

    def _multi_metric_match(
        self,
        entity_type: EntityType,
        candidate_pairs: list[tuple[str, frozenset[str]]],
        *,
        blocked_entity_ids: set[str],
        blocked_canonical_keys: set[str],
    ) -> tuple[RegistryEntry | None, SimilarityResult]:
        _zero = SimilarityResult(0.0, 0.0, 0.0, 0.0, 0.0)
        best_entry: RegistryEntry | None = None
        best_result = _zero

        for entry in self._entries_by_type[entity_type]:
            if _entry_is_blocked(entry, blocked_entity_ids, blocked_canonical_keys):
                continue
            entry_key_pairs = [
                (ekey, NameNormalizer.token_set(ekey, entity_type)) for ekey in entry.keys
            ]
            for ckey, cts in candidate_pairs:
                for ekey, ets in entry_key_pairs:
                    if not ekey:
                        continue
                    result = MultiMetricScorer.score(ckey, cts, ekey, ets)
                    if result.composite > best_result.composite:
                        best_result = result
                        best_entry = entry

        return best_entry, best_result

    async def _graph_relationship_blocks_merge(
        self,
        entry: RegistryEntry,
        *,
        entity_type: EntityType,
        candidate_name: str,
        candidate_names: list[str],
        method: ResolutionMethod,
        similarity: float | None = None,
    ) -> bool:
        if self._graph is None:
            return False
        has_relationship = getattr(
            self._graph,
            "has_relationship_to_named_entity",
            None,
        )
        if not callable(has_relationship):
            return False
        try:
            blocked = await has_relationship(
                target_id=entry.entity.id,
                candidate_label=entity_type,
                candidate_names=candidate_names,
            )
        except Exception as exc:
            self.logger.warning(
                "entity_merge_graph_guard_failed",
                extra={
                    "event": "resolution",
                    "workflow_step": "entity_resolution",
                    "entity_type": entity_type,
                    "error": str(exc),
                    "detail": (
                        f"could not check existing graph relationships before {method} "
                        f'merge "{candidate_name}" -> "{entry.entity.canonical_name}"'
                    ),
                },
            )
            return False
        if not blocked:
            return False

        detail = (
            f'blocked {method} merge "{candidate_name}" -> '
            f'"{entry.entity.canonical_name}" because the graph already has '
            "a relationship between the merge target and a same-named entity"
        )
        if similarity is not None:
            detail += f" score={similarity:.3f}"
        self.logger.info(
            "entity_merge_blocked_by_existing_graph_relationship",
            extra={
                "event": "resolution",
                "workflow_step": "entity_resolution",
                "entity_type": entity_type,
                "detail": detail,
            },
        )
        return True

    def _merge_into_entry(
        self,
        entry: RegistryEntry,
        entity: ExtractedEntity,
        display: str,
        candidate_pairs: list[tuple[str, frozenset[str]]],
    ) -> NormalizedEntity:
        all_aliases = _unique(
            [*entry.entity.aliases, display, entity.name, *entity.aliases],
            entry.entity.label,
        )
        entry.entity.aliases = all_aliases
        entry.entity.evidence_status = strongest_evidence_status(
            entry.entity.evidence_status, entity.evidence_status
        )
        entry.entity.canonical_name = _select_canonical(entry.entity.canonical_name, display)

        # Append incoming description to pool (max 5, deduplicated); primary description unchanged
        incoming_desc = (entity.description or "").strip()
        if incoming_desc and incoming_desc not in entry.descriptions:
            if len(entry.descriptions) < _MAX_DESCRIPTIONS:
                entry.descriptions.append(incoming_desc)

        for ckey, cts in candidate_pairs:
            if ckey:
                entry.keys.add(ckey)
                entry.token_sets.append(cts)
                self._alias_index[(entry.entity.label, ckey)] = entry

        # Sync description pool back onto the entity model for persistence
        entry.entity.descriptions = list(entry.descriptions)

        return entry.entity

    def _entry_by_id(self, entity_type: EntityType, entity_id: str) -> RegistryEntry | None:
        for entry in self._entries_by_type[entity_type]:
            if entry.entity.id == entity_id:
                return entry
        return None

    def _add_entry(self, entity: NormalizedEntity) -> RegistryEntry:
        names = [entity.canonical_name, entity.name, *entity.aliases]
        pairs = [
            (NameNormalizer.key(n, entity.label), NameNormalizer.token_set(n, entity.label))
            for n in names
            if n
        ]
        keys = {k for k, _ in pairs if k}
        token_sets = [ts for k, ts in pairs if k]
        entry = RegistryEntry(entity=entity, keys=keys, token_sets=token_sets)
        self._entries_by_type[entity.label].append(entry)
        for k in keys:
            self._alias_index[(entity.label, k)] = entry
        return entry


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _select_canonical(existing: str, _candidate: str) -> str:
    """Keep the first canonical name stable; aliases carry later variants."""
    return existing


def _entry_is_blocked(
    entry: RegistryEntry,
    blocked_entity_ids: set[str],
    blocked_canonical_keys: set[str],
) -> bool:
    if entry.entity.id in blocked_entity_ids:
        return True
    if not blocked_canonical_keys:
        return False
    canonical_keys = {
        NameNormalizer.key(value, entry.entity.label)
        for value in (entry.entity.canonical_name, entry.entity.name)
        if value
    }
    return bool(canonical_keys & blocked_canonical_keys)


def _unique(values: list[str], entity_type: EntityType | None = None) -> list[str]:
    """Deduplicate names, preserving the first occurrence of each normalised key."""
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        cleaned = NameNormalizer.display(value)
        if not cleaned:
            continue
        key = NameNormalizer.key(cleaned, entity_type or "Startup")
        if key and key not in seen:
            seen.add(key)
            result.append(cleaned)
    return result
