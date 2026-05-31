# Dimensions per provider. If you switch providers, drop existing vector indexes
# manually before restarting — Neo4j's IF NOT EXISTS guard will not recreate them
# with the new dimension.
EMBEDDING_DIMS: dict[str, int] = {
    "openai": 1536,
    "sentence-transformers": 1024,  # Qwen3-Embedding
}

_ENTITY_LABELS = ["Startup", "Investor", "Person", "Topic", "Company"]

_BASE_STATEMENTS = [
    "CREATE CONSTRAINT startup_id IF NOT EXISTS FOR (n:Startup) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT investor_id IF NOT EXISTS FOR (n:Investor) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (n:Person) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (n:Article) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT topic_id IF NOT EXISTS FOR (n:Topic) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (n:Source) REQUIRE n.id IS UNIQUE",
    "CREATE CONSTRAINT company_id IF NOT EXISTS FOR (n:Company) REQUIRE n.id IS UNIQUE",
    "CREATE INDEX startup_canonical_name IF NOT EXISTS FOR (n:Startup) ON (n.canonical_name)",
    "CREATE INDEX investor_canonical_name IF NOT EXISTS FOR (n:Investor) ON (n.canonical_name)",
    "CREATE INDEX person_canonical_name IF NOT EXISTS FOR (n:Person) ON (n.canonical_name)",
    "CREATE INDEX topic_canonical_name IF NOT EXISTS FOR (n:Topic) ON (n.canonical_name)",
    "CREATE INDEX company_canonical_name IF NOT EXISTS FOR (n:Company) ON (n.canonical_name)",
    "CREATE INDEX article_published_at IF NOT EXISTS FOR (n:Article) ON (n.published_at)",
    """
    CREATE FULLTEXT INDEX entitySearch IF NOT EXISTS
    FOR (n:Startup|Investor|Person|Topic|Company)
    ON EACH [n.name, n.canonical_name, n.aliases, n.description]
    """,
    """
    CREATE FULLTEXT INDEX articleSearch IF NOT EXISTS
    FOR (n:Article)
    ON EACH [n.title, n.summary, n.text]
    """,
]


def build_schema_statements(provider: str = "openai") -> list[str]:
    dims = EMBEDDING_DIMS.get(provider, EMBEDDING_DIMS["openai"])
    vector_indexes = [
        f"""
    CREATE VECTOR INDEX {label.lower()}_embedding IF NOT EXISTS
    FOR (n:{label}) ON (n.embedding)
    OPTIONS {{indexConfig: {{`vector.dimensions`: {dims}, `vector.similarity_function`: 'cosine'}}}}
    """
        for label in _ENTITY_LABELS
    ]
    return _BASE_STATEMENTS + vector_indexes


# Default schema statements kept for backwards-compatibility
SCHEMA_STATEMENTS = build_schema_statements("openai")
