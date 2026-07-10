from __future__ import annotations

import json

from app.evidence import build_evidence_catalog
from app.models.extraction import ArticleIn
from app.topic_ontology import get_topic_ontology


ENTITY_TYPES = ["startup", "investor", "person", "topic", "company"]
STRUCTURED_OUTPUT_CONTRACT = "STRICT_JSON_EVIDENCE_REFS_V1"
ENTITY_OUTPUT_SCHEMA = (
    '{"name","entity_type","type_basis","evidence_status","description","evidence_ref"}'
)
RELATIONSHIP_OUTPUT_SCHEMA = (
    '{"source","target","relationship_type","evidence_status","keywords","evidence_ref"}'
)


EXTRACTION_SYSTEM_PROMPT = """\
Extract a startup knowledge graph from the article body. Return JSON matching the supplied strict
schema. Do not output reasoning, headings, commentary, Markdown, or fields outside the schema.

Contract: STRICT_JSON_EVIDENCE_REFS_V1

Use German for generated descriptions and relationship keywords. Keep official names, canonical
Topic names, and article wording unchanged.

WORK BLOCK BY BLOCK
1. Each <EvidenceBlock> is one article paragraph, bullet point, or numbered entry. A Markdown
   heading may classify the block below it, but is context only and never evidence.
2. Decide whether the entire block is eligible before extracting anything from it.
3. Inventory supported named entities in the block and assign each intrinsic type once.
4. Emit every supported relationship using the fixed types and arrows below.
5. Validate scope, endpoints, direction, Topics, evidence references, and duplicates.
6. Continue through the highest-numbered EvidenceBlock. Do not stop after the first eligible block
   or after the first long entity list.

BLOCK ADMISSION
- Use only explicit facts from one block. Do not connect facts across blocks or infer a relationship
  from co-occurrence.
- Ignore navigation, tags, title/intro name lists, and bare lists with no factual predicate.
- A compact factual list is eligible: "A sammelt 5 Millionen ein; B erhält eine Finanzierung"
  supports A and B as entities, but no INVESTED_IN relationship without a named capital provider.
- Ignore an entire block framed as a job advertisement or event promotion, including employers,
  organizers, speakers, roles, partners, tickets, dates, applications, and logistics.
- A publisher call to action does not invalidate explicit startup facts elsewhere in an eligible
  block. Ignore only the promotional wording.
- Customer use, portfolio membership, event participation, shared industry, or co-occurrence does
  not prove partnership, employment, founding, or investment.
- Extract every explicitly named founder, capital provider, buyer, seller, partner, and employee in
  an eligible block. Long investor or founder lists are still facts; do not stop after three names.

INTRINSIC ENTITY TYPES
Use exactly one of: startup, investor, person, topic, company.
- startup: explicitly a startup, scaleup, spinoff, or clearly presented young operating venture
- investor: a dedicated capital-provider organization such as a VC/PE fund, investment arm, family
  office, accelerator, or investment firm
- person: a named human, including a founder, executive, partner, or angel investor
- company: an operating organization not established as a startup or dedicated investor, including
  a bank, acquirer, corporate investor, or acquired established business
- topic: an atomic reusable domain concept admitted by the closed Topic ontology below

Entity identity takes priority over its role. A person remains person when investing. An operating
company or bank remains company when investing or acquiring. An acquired company is not thereby a
startup. Never create unnamed or generic entities such as "Business Angel", "Investors", or "Team".
When startup or dedicated-investor identity is not established, use company.

TYPE BASIS
For every entity use exactly one:
- explicit: the same block explicitly identifies its intrinsic type
- contextual: the type is supported by its relationship role or surrounding context
- fallback: no more specific organization identity is established, so company is used

Do not call a capital provider an explicit investor merely because it invests. Do not call a
financing recipient an explicit startup merely because it raises money. Named people and admitted
Topics are explicit.

TOPICS: CLOSED ONTOLOGY, NO FALLBACK
- Output only exact canonical Topic names from the catalog. An alias may identify a catalog entry
  but must not appear as the output name.
- Start from an explicit category, product, or technology fact in the same evidence block. Select a
  Topic only when the fact satisfies its definition and semantic boundary.
- Never invent, combine, translate, fuzzily approximate, or choose the nearest Topic.
- Reject entity and fund names, self-name Topics, geography, audiences, amounts, metrics, funding
  stages, transactions, event/article labels, headlines, generic activities, and one-product
  descriptions.
- Prefer no Topic when uncertain. Every Topic requires a HAS_TOPIC relationship from the same block.
- Copy the Topic description exactly from the catalog definition.

CLOSED TOPIC ONTOLOGY
Version: {topic_ontology_version}

{topic_ontology_catalog}

EVIDENCE STATUS
Use exactly stated, attributed, or unsure:
- stated: the article directly presents the fact, including a clearly stated plan
- attributed: the definite claim is explicitly credited to a source or speaker
- unsure: the article presents the claim as rumored, ambiguous, or unconfirmed

EVIDENCE REFERENCES
- Every entity and relationship must contain one evidence_ref.
- Use one sentence ID such as B003.S001, or one contiguous same-block range such as
  B003.S001-B003.S003.
- Choose the smallest range that fully proves the fact. Include adjacent sentences when one
  sentence does not support the entity type, description, endpoints, relationship type, and
  direction.
- Never cite a heading, an unknown ID, separate blocks, a reversed range, or non-adjacent sentences.
- For an entity, the cited text must support the chosen type and every factual description claim.
- For a relationship, the cited text must establish both endpoints, exact type, and direction.
- Topic descriptions are the only description exception: use the ontology definition while the
  evidence_ref proves that the source entity has that Topic.
- The backend resolves references to exact article substrings. Do not repeat the article passage in
  any output field.

RELATIONSHIP TYPES AND FIXED ARROWS
- INVESTED_IN: organization/person -> organization. Requires explicit financing, investment,
  capital, round participation, or minority stake. "Supported by" alone is insufficient.
- ACQUIRED: buyer organization/person -> acquired organization. Requires a completed, agreed,
  signed, or announced acquisition, takeover, purchase, exit, or majority stake.
- MERGED_WITH: organization -> organization. Use only for a completed, agreed, signed, or announced
  merger with no buyer. It is symmetric for matching; output article order.
- FOUNDED_BY: organization -> person for explicit founders of a named organization.
- EMPLOYED_BY: non-founder person -> organization for an explicit role.
- PARTNERED_WITH: organization -> organization. Requires an explicit commercial or strategic
  partnership. Customers, users, portfolio companies, speakers, and shared sectors are not partners.
- HAS_TOPIC: any non-topic entity -> topic, supported by the same block.

If an explicit fact does not fit one allowed relationship exactly, emit no relationship.
Never approximate it with the closest predicate.

MECHANICAL DIRECTION CHECK
- "Fonds X investiert in Startup Y" and "Startup Y erhält Geld von Unternehmen X"
  -> Unternehmen X INVESTED_IN Startup Y.
- "Unternehmen A kauft Startup B" and "Startup B wird von Unternehmen A gekauft"
  -> A ACQUIRED B.
- "Organisation B wurde von Person P gegründet" -> Organisation B FOUNDED_BY Person P.
- "Person P ist CEO von Unternehmen A" -> P EMPLOYED_BY A.
- "Startup S geht eine Partnerschaft mit Unternehmen C ein" -> S PARTNERED_WITH C.

OUTPUT VALIDATION
- Both top-level arrays are required and may be empty.
- Each entity has exactly: name, entity_type, type_basis, evidence_status, description, evidence_ref.
- Each relationship has exactly: source, target, relationship_type, evidence_status, keywords,
  evidence_ref.
- Every relationship endpoint must have an entity with the exact same name and a compatible type.
- Use consistent official names and no duplicate entities or relationships.
- For an audit request, return additions only. Never remove, correct, retype, rename, reverse, or
  reword an existing record.

EXAMPLE
Input evidence block:
[B001.S001] Der VC Nordkap investiert in das FinTech LedgerFox, das Mia Kern gegründet hat.

Output:
{{
  "entities": [
    {{
      "name": "Nordkap",
      "entity_type": "investor",
      "type_basis": "explicit",
      "evidence_status": "stated",
      "description": "Nordkap ist ein VC und investiert in LedgerFox.",
      "evidence_ref": "B001.S001"
    }},
    {{
      "name": "LedgerFox",
      "entity_type": "startup",
      "type_basis": "explicit",
      "evidence_status": "stated",
      "description": "LedgerFox ist ein FinTech-Startup.",
      "evidence_ref": "B001.S001"
    }},
    {{
      "name": "Mia Kern",
      "entity_type": "person",
      "type_basis": "explicit",
      "evidence_status": "stated",
      "description": "Mia Kern ist Gründerin von LedgerFox.",
      "evidence_ref": "B001.S001"
    }},
    {{
      "name": "FinTech",
      "entity_type": "topic",
      "type_basis": "explicit",
      "evidence_status": "stated",
      "description": "Technologiegestützte Produkte und Dienstleistungen für Zahlungsverkehr, Bankwesen, Finanzen, Investitionen, Buchhaltung oder Finanzprozesse.",
      "evidence_ref": "B001.S001"
    }}
  ],
  "relationships": [
    {{
      "source": "Nordkap",
      "target": "LedgerFox",
      "relationship_type": "INVESTED_IN",
      "evidence_status": "stated",
      "keywords": "Finanzierung",
      "evidence_ref": "B001.S001"
    }},
    {{
      "source": "LedgerFox",
      "target": "Mia Kern",
      "relationship_type": "FOUNDED_BY",
      "evidence_status": "stated",
      "keywords": "Gründung",
      "evidence_ref": "B001.S001"
    }},
    {{
      "source": "LedgerFox",
      "target": "FinTech",
      "relationship_type": "HAS_TOPIC",
      "evidence_status": "stated",
      "keywords": "Branche",
      "evidence_ref": "B001.S001"
    }}
  ]
}}"""


EXTRACTION_USER_PROMPT = """\
Extract every supported record from the article below, one evidence block at a time.

Use German for descriptions and relationship keywords. Keep official names and canonical Topic
names unchanged. Return only the strict JSON object. When nothing qualifies, return
{{"entities":[],"relationships":[]}}.

<Article>
{input_text}
</Article>

The title, primary type, and Markdown headings are context only. Cite only sentence IDs inside
<EvidenceBlock> elements. Use one ID or the smallest contiguous same-block range that fully proves
each fact. Do not copy evidence text into the output."""


GLEANING_PROMPT = """\
Perform one conservative, recall-only audit of the article and previous JSON output.

Contract: STRICT_JSON_EVIDENCE_REFS_V1

- Return only genuinely missing records in the same strict JSON schema.
- If no safe additions exist, return {{"entities":[],"relationships":[]}}.
- Never repeat, remove, correct, replace, retype, rename, reverse, or reword an existing record.
- Do not add Topic entities or HAS_TOPIC relationships; Topic precision is decided initially.
- Add only high-confidence named non-Topic entities and exact relationships explicitly supported
  inside one eligible EvidenceBlock.
- Every record needs one evidence_ref: one sentence ID or the smallest contiguous same-block range.
  Never copy the evidence passage into an output field.
- Use German for generated descriptions and relationship keywords.
- Audit long founder and investor lists carefully. Add every genuinely missing named participant
  and its supported relationship; do not stop after three names.
- Prioritize mechanically checkable omissions: named founders, named capital providers, both sides
  of an acquisition or merger, and named non-founder employees.
- Add a relationship only when both endpoints exist or are safely added now and satisfy:
  INVESTED_IN organization/person -> organization
  ACQUIRED buyer organization/person -> acquired organization
  MERGED_WITH organization -> organization
  FOUNDED_BY organization -> person
  EMPLOYED_BY person -> organization
  PARTNERED_WITH organization -> organization
- Add nothing from job/event blocks, bare name lists, navigation, tags, title/intro lists, customer
  lists, speaker lists, unsupported portfolio lists, or cross-block combinations.
- Customer use, shared industry, event participation, portfolio membership, and co-occurrence do
  not prove partnership or employment.
- If scope, type, direction, or evidence is uncertain, add nothing."""


DESCRIPTION_MERGE_PROMPT = """\
---Role---
You are a Knowledge Graph Specialist, proficient in data curation and synthesis.

---Task---
Synthesize a list of descriptions of a given startup ecosystem entity or relationship
into a single, comprehensive, and cohesive summary.

---Instructions---
1. Input Format: The description list is provided in JSONL format. Each JSON object
   (one description) appears on a new line within the `Description List` section.
2. Output Format: Return the merged description as plain text, in multiple paragraphs
   if necessary. No markdown, no extra commentary before or after.
3. Comprehensiveness: Integrate all key facts from *every* provided description.
   Do not omit important details about the entity's funding, role, or relationships.
4. Context: Write in the objective third person; explicitly mention the entity or relation name.
5. Conflict Handling:
   - If descriptions describe distinct entities sharing a name, summarize each separately.
   - If they describe the same entity at different points in time or with conflicting details,
     reconcile them or present both viewpoints with noted uncertainty.
6. Topic summaries: If the input starts with `Topic Name:`, produce a reusable,
   article-independent definition of the topic. Do not mention startup/company/person names,
   article titles, funding amounts, locations, or one-off article facts. If the provided
   descriptions contain examples, abstract them into the general meaning of the topic.
7. Length: The summary must not exceed {summary_length} tokens.
8. Language: Write the summary in German.

---Input---
{description_type} Name: {description_name}
Description List:
```{description_list}```

---Output---\
"""


def build_extraction_system_prompt() -> str:
    ontology = get_topic_ontology()
    return EXTRACTION_SYSTEM_PROMPT.format(
        topic_ontology_version=ontology.ontology_version,
        topic_ontology_catalog=ontology.prompt_catalog(),
    )


def build_extraction_user_prompt(article: ArticleIn) -> str:
    return EXTRACTION_USER_PROMPT.format(input_text=build_article_prompt_input(article))


def build_extraction_user_prompt_template() -> str:
    return EXTRACTION_USER_PROMPT.format(input_text="{{input_text}}")


def build_article_prompt_input(article: ArticleIn) -> str:
    body = article_prompt_body(article)
    return (
        '<Metadata context_only="true">\n'
        f"Primary type: {article.primary_type or 'unknown'}\n"
        f"Title: {article.title}\n"
        "</Metadata>\n"
        "<Body>\n"
        f"{body}\n"
        "</Body>"
    )


def article_prompt_body(article: ArticleIn) -> str:
    return build_evidence_catalog(article.text).rendered_body


def article_prompt_audit(article: ArticleIn) -> dict[str, int | bool]:
    sent_chars = len(article_prompt_body(article))
    return {
        "prompt_body_chars_available": len(article.text),
        "prompt_body_chars_sent": sent_chars,
        "prompt_body_truncated": False,
    }


def build_extraction_prompt_registry_template() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_extraction_system_prompt()},
        {"role": "user", "content": build_extraction_user_prompt_template()},
    ]


def build_gleaning_prompt_registry_template() -> list[dict[str, str]]:
    return [{"role": "user", "content": build_gleaning_prompt()}]


def build_gleaning_prompt() -> str:
    return GLEANING_PROMPT


def build_description_merge_prompt(
    *,
    description_type: str,
    description_name: str,
    descriptions: list[str],
    summary_length: int = 500,
) -> str:
    jsonl = "\n".join(json.dumps({"Description": d}, ensure_ascii=False) for d in descriptions)
    return DESCRIPTION_MERGE_PROMPT.format(
        description_type=description_type,
        description_name=description_name,
        description_list=jsonl,
        summary_length=summary_length,
    )
