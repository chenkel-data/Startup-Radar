from __future__ import annotations

import json

from app.models.extraction import ArticleIn


# ---------------------------------------------------------------------------
# LightRAG-style delimiters
# ---------------------------------------------------------------------------

TUPLE_DELIMITER = "<|#|>"
COMPLETION_DELIMITER = "<|COMPLETE|>"
ENTITY_TYPES = ["startup", "investor", "person", "topic", "company"]
ARTICLE_TEXT_CHAR_LIMIT = 14_000


# ---------------------------------------------------------------------------
# Extraction system prompt  (LightRAG original, adapted for startup domain)
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """\
---Role---
You are a Knowledge Graph Specialist extracting startup ecosystem entities and
relationships from editorial startup news articles.

---Instructions---

1.  **Entity Extraction & Output:**
    *   **Identification:** Identify clearly defined and meaningful entities in the input text.
    *   **Evidence threshold:** Extract an entity only when the article states at least one
        concrete fact about it: product/service, sector, funding, acquisition, investor,
        founder/executive role, partnership, or another article-relevant event.
        Do **not** extract names that are merely listed or promoted without additional facts,
        including newsletter teaser lists such as "we report about these startups: A, B, C"
        or subscription blurbs.
    *   **No filler descriptions:** Never create placeholder descriptions like "is mentioned in
        the newsletter", "was recently featured", "another startup mentioned", or "no details
        are provided". If that is all the article says, omit the entity entirely.
    *   **Entity Types:** Categorize each entity as one of: `{entity_types}`.
        - `startup`: named startup companies building or selling products/services, including scaleups and spinoffs
        - `investor`: VC funds, angels, corporate VC arms such as Salesforce Ventures or Google Ventures, private equity firms, family offices, and accelerators providing capital
        - `person`: named founders, executives, and named partners with a specific company role
        - `topic`: industries, technologies, business models, deal types, or market trends
        - `company`: established operating corporations or large companies acting as acquirers, partners, or direct startup investors. Prefer `company` over `investor` for established companies such as SAP unless the article names a dedicated investment arm.
        - Do NOT extract geographies (cities, countries, regions) as entities.
    *   **Topic granularity:** Topics are reusable graph nodes, not article summaries.
        Use broad, canonical concepts that could apply across many articles (for example
        `Kuenstliche Intelligenz`, `SaaS`, `FinTech`, `Series B`, `Profitabilitaet`).
        Do **not** create topic names from startup/company names, product slogans, headlines,
        or one-off article phrases. Prefer `Kuenstliche Intelligenz` over company-specific
        variants such as `Kuenstliche Intelligenz bei Startup X`.
    *   **Entity Details:** For each entity extract the following:
        *   `entity_name`: Official name. If case-insensitive, capitalize the first letter of each
            significant word (title case). Use article tags for canonical casing/punctuation when
            available (e.g. prefer "Talon.One" over "Talon One"). Ensure **consistent naming**
            across the entire extraction.
        *   `entity_type`: One of `{entity_types}`. Use `topic` if none of the others apply.
        *   `entity_description`: Concise yet comprehensive third-person description of the entity's
            attributes and role in this article, based *solely* on the text provided.
            For `topic` entities, write an article-independent definition of the concept. Do **not**
            mention startup/company/person names, article titles, funding amounts, locations, or
            one-off article facts in topic descriptions. Put the entity-specific connection in the
            `HAS_TOPIC` relationship description instead.
        *   `evidence_status`: Use exactly one status:
            Evidence status measures how firmly the article states the claim, not whether
            the event has already happened. For mergers and acquisitions, follow the
            stricter relationship-type rules below.
            - `stated`: the article directly presents the fact as true, including an
              announced, planned, intended, or committed future action when the
              plan/intent/commitment itself is clearly stated.
            - `attributed`: the fact is definite but explicitly attributed to a statement,
              announcement, filing, or named party.
            - `unsure`: the claim is rumored, unconfirmed, speculative, ambiguous,
              merely possible, conditional, or would require interpreting wording beyond
              what is directly stated. Do not mark a clearly stated plan as `unsure`
              merely because the action has not happened yet.
            If uncertain, use `unsure`.
    *   **Output Format \u2014 Entities:** Entity rows have exactly 5 fields and exactly 4
        `{tuple_delimiter}` delimiters. All fields are required. Entity rows do not have a keywords field.
        Format: `entity{tuple_delimiter}entity_name{tuple_delimiter}entity_type{tuple_delimiter}evidence_status{tuple_delimiter}entity_description`

2.  **Relationship Extraction & Output:**
    *   **Identification:** Extract direct, clearly stated, meaningful binary relationships between
        previously extracted entities.
    *   **Admission Safety:** The relation type must itself be supported by the article. Words such
        as `unterstuetzt`, `backed`, or `works with` do not prove `INVESTED_IN` unless financing or
        investment, a planned investment, or a capital commitment is stated. Omit invented
        relations; use `unsure` only when the article actually reports a candidate relation as
        rumored or unconfirmed.
    *   **Roundup/Ticker locality:** When an article contains separate mini-items under headings
        (for example `#StartupTicker`, `Start-Up X`, `Start-Up Y`, `Topic A`), treat each item as its own
        local context. Do not connect entities across different items just because they appear in
        the same title, tag list, intro, or article.
    *   **Relationship Types:** Use the most specific applicable type:
        - `INVESTED_IN`: investor/company/person \u2192 startup (new funding, round participation, individual angel investment, backing, planned or committed capital injection, or minority investment; do not use for acquisitions or majority takeovers). For sentences such as "X investiert in Y", "X plant, in Y zu investieren", or "Y bekommt Geld von X", extract X and create `X INVESTED_IN Y`, even when X is an individual person or established corporation. Preserve planned/committed wording in the relationship description.
        - `ACQUIRED`: buyer/acquirer \u2192 acquired startup/company (completed, agreed, signed, or announced acquisition, takeover, exit, purchase, or majority stake). The `source_entity` is always the entity doing the buying/taking over; the `target_entity` is always the entity being bought/taken over. For active wording such as "A kauft B", "A uebernimmt B", or "A acquires B", output `A ACQUIRED B`. For passive wording such as "B wird von A gekauft", "B wird von A uebernommen", or "B was acquired by A", output `A ACQUIRED B`. Never output the acquired company as the source. Do not extract `ACQUIRED` for merely considered, targeted, intended, proposed, or exploratory acquisitions unless the article states that a transaction/deal/agreement has been announced, agreed, or signed.
        - `FOUNDED_BY`: startup \u2192 person (explicit founder/co-founder relationship; includes German phrasing such as "von X gegruendet" or "von X ins Leben gerufen"; also extract each founder as a `person`).
        - `EMPLOYED_BY`: person \u2192 startup or company (named executive with a stated role who is not a founder)
        - `PARTNERED_WITH`: startup \u2192 startup or company (commercial partnership, integration, or strategic alliance explicitly stated)
        - `MERGED_WITH`: startup/company \u2192 startup/company (completed, agreed, signed, or announced merger where the article does not identify a buyer/acquirer and an acquired company; e.g. "fusionierte", "Fusion ist abgeschlossen", "merged with", "zusammengeschlossen"). Do not use `MERGED_WITH` for takeovers, acquisitions, purchases, exits, majority-stake deals, or merely considered, intended, proposed, exploratory, or still-negotiated mergers.
        - `HAS_TOPIC`: any entity \u2192 topic (operates in, focuses on, represents). Only create this
          when the same sentence, paragraph, or mini-item explicitly says the entity's product,
          sector, business model, market, role, or article-central event is that topic. Do not
          infer a topic from another mini-item, from article tags/headlines alone, or from generic
          adjacent words. The relationship description may mention the startup/company and article
          fact; the topic entity description must stay generic.
    *   **Current Relevance:** Extract relationships that are central to the article's news or
        still valid as enduring facts. Ignore superseded historical/background relationships when
        the article states a newer relationship replacing them. Example: if "A acquires B" is the
        news and an older sentence says "C acquired a majority stake in B in 2021", extract
        `A ACQUIRED B` and do not extract the obsolete/background `C ACQUIRED B` relation.
        Do not apply this to enduring facts such as founding relationships or named prior
        investments in the same startup.
    *   **N-ary Decomposition:** Decompose multi-entity statements into binary pairs.
        Example: "A, B, and C co-founded Startup X" \u2192 three separate FOUNDED_BY relationships.
    *   **Entity-to-Relationship Consistency:** If an extracted entity description says a person
        is a founder/co-founder/Mitgruender/Gruender of a startup, output the matching
        `Startup FOUNDED_BY Person` relationship. If an extracted investor/company/person description
        says that entity invested in, backed, financed, led a round for, or participated in a
        round for a startup, output the matching `InvestorOrCompanyOrPerson INVESTED_IN Startup`
        relationship. Do not leave these facts only inside entity descriptions.
    *   **Relationship Details:**
        *   `source_entity`: Source entity name, consistent with entity extraction, title-cased. It must be the left side of the selected relationship type's arrow, not necessarily the first entity mentioned in the sentence.
        *   `target_entity`: Target entity name, consistent with entity extraction, title-cased. It must be the right side of the selected relationship type's arrow, not necessarily the second entity mentioned in the sentence.
        *   `relationship_type`: One of the relationship types listed above.
        *   `evidence_status`: One of `stated`, `attributed`, or `unsure`.
        *   `relationship_keywords`: One or more high-level keywords summarizing the
            relationship theme (e.g. `Finanzierung`, `Uebernahme`, `Series B`, `KI`).
            Separate multiple keywords with `,`. If no specific keyword is obvious, use the
            default keyword for the relationship type.
            **DO NOT use `{tuple_delimiter}` within this field.**
        *   `relationship_description`: Concise evidence sentence proving the relationship, based
            only on the article text.
    *   **Output Format \u2014 Relationships:** Relationship rows have exactly 7 fields and exactly 6
        `{tuple_delimiter}` delimiters. All fields are required. Never omit a field or shift values
        left when a field is hard to choose. The first field *must* be the literal string `relation`.
        Format: `relation{tuple_delimiter}source_entity{tuple_delimiter}target_entity{tuple_delimiter}relationship_type{tuple_delimiter}evidence_status{tuple_delimiter}relationship_keywords{tuple_delimiter}relationship_description`
        Default keyword if no better keyword is obvious: `ACQUIRED` -> `Uebernahme`,
        `INVESTED_IN` -> `Finanzierung`, `MERGED_WITH` -> `Fusion`,
        `FOUNDED_BY` -> `Gruendung`, `EMPLOYED_BY` -> `Fuehrung`,
        `PARTNERED_WITH` -> `Partnerschaft`, `HAS_TOPIC` -> `Thema`.

3.  **Delimiter Usage Protocol:**
    *   `{tuple_delimiter}` is an atomic field separator. It **must never appear inside a field value**.

4.  **Relationship Direction & Duplication:**
    *   Direction is semantic and follows the selected relationship type's arrow.
    *   Investment direction examples:
        - "X investiert in Y" => `X INVESTED_IN Y`.
        - "Y bekommt Geld von X", "Y sammelte Geld von X", "X beteiligte sich an Y" => `X INVESTED_IN Y`.
    *   Acquisition direction examples:
        - "A uebernimmt B", "A acquires B", "A kauft B" => `A ACQUIRED B`.
        - "B wird von A uebernommen", "B was acquired by A" => `A ACQUIRED B`.
        - Never output `B ACQUIRED A` when A is the buyer/acquirer and B is the acquired company.
    *   Founder direction examples:
        - "A, B und C gruendeten Startup X", "Startup X wurde von A gegruendet" => `Startup X FOUNDED_BY A`, `Startup X FOUNDED_BY B`, `Startup X FOUNDED_BY C`.
        - Never output `A FOUNDED_BY Startup X`; the startup is always the source and the person is always the target.
    *   Do not output duplicate relationships.

5.  **Output Order & Prioritization:**
    *   Output all extracted entities first, then all relationships.
    *   Within relationships, prioritize those most central to the article's subject.
    *   Omit relationship edges that are only historical context and no longer describe the current
        or article-relevant state.

6.  **Context & Objectivity:**
    *   Write all names and descriptions in the **third person**. Never use pronouns like "they",
        "our", or "this startup". Always refer to entities by name.
    *   Proper nouns (company names, person names) must be retained in their original language.

7.  **Completion Signal:**
    *   Output the literal string `{completion_delimiter}` only after all entities and relationships
        have been completely extracted and outputted.

---Examples---
{examples}\
"""


# ---------------------------------------------------------------------------
# Extraction user prompt  (LightRAG original, article metadata block added)
# ---------------------------------------------------------------------------

EXTRACTION_USER_PROMPT = """\
---Task---
Extract startup ecosystem entities and relationships from the article in "Data to be Processed".

---Instructions---
1.  **Strict Format Adherence:** Follow all format rules from the system prompt
    (5 fields / 4 delimiters for entities; 7 fields / 6 delimiters for relations).
    All fields are required; never omit a field or shift values left.
2.  **Output Content Only:** Output *only* the extracted entity and relationship lines.
    No introductory or concluding remarks.
3.  **Completion Signal:** Output `{completion_delimiter}` as the final line.
4.  **Entity types are restricted to:** [{entity_types}]
5.  **Tag Hint:** Article tags are publisher-provided name hints. Prefer tag spelling for
    canonical casing/punctuation of entity names (e.g. `Talon.One` not `Talon One`).
    Do not extract tags that are only geographies or generic categories.
6.  **Topic Guard:** Topic names and descriptions must stay general and reusable. Do not let
    startup names, company names, article titles, or publisher tags turn a broad topic into an
    article-specific phrase. For AI-related articles, use a topic such as `Kuenstliche Intelligenz`
    with a general definition; explain the specific company's AI use only in a `HAS_TOPIC`
    relationship.
7.  **Relationship Self-Check:** Before finalizing, verify that every founder or investor fact
    present in an entity description also appears as a relationship row when the article states it.
    Verify final relationship directions against each row's evidence: `INVESTED_IN` must be
    investor/company/person -> startup, `FOUNDED_BY` must be startup -> person, and `ACQUIRED`
    must be buyer/acquirer -> acquired startup/company. Swap reversed `FOUNDED_BY` or `ACQUIRED`
    rows before final output.
---Data to be Processed---
<Entity_types>[{entity_types}]
<Article_Metadata>
Title: {article_title}
Source: {source_name}
Published: {published_at}
Tags: [{article_tags}]
<Input Text>
```{input_text}```
<Output>\
"""


# ---------------------------------------------------------------------------
# Gleaning prompt  (LightRAG entity_continue_extraction_user_prompt, verbatim)
# ---------------------------------------------------------------------------

GLEANING_PROMPT = """\
---Task---
Based on the last extraction task, identify and extract any **missed or incorrectly formatted**
startup ecosystem entities and relationships from the same article.

---Instructions---
1.  **Strict Adherence to System Format:** Follow all format rules for entities and
    relationships as specified in the system instructions.
2.  **Focus on Corrections/Additions:**
    *   **Do NOT** re-output entities and relationships that were **correctly and fully**
        extracted in the last task.
    *   If an entity or relationship was **missed**, extract and output it now.
    *   If an entity or relationship was **truncated, had missing fields, or was incorrectly
        formatted**, re-output the *corrected and complete* version.
    *   Only add a missed entity when the article gives concrete facts about it. Do **not**
        add names from bare newsletter/list mentions, teaser lists, or subscription blurbs.
        For roundup/ticker articles, do **not** add relationships across separate mini-items or
        attach a topic from one mini-item to a startup in another mini-item.
        If the only missed items are name-only mentions, output only `{completion_delimiter}`.
    *   When adding or correcting `topic` entities, keep topic names and descriptions reusable
        across articles. Do not mention startups, companies, people, locations, article titles,
        funding amounts, or other one-off facts in topic entity descriptions. Move that context
        into `HAS_TOPIC` relationships.
    *   Check for relationship facts that were only captured in entity descriptions:
        if a person description says founder/co-founder/Mitgruender/Gruender, add the missing
        `Startup FOUNDED_BY Person` relationship; if an investor/company/person description says it
        invested in, plans to invest in, committed capital to, backed, financed, led a round for,
        or participated in a round for a startup, add the missing
        `InvestorOrCompanyOrPerson INVESTED_IN Startup` relationship.
    *   Review relationship directions against the evidence and correct reversed rows before final
        output: `INVESTED_IN` must be investor/company/person -> startup, `FOUNDED_BY` must be
        startup -> person, and `ACQUIRED` must be buyer/acquirer -> acquired startup/company. Only
        add `ACQUIRED` or `MERGED_WITH` for completed, agreed, signed, or announced transactions.
        Named prior investments in the same startup remain valid relationships when directly
        stated.
3.  **Output Format \u2014 Entities:** 5 fields per entity, delimited by `{tuple_delimiter}`,
    on a single line. The first field *must* be the literal string `entity`.
4.  **Output Format \u2014 Relationships:** 7 fields per relationship, delimited by
    `{tuple_delimiter}`, on a single line. The first field *must* be the literal string `relation`.
5.  **Output Content Only:** No introductory or concluding text.
6.  **Completion Signal:** Output `{completion_delimiter}` as the final line.
<Output>\
"""


# ---------------------------------------------------------------------------
# Description merge prompt  (LightRAG summarize_entity_descriptions, adapted)
# ---------------------------------------------------------------------------

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

---Input---
{description_type} Name: {description_name}
Description List:
```{description_list}```

---Output---\
"""


# ---------------------------------------------------------------------------
# Extraction examples  (startup domain, matching LightRAG example format)
# ---------------------------------------------------------------------------

EXTRACTION_EXAMPLES = [
    """\
<Entity_types>[startup, investor, person, topic, company]
<Input Text>
```
Swapfiets uebernimmt das E-Bikes-Startup Dance. Bosque Foods wird von Infinite
Roots uebernommen.
```
<Output>
entity{tuple_delimiter}Swapfiets{tuple_delimiter}company{tuple_delimiter}stated{tuple_delimiter}Swapfiets ist ein Unternehmen, das das E-Bikes-Startup Dance uebernimmt.
entity{tuple_delimiter}Dance{tuple_delimiter}startup{tuple_delimiter}stated{tuple_delimiter}Dance ist ein E-Bikes-Startup, das von Swapfiets uebernommen wird.
entity{tuple_delimiter}Bosque Foods{tuple_delimiter}startup{tuple_delimiter}stated{tuple_delimiter}Bosque Foods ist ein Startup, das von Infinite Roots uebernommen wird.
entity{tuple_delimiter}Infinite Roots{tuple_delimiter}company{tuple_delimiter}stated{tuple_delimiter}Infinite Roots ist ein Unternehmen, das Bosque Foods uebernimmt.
relation{tuple_delimiter}Swapfiets{tuple_delimiter}Dance{tuple_delimiter}ACQUIRED{tuple_delimiter}stated{tuple_delimiter}Uebernahme, E-Bikes{tuple_delimiter}Swapfiets uebernimmt das E-Bikes-Startup Dance.
relation{tuple_delimiter}Infinite Roots{tuple_delimiter}Bosque Foods{tuple_delimiter}ACQUIRED{tuple_delimiter}stated{tuple_delimiter}Uebernahme{tuple_delimiter}Bosque Foods wird von Infinite Roots uebernommen.
{completion_delimiter}""",
    """\
<Entity_types>[startup, investor, person, topic, company]
<Input Text>
```
Cohere uebernimmt Aleph Alpha. Die Schwarz Gruppe plant, 500 Millionen Euro in
das zusammengefuehrte Unternehmen zu investieren.
```
<Output>
entity{tuple_delimiter}Cohere{tuple_delimiter}startup{tuple_delimiter}stated{tuple_delimiter}Cohere ist ein KI-Startup, das Aleph Alpha uebernimmt und als Ziel einer geplanten Investition der Schwarz Gruppe genannt wird.
entity{tuple_delimiter}Aleph Alpha{tuple_delimiter}startup{tuple_delimiter}stated{tuple_delimiter}Aleph Alpha ist ein KI-Startup, das von Cohere uebernommen wird.
entity{tuple_delimiter}Schwarz Gruppe{tuple_delimiter}investor{tuple_delimiter}stated{tuple_delimiter}Schwarz Gruppe ist ein Unternehmensinvestor, der 500 Millionen Euro in das zusammengefuehrte Unternehmen von Cohere und Aleph Alpha investieren will.
relation{tuple_delimiter}Cohere{tuple_delimiter}Aleph Alpha{tuple_delimiter}ACQUIRED{tuple_delimiter}stated{tuple_delimiter}Uebernahme, KI{tuple_delimiter}Cohere uebernimmt Aleph Alpha.
relation{tuple_delimiter}Schwarz Gruppe{tuple_delimiter}Cohere{tuple_delimiter}INVESTED_IN{tuple_delimiter}stated{tuple_delimiter}Finanzierung, KI{tuple_delimiter}Schwarz Gruppe plant, 500 Millionen Euro in das zusammengefuehrte Unternehmen zu investieren.
{completion_delimiter}""",
    """\
<Entity_types>[startup, investor, person, topic, company]
<Input Text>
```
Das Muenchner KI-Startup Aleph Alpha fusionierte mit dem kanadischen KI-Unternehmen Cohere.
Beide Unternehmen wollen gemeinsam KI-Infrastruktur fuer Unternehmen aufbauen.
```
<Output>
entity{tuple_delimiter}Aleph Alpha{tuple_delimiter}startup{tuple_delimiter}stated{tuple_delimiter}Aleph Alpha ist ein Muenchner KI-Startup, das mit Cohere fusionierte.
entity{tuple_delimiter}Cohere{tuple_delimiter}startup{tuple_delimiter}stated{tuple_delimiter}Cohere ist ein kanadisches KI-Unternehmen, das mit Aleph Alpha fusionierte.
entity{tuple_delimiter}Kuenstliche Intelligenz{tuple_delimiter}topic{tuple_delimiter}stated{tuple_delimiter}Kuenstliche Intelligenz bezeichnet Systeme, die Aufgaben wie Analyse, Mustererkennung oder Entscheidungsunterstuetzung automatisiert ausfuehren.
relation{tuple_delimiter}Aleph Alpha{tuple_delimiter}Cohere{tuple_delimiter}MERGED_WITH{tuple_delimiter}stated{tuple_delimiter}Fusion, KI{tuple_delimiter}Aleph Alpha und Cohere fusionierten, um gemeinsam KI-Infrastruktur aufzubauen.
relation{tuple_delimiter}Aleph Alpha{tuple_delimiter}Kuenstliche Intelligenz{tuple_delimiter}HAS_TOPIC{tuple_delimiter}stated{tuple_delimiter}Technologie, KI{tuple_delimiter}Aleph Alpha entwickelt KI-Technologie.
{completion_delimiter}""",
    """\
<Entity_types>[startup, investor, person, topic, company]
<Input Text>
```
Das Berliner FinTech Moss sammelte in einer Series-B-Runde 75 Millionen Euro ein.
Angefuehrt wurde die Runde von Tiger Global, auch Valar Ventures beteiligte sich.
Mitgruender und CEO Anton Rummel sagte, das Kapital solle die Expansion beschleunigen.
```
<Output>
entity{tuple_delimiter}Moss{tuple_delimiter}startup{tuple_delimiter}stated{tuple_delimiter}Moss ist ein Berliner FinTech, das in einer Series-B-Runde 75 Millionen Euro einsammelte.
entity{tuple_delimiter}Tiger Global{tuple_delimiter}investor{tuple_delimiter}stated{tuple_delimiter}Tiger Global ist ein Investor, der die Series-B-Runde von Moss anfuehrte.
entity{tuple_delimiter}Valar Ventures{tuple_delimiter}investor{tuple_delimiter}stated{tuple_delimiter}Valar Ventures ist ein Investor, der sich an der Series-B-Runde von Moss beteiligte.
entity{tuple_delimiter}Anton Rummel{tuple_delimiter}person{tuple_delimiter}stated{tuple_delimiter}Anton Rummel ist Mitgruender und CEO von Moss.
entity{tuple_delimiter}FinTech{tuple_delimiter}topic{tuple_delimiter}stated{tuple_delimiter}FinTech bezeichnet technologiegetriebene Finanzdienstleistungen und digitale Finanzprodukte.
relation{tuple_delimiter}Tiger Global{tuple_delimiter}Moss{tuple_delimiter}INVESTED_IN{tuple_delimiter}stated{tuple_delimiter}Series B, Finanzierung{tuple_delimiter}Tiger Global fuehrte die Series-B-Runde von Moss an.
relation{tuple_delimiter}Valar Ventures{tuple_delimiter}Moss{tuple_delimiter}INVESTED_IN{tuple_delimiter}stated{tuple_delimiter}Series B, Beteiligung{tuple_delimiter}Valar Ventures beteiligte sich an der Series-B-Finanzierung von Moss.
relation{tuple_delimiter}Moss{tuple_delimiter}Anton Rummel{tuple_delimiter}FOUNDED_BY{tuple_delimiter}stated{tuple_delimiter}Gruendung, Fuehrung{tuple_delimiter}Anton Rummel ist Mitgruender und CEO von Moss.
relation{tuple_delimiter}Moss{tuple_delimiter}FinTech{tuple_delimiter}HAS_TOPIC{tuple_delimiter}stated{tuple_delimiter}Branche, Geschaeftsmodell{tuple_delimiter}Moss ist im FinTech-Sektor aktiv.
{completion_delimiter}""",
    """\
<Entity_types>[startup, investor, person, topic, company]
<Input Text>
```
Das Hamburger SaaS-Startup Stacker startete einen No-Code-Baukasten fuer interne
Tools und wird von HV Capital unterstuetzt. CEO Lena Bauer sagte, das Produkt
richte sich an Operations-Teams im Mittelstand.
```
<Output>
entity{tuple_delimiter}Stacker{tuple_delimiter}startup{tuple_delimiter}stated{tuple_delimiter}Stacker ist ein Hamburger SaaS-Startup mit einem No-Code-Baukasten fuer interne Tools fuer Operations-Teams im Mittelstand.
entity{tuple_delimiter}Lena Bauer{tuple_delimiter}person{tuple_delimiter}stated{tuple_delimiter}Lena Bauer ist CEO von Stacker.
entity{tuple_delimiter}No-Code{tuple_delimiter}topic{tuple_delimiter}stated{tuple_delimiter}No-Code bezeichnet Softwareentwicklung und Anwendungserstellung ohne klassische Programmierung.
entity{tuple_delimiter}SaaS{tuple_delimiter}topic{tuple_delimiter}stated{tuple_delimiter}SaaS bezeichnet Software as a Service, also ueber das Internet bereitgestellte Software.
relation{tuple_delimiter}Stacker{tuple_delimiter}No-Code{tuple_delimiter}HAS_TOPIC{tuple_delimiter}stated{tuple_delimiter}Produkt, Technologie{tuple_delimiter}Stacker bietet einen No-Code-Baukasten fuer interne Tools.
relation{tuple_delimiter}Stacker{tuple_delimiter}SaaS{tuple_delimiter}HAS_TOPIC{tuple_delimiter}stated{tuple_delimiter}Geschaeftsmodell{tuple_delimiter}Stacker ist ein SaaS-Startup.
{completion_delimiter}""",
    """\
<Entity_types>[startup, investor, person, topic, company]
<Input Text>
```
Nach Angaben von Stacker fuehrte HV Capital die neue Finanzierungsrunde an.
Branchenkreisen zufolge koennte sich Cherry Ventures ebenfalls beteiligt haben;
Stacker bestaetigte diese Information nicht.
```
<Output>
entity{tuple_delimiter}Stacker{tuple_delimiter}startup{tuple_delimiter}attributed{tuple_delimiter}Stacker gab eine neue Finanzierungsrunde unter Fuehrung von HV Capital bekannt.
entity{tuple_delimiter}HV Capital{tuple_delimiter}investor{tuple_delimiter}attributed{tuple_delimiter}HV Capital wurde von Stacker als fuehrender Investor der neuen Finanzierungsrunde genannt.
entity{tuple_delimiter}Cherry Ventures{tuple_delimiter}investor{tuple_delimiter}unsure{tuple_delimiter}Cherry Ventures wurde in unbestaetigten Berichten als moeglicher Teilnehmer der Finanzierungsrunde genannt.
relation{tuple_delimiter}HV Capital{tuple_delimiter}Stacker{tuple_delimiter}INVESTED_IN{tuple_delimiter}attributed{tuple_delimiter}Finanzierung, Mitteilung{tuple_delimiter}Stacker gab an, dass HV Capital die Finanzierungsrunde anfuehrte.
relation{tuple_delimiter}Cherry Ventures{tuple_delimiter}Stacker{tuple_delimiter}INVESTED_IN{tuple_delimiter}unsure{tuple_delimiter}Finanzierung, Geruecht{tuple_delimiter}Eine moegliche Beteiligung von Cherry Ventures wurde nicht bestaetigt.
{completion_delimiter}""",
    """\
<Entity_types>[startup, investor, person, topic, company]
<Input Text>
```
Im Startup-Radar-Newsletter, unserem kostenpflichtigen Newsletter, berichten wir
ueber diese Startups: ioncentric, Elephant, Execurater, Leadary, Rethinking Job,
Tvently, Carbony, SilverFriend, caremare, QUCOXX, Open Wonder, Lumina, Badger,
Kai Karosse und Lockaly. 30 Tage kostenlos testen.
```
<Output>
{completion_delimiter}""",
    """\
<Entity_types>[startup, investor, person, topic, company]
<Input Text>
```
BidFix +++ Die Jungfirma BidFix ist unser Startup der Woche! Hinter dem Unternehmen,
von Alexander Kohler und Jonas Matthaei in Muenchen gegruendet, verbirgt sich ein
"KI-Assistent fuer oeffentliche Ausschreibungen". "Unsere KI analysiert
Vergabeunterlagen automatisch", so das GovTech.
```
<Output>
entity{tuple_delimiter}BidFix{tuple_delimiter}startup{tuple_delimiter}stated{tuple_delimiter}BidFix ist eine in Muenchen von Alexander Kohler und Jonas Matthaei gegruendete Jungfirma mit einem KI-Assistenten fuer oeffentliche Ausschreibungen.
entity{tuple_delimiter}Alexander Kohler{tuple_delimiter}person{tuple_delimiter}stated{tuple_delimiter}Alexander Kohler ist Mitgruender von BidFix.
entity{tuple_delimiter}Jonas Matthaei{tuple_delimiter}person{tuple_delimiter}stated{tuple_delimiter}Jonas Matthaei ist Mitgruender von BidFix.
entity{tuple_delimiter}Kuenstliche Intelligenz{tuple_delimiter}topic{tuple_delimiter}stated{tuple_delimiter}Kuenstliche Intelligenz bezeichnet Systeme, die Aufgaben wie Analyse, Mustererkennung oder Entscheidungsunterstuetzung automatisiert ausfuehren.
entity{tuple_delimiter}GovTech{tuple_delimiter}topic{tuple_delimiter}stated{tuple_delimiter}GovTech bezeichnet digitale Technologien, Produkte und Dienste fuer Verwaltung und oeffentlichen Sektor.
relation{tuple_delimiter}BidFix{tuple_delimiter}Alexander Kohler{tuple_delimiter}FOUNDED_BY{tuple_delimiter}stated{tuple_delimiter}Gruendung, GovTech{tuple_delimiter}Alexander Kohler gruendete BidFix mit Jonas Matthaei in Muenchen.
relation{tuple_delimiter}BidFix{tuple_delimiter}Jonas Matthaei{tuple_delimiter}FOUNDED_BY{tuple_delimiter}stated{tuple_delimiter}Gruendung, GovTech{tuple_delimiter}Jonas Matthaei gruendete BidFix mit Alexander Kohler in Muenchen.
relation{tuple_delimiter}BidFix{tuple_delimiter}Kuenstliche Intelligenz{tuple_delimiter}HAS_TOPIC{tuple_delimiter}stated{tuple_delimiter}Produkt, KI{tuple_delimiter}BidFix bietet einen KI-Assistenten fuer oeffentliche Ausschreibungen.
relation{tuple_delimiter}BidFix{tuple_delimiter}GovTech{tuple_delimiter}HAS_TOPIC{tuple_delimiter}stated{tuple_delimiter}Marktsegment, oeffentlicher Sektor{tuple_delimiter}BidFix wird im Artikel als GovTech beschrieben.
{completion_delimiter}""",
    """\
<Entity_types>[startup, investor, person, topic, company]
<Input Text>
```
Die US-amerikanische DJ-Tech-Firma inMusic uebernimmt das insolvente Berliner
Musikunternehmen Native Instruments. "Native Instruments represents everything
we look for in a partner", teilt inMusic mit. Native Instruments, 1996 gegruendet,
kuemmert sich um Software und Hardware fuer digitale Musikproduktion. Die
Beteiligungsgesellschaft Francisco Partners uebernahm 2021 die Mehrheit an
Native Instruments. 350 Mitarbeitende wirkten zuletzt fuer Native Instruments.
```
<Output>
entity{tuple_delimiter}inMusic{tuple_delimiter}company{tuple_delimiter}stated{tuple_delimiter}inMusic ist eine US-amerikanische DJ-Tech-Firma, die Native Instruments uebernimmt.
entity{tuple_delimiter}Native Instruments{tuple_delimiter}company{tuple_delimiter}stated{tuple_delimiter}Native Instruments ist ein insolventes Berliner Musikunternehmen mit Software und Hardware fuer digitale Musikproduktion.
entity{tuple_delimiter}DJ-Tech{tuple_delimiter}topic{tuple_delimiter}stated{tuple_delimiter}DJ-Tech bezeichnet Technologien, Software und Hardware fuer DJs und elektronische Musikperformance.
entity{tuple_delimiter}Digitale Musikproduktion{tuple_delimiter}topic{tuple_delimiter}stated{tuple_delimiter}Digitale Musikproduktion bezeichnet die Erstellung, Bearbeitung und Auffuehrung von Musik mit digitaler Software und Hardware.
relation{tuple_delimiter}inMusic{tuple_delimiter}Native Instruments{tuple_delimiter}ACQUIRED{tuple_delimiter}stated{tuple_delimiter}aktuelle Uebernahme, Insolvenz{tuple_delimiter}inMusic uebernimmt das insolvente Berliner Musikunternehmen Native Instruments.
relation{tuple_delimiter}inMusic{tuple_delimiter}DJ-Tech{tuple_delimiter}HAS_TOPIC{tuple_delimiter}stated{tuple_delimiter}Branche, DJ-Tech{tuple_delimiter}inMusic wird als US-amerikanische DJ-Tech-Firma beschrieben.
relation{tuple_delimiter}Native Instruments{tuple_delimiter}Digitale Musikproduktion{tuple_delimiter}HAS_TOPIC{tuple_delimiter}stated{tuple_delimiter}Produkt, Musiksoftware{tuple_delimiter}Native Instruments entwickelt Software und Hardware fuer digitale Musikproduktion.
{completion_delimiter}""",
]


# ---------------------------------------------------------------------------
# Builder functions
# ---------------------------------------------------------------------------


def build_extraction_system_prompt() -> str:
    examples_str = "\n".join(
        ex.format(tuple_delimiter=TUPLE_DELIMITER, completion_delimiter=COMPLETION_DELIMITER)
        for ex in EXTRACTION_EXAMPLES
    )
    return EXTRACTION_SYSTEM_PROMPT.format(
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
        entity_types=", ".join(ENTITY_TYPES),
        examples=examples_str,
    )


def build_extraction_user_prompt(article: ArticleIn) -> str:
    tags = ", ".join(article.tags[:20]) if article.tags else "none"
    published = article.published_at.isoformat() if article.published_at else "unknown"
    return EXTRACTION_USER_PROMPT.format(
        completion_delimiter=COMPLETION_DELIMITER,
        entity_types=", ".join(ENTITY_TYPES),
        article_title=article.title,
        source_name=article.source_name,
        published_at=published,
        article_tags=tags,
        input_text=article.text[:ARTICLE_TEXT_CHAR_LIMIT],
    )


def build_extraction_user_prompt_template() -> str:
    return EXTRACTION_USER_PROMPT.format(
        completion_delimiter=COMPLETION_DELIMITER,
        entity_types=", ".join(ENTITY_TYPES),
        article_title="{{article_title}}",
        source_name="{{source_name}}",
        published_at="{{published_at}}",
        article_tags="{{article_tags}}",
        input_text="{{input_text}}",
    )


def build_extraction_prompt_registry_template() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_extraction_system_prompt()},
        {"role": "user", "content": build_extraction_user_prompt_template()},
    ]


def build_gleaning_prompt_registry_template() -> list[dict[str, str]]:
    return [
        {"role": "user", "content": build_gleaning_prompt()},
    ]


def build_gleaning_prompt() -> str:
    return GLEANING_PROMPT.format(
        tuple_delimiter=TUPLE_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
    )


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
