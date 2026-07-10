from __future__ import annotations

import re
from dataclasses import dataclass


_REFERENCE_RE = re.compile(
    r"^(?P<start>B(?P<start_block>\d{3,})\.S(?P<start_sentence>\d{3,}))"
    r"(?:-(?P<end>B(?P<end_block>\d{3,})\.S(?P<end_sentence>\d{3,})))?$"
)
_SENTENCE_END_RE = re.compile(r'[.!?]+(?:["”’»)\]]+)?(?=\s+|$)')


@dataclass(frozen=True)
class EvidenceSentence:
    reference: str
    block_number: int
    sentence_number: int
    start: int
    end: int


@dataclass(frozen=True)
class EvidenceCatalog:
    article_text: str
    rendered_body: str
    sentences: dict[str, EvidenceSentence]

    def resolve(self, reference: str) -> tuple[str | None, str | None]:
        candidate = reference.strip()
        if not candidate:
            return None, "missing_ref"
        if "," in candidate or "+" in candidate:
            return None, "noncontiguous_ref"

        match = _REFERENCE_RE.fullmatch(candidate)
        if match is None:
            return None, "malformed_ref"

        start_ref = match.group("start")
        end_ref = match.group("end") or start_ref
        if match.group("start_block") != (match.group("end_block") or match.group("start_block")):
            return None, "cross_block_ref"

        start_sentence = int(match.group("start_sentence"))
        end_sentence = int(match.group("end_sentence") or match.group("start_sentence"))
        if start_sentence > end_sentence:
            return None, "reversed_ref"

        start = self.sentences.get(start_ref)
        end = self.sentences.get(end_ref)
        if start is None or end is None:
            return None, "unknown_ref"
        return self.article_text[start.start : end.end], None


def build_evidence_catalog(article_text: str) -> EvidenceCatalog:
    sentences: dict[str, EvidenceSentence] = {}
    rendered_lines: list[str] = []
    block_number = 0
    cursor = 0

    for raw_line in article_text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        line_start = cursor
        cursor += len(raw_line)

        if not line.strip():
            continue
        if line.startswith("## "):
            rendered_lines.append(line)
            continue

        block_number += 1
        block_id = f"B{block_number:03d}"
        rendered_lines.append(f'<EvidenceBlock id="{block_id}">')
        for sentence_number, (relative_start, relative_end) in enumerate(
            _sentence_spans(line),
            start=1,
        ):
            reference = f"{block_id}.S{sentence_number:03d}"
            sentences[reference] = EvidenceSentence(
                reference=reference,
                block_number=block_number,
                sentence_number=sentence_number,
                start=line_start + relative_start,
                end=line_start + relative_end,
            )
            rendered_lines.append(f"[{reference}] {line[relative_start:relative_end]}")
        rendered_lines.append("</EvidenceBlock>")

    return EvidenceCatalog(
        article_text=article_text,
        rendered_body="\n".join(rendered_lines),
        sentences=sentences,
    )


def _sentence_spans(block: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(block):
        end = match.end()
        if block[start:end].strip():
            sentence_start = start
            while sentence_start < end and block[sentence_start].isspace():
                sentence_start += 1
            spans.append((sentence_start, end))
        start = end
        while start < len(block) and block[start].isspace():
            start += 1

    if start < len(block) and block[start:].strip():
        end = len(block)
        while end > start and block[end - 1].isspace():
            end -= 1
        spans.append((start, end))

    if not spans and block.strip():
        start = len(block) - len(block.lstrip())
        end = len(block.rstrip())
        spans.append((start, end))
    return spans
