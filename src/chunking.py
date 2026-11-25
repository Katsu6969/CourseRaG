"""Structure-aware text chunker preserving section headings and code blocks."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

from src.ingestion import PageRecord

CHUNK_NAMESPACE = uuid.UUID("7f9b13a4-3d82-48d0-9f5c-2189f4e2a101")

_HEADING_PATTERNS = [
    re.compile(r"^#{1,4}\s+(.+)$"),
    re.compile(r"^(Chapter\s+\d+\s*[:\-–]\s*.+)$", re.IGNORECASE),
    re.compile(r"^(\d+\.\d+(?:\.\d+)?\s+[A-Z][A-Za-z0-9\s,&/\-()'\"+*—–:]{3,85})$"),
    re.compile(r"^([A-Z][A-Z0-9\s,&/\-()']{5,65})$"),
]


@dataclass
class TextChunk:
    """Indexed document chunk with provenance metadata."""

    chunk_id: str
    text: str
    source_name: str
    page_number: int
    section_title: str
    file_hash: str
    char_count: int

    @property
    def embedding_input(self) -> str:
        if self.section_title and self.section_title.lower() != "general":
            return f"{self.section_title}\n{self.text}"
        return self.text


def _detect_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    # Skip lines that look like code or table rows
    if any(tok in stripped for tok in ("def ", "class ", "return ", "import ", "|", "=", "{", "}")):
        return None
    for pat in _HEADING_PATTERNS:
        m = pat.match(stripped)
        if m:
            heading = m.group(1).strip()
            # Ignore running page footers like "PAGE 4"
            if re.match(r"^page\s+\d+$", heading, re.IGNORECASE):
                return None
            return heading
    return None


def _split_oversized_unit(unit: str, max_chars: int, overlap: int) -> list[str]:
    """Break a single long paragraph or code block cleanly along line or sentence boundaries."""
    if len(unit) <= max_chars:
        return [unit]

    # Prefer newline splits first (critical for code listings), then sentence boundaries
    lines = unit.split("\n")
    if len(lines) > 1:
        pieces: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for line in lines:
            if buf_len + len(line) + 1 > max_chars and buf:
                pieces.append("\n".join(buf).strip())
                # Keep trailing lines up to overlap budget
                tail: list[str] = []
                tail_len = 0
                for prev in reversed(buf):
                    if tail_len + len(prev) + 1 <= overlap:
                        tail.insert(0, prev)
                        tail_len += len(prev) + 1
                    else:
                        break
                buf = tail + [line]
                buf_len = sum(len(x) + 1 for x in buf)
            else:
                buf.append(line)
                buf_len += len(line) + 1
        if buf:
            pieces.append("\n".join(buf).strip())
        return [p for p in pieces if p]

    # Single giant paragraph without newlines: split by sentences
    sentences = re.split(r"(?<=[.!?])\s+", unit)
    pieces = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > max_chars and current:
            pieces.append(current.strip())
            tail = current[-overlap:].strip() if overlap > 0 else ""
            current = f"{tail} {sent}".strip()
        else:
            current = f"{current} {sent}".strip() if current else sent
    if current:
        pieces.append(current.strip())
    return pieces


def chunk_pages(
    pages: list[PageRecord],
    chunk_size: int = 850,
    chunk_overlap: int = 140,
) -> list[TextChunk]:
    """Split extracted pages into overlapping, section-tagged chunks."""
    chunks: list[TextChunk] = []
    active_section = "Overview"

    for page in pages:
        raw_blocks = [b.strip() for b in page.text.split("\n\n") if b.strip()]
        tagged_units: list[tuple[str, str]] = []

        for block in raw_blocks:
            lines = block.split("\n")
            first_heading = _detect_heading(lines[0])
            if first_heading:
                active_section = first_heading
                if len(lines) == 1:
                    # Keep heading attached to next content unit instead of emitting a tiny orphan chunk
                    continue

            for sub_unit in _split_oversized_unit(block, chunk_size, chunk_overlap):
                tagged_units.append((active_section, sub_unit))

        if not tagged_units and page.text.strip():
            tagged_units.append((active_section, page.text.strip()[:chunk_size]))

        buf_texts: list[str] = []
        buf_section = active_section
        buf_len = 0
        local_idx = 0

        for sec, unit in tagged_units:
            if not buf_texts:
                buf_section = sec

            section_changed = bool(buf_texts and sec != buf_section)
            projected = buf_len + len(unit) + (2 if buf_texts else 0)
            if (projected > chunk_size or section_changed) and buf_texts:
                merged = "\n\n".join(buf_texts).strip()
                cid = str(uuid.uuid5(CHUNK_NAMESPACE, f"{page.file_hash}:{page.page_number}:{local_idx}"))
                chunks.append(
                    TextChunk(
                        chunk_id=cid,
                        text=merged,
                        source_name=page.source_name,
                        page_number=page.page_number,
                        section_title=buf_section,
                        file_hash=page.file_hash,
                        char_count=len(merged),
                    )
                )
                local_idx += 1

                # Carry overlap from the end of previous buffer if staying in same section
                overlap_blocks: list[str] = []
                overlap_chars = 0
                if chunk_overlap > 0 and sec == buf_section:
                    for prev_unit in reversed(buf_texts):
                        if overlap_chars + len(prev_unit) <= chunk_overlap:
                            overlap_blocks.insert(0, prev_unit)
                            overlap_chars += len(prev_unit) + 2
                        else:
                            # Take last sentence/lines of prev_unit if overlap_blocks is still empty
                            if not overlap_blocks and chunk_overlap >= 80:
                                tail_snippet = prev_unit[-chunk_overlap:]
                                nl_pos = tail_snippet.find("\n")
                                sp_pos = tail_snippet.find(" ")
                                cut = nl_pos + 1 if nl_pos != -1 else (sp_pos + 1 if sp_pos != -1 else 0)
                                trimmed = tail_snippet[cut:].strip()
                                if trimmed:
                                    overlap_blocks.append(trimmed)
                                    overlap_chars = len(trimmed)
                            break

                buf_texts = overlap_blocks + [unit]
                buf_section = sec
                buf_len = sum(len(x) for x in buf_texts) + 2 * max(0, len(buf_texts) - 1)
            else:
                buf_texts.append(unit)
                buf_len = projected

        if buf_texts:
            merged = "\n\n".join(buf_texts).strip()
            cid = str(uuid.uuid5(CHUNK_NAMESPACE, f"{page.file_hash}:{page.page_number}:{local_idx}"))
            chunks.append(
                TextChunk(
                    chunk_id=cid,
                    text=merged,
                    source_name=page.source_name,
                    page_number=page.page_number,
                    section_title=buf_section,
                    file_hash=page.file_hash,
                    char_count=len(merged),
                )
            )

    return chunks
