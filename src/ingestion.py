"""Document loading and page-level parse caching for PDF, DOCX, Markdown, and TXT files."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pymupdf
from docx import Document as DocxDocument

from src.config import PARSE_CACHE_DIR

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".txt"}


@dataclass
class PageRecord:
    """Extracted text for a single logical page of a document."""

    source_name: str
    page_number: int  # 1-indexed for human-readable citations
    text: str
    file_hash: str


def compute_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def compute_sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _normalize_extracted_text(raw: str) -> str:
    """Clean PDF extraction artifacts without flattening code indentation."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    replacements = {
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\u00a0": " ",
        "\u2013": "-",
        "\u2014": "--",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)

    lines = [line.rstrip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _try_ocr_page(page: pymupdf.Page) -> str:
    """Fallback OCR for scanned PDF pages when RapidOCR is installed."""
    try:
        import numpy as np
        from PIL import Image
        import io
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        return ""

    pix = page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    ocr_engine = RapidOCR()
    result, _ = ocr_engine(np.array(img))
    if not result:
        return ""
    lines = [entry[1].strip() for entry in result if len(entry) >= 2 and entry[1].strip()]
    return "\n".join(lines)


def _looks_like_code_text(text: str) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return False
    code_prefixes = (
        "def ", "class ", "import ", "from ", "return ", "for ", "while ", "if ", "@", "#"
    )
    indented = sum(1 for ln in lines if ln.startswith("    ") or ln.startswith("  "))
    prefixed = sum(1 for ln in lines if ln.lstrip().startswith(code_prefixes))
    return prefixed >= 1 and (indented >= 1 or len(lines) >= 2)


def _format_pdf_blocks(page: pymupdf.Page) -> str:
    """Reconstruct clean paragraphs, code listings, and pipe-delimited tables from PDF layout blocks."""
    page_h = page.rect.height
    table_rects: list[pymupdf.Rect] = []
    positioned_items: list[tuple[float, float, str]] = []

    # Detect structured bordered tables first so columns stay pipe-delimited
    try:
        detected_tables = page.find_tables()
        for tbl in detected_tables.tables:
            t_rect = pymupdf.Rect(tbl.bbox)
            extracted_rows = tbl.extract()
            pipe_rows: list[str] = []
            for r in extracted_rows:
                if not r:
                    continue
                cells = [" ".join((c or "").split()) for c in r]
                if any(cells):
                    pipe_rows.append(" | ".join(cells))
            if len(pipe_rows) >= 2:
                table_rects.append(t_rect)
                positioned_items.append((t_rect.y0, t_rect.x0, "\n".join(pipe_rows)))
    except Exception:
        pass

    raw_blocks = page.get_text("blocks")
    for blk in raw_blocks:
        if len(blk) < 7 or blk[6] != 0:
            continue
        x0, y0, x1, y1, content = blk[0], blk[1], blk[2], blk[3], blk[4]
        b_rect = pymupdf.Rect(x0, y0, x1, y1)
        # Skip text blocks already captured inside a detected table
        if any(
            (b_rect & tr).get_area() >= 0.55 * max(b_rect.get_area(), 1.0)
            for tr in table_rects
        ):
            continue

        cleaned = _normalize_extracted_text(content)
        if not cleaned:
            continue
        if (y0 < 48 or y1 > page_h - 48) and (
            "CourseRAG Knowledge Base" in cleaned or re.match(r"^Page\s+\d+$", cleaned, re.IGNORECASE)
        ):
            continue

        if _looks_like_code_text(cleaned):
            positioned_items.append((y0, x0, cleaned))
        else:
            unwrapped = re.sub(r"\s*\n\s*", " ", cleaned).strip()
            positioned_items.append((y0, x0, unwrapped))

    if not positioned_items:
        return ""

    positioned_items.sort(key=lambda item: (round(item[0], 1), round(item[1], 1)))
    return "\n\n".join(item[2] for item in positioned_items).strip()


def _extract_pdf_bytes(raw_bytes: bytes, source_name: str, file_hash: str) -> list[PageRecord]:
    records: list[PageRecord] = []
    with pymupdf.open(stream=raw_bytes, filetype="pdf") as doc:
        for idx in range(len(doc)):
            page = doc[idx]
            text = _format_pdf_blocks(page)
            if len(text) < 60:
                fallback_plain = _normalize_extracted_text(page.get_text("text"))
                if len(fallback_plain) > len(text):
                    text = fallback_plain
            if len(text) < 60 and page.get_images():
                ocr_text = _normalize_extracted_text(_try_ocr_page(page))
                if len(ocr_text) > len(text):
                    text = ocr_text
            if text:
                records.append(
                    PageRecord(
                        source_name=source_name,
                        page_number=idx + 1,
                        text=text,
                        file_hash=file_hash,
                    )
                )
    return records


def _extract_docx_bytes(raw_bytes: bytes, source_name: str, file_hash: str) -> list[PageRecord]:
    import io

    doc = DocxDocument(io.BytesIO(raw_bytes))
    blocks: list[str] = []

    for para in doc.paragraphs:
        content = para.text.strip()
        if not content:
            continue
        style_name = (para.style.name or "").lower()
        if "heading" in style_name:
            blocks.append(f"\n## {content}\n")
        else:
            blocks.append(content)

    for table in doc.tables:
        rows = []
        for row in table.rows:
            cells = [c.text.strip().replace("\n", " ") for c in row.cells]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            blocks.append("\n".join(rows))

    return _paginate_text_blocks(blocks, source_name, file_hash, chars_per_page=2400)


def _extract_plain_text(raw_bytes: bytes, source_name: str, file_hash: str) -> list[PageRecord]:
    decoded = raw_bytes.decode("utf-8", errors="replace")
    normalized = _normalize_extracted_text(decoded)
    paragraphs = [p for p in normalized.split("\n\n") if p.strip()]
    return _paginate_text_blocks(paragraphs, source_name, file_hash, chars_per_page=2600)


def _paginate_text_blocks(
    blocks: list[str],
    source_name: str,
    file_hash: str,
    chars_per_page: int = 2500,
) -> list[PageRecord]:
    records: list[PageRecord] = []
    buffer: list[str] = []
    running_len = 0
    page_num = 1

    for block in blocks:
        if running_len + len(block) > chars_per_page and buffer:
            records.append(
                PageRecord(
                    source_name=source_name,
                    page_number=page_num,
                    text="\n\n".join(buffer).strip(),
                    file_hash=file_hash,
                )
            )
            page_num += 1
            buffer = [block]
            running_len = len(block)
        else:
            buffer.append(block)
            running_len += len(block) + 2

    if buffer:
        records.append(
            PageRecord(
                source_name=source_name,
                page_number=page_num,
                text="\n\n".join(buffer).strip(),
                file_hash=file_hash,
            )
        )
    return records


def load_document_bytes(
    filename: str,
    raw_bytes: bytes,
    cache_dir: Path = PARSE_CACHE_DIR,
) -> list[PageRecord]:
    """Parse raw document bytes with SHA-256 disk caching."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    file_hash = compute_sha256_bytes(raw_bytes)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{file_hash[:20]}.json"

    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("file_hash") == file_hash and cached.get("pages"):
                return [
                    PageRecord(
                        source_name=filename,
                        page_number=item["page_number"],
                        text=item["text"],
                        file_hash=file_hash,
                    )
                    for item in cached["pages"]
                ]
        except (json.JSONDecodeError, KeyError):
            pass

    if suffix == ".pdf":
        records = _extract_pdf_bytes(raw_bytes, filename, file_hash)
    elif suffix == ".docx":
        records = _extract_docx_bytes(raw_bytes, filename, file_hash)
    else:
        records = _extract_plain_text(raw_bytes, filename, file_hash)

    if not records:
        raise RuntimeError(f"No readable text could be extracted from '{filename}'.")

    payload = {
        "source_name": filename,
        "file_hash": file_hash,
        "pages": [asdict(r) for r in records],
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def load_document_file(path: Path, cache_dir: Path = PARSE_CACHE_DIR) -> list[PageRecord]:
    """Load a local document from disk."""
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Document not found: {resolved}")
    return load_document_bytes(resolved.name, resolved.read_bytes(), cache_dir=cache_dir)
