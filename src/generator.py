"""Conversational query contextualization, multi-provider cloud LLM calls, and local MMR synthesis."""

from __future__ import annotations

import re
import httpx

from src.config import RAGSettings
from src.retriever import RetrievedChunk, code_aware_tokenize

_FOLLOWUP_PRONOUNS = {
    "it", "its", "they", "their", "them", "that", "this", "those", "these", "same",
}

SYSTEM_PROMPT = """You are CourseRAG, a precise computer science and technical study assistant.
Answer the user's question primarily using the supplied document excerpts.

Guidelines:
1. Ground every technical claim, definition, complexity figure, and code pattern in the retrieved excerpts.
2. Cite sources inline using the exact format [DocumentName, p. X] whenever referencing information from an excerpt.
3. When code examples or complexity trade-offs are relevant, include clean, runnable Python examples and clearly state time/space complexity.
4. If the retrieved excerpts do not contain enough information to answer the question, state clearly what is missing rather than guessing or fabricating page citations.
5. Write clearly and concisely like a knowledgeable teaching assistant—avoid filler intros or robotic disclaimers."""


def contextualize_query(question: str, chat_history: list[dict[str, str]] | None = None) -> str:
    """Expand pronoun-heavy follow-up queries with recent user context for better retrieval."""
    cleaned = question.strip()
    if not chat_history:
        return cleaned

    words = {w.strip(".,?!:;()[]\"'").lower() for w in cleaned.split()}
    needs_context = bool(words & _FOLLOWUP_PRONOUNS) or len(words) <= 5
    if not needs_context:
        return cleaned

    # Pull key technical terms from the most recent prior user turn
    prior_user_msgs = [m["content"] for m in chat_history if m.get("role") == "user" and m.get("content")]
    if not prior_user_msgs:
        return cleaned

    last_topic_tokens = code_aware_tokenize(prior_user_msgs[-1])
    if not last_topic_tokens:
        return cleaned

    seen: list[str] = []
    for tok in last_topic_tokens:
        if tok not in seen and tok not in words:
            seen.append(tok)
        if len(seen) >= 6:
            break

    if not seen:
        return cleaned
    return f"{cleaned} (context: {' '.join(seen)})"


def format_context_for_llm(retrieved: list[RetrievedChunk]) -> str:
    blocks: list[str] = []
    for idx, item in enumerate(retrieved, start=1):
        ch = item.chunk
        header = (
            f"[Excerpt {idx} | Source: {ch.source_name}, p. {ch.page_number} "
            f"| Section: {ch.section_title}]"
        )
        blocks.append(f"{header}\n{ch.text}")
    return "\n\n---\n\n".join(blocks)


def _build_messages(
    question: str,
    context_str: str,
    chat_history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        # Include up to the last 4 turns for conversational continuity
        for turn in chat_history[-4:]:
            role = turn.get("role", "user")
            if role in {"user", "assistant"} and turn.get("content"):
                messages.append({"role": role, "content": turn["content"][:1200]})

    user_payload = (
        f"Retrieved Document Excerpts:\n\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Provide a well-structured answer with inline [DocumentName, p. X] citations."
    )
    messages.append({"role": "user", "content": user_payload})
    return messages


def _call_openai_compatible(
    endpoint: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    with httpx.Client(timeout=40.0) as client:
        resp = client.post(endpoint, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def _call_gemini(
    api_key: str,
    model: str,
    question: str,
    context_str: str,
    chat_history: list[dict[str, str]] | None,
    temperature: float,
    max_tokens: int,
) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    history_text = ""
    if chat_history:
        recent = [
            f"{t['role'].capitalize()}: {t['content'][:600]}"
            for t in chat_history[-4:]
            if t.get("content")
        ]
        if recent:
            history_text = "Recent conversation:\n" + "\n".join(recent) + "\n\n"

    prompt_text = (
        f"{history_text}"
        f"Retrieved Document Excerpts:\n\n{context_str}\n\n"
        f"Question: {question}\n\n"
        "Provide a well-structured answer with inline [DocumentName, p. X] citations."
    )
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": prompt_text}]}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
        },
    }
    with httpx.Client(timeout=40.0) as client:
        resp = client.post(url, params={"key": api_key}, json=payload)
        resp.raise_for_status()
        data = resp.json()
    parts = data["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip()


def _looks_like_code_block(block: str) -> bool:
    lines = [ln for ln in block.split("\n") if ln.strip()]
    if not lines:
        return False
    code_starters = ("def ", "class ", "for ", "while ", "if ", "return ", "from ", "import ", "@")
    indented = sum(1 for ln in lines if ln.startswith("    ") or ln.startswith("  "))
    starter_hits = sum(1 for ln in lines if ln.lstrip().startswith(code_starters))
    return starter_hits >= 1 and (indented >= 1 or len(lines) >= 2)


def _looks_like_table_block(block: str) -> bool:
    lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
    pipe_lines = sum(1 for ln in lines if "|" in ln)
    return len(lines) >= 2 and pipe_lines >= 2


def synthesize_locally(question: str, retrieved: list[RetrievedChunk]) -> str:
    """Deterministic MMR-style extractive synthesizer when running without a cloud LLM API key."""
    if not retrieved:
        return (
            "No matching sections were found in the indexed documents for that query. "
            "Try rephrasing with specific computer science, Python, or data-structure terms, "
            "or upload an additional reference file in the sidebar."
        )

    best_dense = max((r.dense_score for r in retrieved), default=0.0)
    best_bm25 = max((r.bm25_score for r in retrieved), default=0.0)
    if best_dense < 0.58 and best_bm25 < 2.2:
        sources_list = sorted({r.chunk.source_name for r in retrieved})
        return (
            "I could not find relevant coverage for that topic in the currently indexed material "
            f"({', '.join(sources_list)}). Try asking about topics covered in your loaded documents—"
            "such as Python memory management, hash tables, trees, graph algorithms, sorting, "
            "or dynamic programming—or upload another document."
        )

    q_tokens = set(code_aware_tokenize(question))
    wants_code = any(
        kw in question.lower()
        for kw in ("code", "implement", "example", "write", "function", "python", "algorithm", "how")
    )

    prose_candidates: list[tuple[float, set[str], str, TextChunkRef]] = []
    code_candidates: list[tuple[float, str, TextChunkRef]] = []
    table_candidates: list[tuple[float, str, TextChunkRef]] = []

    for rank_idx, item in enumerate(retrieved):
        ch = item.chunk
        rank_Decay = 1.0 / (1.0 + 0.28 * rank_idx)
        ref = TextChunkRef(ch.source_name, ch.page_number, ch.section_title)

        raw_segments = [seg.strip() for seg in ch.text.split("\n\n") if seg.strip()]
        for seg in raw_segments:
            # Skip bare heading echoes or mid-sentence overlap fragments
            if len(seg) < 35 and "\n" not in seg and not seg.endswith("."):
                continue
            if not _looks_like_code_block(seg) and not _looks_like_table_block(seg) and seg[0].islower():
                continue

            seg_tokens = set(code_aware_tokenize(seg))
            overlap = len(q_tokens & seg_tokens) / max(len(q_tokens), 1)
            salience = rank_Decay * (0.55 + 0.45 * overlap)

            if _looks_like_code_block(seg):
                # Only attach code blocks that belong to the top-ranked section or match query terms
                if overlap > 0 or (rank_idx == 0 and ch.section_title == retrieved[0].chunk.section_title):
                    code_candidates.append((salience + (0.15 if wants_code else 0.0), seg, ref))
            elif _looks_like_table_block(seg):
                if overlap > 0 or rank_idx == 0:
                    table_candidates.append((salience, seg, ref))
            else:
                prose_candidates.append((salience, seg_tokens, seg, ref))

    # Select up to 3 non-redundant prose paragraphs via Maximal Marginal Relevance (MMR)
    selected_prose: list[tuple[str, TextChunkRef]] = []
    selected_token_sets: list[set[str]] = []
    prose_candidates.sort(key=lambda x: x[0], reverse=True)

    for base_score, tok_set, text, ref in prose_candidates:
        if len(selected_prose) >= 3:
            break
        if selected_token_sets:
            max_sim = max(
                len(tok_set & prev) / max(len(tok_set | prev), 1)
                for prev in selected_token_sets
            )
            if max_sim > 0.68:
                continue
        selected_prose.append((text, ref))
        selected_token_sets.append(tok_set)

    top_section = retrieved[0].chunk.section_title
    parts: list[str] = [f"### {top_section}"]

    if selected_prose:
        lead_text, lead_ref = selected_prose[0]
        parts.append(f"{lead_text} `[{lead_ref.source_name}, p. {lead_ref.page_number}]`")

        if len(selected_prose) > 1:
            bullets: list[str] = []
            for extra_text, extra_ref in selected_prose[1:]:
                compact = re.sub(r"\s+", " ", extra_text).strip()
                bullets.append(
                    f"- {compact} `[{extra_ref.source_name}, p. {extra_ref.page_number}]`"
                )
            parts.append("**Key Details & Mechanics**\n" + "\n".join(bullets))

    if table_candidates:
        table_candidates.sort(key=lambda x: x[0], reverse=True)
        best_table, tbl_ref = table_candidates[0][1], table_candidates[0][2]
        md_table = _format_pipe_table(best_table)
        parts.append(
            f"**Summary Reference** `[{tbl_ref.source_name}, p. {tbl_ref.page_number}]`\n\n{md_table}"
        )

    if code_candidates:
        code_candidates.sort(key=lambda x: x[0], reverse=True)
        best_code, code_ref = code_candidates[0][1], code_candidates[0][2]
        clean_code = best_code.strip("` \n")
        if clean_code.startswith("python\n"):
            clean_code = clean_code[len("python\n") :]
        parts.append(
            f"**Reference Implementation** `[{code_ref.source_name}, p. {code_ref.page_number}]`\n"
            f"```python\n{clean_code}\n```"
        )

    # Distinct cited sources footer
    seen_refs: list[str] = []
    for r in retrieved[:3]:
        tag = f"{r.chunk.source_name} (p. {r.chunk.page_number} — {r.chunk.section_title})"
        if tag not in seen_refs:
            seen_refs.append(tag)
    parts.append("**Matched Sources:** " + " · ".join(seen_refs))

    return "\n\n".join(parts)


class TextChunkRef:
    __slots__ = ("source_name", "page_number", "section_title")

    def __init__(self, source_name: str, page_number: int, section_title: str) -> None:
        self.source_name = source_name
        self.page_number = page_number
        self.section_title = section_title


def _format_pipe_table(raw_block: str) -> str:
    rows = [ln.strip() for ln in raw_block.split("\n") if "|" in ln]
    if len(rows) < 2:
        return raw_block
    formatted: list[str] = []
    for idx, row in enumerate(rows):
        cells = [c.strip() for c in row.strip("|").split("|")]
        formatted.append("| " + " | ".join(cells) + " |")
        if idx == 0:
            formatted.append("| " + " | ".join(["---"] * len(cells)) + " |")
    return "\n".join(formatted)


def generate_answer(
    question: str,
    retrieved: list[RetrievedChunk],
    settings: RAGSettings,
    chat_history: list[dict[str, str]] | None = None,
) -> tuple[str, str, str]:
    """Generate a grounded response using the active cloud LLM or local MMR fallback.

    Returns:
        (answer_markdown, provider_used, model_used)
    """
    provider, model_name, api_key = settings.resolve_active_provider()
    if provider == "local" or not retrieved:
        return (synthesize_locally(question, retrieved), "local", "extractive-mmr-v1")

    context_str = format_context_for_llm(retrieved)
    messages = _build_messages(question, context_str, chat_history)

    try:
        if provider == "groq":
            ans = _call_openai_compatible(
                endpoint="https://api.groq.com/openai/v1/chat/completions",
                api_key=api_key,
                model=model_name,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            return (ans, "groq", model_name)

        if provider == "openai":
            ans = _call_openai_compatible(
                endpoint="https://api.openai.com/v1/chat/completions",
                api_key=api_key,
                model=model_name,
                messages=messages,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            return (ans, "openai", model_name)

        if provider == "gemini":
            ans = _call_gemini(
                api_key=api_key,
                model=model_name,
                question=question,
                context_str=context_str,
                chat_history=chat_history,
                temperature=settings.temperature,
                max_tokens=settings.max_tokens,
            )
            return (ans, "gemini", model_name)
    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code
        fallback = synthesize_locally(question, retrieved)
        notice = (
            f"> **Note:** Cloud provider `{provider}` returned HTTP {status_code} "
            "(check your API key or model quota). Showing local extractive synthesis instead.\n\n"
        )
        return (notice + fallback, "local (fallback)", "extractive-mmr-v1")
    except Exception as exc:
        fallback = synthesize_locally(question, retrieved)
        notice = (
            f"> **Note:** Could not reach `{provider}` ({exc.__class__.__name__}). "
            "Showing local extractive synthesis instead.\n\n"
        )
        return (notice + fallback, "local (fallback)", "extractive-mmr-v1")

    return (synthesize_locally(question, retrieved), "local", "extractive-mmr-v1")
