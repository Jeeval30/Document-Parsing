"""LLM calls: classify, design a schema for unseen types, and extract.

Three providers, chosen with LLM_PROVIDER (model: LLM_MODEL):
  * anthropic - Claude via the Anthropic SDK (structured outputs via messages.parse)
  * gemini    - Gemini via the google-genai SDK (structured outputs via response_json_schema)
  * ollama    - local Ollama server via /api/chat (structured outputs via a JSON schema `format`)
Claude and Gemini read PDFs and images natively. Ollama only accepts images, so PDFs are rasterised
to PNG pages first and the model must be vision-capable. Everything above _parse() is provider-neutral:
prompts, schemas and the returned Pydantic objects are the same across all three.
"""
import asyncio
import base64
import logging
from pathlib import Path

import anthropic
import httpx
from pydantic import BaseModel, ValidationError

from app import config, dynamic_schema
from app.doc_schemas import Classification, DocTypeSpec, GeneratedSchema

log = logging.getLogger("llm")
_client: anthropic.AsyncAnthropic | None = None
_gemini_client = None


def client() -> anthropic.AsyncAnthropic:
    global _client
    if _client is None:
        _client = anthropic.AsyncAnthropic(max_retries=4)
    return _client


def gemini_client():
    global _gemini_client
    if _gemini_client is None:
        from google import genai  # imported lazily so the Claude-only setup doesn't need it

        if not config.GEMINI_API_KEY:
            raise LLMError("Gemini API key missing. Set GEMINI_API_KEY (or GOOGLE_API_KEY).")
        _gemini_client = genai.Client(api_key=config.GEMINI_API_KEY)
    return _gemini_client


class LLMError(Exception):
    pass


class SchemaRejected(LLMError):
    """The API refused the request, e.g. because a generated schema was not acceptable."""


SYSTEM_PROMPT = """You are the document-understanding engine of an intelligent document processing platform.
Output from you is written into business records, so accuracy beats completeness:
- Copy identifiers (document numbers, reference numbers, tax IDs) exactly as printed.
- Dates as YYYY-MM-DD. Resolve ambiguous formats (03/04/2025) using the document's locale cues; if still ambiguous, pick the likelier reading and mark it as uncertain.
- Amounts as plain numbers, no currency symbols or thousands separators. Credits/refunds are negative.
- Use null for anything not present on the document. Never invent or compute a value that isn't printed, except currency which you may infer from symbols.
- Flag every value that was blurry, handwritten, cut off, or otherwise uncertain."""


def file_block(path: Path, media_type: str) -> dict:
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    block_type = "document" if media_type == "application/pdf" else "image"
    return {"type": block_type, "source": {"type": "base64", "media_type": media_type, "data": data}}


async def _parse(content: list[dict], schema: type[BaseModel], effort: str, max_tokens: int):
    """Send the document + prompt, get back a validated instance of `schema` and a usage dict.

    `content` uses Anthropic-style blocks ({"type": "document"|"image", "source": {...}} and
    {"type": "text", "text": ...}); the Gemini backend converts them."""
    if config.LLM_PROVIDER == "gemini":
        return await _parse_gemini(content, schema, max_tokens)
    if config.LLM_PROVIDER == "ollama":
        return await _parse_ollama(content, schema, max_tokens)
    return await _parse_anthropic(content, schema, effort, max_tokens)


# --------------------------------------------------------------------------- Gemini backend

_GEMINI_RETRIES = 3


def _gemini_parts(content: list[dict]) -> list:
    from google.genai import types

    parts = []
    for block in content:
        if block["type"] == "text":
            parts.append(types.Part.from_text(text=block["text"]))
        else:
            src = block["source"]
            parts.append(types.Part.from_bytes(data=base64.b64decode(src["data"]), mime_type=src["media_type"]))
    return parts


async def _parse_gemini(content: list[dict], schema: type[BaseModel], max_tokens: int):
    from google.genai import errors, types

    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=schema.model_json_schema(),
        max_output_tokens=max_tokens,
    )
    parts = _gemini_parts(content)
    for attempt in range(_GEMINI_RETRIES + 1):
        try:
            response = await gemini_client().aio.models.generate_content(
                model=config.LLM_MODEL, contents=[types.Content(role="user", parts=parts)], config=cfg)
            break
        except errors.ClientError as e:
            if e.code in (401, 403):
                raise LLMError("Gemini API key missing or invalid. Set GEMINI_API_KEY.") from e
            if e.code == 404:
                raise LLMError(f"Gemini model '{config.LLM_MODEL}' not found. Check LLM_MODEL.") from e
            if e.code == 429 and attempt < _GEMINI_RETRIES:
                await asyncio.sleep(2 ** attempt * 2)
                continue
            if e.code == 429:
                raise LLMError("Rate limited by the Gemini API after retries. Try reprocessing shortly.") from e
            raise SchemaRejected(f"Gemini rejected the request: {e.message}") from e
        except errors.ServerError as e:
            if attempt < _GEMINI_RETRIES:
                await asyncio.sleep(2 ** attempt * 2)
                continue
            raise LLMError(f"Gemini API error {e.code}: {e.message}") from e
        except errors.APIError as e:
            raise LLMError(f"Gemini API error {e.code}: {e.message}") from e
        except (httpx.HTTPError, OSError, TimeoutError) as e:
            raise LLMError("Could not reach the Gemini API (network error).") from e

    feedback = getattr(response, "prompt_feedback", None)
    if feedback is not None and getattr(feedback, "block_reason", None):
        raise LLMError(f"Gemini blocked this document ({feedback.block_reason}).")
    candidate = (response.candidates or [None])[0]
    finish = getattr(getattr(candidate, "finish_reason", None), "name", None)
    if finish == "MAX_TOKENS":
        raise LLMError("Output was cut off (document too long for one pass).")
    if finish in {"SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "SPII", "RECITATION"}:
        raise LLMError(f"Gemini declined to process this document ({finish}).")
    text = response.text
    if not text:
        raise LLMError("Model returned no structured output.")
    try:
        parsed = schema.model_validate_json(text)
    except ValidationError as e:
        log.warning("Gemini output did not match schema: %s", e)
        raise LLMError("Model output did not match the expected schema.") from e

    meta = response.usage_metadata
    usage = {
        "model": response.model_version or config.LLM_MODEL,
        "input_tokens": getattr(meta, "prompt_token_count", None),
        "output_tokens": getattr(meta, "candidates_token_count", None),
    }
    return parsed, usage


# --------------------------------------------------------------------------- Ollama backend
# Local Ollama server (http://localhost:11434 by default). Uses the native /api/chat endpoint
# with JSON-schema structured outputs. Ollama only accepts images, so PDFs are rasterised to PNG
# pages first — the chosen model must be vision-capable. When a PDF carries a real text layer we
# also pass that text: small local models extract far more reliably from exact characters than by
# OCR-ing their own render, and the image still supplies layout for tables and totals.

_OLLAMA_RETRIES = 2
_OLLAMA_PDF_ZOOM = 3.0  # render PDF pages at 3x so small table text stays legible for the vision model
_OLLAMA_MIN_TEXT_CHARS = 20  # below this a PDF is treated as scanned (no usable text layer)


def _pdf_render(pdf_bytes: bytes) -> tuple[list[str], str]:
    """Return (base64 PNG per page, concatenated text-layer text) for a PDF."""
    try:
        import pymupdf  # PyMuPDF (the older `import fitz` name is deprecated)
    except ImportError as e:
        raise LLMError(
            "Ollama needs PDFs converted to images first. Install PyMuPDF: pip install pymupdf."
        ) from e
    images, texts = [], []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        matrix = pymupdf.Matrix(_OLLAMA_PDF_ZOOM, _OLLAMA_PDF_ZOOM)
        for page in doc:
            images.append(base64.b64encode(page.get_pixmap(matrix=matrix).tobytes("png")).decode("ascii"))
            texts.append(page.get_text())
    return images, "\n".join(texts).strip()


def _pdf_to_images_b64(pdf_bytes: bytes) -> list[str]:  # kept for callers/tests that only need images
    return _pdf_render(pdf_bytes)[0]


def _ollama_user_message(content: list[dict]) -> dict:
    texts, images, pdf_text = [], [], ""
    for block in content:
        if block["type"] == "text":
            texts.append(block["text"])
        else:
            src = block["source"]
            if src["media_type"] == "application/pdf":
                imgs, extracted = _pdf_render(base64.b64decode(src["data"]))
                images.extend(imgs)
                if len(extracted) >= _OLLAMA_MIN_TEXT_CHARS:
                    pdf_text = extracted
            else:
                images.append(src["data"])  # already base64, no data: prefix
    if pdf_text:
        texts.append(
            "Text extracted from the document's text layer (order may be imperfect; use the image "
            "for layout and to resolve which number belongs to which field):\n"
            f"-----\n{pdf_text}\n-----"
        )
    msg = {"role": "user", "content": "\n\n".join(texts) or "Extract the requested information."}
    if images:
        msg["images"] = images
    return msg


async def _parse_ollama(content: list[dict], schema: type[BaseModel], max_tokens: int):
    payload = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, _ollama_user_message(content)],
        "format": schema.model_json_schema(),  # structured output (Ollama >= 0.5)
        "stream": False,
        "options": {"temperature": 0, "num_predict": max_tokens, "num_ctx": config.OLLAMA_NUM_CTX},
    }
    url = f"{config.OLLAMA_HOST}/api/chat"
    response = None
    for attempt in range(_OLLAMA_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=config.OLLAMA_TIMEOUT) as http:
                response = await http.post(url, json=payload)
            if response.status_code == 404:
                raise LLMError(
                    f"Ollama model '{config.LLM_MODEL}' not found. Pull it first: "
                    f"ollama pull {config.LLM_MODEL}"
                )
            response.raise_for_status()
            break
        except httpx.HTTPStatusError as e:
            raise LLMError(f"Ollama API error {e.response.status_code}: {e.response.text[:200]}") from e
        except (httpx.HTTPError, OSError, TimeoutError) as e:
            if attempt < _OLLAMA_RETRIES:
                await asyncio.sleep(2 ** attempt)
                continue
            raise LLMError(
                f"Could not reach Ollama at {config.OLLAMA_HOST}. Is `ollama serve` running?"
            ) from e

    data = response.json()
    if data.get("done_reason") == "length":
        raise LLMError("Output was cut off (document too long for one pass).")
    text = (data.get("message") or {}).get("content")
    if not text:
        raise LLMError("Model returned no structured output.")
    try:
        parsed = schema.model_validate_json(text)
    except ValidationError as e:
        log.warning("Ollama output did not match schema: %s", e)
        raise LLMError("Model output did not match the expected schema.") from e

    usage = {
        "model": data.get("model") or config.LLM_MODEL,
        "input_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
    }
    return parsed, usage


# --------------------------------------------------------------------------- Claude backend


async def _parse_anthropic(content: list[dict], schema: type[BaseModel], effort: str, max_tokens: int):
    kwargs = {}
    if config.LLM_FALLBACKS:
        kwargs["extra_headers"] = {"anthropic-beta": "server-side-fallback-2026-07-01"}
        kwargs["extra_body"] = {"fallbacks": "default"}
    try:
        response = await client().messages.parse(
            model=config.LLM_MODEL,
            max_tokens=max_tokens,
            system=SYSTEM_PROMPT,
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            output_format=schema,
            messages=[{"role": "user", "content": content}],
            **kwargs,
        )
    except anthropic.AuthenticationError as e:
        raise LLMError("Anthropic API key missing or invalid. Set ANTHROPIC_API_KEY.") from e
    except anthropic.BadRequestError as e:
        raise SchemaRejected(f"Claude rejected the request: {e.message}") from e
    except anthropic.RateLimitError as e:
        raise LLMError("Rate limited by the Claude API after retries. Try reprocessing shortly.") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"Claude API error {e.status_code}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise LLMError("Could not reach the Claude API (network error).") from e

    if response.stop_reason == "refusal":
        category = response.stop_details.category if response.stop_details else None
        raise LLMError(f"Model declined to process this document (category: {category}).")
    if response.stop_reason == "max_tokens":
        raise LLMError("Output was cut off (document too long for one pass).")
    if response.parsed_output is None:
        raise LLMError("Model returned no structured output.")

    usage = {
        "model": response.model,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return response.parsed_output, usage


# --------------------------------------------------------------------------- classify


def _catalog_text(catalog) -> str:
    built_in = [t for t in catalog if t.kind == "built_in"]
    discovered = [t for t in catalog if t.kind == "discovered"]
    lines = ["Built-in types (use these exact ids when the document fits):"]
    lines += [f"- {t.slug}: {t.purpose}" for t in built_in]
    if discovered:
        lines.append("Previously discovered types (reuse the id if the document is the same kind):")
        lines += [f"- {t.slug}: {t.display_name} - {t.purpose or ''}" for t in discovered]
    return "\n".join(lines)


async def classify(block: dict, catalog) -> tuple[Classification, dict]:
    prompt = (
        "Identify what kind of business document this is.\n\n"
        f"{_catalog_text(catalog)}\n\n"
        "If none of the listed types fits, name a new type: a short, general snake_case id for the kind of "
        "document (e.g. bill_of_lading, lease_agreement, bank_statement, lab_report), not for this specific "
        "instance or company. Never answer 'unknown'. Also give a readable name, the document's purpose in "
        "one sentence, and your confidence from 0 to 1."
    )
    return await _parse([block, {"type": "text", "text": prompt}], Classification, config.CLASSIFY_EFFORT, 2000)


# --------------------------------------------------------------------------- schema design (unseen types)


async def generate_schema(block: dict, classification: Classification) -> tuple[list[dict], dict]:
    prompt = (
        f"This document is a '{classification.display_name}' ({classification.purpose}). No extraction schema "
        "exists for this document type yet. Design one.\n\n"
        "- Capture the business-relevant information someone processing this kind of document would need "
        "(identifiers, parties, dates, amounts, terms, statuses, references), not layout or boilerplate.\n"
        "- Design for the document TYPE, so it also fits other documents of this kind, not just this one.\n"
        "- Use snake_case keys, clear labels and a one-line description per field.\n"
        "- Use type 'table' with columns for repeating rows (line items, schedules, transactions, results).\n"
        "- Mark required=true only for fields every document of this type must contain.\n"
        "- Assign roles where they apply (document_number, document_date, due_date, period_start, period_end, "
        "party = issuer, counterparty = recipient, currency, subtotal, tax, total; table columns: description, "
        "quantity, unit_price, line_amount). Use 'none' otherwise. Each role at most once.\n"
        f"- At most {dynamic_schema.MAX_FIELDS} fields."
    )
    result, usage = await _parse([block, {"type": "text", "text": prompt}], GeneratedSchema, "medium", 8000)
    return dynamic_schema.sanitize_schema([f.model_dump() for f in result.fields]), usage


# --------------------------------------------------------------------------- extraction


async def extract(block: dict, spec: DocTypeSpec) -> tuple[BaseModel, dict]:
    """Built-in types: specialised schema."""
    prompt = (
        f"This document is a {spec.label.lower()}. Extract its fields and every line item. "
        "Include tax, shipping and discount lines only in the totals fields, not as line items, "
        "unless they are printed as regular item rows. List uncertain fields in low_confidence_fields."
    )
    return await _parse([block, {"type": "text", "text": prompt}], spec.model, config.EXTRACT_EFFORT, 16000)


async def extract_dynamic(block: dict, schema: list[dict], display_name: str) -> tuple[dict, dict]:
    """Discovered types: extract against a runtime schema. Returns ({key: {value|rows, confidence}}, usage)."""
    prompt = (
        f"This document is a '{display_name}'. Extract every field of the schema, including all table rows, "
        "and give a confidence from 0 to 1 for each field."
    )
    try:
        model = dynamic_schema.extraction_model(schema)
        result, usage = await _parse([block, {"type": "text", "text": prompt}], model, config.EXTRACT_EFFORT, 16000)
        return result.model_dump(), usage
    except SchemaRejected:
        # Fall back to a fixed generic format and convert types ourselves.
        fallback = (
            f"{prompt}\n\nReturn fields and tables using these keys:\n{dynamic_schema.describe_for_envelope(schema)}\n"
            "Write every value as text exactly as it should be recorded (dates YYYY-MM-DD, plain numbers)."
        )
        result, usage = await _parse([block, {"type": "text", "text": fallback}], dynamic_schema.Envelope,
                                     config.EXTRACT_EFFORT, 16000)
        usage["fallback_format"] = True
        return dynamic_schema.envelope_to_raw(schema, result.model_dump()), usage
