import os
import re
import tempfile
import httpx
import json
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.base_models import InputFormat
from docling_core.types.doc import ImageRefMode

# ── Config ────────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY = os.environ["OPENROUTER_API_KEY"]
SUPABASE_URL       = os.environ["SUPABASE_URL"]
SUPABASE_KEY       = os.environ["SUPABASE_KEY"]
EMBED_MODEL        = "google/gemini-embedding-2"

# ── Docling setup (initialised once at startup, models cached in container) ──
pipeline_options = PdfPipelineOptions()
pipeline_options.do_ocr = True
pipeline_options.generate_picture_images = True
pipeline_options.generate_table_images   = False
pipeline_options.images_scale            = 2.0

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

# ── App ───────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Miss MoMo Ingestion Server ready.")
    yield

app = FastAPI(title="MoMo Ingestion Server", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Core pipeline helpers ─────────────────────────────────────────────────────
def parse_pdf(path: Path) -> str:
    result = converter.convert(str(path))
    try:
        md = result.document.export_to_markdown(image_mode=ImageRefMode.EMBEDDED)
    except Exception:
        md = result.document.export_to_markdown()
    return md


def chunk_markdown(md: str, chunk_size: int = 3000, overlap: int = 200) -> list[str]:
    paragraphs = md.split("\n\n")
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 < chunk_size:
            current += para + "\n\n"
        else:
            if current.strip():
                chunks.append(current.strip())
            overlap_text = current[-overlap:] if len(current) > overlap else current
            current = overlap_text + para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def embed_text(text: str) -> list[float]:
    clean = re.sub(r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+', '[IMAGE]', text)
    resp = httpx.post(
        "https://openrouter.ai/api/v1/embeddings",
        headers={"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"},
        json={"model": EMBED_MODEL, "input": clean},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()["data"][0]["embedding"]


def save_chunks(chunks: list[str], source: str, product_handles: list[str]) -> tuple[int, list[str]]:
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    errors = []
    saved  = 0
    for i, chunk in enumerate(chunks):
        embedding = embed_text(chunk)
        payload = {
            "content":   chunk,
            "embedding": embedding,
            "metadata": {
                "source":          source,
                "chunk":           i,
                "engine":          "docling+gemini-embedding-2",
                "has_image":       "data:image" in chunk,
                "product_handles": product_handles,
            },
        }
        r = httpx.post(f"{SUPABASE_URL}/rest/v1/documents_gemini", headers=headers, json=payload, timeout=30.0)
        if r.status_code in (200, 201):
            saved += 1
        else:
            errors.append(f"chunk {i}: {r.status_code} {r.text[:120]}")
    return saved, errors


def delete_source(source: str):
    """Remove all existing chunks for a document before re-ingesting."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    httpx.delete(
        f"{SUPABASE_URL}/rest/v1/documents_gemini?metadata->>source=eq.{source}",
        headers=headers,
        timeout=30.0,
    )


# ── V3 folder ingestion ───────────────────────────────────────────────────────
V3_TABLE = "documents_gemini_v3"
V3_ENGINE = "docling+gemini-embedding-2"
V3_REFERENCE_FOLDERS = {
    "acronyms", "price list", "product history", "ref information"
}


def _supabase_headers(prefer: str | None = None) -> dict:
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _safe_relative_path(value: str) -> str:
    value = (value or "").replace("\\", "/").strip().lstrip("/")
    parts = [p for p in value.split("/") if p not in ("", ".")]
    if not parts or any(p == ".." for p in parts):
        raise HTTPException(400, f"Invalid relative path: {value!r}")
    return "/".join(parts)


def _v3_folder_from_path(relative_path: str) -> str:
    parts = relative_path.split("/")
    # Browser directory uploads normally look like RootFolder/Product/file.pdf.
    # The first directory below the selected root is the authoritative V3 scope.
    if len(parts) >= 3:
        return parts[1]
    if len(parts) == 2:
        return parts[0]
    raise HTTPException(400, f"File must be inside a folder: {relative_path}")


def _v3_scope(folder: str) -> str:
    return "reference" if folder.strip().lower() in V3_REFERENCE_FOLDERS else "product"


def delete_v3_source(source_path: str):
    httpx.delete(
        f"{SUPABASE_URL}/rest/v1/{V3_TABLE}",
        params={"metadata->>source_path": f"eq.{source_path}"},
        headers=_supabase_headers(),
        timeout=30.0,
    )


def save_v3_chunks(chunks: list[str], metadata_base: dict) -> tuple[int, list[str]]:
    errors = []
    saved = 0
    headers = _supabase_headers("return=minimal")
    for i, chunk in enumerate(chunks):
        try:
            embedding = embed_text(chunk)
            metadata = {
                **metadata_base,
                "chunk": i,
                "engine": V3_ENGINE,
                "has_image": "data:image" in chunk,
            }
            payload = {"content": chunk, "embedding": embedding, "metadata": metadata}
            r = httpx.post(
                f"{SUPABASE_URL}/rest/v1/{V3_TABLE}",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            if r.status_code in (200, 201):
                saved += 1
            else:
                errors.append(f"chunk {i}: {r.status_code} {r.text[:160]}")
        except Exception as exc:
            errors.append(f"chunk {i}: {type(exc).__name__}: {str(exc)[:160]}")
    return saved, errors


@app.post("/ingest/v3/folder")
async def ingest_v3_folder(
    files: list[UploadFile] = File(...),
    paths: str = Form("[]"),
    replace: bool = Form(True),
):
    """Ingest a browser-selected knowledge folder recursively.

    V3 uses the folder structure as knowledge identity and writes only to the
    isolated documents_gemini_v3 table. It supports PDF, Markdown, and text
    knowledge files without requiring Shopify tagging.
    """
    try:
        relative_paths = json.loads(paths)
    except json.JSONDecodeError:
        raise HTTPException(400, "paths must be a JSON array")

    if not isinstance(relative_paths, list) or len(relative_paths) != len(files):
        raise HTTPException(400, "paths must contain one relative path per uploaded file")

    results = []
    totals = {
        "files": 0,
        "pdfs": 0,
        "text_files": 0,
        "unsupported": 0,
        "chunks_saved": 0,
        "chunks_total": 0,
    }
    unsupported = []

    for upload, raw_path in zip(files, relative_paths):
        relative_path = _safe_relative_path(str(raw_path))
        folder = _v3_folder_from_path(relative_path)
        filename = Path(relative_path).name
        suffix = Path(filename).suffix.lower()
        totals["files"] += 1

        if suffix not in {".pdf", ".md", ".txt"}:
            totals["unsupported"] += 1
            unsupported.append({
                "path": relative_path,
                "reason": "unsupported file type",
            })
            continue

        source_path = relative_path
        source_scope = _v3_scope(folder)
        contents = await upload.read()

        if not contents:
            results.append({
                "source_path": source_path,
                "folder": folder,
                "status": "error",
                "error": "empty file",
            })
            continue

        try:
            if replace:
                delete_v3_source(source_path)

            # PDF: use the existing Docling pipeline.
            if suffix == ".pdf":
                totals["pdfs"] += 1

                with tempfile.NamedTemporaryFile(
                    suffix=".pdf",
                    delete=False,
                ) as tmp:
                    tmp.write(contents)
                    tmp_path = Path(tmp.name)

                try:
                    md = parse_pdf(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)

            # Markdown/text: use the content directly.
            else:
                totals["text_files"] += 1
                md = contents.decode("utf-8", errors="replace")

            chunks = chunk_markdown(md)

            metadata_base = {
                "source": filename,
                "source_path": source_path,
                "source_folder": folder,
                "knowledge_scope": source_scope,
                "product_name": None if source_scope == "reference" else folder,
                "product_handles": [],
                "content_type": "text",
                "kb_version": "v3",
            }

            saved, errors = save_v3_chunks(chunks, metadata_base)

            totals["chunks_saved"] += saved
            totals["chunks_total"] += len(chunks)

            results.append({
                "source_path": source_path,
                "folder": folder,
                "chunks_total": len(chunks),
                "chunks_saved": saved,
                "errors": errors,
                "status": "ok" if not errors else "partial",
            })

        except Exception as exc:
            results.append({
                "source_path": source_path,
                "folder": folder,
                "status": "error",
                "error": str(exc)[:300],
            })

    return {
        "version": "v3",
        "summary": totals,
        "unsupported": unsupported,
        "files": results,
    }

@app.get("/documents/v3")
def list_v3_documents():
    rows = httpx.get(
        f"{SUPABASE_URL}/rest/v1/{V3_TABLE}?select=metadata&metadata->>chunk=eq.0&metadata->>kb_version=eq.v3&order=id.asc",
        headers=_supabase_headers(),
        timeout=30.0,
    )
    rows.raise_for_status()
    docs = []
    for row in rows.json():
        m = row.get("metadata", {})
        docs.append({
            "source": m.get("source"),
            "source_path": m.get("source_path"),
            "source_folder": m.get("source_folder"),
            "knowledge_scope": m.get("knowledge_scope"),
            "product_name": m.get("product_name"),
            "product_handles": m.get("product_handles", []),
            "engine": m.get("engine"),
        })
    return docs


@app.delete("/documents/v3/{source_path:path}")
def delete_v3_document(source_path: str):
    source_path = _safe_relative_path(source_path)
    delete_v3_source(source_path)
    return {"deleted": source_path, "version": "v3"}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(
    file: UploadFile = File(...),
    product_handles: str = "",   # comma-separated, e.g. "load-sentinel,pfr-1750"
    replace: bool = True,        # delete existing chunks for this doc first
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    handles = [h.strip() for h in product_handles.split(",") if h.strip()]
    source  = file.filename

    contents = await file.read()

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(contents)
        tmp_path = Path(tmp.name)

    try:
        if replace:
            delete_source(source)

        md     = parse_pdf(tmp_path)
        chunks = chunk_markdown(md)
        saved, errors = save_chunks(chunks, source, handles)
    finally:
        tmp_path.unlink(missing_ok=True)

    return {
        "source":          source,
        "chunks_total":    len(chunks),
        "chunks_saved":    saved,
        "product_handles": handles,
        "errors":          errors,
    }


@app.get("/documents")
def list_documents():
    """Return distinct document names and their product mappings."""
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }
    r = httpx.get(
        f"{SUPABASE_URL}/rest/v1/documents_gemini?select=metadata&metadata->>chunk=eq.0",
        headers=headers,
        timeout=30.0,
    )
    r.raise_for_status()
    rows = r.json()
    docs = []
    for row in rows:
        m = row.get("metadata", {})
        docs.append({
            "source":          m.get("source"),
            "product_handles": m.get("product_handles", []),
            "engine":          m.get("engine"),
        })
    return docs


@app.delete("/documents/{source}")
def delete_document(source: str):
    """Delete all chunks for a document."""
    delete_source(source)
    return {"deleted": source}
