"""
modal_worker.py — NEURA AI Serverless Document Ingestion Worker
================================================================
Deployed on Modal.com (https://modal.com).

Each document upload fires a POST to this web endpoint from main.py
(awaiting up to 120s from BackgroundTask). Modal spins up ONE isolated
container per document — no queues, no waiting — every student's upload
runs simultaneously.

Deploy (once):
    pip install modal
    modal setup          # authenticate with your Modal account
    modal deploy modal_worker.py

Secrets — create a Secret Group named "neura-secrets" at modal.com/secrets:
    WHATSAPP_TOKEN      — Meta Cloud API permanent token
    PHONE_NUMBER_ID     — Meta WhatsApp phone number ID (e.g. 1150180661520951)
    QDRANT_URL          — Qdrant Cloud cluster URL
    QDRANT_API_KEY      — Qdrant Cloud API key
    MONGO_URI           — MongoDB Atlas connection string
    OPENROUTER_API_KEY  — For AI Medical Gatekeeper
    GROQ_API_KEY        — Gatekeeper fallback
"""

import os
import gc
import io
import re
import json
import uuid
import hashlib
import asyncio
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import modal

# ---------------------------------------------------------------------------
# GATEKEEPER MODEL CONFIGURATION
# ---------------------------------------------------------------------------
GATEKEEPER_OPENROUTER_MODELS = ["google/gemini-2.5-flash-lite", "openai/gpt-4o-mini"]
GATEKEEPER_GROQ_MODELS = ["groq/compound-mini", "llama-3.3-70b-versatile", "qwen/qwen3.6-27b"]
CANDIDATE_MODELS = GATEKEEPER_OPENROUTER_MODELS + GATEKEEPER_GROQ_MODELS

# ── Character-based text chunking with delimiter boundary detection ────────
def chunk_text_with_overlap(text: str, chunk_size: int = 850, overlap: int = 150) -> list:
    """Splits text into overlapping chunks (character-based), respecting sentence/paragraph boundaries."""
    if not text or not text.strip():
        return []
    clean_text = re.sub(r'[ \t]+', ' ', text).strip()
    if len(clean_text) <= chunk_size:
        return [clean_text]
    
    chunks = []
    start = 0
    text_len = len(clean_text)
    
    while start < text_len:
        end = start + chunk_size
        if end >= text_len:
            chunks.append(clean_text[start:].strip())
            break
        
        split_idx = -1
        sub = clean_text[start:end]
        for delim in ["\n\n", "\n", ". ", "; ", ", "]:
            last_pos = sub.rfind(delim, chunk_size - overlap)
            if last_pos != -1:
                split_idx = start + last_pos + len(delim)
                break
        
        if split_idx == -1:
            split_idx = end
            
        chunk_str = clean_text[start:split_idx].strip()
        if len(chunk_str) > 30:
            chunks.append(chunk_str)
            
        start = max(split_idx - overlap, start + 1)
        
    return chunks

# ── Multi-format text extractors matching main.py ─────────────────────────
def extract_pdf(raw: bytes, fname: str):
    if not raw or len(raw) < 100:
        return False, "EMPTY_FILE", [], {"error": "Uploaded file is empty or corrupted."}
    
    # Method 1: PyMuPDF (fitz)
    try:
        import fitz
        doc = fitz.open(stream=raw, filetype="pdf")
        if doc.is_encrypted:
            doc.close()
            return False, "ENCRYPTED", [], {"error": "PDF is password protected."}
        page_count = len(doc)
        if page_count == 0:
            doc.close()
            return False, "EMPTY_PAGES", [], {"error": "PDF contains 0 pages."}
        if page_count > 200:
            doc.close()
            return False, "TOO_MANY_PAGES", [], {"error": f"Document has {page_count} pages (limit is 200).", "page_count": page_count}

        pages_data = []
        total_words = 0
        empty_pages = 0
        for idx in range(page_count):
            txt = doc[idx].get_text("text").strip()
            words = len(txt.split()) if txt else 0
            total_words += words
            if words >= 6:
                pages_data.append((idx + 1, txt))
            else:
                empty_pages += 1
        doc.close()

        empty_ratio = empty_pages / page_count
        avg_words = total_words / max(page_count, 1)

        # Hardened 35% empty-page ratio check matching main.py
        if page_count == 1 and total_words < 15:
            return False, "SCANNED_IMAGE", [], {
                "error": "Single-page document contains insufficient readable text.",
                "total_words": total_words, "page_count": 1, "empty_pages": 1, "empty_pct": 100.0
            }
        if page_count >= 2 and (empty_ratio > 0.35 or avg_words < 12.0):
            return False, "SCANNED_IMAGE", [], {
                "error": f"Scanned or image-only document ({empty_pages}/{page_count} pages have no extractable text).",
                "total_words": total_words, "page_count": page_count, "empty_pages": empty_pages, "empty_pct": round(empty_ratio * 100, 1)
            }

        return True, "OK", pages_data, {"page_count": page_count, "total_words": total_words, "empty_pages": empty_pages, "empty_pct": round(empty_ratio * 100, 1), "doc_type": "pdf"}
    except Exception as fitz_err:
        print(f"⚠️ PyMuPDF extraction failed for {fname}, attempting pypdf fallback: {fitz_err}")

    # Method 2: pypdf fallback
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            return False, "ENCRYPTED", [], {"error": "PDF is password protected."}
        page_count = len(reader.pages)
        if page_count == 0:
            return False, "EMPTY_PAGES", [], {"error": "PDF contains 0 pages."}
        if page_count > 200:
            return False, "TOO_MANY_PAGES", [], {"error": f"Document has {page_count} pages (limit is 200).", "page_count": page_count}

        pages_data = []
        total_words = 0
        empty_pages = 0
        for idx, p in enumerate(reader.pages):
            txt = (p.extract_text() or "").strip()
            words = len(txt.split()) if txt else 0
            total_words += words
            if words >= 6:
                pages_data.append((idx + 1, txt))
            else:
                empty_pages += 1

        empty_ratio = empty_pages / page_count
        avg_words = total_words / max(page_count, 1)

        if page_count == 1 and total_words < 15:
            return False, "SCANNED_IMAGE", [], {
                "error": "Single-page document contains insufficient readable text.",
                "total_words": total_words, "page_count": 1, "empty_pages": 1, "empty_pct": 100.0
            }
        if page_count >= 2 and (empty_ratio > 0.35 or avg_words < 12.0):
            return False, "SCANNED_IMAGE", [], {
                "error": f"Scanned or image-only document ({empty_pages}/{page_count} pages have no extractable text).",
                "total_words": total_words, "page_count": page_count, "empty_pages": empty_pages, "empty_pct": round(empty_ratio * 100, 1)
            }

        return True, "OK", pages_data, {"page_count": page_count, "total_words": total_words, "empty_pages": empty_pages, "empty_pct": round(empty_ratio * 100, 1), "doc_type": "pdf"}
    except Exception as pypdf_err:
        print(f"❌ pypdf fallback failed for {fname}: {pypdf_err}")

    return False, "CORRUPTED", [], {"error": "Unable to extract text from this PDF."}

def extract_pptx(raw: bytes, fname: str):
    import zipfile, xml.etree.ElementTree as ET
    pages_data = []
    total_words = 0
    empty_pages = 0

    # Method 1: python-pptx (with speaker notes & tables)
    try:
        import pptx
        prs = pptx.Presentation(io.BytesIO(raw))
        slide_count = len(prs.slides)
        if slide_count == 0:
            return False, "EMPTY_DOCUMENT", [], {"error": "Presentation contains 0 slides."}
        if slide_count > 200:
            return False, "TOO_MANY_PAGES", [], {"error": f"Presentation has {slide_count} slides (limit is 200).", "page_count": slide_count}

        for s_idx, slide in enumerate(prs.slides, 1):
            slide_texts = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for paragraph in shape.text_frame.paragraphs:
                        p_text = "".join(run.text for run in paragraph.runs if run.text).strip()
                        if p_text:
                            slide_texts.append(p_text)
                elif shape.has_table:
                    for row in shape.table.rows:
                        row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if row_cells:
                            slide_texts.append(" | ".join(row_cells))

            if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
                notes_txt = slide.notes_slide.notes_text_frame.text.strip()
                if notes_txt:
                    slide_texts.append(f"[Speaker Notes: {notes_txt}]")

            combined_text = "\n".join(slide_texts).strip()
            word_count = len(combined_text.split())
            total_words += word_count
            if word_count < 6:
                empty_pages += 1
            pages_data.append((s_idx, combined_text))
    except Exception as pptx_err:
        print(f"⚠️ python-pptx parser error on {fname} ({pptx_err}), attempting XML fallback...")
        pages_data = []

    # Method 2: Pure-Python zipfile + XML fallback
    if not pages_data:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                slide_files = [f for f in zf.namelist() if f.startswith("ppt/slides/slide") and f.endswith(".xml")]
                if not slide_files:
                    return False, "EMPTY_DOCUMENT", [], {"error": "No slide XMLs found in PPTX archive."}
                slide_files.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()) if re.search(r'\d+', os.path.basename(x)) else 0)
                slide_count = len(slide_files)
                if slide_count > 200:
                    return False, "TOO_MANY_PAGES", [], {"error": f"Presentation has {slide_count} slides (limit is 200).", "page_count": slide_count}

                total_words = 0
                empty_pages = 0
                for s_idx, sfile in enumerate(slide_files, 1):
                    root = ET.fromstring(zf.read(sfile))
                    texts = [elem.text for elem in root.iter() if elem.text and elem.tag.endswith('}t')]
                    combined_text = " ".join(" ".join(texts).split())
                    word_count = len(combined_text.split())
                    total_words += word_count
                    if word_count < 6:
                        empty_pages += 1
                    pages_data.append((s_idx, combined_text))
        except Exception as e:
            return False, "CORRUPTED", [], {"error": str(e)}

    slide_count = len(pages_data)
    empty_ratio = (empty_pages / slide_count) if slide_count > 0 else 1.0
    avg_words = (total_words / slide_count) if slide_count > 0 else 0.0

    stats = {
        "page_count": slide_count, "total_words": total_words, "empty_pages": empty_pages,
        "empty_pct": round(empty_ratio * 100, 1), "doc_type": "presentation"
    }
    if total_words == 0 or empty_ratio > 0.45 or (slide_count > 3 and avg_words < 8.0):
        stats["error"] = f"Image-only slides ({empty_pages}/{slide_count} slides have no selectable text)."
        return False, "SCANNED_IMAGE", [], stats

    return True, "OK", pages_data, stats

def extract_docx(raw: bytes, fname: str):
    import zipfile, xml.etree.ElementTree as ET
    pages_data = []
    total_words = 0

    # Method 1: python-docx
    try:
        import docx
        doc = docx.Document(io.BytesIO(raw))
        current_page = 1
        current_page_text = []
        current_word_count = 0

        units = []
        for p in doc.paragraphs:
            txt = p.text.strip()
            if txt:
                is_heading = p.style.name.startswith("Heading") if p.style else False
                units.append((txt, is_heading, "paragraph"))

        for table in doc.tables:
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_cells:
                    units.append((" | ".join(row_cells), False, "table"))

        if not units:
            return False, "EMPTY_DOCUMENT", [], {"error": "Word document has no readable text."}

        for txt, is_heading, u_type in units:
            w_count = len(txt.split())
            if (current_word_count > 450) or (is_heading and current_word_count > 150):
                if current_page_text:
                    pages_data.append((current_page, "\n\n".join(current_page_text).strip()))
                    total_words += current_word_count
                    current_page += 1
                    current_page_text = []
                    current_word_count = 0
            current_page_text.append(txt)
            current_word_count += w_count

        if current_page_text:
            pages_data.append((current_page, "\n\n".join(current_page_text).strip()))
            total_words += current_word_count
    except Exception as docx_err:
        print(f"⚠️ python-docx parser error on {fname} ({docx_err}), attempting XML fallback...")
        pages_data = []

    # Method 2: Pure-Python zipfile + XML fallback
    if not pages_data:
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                if "word/document.xml" not in zf.namelist():
                    return False, "EMPTY_DOCUMENT", [], {"error": "word/document.xml missing in docx archive."}
                root = ET.fromstring(zf.read("word/document.xml"))
                paragraphs = []
                for pe in root.iter():
                    if pe.tag.endswith("}p"):
                        ts = [e.text for e in pe.iter() if e.text and e.tag.endswith("}t")]
                        if ts:
                            p_text = "".join(ts).strip()
                            if p_text:
                                paragraphs.append(p_text)
                if not paragraphs:
                    return False, "EMPTY_DOCUMENT", [], {"error": "No readable text found in Word document."}

                current_page = 1
                current_page_text = []
                current_word_count = 0
                total_words = 0
                for p_text in paragraphs:
                    w_count = len(p_text.split())
                    if current_word_count > 450:
                        pages_data.append((current_page, "\n\n".join(current_page_text).strip()))
                        total_words += current_word_count
                        current_page += 1
                        current_page_text = []
                        current_word_count = 0
                    current_page_text.append(p_text)
                    current_word_count += w_count

                if current_page_text:
                    pages_data.append((current_page, "\n\n".join(current_page_text).strip()))
                    total_words += current_word_count
        except Exception as e:
            return False, "CORRUPTED", [], {"error": str(e)}

    page_count = len(pages_data)
    stats = {
        "page_count": page_count, "total_words": total_words, "empty_pages": 0,
        "empty_pct": 0.0, "doc_type": "word"
    }
    if page_count > 200:
        return False, "TOO_MANY_PAGES", [], {"error": f"Word document exceeds 200 virtual pages ({page_count} pages).", "page_count": page_count}
    if total_words < 15:
        return False, "SCANNED_IMAGE", [], {"error": "Document contains almost no readable text."}

    return True, "OK", pages_data, stats

def extract_document(raw: bytes, fname: str, mime: str):
    fn = fname.lower()
    if fn.endswith((".doc", ".ppt")) or mime in ("application/msword", "application/vnd.ms-powerpoint"):
        if fn.endswith(".ppt") or mime == "application/vnd.ms-powerpoint":
            r = extract_pptx(raw, fname)
            if r[0]: return r
        if fn.endswith(".doc") or mime == "application/msword":
            r = extract_docx(raw, fname)
            if r[0]: return r
        return False, "LEGACY_BINARY", [], {"error": "Legacy 97-2003 binary format. Please open in Word/PowerPoint and save as .docx or .pptx (or export to PDF)!"}
    if fn.endswith(".pptx") or "presentation" in mime:
        return extract_pptx(raw, fname)
    if fn.endswith(".docx") or "wordprocessing" in mime or "officedocument.word" in mime:
        return extract_docx(raw, fname)
    return extract_pdf(raw, fname)

# ---------------------------------------------------------------------------
# APP DEFINITION
# ---------------------------------------------------------------------------
app = modal.App("neura-ai-ingestion")

# ---------------------------------------------------------------------------
# CONTAINER IMAGE
# Built once and cached. Subsequent deploys reuse the cached image.
# ---------------------------------------------------------------------------
ingestion_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi[standard]",       # Required by @modal.fastapi_endpoint
        "pymupdf",                 # Fast PDF text extraction (fitz)
        "pypdf",                   # Lightweight PDF fallback
        "python-docx",             # .docx Word document parsing
        "python-pptx",             # .pptx PowerPoint parsing
        "fastembed==0.3.6",        # BAAI/bge-small-en-v1.5 ONNX embeddings (CPU)
        "qdrant-client",           # Vector database client
        "motor",                   # Async MongoDB driver
        "httpx",                   # Async HTTP (Meta CDN + WhatsApp API + LLMs)
        "python-dotenv",           # Optional: load .env in local testing
    )
)

neura_secrets = modal.Secret.from_name("neura-secrets")

# ---------------------------------------------------------------------------
# MAIN INGESTION FUNCTION
# ---------------------------------------------------------------------------
@app.function(
    image=ingestion_image,
    secrets=[neura_secrets],
    cpu=2.0,       # 2 vCPUs — FastEmbed ONNX is CPU-bound
    memory=2048,   # 2 GB RAM — PyMuPDF + ONNX model + Qdrant client
    timeout=300,   # 5 min hard limit (large textbooks ~2 min)
    max_containers=50,  # Safety cap: at most 50 parallel containers (DB protection)
)
@modal.fastapi_endpoint(method="POST", label="neura-ingest")
async def ingest(request: dict) -> dict:
    """
    Called by main.py (Render) when a WhatsApp document arrives.

    Expected JSON body:
        {
            "sender_phone": "2348012345678",
            "media_id":     "1234567890",
            "filename":     "Pathology_Lecture_3.pdf",
            "caption":      "summarise slide 5",   <- optional
            "mime_type":    "application/pdf"
        }
    """
    import os, gc, io, re, json, uuid, hashlib, asyncio, traceback
    from concurrent.futures import ThreadPoolExecutor
    from datetime import datetime, timedelta, timezone

    # ── Read inputs ───────────────────────────────────────────────────────────
    sender_phone = request.get("sender_phone", "")
    media_id     = request.get("media_id", "")
    filename     = request.get("filename", "document.pdf")
    mime_type    = request.get("mime_type", "application/pdf")
    dispatch_id  = request.get("dispatch_id", "")

    if not sender_phone or not media_id:
        return {"ok": False, "error": "Missing sender_phone or media_id"}

    print(f"\n[MODAL] START user={sender_phone} file={filename!r} media_id={media_id} dispatch_id={dispatch_id}")

    # ── Environment variables (injected by Modal Secret "neura-secrets") ──────
    WHATSAPP_TOKEN  = os.environ["WHATSAPP_TOKEN"]
    PHONE_NUMBER_ID = os.environ["PHONE_NUMBER_ID"]
    QDRANT_URL      = os.environ["QDRANT_URL"]
    QDRANT_API_KEY  = os.environ["QDRANT_API_KEY"]
    MONGO_URI       = os.environ.get("MONGO_URI", "")
    OPENROUTER_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
    GROQ_KEY        = os.environ.get("GROQ_API_KEY", "")
    COLLECTION      = "neura_medical_knowledge"

    # ── Lazy imports ──────────────────────────────────────────────────────────
    import httpx
    from qdrant_client import AsyncQdrantClient
    from qdrant_client import models as qm
    from motor.motor_asyncio import AsyncIOMotorClient
    from fastembed import TextEmbedding

    # ── Initialise clients (one per container, warm for this call) ────────────
    qdrant_cl = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    mongo_cl  = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
    users_col = mongo_cl["neura_db"]["users"] if mongo_cl is not None else None
    embedder  = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=2)
    emb_pool  = ThreadPoolExecutor(max_workers=2)
    loop      = asyncio.get_event_loop()

    # ── WhatsApp: send plain text message ────────────────────────────────────
    async def wa_send(to: str, text: str):
        url  = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
        hdrs = {"Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}", "Content-Type": "application/json"}
        body = {"messaging_product": "whatsapp", "recipient_type": "individual", "to": to,
                "type": "text", "text": {"preview_url": False, "body": text}}
        try:
            async with httpx.AsyncClient(timeout=20.0) as c:
                r = await c.post(url, headers=hdrs, json=body)
                print(f"[WA text] {r.status_code}")
        except Exception as e:
            print(f"[WA text error] {e}")

    # ── WhatsApp: send interactive button message (max 3 buttons) ────────────
    async def wa_buttons(to: str, body_text: str, buttons: list):
        url  = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
        hdrs = {"Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}", "Content-Type": "application/json"}
        btns = [
            {"type": "reply", "reply": {"id": str(b.get("id", ""))[:256], "title": str(b.get("title", ""))[:20].strip()}}
            for b in buttons[:3]
        ]
        body = {
            "messaging_product": "whatsapp", "recipient_type": "individual", "to": to,
            "type": "interactive",
            "interactive": {"type": "button", "body": {"text": body_text}, "action": {"buttons": btns}}
        }
        try:
            async with httpx.AsyncClient(timeout=20.0) as c:
                r = await c.post(url, headers=hdrs, json=body)
                print(f"[WA buttons] {r.status_code}")
                if r.status_code != 200:
                    await wa_send(to, body_text)
        except Exception as e:
            print(f"[WA buttons error] {e}")
            await wa_send(to, body_text)

    # ── Download file from Meta CDN ───────────────────────────────────────────
    async def download_media(mid: str):
        hdrs = {"Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}"}
        try:
            async with httpx.AsyncClient(timeout=40.0) as c:
                r = await c.get(f"https://graph.facebook.com/v19.0/{mid}", headers=hdrs)
                if r.status_code != 200:
                    print(f"[DL] Meta lookup failed {r.status_code}")
                    return b"", ""
                info = r.json()
                dl   = info.get("url", "")
                mime = info.get("mime_type", "application/pdf")
                if not dl:
                    return b"", ""
                r2 = await c.get(dl, headers=hdrs)
                if r2.status_code == 200:
                    print(f"[DL] {len(r2.content):,} bytes ok")
                    return r2.content, mime
                print(f"[DL] CDN failed {r2.status_code}")
                return b"", ""
        except Exception as e:
            print(f"[DL error] {e}")
            return b"", ""

    # ── AI Medical Gatekeeper ─────────────────────────────────────────────────
    async def is_medical_doc(sample: str, fname: str) -> dict:
        dtitle = os.path.splitext(fname)[0].replace("_", " ").replace("-", " ").title()
        if len(dtitle) > 50:
            dtitle = dtitle[:47] + "..."
        if len(sample.strip()) < 30:
            return {"is_medical": False, "category": "Unclassified", "title": dtitle, "rejection_reason": "Insufficient text."}

        prompt = (
            "You are a strict academic gatekeeper for NEURA AI, an MBBS medical school study assistant.\n"
            f"File Name: {fname}\n"
            "Document Sample Text:\n"
            f"\"\"\"{sample[:2000]}\"\"\"\n\n"
            "Evaluate whether this document is genuine MEDICAL, CLINICAL, BIOMEDICAL, PHARMACEUTICAL, ANATOMICAL, or HEALTH SCIENCES study material (e.g. medical textbooks, lecture slides, clinical guidelines, hospital case notes, pathology slides, pharmacology handouts, journal articles, exam prep questions).\n\n"
            "REJECT non-medical files, including:\n"
            "- Receipts, invoices, bank statements, transaction slips, payment receipts\n"
            "- CVs, resumes, job applications, cover letters\n"
            "- Administrative documents (admission letters, fee slips, course registration forms, general timetables)\n"
            "- Non-medical coursework (law, computer science, business, pure literature, non-medical math)\n"
            "- Personal letters, stories, jokes, memes, spam\n\n"
            "Output ONLY a valid JSON object in this exact schema (no markdown, no code blocks):\n"
            "{\n"
            '  "is_medical": true,\n'
            '  "category": "Cardiology / Renal Physiology / Pathology / Pharmacology / Anatomy / etc.",\n'
            '  "title": "Concise, descriptive title for this document (max 50 characters)",\n'
            '  "rejection_reason": ""\n'
            "}\n"
            "If is_medical is false, provide a polite, concise rejection_reason (e.g., 'This file appears to be a financial receipt or payment invoice rather than medical study notes.')."
        )
        res = ""
        # Primary: OpenRouter multi-candidate array
        if OPENROUTER_KEY:
            for model_id in GATEKEEPER_OPENROUTER_MODELS:
                try:
                    async with httpx.AsyncClient(timeout=25.0) as c:
                        r = await c.post("https://openrouter.ai/api/v1/chat/completions",
                            headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
                            json={"model": model_id,
                                  "messages": [{"role": "system", "content": "You are an academic gatekeeper. Output ONLY JSON."},
                                               {"role": "user", "content": prompt}],
                                  "temperature": 0.0, "max_tokens": 250})
                        if r.status_code == 200:
                            data = r.json()
                            res = data["choices"][0]["message"]["content"]
                            if res:
                                break
                except Exception as e:
                    print(f"[GK openrouter {model_id}] {e}")

        # Fallback: Groq active candidate models
        if not res and GROQ_KEY:
            for groq_cand in GATEKEEPER_GROQ_MODELS:
                try:
                    async with httpx.AsyncClient(timeout=15.0) as c:
                        r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                            json={"model": groq_cand,
                                  "messages": [{"role": "user", "content": prompt}],
                                  "temperature": 0.0, "max_tokens": 250})
                        if r.status_code == 200:
                            data = r.json()
                            content = data["choices"][0]["message"]["content"].strip()
                            content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
                            if content:
                                res = content
                                break
                except Exception as e:
                    print(f"[GK groq {groq_cand}] {e}")

        # Parse JSON
        if res:
            try:
                c2 = res.strip()
                for d in ["```json", "```"]:
                    if d in c2:
                        c2 = c2.split(d)[-1].split("```")[0].strip()
                s, ei = c2.find("{"), c2.rfind("}")
                if s != -1 and ei != -1:
                    d2 = json.loads(c2[s:ei+1])
                    return {"is_medical": bool(d2.get("is_medical", True)),
                            "category":   d2.get("category", "General Medicine"),
                            "title":      d2.get("title", dtitle) or dtitle,
                            "rejection_reason": d2.get("rejection_reason", "")}
            except Exception as pe:
                print(f"[GK parse] {pe}")

        # Full 57-stem offline heuristic fallback matching main.py
        MEDICAL_STEMS = [
            "pharm", "pathol", "physiol", "anatom", "clini", "diagnos", "therap", "syndrom",
            "diseas", "symptom", "patient", "hyperten", "diabet", "arter", "vein", "nerv",
            "muscl", "infect", "bacter", "viru", "inhibit", "receptor", "enzym", "protein",
            "tissu", "cardio", "renal", "nephr", "pulmon", "hepat", "gastr", "cerebr", "vascul",
            "lesion", "carcin", "biops", "histol", "etiol", "antibiot", "neoplasm", "inflamm",
            "immun", "antibod", "antigen", "hormon", "endocrin", "metabol", "hematol", "haematol",
            "leukocyt", "erythrocyt", "platelet", "hemoglobin", "haemoglobin", "surger", "pediatr",
            "obstetr", "gynecol", "anesthes", "toxicol", "tubul", "glomerul", "microbiol"
        ]
        sample_lower = sample.lower()
        if sum(1 for st in MEDICAL_STEMS if st in sample_lower) >= 2:
            return {"is_medical": True, "category": "Medical Study Material", "title": dtitle, "rejection_reason": ""}
        return {"is_medical": False, "category": "Unverified", "title": dtitle,
                "rejection_reason": "Could not verify authentic medical, clinical, or health-sciences study content in this document."}


    # ── FastEmbed + Qdrant upsert ─────────────────────────────────────────────
    async def embed_upsert(chunk_items: list, start_idx: int, phone: str,
                           fname: str, title: str, category: str) -> int:
        BATCH  = 64
        cfn    = re.sub(r"[^a-zA-Z0-9_]", "", os.path.splitext(fname)[0].lower()) or "doc"
        points = []
        for bs in range(0, len(chunk_items), BATCH):
            sl   = chunk_items[bs:bs+BATCH]
            txts = [x["text"] for x in sl]
            embs = await loop.run_in_executor(
                emb_pool,
                lambda t=txts: list(embedder.embed(t))
            )
            for si, (item, emb) in enumerate(zip(sl, embs)):
                ci  = start_idx + bs + si
                ch  = hashlib.sha256(item["text"].encode("utf-8")).hexdigest()[:12]
                pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{phone}_{cfn}_{ci}_{ch}"))
                points.append(qm.PointStruct(
                    id=pid, vector=emb.tolist(),
                    payload={"text": item["text"], "book_title": title, "source_file": fname,
                             "user_id": str(phone), "is_user_doc": True, "category": category,
                             "page_number": item["page_number"], "chunk_index": ci,
                             "uploaded_at": datetime.utcnow().isoformat()}
                ))
        if points:
            await qdrant_cl.upsert(collection_name=COLLECTION, points=points)
        return len(points)

    # ── Distributed Fencing Check (prevents race with local fallback) ─────────
    async def is_dispatch_active() -> bool:
        """Returns True if this worker's dispatch_id still holds the active lease in MongoDB."""
        if not dispatch_id or users_col is None:
            return True
        try:
            ud = await users_col.find_one({"user_id": sender_phone}, {"active_upload.dispatch_id": 1})
            current_did = (ud or {}).get("active_upload", {}).get("dispatch_id")
            if current_did and current_did != dispatch_id:
                print(f"[MODAL FENCING] Dispatch {dispatch_id} superseded by active lease {current_did}. Aborting writes.")
                return False
        except Exception as e:
            print(f"[MODAL FENCING error] {e}")
        return True

    # ── MongoDB helpers ───────────────────────────────────────────────────────
    async def mongo_set_upload(status: str, **kw):
        if users_col is None:
            return
        if not await is_dispatch_active():
            return
        try:
            await users_col.update_one(
                {"user_id": sender_phone},
                {"$set": {"active_upload.status": status, **kw}},
                upsert=True
            )
        except Exception as e:
            print(f"[MONGO set] {e}")

    async def mongo_clear_upload():
        if users_col is None:
            return
        try:
            # Only clear active_upload if THIS worker's dispatch_id is still the active one!
            if dispatch_id:
                await users_col.update_one(
                    {"user_id": sender_phone, "active_upload.dispatch_id": dispatch_id},
                    {"$unset": {"active_upload": ""}}
                )
            else:
                await users_col.update_one(
                    {"user_id": sender_phone},
                    {"$unset": {"active_upload": ""}}
                )
        except Exception as e:
            print(f"[MONGO clear] {e}")

    # ── MAIN PIPELINE ─────────────────────────────────────────────────────────
    try:
        await mongo_set_upload("processing")

        # Step 1: Download from Meta CDN
        raw, _ = await download_media(media_id)
        if not raw:
            await wa_send(sender_phone,
                "⚠️ *Download Failed*\n\n"
                "I couldn't retrieve your document from WhatsApp. Please try sending it again! 📄")
            return {"ok": False, "error": "download_failed"}

        # Step 2: File size check (150 MB limit)
        mb = len(raw) / (1024 * 1024)
        if mb > 150.0:
            await wa_send(sender_phone,
                f"⚠️ *File Too Large*\n\n"
                f"Your document is *{mb:.1f} MB*. Please keep uploads under *150 MB*.\n\n"
                "💡 Tip: Split large slide decks into individual lecture modules!")
            return {"ok": False, "error": "too_large"}

        # Step 3: Detect format labels
        fn = filename.lower()
        is_pres = fn.endswith((".pptx", ".ppt")) or "presentation" in mime_type or "powerpoint" in mime_type
        is_word = fn.endswith((".docx", ".doc")) or "word" in mime_type or "officedocument" in mime_type
        unit = "slides" if is_pres else "pages"
        icon = "📊" if is_pres else ("📝" if is_word else "📑")

        # Step 4: Extract text (CPU-bound → threadpool)
        ok, err, pages, stats = await loop.run_in_executor(
            emb_pool, extract_document, raw, filename, mime_type
        )
        del raw
        gc.collect()

        if not ok:
            ERR_MSGS = {
                "LEGACY_BINARY": (
                    f"💾 *Legacy Office Format*\n\n*{filename}* is a 97–2003 binary format.\n\n"
                    "Please save as *.docx*, *.pptx*, or *.pdf* and re-upload! 🩺📚"
                ),
                "ENCRYPTED": (
                    f"🔒 *Password-Protected Document*\n\n*{filename}* is encrypted.\n\n"
                    "Remove the password and try again! 🔑"
                ),
                "TOO_MANY_PAGES": (
                    f"📑 *Document Too Large*\n\n*{filename}* has *{stats.get('page_count')} {unit}* — max is 200.\n\n"
                    "💡 Upload individual lecture modules or chapters for best results!"
                ),
            }
            if err in ERR_MSGS:
                msg = ERR_MSGS[err]
            elif err == "SCANNED_IMAGE":
                ep  = stats.get("empty_pages", 0)
                tp  = stats.get("page_count", 0)
                pct = stats.get("empty_pct", 0)
                if is_pres:
                    msg = (f"📷 *Image-Only Slides Detected*\n\n*{filename}* has {ep}/{tp} slides "
                           f"(~{pct:.0f}%) with no selectable text.\n\nPlease export slides with digital text! 💡")
                else:
                    msg = (f"📷 *Scanned Image Detected*\n\n*{filename}* appears to be a scanned PDF "
                           f"({ep}/{tp} pages with no selectable text).\n\nPlease run OCR (Adobe Scan, CamScanner, Google Drive OCR) and re-upload! 💡🔍")
            else:
                msg = (f"⚠️ *Unable to Read Document*\n\n"
                       f"I had trouble reading *{filename}*. It may be empty or corrupted.\n\n"
                       "Please check the file and try again!")
            await wa_send(sender_phone, msg)
            return {"ok": False, "error": err}

        # Step 5: AI Medical Gatekeeper
        pfx = "Slide" if is_pres else "Page"
        snips = [
            f"--- {pfx} {pn} ---\n{pt[:600]}"
            for i, (pn, pt) in enumerate(pages)
            if i < 3 or i == len(pages) // 2
        ]
        gk = await is_medical_doc("\n".join(snips), filename)
        print(f"[GK] is_medical={gk['is_medical']} category={gk['category']!r}")

        if not gk.get("is_medical", True):
            rej = gk.get("rejection_reason") or "This document does not appear to contain medical study content."
            await wa_send(sender_phone,
                f"📋 *Document Not Indexed: {filename}*\n\n{rej}\n\n"
                "NEURA AI only indexes medical lecture slides, textbook chapters, and clinical handouts! 🩺📚")
            return {"ok": False, "error": "non_medical"}

        title    = gk.get("title") or os.path.splitext(filename)[0].replace("_", " ").replace("-", " ").title()
        category = gk.get("category", "General Medicine")

        if not await is_dispatch_active():
            print(f"[MODAL] Aborting before gatekeeper notification: dispatch {dispatch_id} superseded by active lease")
            return {"ok": False, "error": "superseded_by_local"}

        await wa_send(sender_phone,
            f"{icon} *Medical Document Verified: {title}*\n\n"
            f"Indexing {len(pages)} {unit} into your personal study vault... ⏳")

        # Step 6: Chunk all pages using character-based chunking with boundary splitting
        chunks = []
        for pn, pt in pages:
            for c in chunk_text_with_overlap(pt, chunk_size=850, overlap=150):
                if len(c.strip()) > 35:
                    chunks.append({"text": c.strip(), "page_number": pn})

        if not chunks:
            await wa_send(sender_phone,
                f"⚠️ *No Study Content Found*\n\n"
                f"I couldn't extract text chunks from *{filename}*. Please try a different version!")
            return {"ok": False, "error": "no_chunks"}

        # Step 7: Delete stale Qdrant points (re-upload deduplication)
        if not await is_dispatch_active():
            print(f"[MODAL] Aborting before Qdrant write: dispatch {dispatch_id} superseded by active lease")
            return {"ok": False, "error": "superseded_by_local"}

        try:
            await qdrant_cl.delete(
                collection_name=COLLECTION,
                points_selector=qm.FilterSelector(filter=qm.Filter(must=[
                    qm.FieldCondition(key="user_id",     match=qm.MatchValue(value=str(sender_phone))),
                    qm.FieldCondition(key="source_file", match=qm.MatchValue(value=filename)),
                ]))
            )
            print(f"[QDRANT] Cleaned stale points for {sender_phone} ({filename})")
        except Exception as de:
            print(f"[QDRANT cleanup] {de}")

        STAGE1_THRESHOLD = 20
        wat_today = datetime.now(timezone(timedelta(hours=1))).strftime("%Y-%m-%d")
        n = 0

        if len(chunks) <= STAGE1_THRESHOLD:
            # Single-stage (small documents — ≤20 chunks)
            n = await embed_upsert(chunks, 0, sender_phone, filename, title, category)
            if users_col is not None:
                try:
                    rec = {"filename": filename, "title": title, "category": category,
                           "page_count": len(pages), "chunk_count": n,
                           "is_partially_indexed": False, "uploaded_at": datetime.utcnow().isoformat()}
                    await users_col.update_one({"user_id": sender_phone}, {"$pull": {"custom_documents": {"filename": filename}}})
                    await users_col.update_one(
                        {"user_id": sender_phone},
                        {"$push": {"custom_documents": rec},
                         "$inc": {"total_uploaded_docs": 1, f"daily_doc_uploads.{wat_today}": 1}},
                        upsert=True
                    )
                except Exception as me:
                    print(f"[MONGO single] {me}")
        else:
            # Progressive 2-stage ingestion (large documents)
            # Stage 1: first 20 chunks → searchable immediately
            stage1_chunks = chunks[:STAGE1_THRESHOLD]
            n1 = await embed_upsert(stage1_chunks, 0, sender_phone, filename, title, category)
            mx = max(c["page_number"] for c in stage1_chunks)
            if users_col is not None:
                try:
                    rec = {"filename": filename, "title": title, "category": category,
                           "page_count": len(pages), "chunk_count": n1, "total_chunks": len(chunks),
                           "is_partially_indexed": True, "uploaded_at": datetime.utcnow().isoformat()}
                    await users_col.update_one({"user_id": sender_phone}, {"$pull": {"custom_documents": {"filename": filename}}})
                    await users_col.update_one(
                        {"user_id": sender_phone},
                        {"$push": {"custom_documents": rec},
                         "$inc": {"total_uploaded_docs": 1, f"daily_doc_uploads.{wat_today}": 1}},
                        upsert=True
                    )
                except Exception as me:
                    print(f"[MONGO stage1] {me}")
            try:
                await wa_send(sender_phone,
                    f"⚡ *Early Search Active: {title}*\n\n"
                    f"First {n1} study chunks (~{unit} 1–{mx}) are searchable right now! 🔍\n\n"
                    f"Indexing {len(chunks) - n1} more chunks in the background... ⏳")
            except Exception:
                pass

            # Stage 2: remaining chunks
            if not await is_dispatch_active():
                print(f"[MODAL] Aborting before Stage 2: dispatch {dispatch_id} superseded by active lease")
                return {"ok": False, "error": "superseded_by_local"}

            n = n1 + await embed_upsert(chunks[STAGE1_THRESHOLD:], STAGE1_THRESHOLD, sender_phone, filename, title, category)
            if users_col is not None:
                try:
                    await users_col.update_one(
                        {"user_id": sender_phone, "custom_documents.filename": filename},
                        {"$set": {"custom_documents.$.is_partially_indexed": False,
                                  "custom_documents.$.chunk_count": n}}
                    )
                except Exception as me:
                    print(f"[MONGO stage2 final] {me}")

        print(f"[MODAL] DONE {n} chunks indexed for {sender_phone} ({title!r})")

        # Step 8: WhatsApp success card with 1-tap study buttons
        if not await is_dispatch_active():
            print(f"[MODAL] Aborting before success card: dispatch {dispatch_id} superseded by active lease")
            return {"ok": False, "error": "superseded_by_local"}

        success_body = (
            f"✅ *Added to Your Personal Study Vault!*\n\n"
            f"📚 *Document:* *{title}*\n"
            f"📑 *Scope:* {len(pages)} {unit} ({n} study chunks)\n"
            f"🩺 *Discipline:* {category}\n\n"
            f"You can now ask questions about this document anytime!\n"
            f"💡 _Tap a quick action below to start studying:_"
        )
        await wa_buttons(sender_phone, success_body, [
            {"id": f"Summarize key clinical concepts in {title}", "title": "🔍 Summarize"},
            {"id": f"Quiz me on {title}",                         "title": "❓ Quiz Me"},
            {"id": "/documents",                                   "title": "📂 My Vault"},
        ])

        return {"ok": True, "chunks": n, "title": title}

    except Exception as exc:
        traceback.print_exc()
        await wa_send(sender_phone,
            f"⚠️ *Indexing Error*\n\n"
            f"Something went wrong while processing *{filename}*.\n\n"
            "Please try uploading it again!")
        return {"ok": False, "error": str(exc)}
    finally:
        await mongo_clear_upload()
        if mongo_cl is not None:
            mongo_cl.close()
