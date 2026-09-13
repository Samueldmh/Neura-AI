"""
modal_worker.py — NEURA AI Serverless Document Ingestion Worker
================================================================
Deployed on Modal.com (https://modal.com).

Each document upload fires a POST to this web endpoint from main.py
(fire-and-forget). Modal spins up ONE isolated container per document —
no queues, no waiting — every student's upload runs simultaneously.

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

import modal

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
    Called by main.py (Render) as fire-and-forget when a WhatsApp document arrives.

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

    if not sender_phone or not media_id:
        return {"ok": False, "error": "Missing sender_phone or media_id"}

    print(f"\n[MODAL] START user={sender_phone} file={filename!r} media_id={media_id}")

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
    users_col = mongo_cl["neura_db"]["users"] if mongo_cl else None
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
        dtitle = re.sub(r"[_\-]+", " ", os.path.splitext(fname)[0]).title()[:50]
        if len(sample.strip()) < 30:
            return {"is_medical": False, "category": "Unclassified", "title": dtitle, "rejection_reason": "Insufficient text."}

        prompt = (
            "You are a strict academic gatekeeper for NEURA AI (MBBS medical school study assistant).\n"
            f"File: {fname}\nSample:\n\"\"\"{sample[:2000]}\"\"\"\n\n"
            "Is this genuine medical/clinical/biomedical/pharmacological/anatomical/health-sciences study material?\n"
            "REJECT: receipts, CVs, admin forms, non-medical coursework, personal letters, spam.\n"
            'Output ONLY valid JSON (no markdown): {"is_medical":true,"category":"...","title":"...","rejection_reason":""}'
        )
        res = ""
        # Primary: OpenRouter
        if OPENROUTER_KEY:
            try:
                async with httpx.AsyncClient(timeout=25.0) as c:
                    r = await c.post("https://openrouter.ai/api/v1/chat/completions",
                        headers={"Authorization": f"Bearer {OPENROUTER_KEY}", "Content-Type": "application/json"},
                        json={"model": "google/gemini-2.5-flash-lite",
                              "messages": [{"role": "system", "content": "Output ONLY JSON."},
                                           {"role": "user", "content": prompt}],
                              "temperature": 0.0, "max_tokens": 250})
                    if r.status_code == 200:
                        res = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[GK openrouter] {e}")

        # Fallback: Groq
        if not res and GROQ_KEY:
            try:
                async with httpx.AsyncClient(timeout=20.0) as c:
                    r = await c.post("https://api.groq.com/openai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
                        json={"model": "llama-3.1-8b-instant",
                              "messages": [{"role": "system", "content": "Output ONLY JSON."},
                                           {"role": "user", "content": prompt}],
                              "temperature": 0.0, "max_tokens": 250})
                    if r.status_code == 200:
                        res = r.json()["choices"][0]["message"]["content"]
            except Exception as e:
                print(f"[GK groq] {e}")

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

        # Offline heuristic fallback
        STEMS = ["pharm","pathol","physiol","anatom","clini","diagnos","therap","syndrom",
                 "diseas","symptom","patient","hyperten","diabet","arter","vein","nerv",
                 "muscl","infect","bacter","viru","inhibit","receptor","enzym","tissu",
                 "cardio","renal","nephr","pulmon","hepat","gastr","cerebr","vascul"]
        if sum(1 for st in STEMS if st in sample.lower()) >= 2:
            return {"is_medical": True, "category": "Medical Study Material", "title": dtitle, "rejection_reason": ""}
        return {"is_medical": False, "category": "Unverified", "title": dtitle,
                "rejection_reason": "Could not verify authentic medical or health-sciences content."}

    # ── Text chunking ─────────────────────────────────────────────────────────
    def chunk_text(text: str, chunk_size: int = 850, overlap: int = 150) -> list:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            end = min(i + chunk_size, len(words))
            chunks.append(" ".join(words[i:end]))
            if end == len(words):
                break
            i += chunk_size - overlap
        return chunks

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
                ch  = hashlib.sha256(item["text"].encode()).hexdigest()[:12]
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

    # ── MongoDB helpers ───────────────────────────────────────────────────────
    async def mongo_set_upload(status: str, **kw):
        if not users_col:
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
        if not users_col:
            return
        try:
            await users_col.update_one(
                {"user_id": sender_phone},
                {"$unset": {"active_upload": ""}}
            )
        except Exception as e:
            print(f"[MONGO clear] {e}")

    # ── Multi-format text extractors ──────────────────────────────────────────
    def extract_pdf(raw: bytes, fname: str):
        try:
            import fitz
            doc = fitz.open(stream=raw, filetype="pdf")
            if doc.is_encrypted:
                return False, "ENCRYPTED", [], {}
            pages, empty = [], 0
            for i, pg in enumerate(doc):
                t = pg.get_text("text").strip()
                if len(t.split()) < 5:
                    empty += 1
                pages.append((i + 1, t))
            tot = len(pages)
            if tot == 0:
                return False, "EMPTY_DOCUMENT", [], {}
            if tot > 200:
                return False, "TOO_MANY_PAGES", [], {"page_count": tot}
            tw = sum(len(t.split()) for _, t in pages)
            er = empty / tot
            if tw < 15 or er > 0.85:
                return False, "SCANNED_IMAGE", [], {"empty_pages": empty, "page_count": tot, "empty_pct": round(er * 100, 1)}
            return True, "OK", pages, {"page_count": tot, "total_words": tw, "empty_pages": empty, "empty_pct": round(er * 100, 1), "doc_type": "pdf"}
        except Exception as e:
            return False, "CORRUPTED", [], {"error": str(e)}

    def extract_pptx(raw: bytes, fname: str):
        import zipfile, xml.etree.ElementTree as ET
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                sfs = sorted(
                    [f for f in zf.namelist() if f.startswith("ppt/slides/slide") and f.endswith(".xml")],
                    key=lambda x: int(re.search(r"\d+", x.split("/")[-1]).group())
                )
                pages, empty, tw = [], 0, 0
                for si, sf in enumerate(sfs, 1):
                    root = ET.fromstring(zf.read(sf))
                    texts = [e.text for e in root.iter() if e.text and e.tag.endswith("}t")]
                    txt = " ".join(" ".join(texts).split())
                    wc = len(txt.split())
                    tw += wc
                    if wc < 6:
                        empty += 1
                    pages.append((si, txt))
            tot = len(pages)
            if tot == 0:
                return False, "EMPTY_DOCUMENT", [], {}
            er = empty / tot
            if tw == 0 or er > 0.45 or (tot > 3 and tw / tot < 8.0):
                return False, "SCANNED_IMAGE", [], {"empty_pages": empty, "page_count": tot, "empty_pct": round(er * 100, 1)}
            return True, "OK", pages, {"page_count": tot, "total_words": tw, "doc_type": "presentation"}
        except Exception as e:
            return False, "CORRUPTED", [], {"error": str(e)}

    def extract_docx(raw: bytes, fname: str):
        import zipfile, xml.etree.ElementTree as ET
        paras = []
        try:
            import docx as _docx
            doc = _docx.Document(io.BytesIO(raw))
            paras = [
                (p.text.strip(), p.style.name.startswith("Heading") if p.style else False)
                for p in doc.paragraphs if p.text.strip()
            ]
            for table in doc.tables:
                for row in table.rows:
                    rt = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
                    if rt:
                        paras.append((rt, False))
        except Exception:
            try:
                with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                    if "word/document.xml" not in zf.namelist():
                        return False, "EMPTY_DOCUMENT", [], {}
                    root = ET.fromstring(zf.read("word/document.xml"))
                    for pe in root.iter():
                        if pe.tag.endswith("}p"):
                            ts = [e.text for e in pe.iter() if e.text and e.tag.endswith("}t")]
                            if ts:
                                paras.append(("".join(ts).strip(), False))
            except Exception as e:
                return False, "CORRUPTED", [], {"error": str(e)}
        if not paras:
            return False, "EMPTY_DOCUMENT", [], {}
        pgs, cp, cw, ct, tw = [], 1, 0, [], 0
        for txt, ih in paras:
            wc = len(txt.split())
            if cw > 450 or (ih and cw > 150):
                if ct:
                    pgs.append((cp, "\n\n".join(ct).strip()))
                    tw += cw; cp += 1; ct, cw = [], 0
            ct.append(txt); cw += wc
        if ct:
            pgs.append((cp, "\n\n".join(ct).strip()))
            tw += cw
        if len(pgs) > 200:
            return False, "TOO_MANY_PAGES", [], {"page_count": len(pgs)}
        if tw < 15:
            return False, "SCANNED_IMAGE", [], {}
        return True, "OK", pgs, {"page_count": len(pgs), "total_words": tw, "doc_type": "word"}

    def extract_document(raw: bytes, fname: str, mime: str):
        fn = fname.lower()
        if fn.endswith((".doc", ".ppt")) or mime in ("application/msword", "application/vnd.ms-powerpoint"):
            if "ppt" in fn or "powerpoint" in mime:
                r = extract_pptx(raw, fname)
                if r[0]:
                    return r
            r = extract_docx(raw, fname)
            if r[0]:
                return r
            return False, "LEGACY_BINARY", [], {"error": "Legacy 97-2003 format. Save as .docx/.pptx/.pdf."}
        if fn.endswith(".pptx") or "presentation" in mime:
            return extract_pptx(raw, fname)
        if fn.endswith(".docx") or "wordprocessing" in mime or "officedocument.word" in mime:
            return extract_docx(raw, fname)
        return extract_pdf(raw, fname)

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

        # Step 2: File size check (28 MB limit)
        mb = len(raw) / (1024 * 1024)
        if mb > 28.0:
            await wa_send(sender_phone,
                f"⚠️ *File Too Large*\n\n"
                f"Your document is *{mb:.1f} MB*. Please keep uploads under *28 MB*.\n\n"
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
                           f"({ep}/{tp} pages).\n\nPlease run OCR (Adobe Scan, CamScanner, Google Drive OCR) and re-upload! 💡🔍")
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

        title    = gk.get("title") or re.sub(r"[_\-]+", " ", os.path.splitext(filename)[0]).title()
        category = gk.get("category", "General Medicine")

        await wa_send(sender_phone,
            f"{icon} *Medical Document Verified: {title}*\n\n"
            f"Indexing {len(pages)} {unit} into your personal study vault... ⏳")

        # Step 6: Chunk all pages
        chunks = []
        for pn, pt in pages:
            for c in chunk_text(pt, chunk_size=850, overlap=150):
                if len(c.strip()) > 35:
                    chunks.append({"text": c.strip(), "page_number": pn})

        if not chunks:
            await wa_send(sender_phone,
                f"⚠️ *No Study Content Found*\n\n"
                f"I couldn't extract text chunks from *{filename}*. Please try a different version!")
            return {"ok": False, "error": "no_chunks"}

        # Step 7: Delete stale Qdrant points (re-upload deduplication)
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

        STAGE1    = 20
        wat_today = datetime.now(timezone(timedelta(hours=1))).strftime("%Y-%m-%d")
        n         = 0

        if len(chunks) <= STAGE1:
            # Single-stage (small documents — ≤20 chunks)
            n = await embed_upsert(chunks, 0, sender_phone, filename, title, category)
            if users_col:
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
            s1     = chunks[:STAGE1]
            n1     = await embed_upsert(s1, 0, sender_phone, filename, title, category)
            mx     = max(c["page_number"] for c in s1)
            if users_col:
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
            n = n1 + await embed_upsert(chunks[STAGE1:], STAGE1, sender_phone, filename, title, category)
            if users_col:
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
        if mongo_cl:
            mongo_cl.close()
