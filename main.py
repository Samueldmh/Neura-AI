import os
import re
import json
import traceback
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.background import BackgroundTask
from pydantic import BaseModel
from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client import models
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT VARIABLES (v2.0 Webhook)
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://76ce5d85-4701-4671-8c3f-02bcc741b078.us-west-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")

# Official Meta WhatsApp Cloud API credentials
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN", "EAAM3F01f3nYBSKwpMPZAU2Nhgdvr7b4481UQ2sCTosr3Hu6UIL3U5BTBiN8I5932PfnEx6GzDWiUfwMYiFok4eZCaMrLPNhhMvnAQ27fVsxxqpxIvES3SYhSi6speeab3FaBq8anZCoPVXS2f9LXA7b7ZA2kWrZBRA8zmBv03cBe2yTR3OWAAhgEh0lEk3ULqfAZDZD")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID", "1150180661520951")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "neura_ai_webhook_secret_2026")

COLLECTION_NAME = "neura_medical_knowledge"

CURRICULUM = {
    "200L": ["Anatomy", "Physiology", "Biochemistry"],
    "300L": ["Anatomy", "Physiology", "Biochemistry"],
    "400L": ["Histopathology", "Chemical Pathology", "Haematology", "Microbiology", "Pharmacology"],
    "500L": ["Obstetrics & Gynaecology"],
    "600L": ["Medicine & Surgery"]
}

AVAILABLE_BOOKS = {
    "Anatomy": ["Clinically Oriented Anatomy 8th Ed by Keith L Moore, Arthur F Dalley"],
    "Physiology": ["K Sembulingam Essentials of Medical Physiology 6th Edition"],
    "Biochemistry": ["Textbook of Biochemistry For Medical Students 7th Edition"],
    "Histopathology": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
    "Haematology": ["Essentials of Haematology"],
    "Microbiology": ["Jawetz_Melnick_Adelbergs_Medical_Microbiology_27_edition_Med_zoneTV"],
    "Pharmacology": ["Lippincott Illustrated Reviews: Pharmacology"]
}

app = FastAPI(title="NEURA AI Backend", version="2.0.0")

# Initialize FastEmbed & Qdrant Client
print("Initializing FastEmbed & Qdrant Client...")
print(f"QDRANT_URL: {QDRANT_URL}")
print(f"QDRANT_API_KEY Present: {bool(QDRANT_API_KEY)}")
print(f"OPENROUTER_API_KEY Present: {bool(OPENROUTER_API_KEY)}")
print(f"PHONE_NUMBER_ID: {PHONE_NUMBER_ID}")

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
shared_http_client = httpx.AsyncClient(timeout=30.0)
embedding_pool = ThreadPoolExecutor(max_workers=4)

def get_embedding_sync(text: str):
    return list(embedder.embed(text))[0]

@app.on_event("startup")
async def startup_event():
    try:
        await qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="book_title",
            field_schema=models.PayloadSchemaType.KEYWORD
        )
        print("✅ Created/verified Qdrant payload index for 'book_title'")
    except Exception as idx_err:
        print(f"ℹ️ Payload index info: {idx_err}")

print(f"MONGO_URI Present: {bool(MONGO_URI)}")
mongo_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = mongo_client.neura_db if mongo_client else None
chat_history_col = db.chat_history if db is not None else None
users_col = db.users if db is not None else None

class QueryRequest(BaseModel):
    user_id: str
    message: str

# ==========================================
# 2. SYSTEM PROMPTS & INTENT ROUTER
# ==========================================
SYSTEM_MEDICAL_PROMPT = """{user_context}You are NEURA AI, an elite medical study assistant and clinical co-pilot designed for Nigerian medical students.
Your goal is to engage students in an intelligent, conversational, back-and-forth Socratic dialogue while anchoring all core medical principles in their textbooks.

CLINICAL EXPLANATION & SIMPLIFICATION RULES:
1. STRICT TEXTBOOK GROUNDING: Answer ONLY using facts explicitly present in the RETRIEVED TEXTBOOK CONTEXT. If the requested medical topic is not covered in the retrieved context, state: "I'm sorry, but this topic is not covered in your currently selected textbooks." Do NOT use outside AI memory, and NEVER output notes about using outside knowledge.
2. NO ROBOT TALK: Never use phrases like "Based on the retrieved context..." or "According to this textbook...". Jump straight into the explanation naturally as if you inherently know the medical facts. You are a confident expert tutor.
3. SIMPLIFY COMPLEX WORDS: Whenever you use complex medical jargon or high-level pathology terms, immediately simplify and explain them in clear, intuitive terms so students can grasp the underlying concepts effortlessly.
4. HIGHLIGHT IMPORTANT TERMS: Bold key terms, mechanisms, and diagnostic criteria so the text is visually clear and easy to read.
5. CONVERSATIONAL SOCRATIC CO-PILOT: Engage students naturally. When they ask hypothetical "what if" questions or follow-ups, synthesize textbook principles with common-sense medical reasoning.

CRITICAL FORMATTING & LAYOUT RULES FOR WHATSAPP:
1. DOUBLE-LINE SPACING: Every section heading, sub-heading, bullet item, and paragraph MUST be separated by a full blank line (`\n\n`). Never stack bullet items or headers back-to-back on consecutive single lines.
2. SHORT PARAGRAPHS: Keep text blocks short (maximum 2 to 3 sentences per paragraph) so the message feels spacious and easy to read on mobile screens.
3. NO HASHTAGS OR RAW SYMBOLS: Never use `#`, `##`, or `###` headers. Never output raw visible asterisks.
4. BOLD HEADINGS & KEY TERMS: Use valid WhatsApp bold syntax (*Heading:* or *Key Term*) for all section headings, sub-headings, and key concepts so WhatsApp renders them in clean bold text.
5. PROPERLY INDENTED LISTS: Format bullet lists neatly using hyphens and proper double-line spacing:
   
   - *Main Section:* Clear explanation.
   
     - *Sub-detail:* Properly indented supporting detail.
6. NO INLINE CITATIONS & NO CITATION COMPLAINTS: Absolutely NEVER include page numbers, figure numbers, or textbook references (e.g. Robbins p. 787, Fig 20.33) in the middle of sentences, bullet points, or paragraphs. All citations MUST be listed strictly at the very end of your response under 📚 CITATIONS. Under 📚 CITATIONS, ONLY list the raw source (e.g., "- Robbins Pathology"). NEVER add conversational notes, complaints about the textbook's relevance, or explanations about where you got the knowledge. Just list the source.
7. NO SMASHED WORDS: Ensure flawless spacing between words, punctuation, and hyphens. Always put a space after colons, periods, and bold words (e.g. write "*Prazosin* is" instead of "*Prazosin*is").
8. Structure responses into clear sections separated by blank lines:
   📖 *IN-DEPTH EXPLANATION*
   
   💡 *KEY CLINICAL PEARLS*
   
   📚 *CITATIONS*
9. NO RAW MARKDOWN TABLES: WhatsApp does NOT render markdown tables. NEVER output pipes or table headers (| Col 1 | Col 2 | or |---|---|). Always structure comparisons and summary tables as clean bulleted list cards (e.g. - *Category:* followed by indented • *Detail:* bullets).
10. WHEN ASKED FOR DIAGRAMS/ILLUSTRATIONS: NEVER apologize or say 'I cannot generate or display diagrams' or 'I am only a text AI'. You ARE fully equipped with a real medical diagram and histology retrieval system that automatically delivers the authentic textbook figure below your explanation. Confidently provide the structured textbook breakdown and refer the student to the attached medical figure/histology slide below.
"""

SYSTEM_QUIZ_PROMPT = """{user_context}You are NEURA AI. Based ONLY on the retrieved medical textbook context, generate exactly 7 rigorous, medical-school standard (MBBS / USMLE Step 1 & 2 style) Multiple Choice Questions (MCQs).

RULES FOR MCQs:
1. Each question must present a realistic clinical vignette, physiological mechanism, or pharmacological scenario appropriate for medical students.
2. Provide 4 distinct options (A, B, C, D) for each of the 7 questions.
3. Structure your response clearly using WhatsApp Markdown:
   - List Question 1 through 7 with their options (A, B, C, D).
   - Provide a separate 🔑 **ANSWER KEY & DETAILED EXPLANATIONS** section at the bottom.
   - For every answer, explain why the correct option is right AND why the key distractor options are wrong, citing the specific textbook title.
"""

SYSTEM_INTERACTIVE_QUIZ_PROMPT = """You are NEURA AI. Based ONLY on the retrieved medical textbook context, generate 5 rigorous, medical-school standard (MBBS / USMLE style) Multiple Choice Questions.

CRITICAL INSTRUCTION: You MUST output ONLY valid JSON without any markdown formatting, code block backticks (no ```json), or outside conversational text.

Output JSON structure:
[
  {
    "q_num": 1,
    "vignette": "A 55-year-old male with hypertension and BPH is prescribed prazosin...",
    "option_a": "Alpha-1 adrenergic receptor antagonist",
    "option_b": "Beta-1 adrenergic receptor antagonist",
    "option_c": "ACE inhibitor",
    "option_d": "Calcium channel blocker",
    "correct_option": "A",
    "explanation": "Prazosin selectively blocks alpha-1 receptors on vascular smooth muscle...",
    "book_source": "Lippincott Illustrated Reviews: Pharmacology"
  }
]
"""

def classify_intent(message: str) -> str:
    msg_lower = message.strip().lower()
    
    if msg_lower in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "who are you", "what is neura ai"]:
        return "GREETING"
    
    if any(k in msg_lower for k in ["mcq", "quiz", "practice question", "test me", "exam question", "questions on", "generate_quiz"]):
        return "QUIZ"
    
    return "MEDICAL"

async def call_openrouter_llm(system_prompt: str, user_prompt: str, chat_history: list = None) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set on Render!")
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neura-ai.org",
        "X-Title": "NEURA AI Medical Assistant"
    }
    
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})
    
    payload = {
        "model": "deepseek/deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2500
    }
    
    try:
        response = await shared_http_client.post(url, headers=headers, json=payload)
    except Exception as http_err:
        async with httpx.AsyncClient(timeout=30.0) as fallback_client:
            response = await fallback_client.post(url, headers=headers, json=payload)
            
    if response.status_code != 200:
        print(f"OpenRouter Error Status {response.status_code}: {response.text}")
        raise HTTPException(status_code=500, detail=f"OpenRouter Error: {response.text}")
    data = response.json()
    return data["choices"][0]["message"]["content"]

async def stream_openrouter_llm_to_whatsapp(system_prompt: str, user_prompt: str, sender_phone: str, chat_history: list = None) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set on Render!")
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neura-ai.org",
        "X-Title": "NEURA AI Medical Assistant"
    }
    
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})
    
    payload = {
        "model": "deepseek/deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2500,
        "stream": True
    }
    
    full_text = ""
    current_chunk = ""
    chunk_sent_count = 0
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as stream_client:
            async with stream_client.stream("POST", url, headers=headers, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data_str = line[6:]
                        try:
                            data_json = json.loads(data_str)
                            if "choices" in data_json and len(data_json["choices"]) > 0:
                                delta = data_json["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    full_text += content
                                    current_chunk += content
                                    
                                    # Send first chunk fast (~220 chars) so student gets instant answer, then larger paragraphs
                                    threshold = 220 if chunk_sent_count == 0 else 750
                                    if "\n\n" in current_chunk and len(current_chunk) > threshold:
                                        parts = current_chunk.rsplit("\n\n", 1)
                                        send_part = parts[0].strip()
                                        if send_part:
                                            await send_whatsapp_cloud_msg(sender_phone, send_part)
                                            chunk_sent_count += 1
                                        current_chunk = parts[1] if len(parts) > 1 else ""
                        except json.JSONDecodeError:
                            pass
                            
            if current_chunk.strip():
                await send_whatsapp_cloud_msg(sender_phone, current_chunk.strip())
                
    except Exception as e:
        print(f"Error streaming LLM: {e}")
        full_text = await call_openrouter_llm(system_prompt, user_prompt, chat_history)
        await send_whatsapp_cloud_msg(sender_phone, full_text)
        
    return full_text

def convert_markdown_tables_to_whatsapp_cards(text: str) -> str:
    """Detects raw markdown tables and transforms them into clean, indented WhatsApp bullet cards."""
    lines = text.split('\n')
    output_lines = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        # Detect if line is part of a markdown table
        if line.startswith('|') and line.endswith('|') and line.count('|') >= 2:
            table_rows = []
            while i < len(lines) and lines[i].strip().startswith('|') and lines[i].strip().endswith('|'):
                row_raw = lines[i].strip()
                cells = [c.strip() for c in row_raw.split('|')[1:-1]]
                # Exclude delimiter rows like |---|---|
                if not all(re.match(r'^[-\s:]+$', c) for c in cells if c):
                    table_rows.append(cells)
                i += 1
            
            if table_rows:
                headers = table_rows[0]
                data_rows = table_rows[1:] if len(table_rows) > 1 else []
                output_lines.append("")
                for r in data_rows:
                    if not r or not any(r): continue
                    primary_title = r[0]
                    output_lines.append(f"- *{primary_title}*")
                    for c_idx in range(1, min(len(headers), len(r))):
                        col_name = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else f"Detail {c_idx}"
                        val = r[c_idx]
                        if val:
                            output_lines.append(f"  • *{col_name}:* {val}")
                    output_lines.append("")
            continue
        else:
            # Strip ugly standalone horizontal rules --- or ___
            if re.match(r'^\s*[-_]{3,}\s*$', line):
                output_lines.append("")
            else:
                output_lines.append(lines[i])
            i += 1
    return '\n'.join(output_lines)

def format_whatsapp_text(text: str) -> str:
    """Master WhatsApp text sanitizer. Fixes layout, tables, and bolding without destroying text."""
    if not text:
        return text

    # 0. Convert raw markdown tables to readable bullet cards & remove --- lines
    text = convert_markdown_tables_to_whatsapp_cards(text)

    # 1. Remove markdown hashes (e.g. ### Header -> Header)
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = text.replace('###', '').replace('##', '')

    # 2. Fix double asterisks **text** -> *text*
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)

    # 3. Add space around bold text if missing to prevent word merging
    text = re.sub(r'([a-zA-Z0-9])(\*[^\*\n]+\*)', r'\1 \2', text)
    text = re.sub(r'(\*[^\*\n]+\*)([a-zA-Z0-9])', r'\1 \2', text)

    # 4. Fix missing space after colons/commas inside bold or right after bold
    text = re.sub(r'(\*[^\*\n]+\*:)([^\s\n])', r'\1 \2', text)

    # 5. Ensure space after bullet hyphens
    text = re.sub(r'^-([a-zA-Z0-9\*])', r'- \1', text, flags=re.MULTILINE)

    # 6. Ensure double-line spacing before list items
    text = re.sub(r'([^\n])\n(-|\*|[0-9]+\.)\s+', r'\1\n\n\2 ', text)

    # 7. Add newline before emojis
    text = re.sub(r'([^\n])([📖💡📚🎯])', r'\1\n\n\2', text)

    # 8. FINAL PASS: Clean up orphan asterisks safely
    def clean_line_asterisks(line: str) -> str:
        placeholders = []
        def store_valid_bold(m):
            placeholders.append(m.group(0))
            return f"__VALID_BOLD_{len(placeholders)-1}__"

        pattern = r'\*([^\s\*][^\*]*[^\s\*]|[^\s\*])\*'
        line_masked = re.sub(pattern, store_valid_bold, line)

        line_masked = re.sub(r'([a-zA-Z0-9])\*([a-zA-Z0-9])', r'\1 \2', line_masked)
        line_masked = line_masked.replace('*', '')

        for idx, ph in enumerate(placeholders):
            line_masked = line_masked.replace(f"__VALID_BOLD_{idx}__", ph)

        return line_masked

    cleaned_lines = [clean_line_asterisks(l) for l in text.split('\n')]
    text = '\n'.join(cleaned_lines)

    # 9. Normalize multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

async def send_whatsapp_cloud_msg(to_number: str, message_text: str):
    """Sends a text response directly to the student via Meta WhatsApp Cloud API. Automatically chunks messages exceeding Meta's 4000 char limit."""
    message_text = format_whatsapp_text(message_text)
    if not message_text:
        return

    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }

    # Meta WhatsApp text messages cap at 4096 chars. Split into ~3500 char chunks by paragraph.
    chunks = []
    if len(message_text) > 3500:
        paragraphs = message_text.split("\n\n")
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) + 2 <= 3500:
                current_chunk += ("\n\n" if current_chunk else "") + p
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = p
        if current_chunk:
            chunks.append(current_chunk)
    else:
        chunks = [message_text]

    for chunk in chunks:
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {
                "preview_url": False,
                "body": chunk
            }
        }
        try:
            res = await shared_http_client.post(url, headers=headers, json=payload)
        except Exception:
            async with httpx.AsyncClient(timeout=20.0) as fallback_client:
                res = await fallback_client.post(url, headers=headers, json=payload)
        print(f"Meta Graph API Send Status {res.status_code}: {res.text}")

async def send_whatsapp_interactive_list(to_number: str, body_text: str, button_text: str, options: list):
    """Sends an Interactive List Message (max 10 options)"""
    body_text = format_whatsapp_text(body_text)
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }
    
    # WhatsApp requires list items to be under 24 chars for ID and title (usually). We will truncate safely.
    rows = []
    for opt in options[:10]: # Max 10 options per list
        if isinstance(opt, dict):
            row_item = {
                "id": str(opt.get("id", ""))[:200],
                "title": str(opt.get("title", ""))[:24].strip()
            }
            if opt.get("description"):
                row_item["description"] = str(opt.get("description"))[:72].strip()
            rows.append(row_item)
        else:
            rows.append({
                "id": str(opt)[:200],
                "title": str(opt)[:24].strip()
            })
        
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {
                "text": body_text
            },
            "action": {
                "button": button_text[:20], # Max 20 chars
                "sections": [
                    {
                        "title": "Available Options",
                        "rows": rows
                    }
                ]
            }
        }
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        res = await client.post(url, headers=headers, json=payload)
        print(f"Meta Graph API List Send Status {res.status_code}: {res.text}")

async def send_whatsapp_interactive_button(to_number: str, body_text: str, buttons: list):
    """Sends an Interactive Button Message (max 3 buttons)"""
    body_text = format_whatsapp_text(body_text)
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }
    
    action_buttons = []
    for btn in buttons[:3]:
        action_buttons.append({
            "type": "reply",
            "reply": {
                "id": btn.get("id", btn.get("title"))[:256],
                "title": btn.get("title")[:20]
            }
        })
        
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": body_text
            },
            "action": {
                "buttons": action_buttons
            }
        }
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        res = await client.post(url, headers=headers, json=payload)
        print(f"Meta Graph API Button Send Status {res.status_code}: {res.text}")

async def send_whatsapp_cta_url_button(to_number: str, body_text: str, button_label: str, url_target: str):
    """Sends an interactive Call-To-Action (CTA) URL button to open In-App WebViews"""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "cta_url",
            "body": {
                "text": body_text
            },
            "action": {
                "name": "cta_url",
                "parameters": {
                    "display_text": button_label[:20],
                    "url": url_target
                }
            }
        }
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        res = await client.post(url, headers=headers, json=payload)
        print(f"Meta CTA URL Button Send Status {res.status_code}: {res.text}")

async def mark_message_as_read(message_id: str):
    """Marks incoming message as read on WhatsApp Cloud API for instant blue ticks (<100ms)"""
    if not message_id or not WHATSAPP_TOKEN:
        return
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "status": "read",
        "message_id": message_id
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            await client.post(url, headers=headers, json=payload)
    except Exception:
        pass

async def send_whatsapp_image_url(to_number: str, image_url: str, caption: str = ""):
    """Sends an image to WhatsApp via Meta Cloud API using an image URL"""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "image",
        "image": {
            "link": image_url,
            "caption": caption[:1024] if caption else ""
        }
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            print(f"Meta Image Send Status {res.status_code}: {res.text}")
    except Exception as e:
        print(f"⚠️ Error sending WhatsApp image: {e}")

def should_generate_medical_illustration(user_msg: str, ai_answer: str) -> bool:
    """Detects if query or medical topic warrants a visual anatomical, histological or clinical illustration (Pre-clinical & Clinical 200L-600L)"""
    msg_low = user_msg.lower()
    # Explicit user visual requests (Pre-clinical & Clinical)
    explicit_triggers = [
        "diagram", "illustration", "picture", "image", "draw", "show me", "illustrate",
        "schematic", "visualize", "get a", "can i get", "layers", "pathway", "flowchart",
        "algorithm", "stages of", "mechanism of", "relations of", "parts of", "structure of",
        "structures of", "scoring", "cross section", "arterial supply", "venous drainage",
        "innervation", "lymphatic drainage", "histopathology", "pathology slide"
    ]
    if any(t in msg_low for t in explicit_triggers):
        return True
    
    # Anatomical / clinical topics where visual diagrams are high-yield across 200L-600L (auto-send image)
    visual_keywords = [
        # Pre-clinical (200L/300L/400L: Anatomy, Physio, Biochem, Path, Pharm, Micro)
        "anatomy", "cross section", "blood supply", "innervation", "artery", "arteries",
        "vein", "veins", "nerve", "plexus", "triangle", "foramen", "fossa", "sulcus",
        "cranial nerve", "circle of willis", "brachial plexus", "inguinal canal", "femoral triangle",
        "histology", "histopathology", "epithelium", "nephron", "sarcomere",
        "krebs", "glycolysis", "action potential", "cardiac cycle", "ecg", "heart", "brain",
        "liver", "kidney", "lungs", "embryology", "germ layer", "branchial arch", "pharyngeal arch",
        "b cell", "t cell", "nk cell", "lymphocyte", "development", "maturation", "lymphopoiesis",
        "renin", "angiotensin", "aldosterone", "raas", "coagulation", "cascade", "complement",
        "immunoglobulin", "antibody", "granuloma", "necrosis", "infarction", "atherosclerosis",
        "epidermis", "dermis", "skin layer", "renal tubule", "nephron", "alveoli", "bronchi",
        "neuromuscular", "synapse", "receptor", "spinal cord", "meninges", "ventricle",

        # Senior Clinical (500L/600L: O&G, Surgery, Pediatrics, Internal Medicine, Comm Health)
        "menstrual cycle", "calot", "tetralogy of fallot", "apgar", "glasgow coma", "gcs",
        "dka", "ketoacidosis", "appendicitis", "placenta previa", "abruptio", "ectopic pregnancy",
        "partograph", "jvp", "jugular venous", "portal hypertension", "portacaval", "portosystemic",
        "rule of nines", "burns", "hernia", "hesselbach", "murmur", "heart murmur", "cardiac murmur",
        "bishop score", "episiotomy", "pph", "postpartum hemorrhage", "imci", "phototherapy",
        "bilirubin nomogram", "curb-65", "wells score", "child-pugh", "meld score", "heparin",
        "warfarin", "atrial fibrillation", "ventricular tachycardia", "stemi", "nstemi", "ctg",
        "cardiotocography", "fetal heart rate", "cardinal movements", "stages of labor", "cholecystectomy"
    ]
    return any(kw in msg_low for kw in visual_keywords)

# Comprehensive list of filler words to aggressively strip from diagram topic searches
_DIAGRAM_FILLER_PATTERN = re.compile(
    r'(?i)\b(show|me|get|a|can|i|diagram|diagrams|illustration|illustrations|picture|pictures|'
    r'image|images|draw|drawing|drawings|photo|photos|pic|pics|sketch|visual|visualize|'
    r'view|of|the|an|with|and|in|for|please|help|want|give|need|display|generate|'
    r'create|make|produce|see|look|at|whats|what|is|are|how|does|do|'
    r'explain|describe|tell|show|present|provide)\b'
)

# Pre-verified high-yield medical diagram atlas (Instant 0ms, zero-rate-limit, 100% verified authentic figures across 200L-600L)
VERIFIED_MEDICAL_ATLAS = [
    # Immunology & Hematology (Pre-clinical & Clinical)
    (["b cell", "b-cell", "b lymph", "b cell develop", "b cell matur", "lymphopoies", "b cell activat", "b cel", "differentiat"], 
     ("B-cell Differentiation & Hematopoietic Lineage Flowchart", "https://upload.wikimedia.org/wikipedia/commons/6/69/Hematopoiesis_%28human%29_diagram.png")),
    (["t cell activat", "t cell develop", "t lymph", "t cell matur", "t cel", "thymus"], 
     ("T-cell Maturation and Activation Pathway", "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/T-cell_and_microvillus.png/1920px-T-cell_and_microvillus.png")),
    (["nk cell", "natural killer"], 
     ("Natural Killer (NK) Cell Cytotoxic Action", "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/Natural_killer_cell.jpg/1920px-Natural_killer_cell.jpg")),
    (["coagulat", "clotting cascade", "clotting factor", "coagulation cascade", "hemostasis"], 
     ("Coagulation Cascade with Clinical Coagulation Tests (Intrinsic & Extrinsic Pathways)", "https://upload.wikimedia.org/wikipedia/commons/4/4b/Coagulation_Cascade_with_Tests.jpg")),
    (["immunoglobulin", "igg", "iga", "igm", "igd", "ige", "antibody structur"], 
     ("Immunoglobulin G (IgG) Molecular Structure", "https://upload.wikimedia.org/wikipedia/commons/3/3a/Antibody_IgG1_surface.png")),
    (["complement system", "complement cascade", "complement activat"], 
     ("Complement Activation Pathways (Classical, Alternative, MBL)", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Complement_pathway.svg/1920px-Complement_pathway.svg.png")),
    (["lymph node", "lymph node structur"], 
     ("Lymph Node Histological Architecture", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/92/Schematic_of_lymph_node.svg/1920px-Schematic_of_lymph_node.svg.png")),
    (["bone marrow", "hematopoiesis", "haematopoiesis"], 
     ("Hematopoiesis and Blood Cell Lineage Differentiation", "https://upload.wikimedia.org/wikipedia/commons/6/69/Hematopoiesis_%28human%29_diagram.png")),

    # Cardiology & Vascular (Medicine & Surgery)
    (["circle of willis", "circle willis", "cerebral arterial circle", "willis"], 
     ("Circle of Willis (Cerebral Arterial Network)", "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Circle_of_Willis_en.svg/1920px-Circle_of_Willis_en.svg.png")),
    (["atrioventricular node", "av node", "cardiac conduction", "heart conduction", "sa node", "bundle of his"], 
     ("Cardiac Electrical Conduction Pathway", "https://upload.wikimedia.org/wikipedia/commons/0/0f/Cardiac_Conduction_System.jpg")),
    (["renin", "angiotensin", "aldosteron", "raas"], 
     ("Renin-Angiotensin-Aldosterone System (RAAS Pathway Illustration)", "https://upload.wikimedia.org/wikipedia/commons/a/a7/2117_Renin_Angiotensin_Aldosterone_Pathway.jpg")),
    (["cardiac cycle", "heart cycle", "wiggers diagram"], 
     ("Cardiac Cycle Wiggers Diagram (Pressures, Volumes & Heart Sounds)", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Wiggers_Diagram_2.svg/1920px-Wiggers_Diagram_2.svg.png")),
    (["ecg", "electrocardiogram", "ekg"], 
     ("Electrocardiography (ECG Waveform Components)", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/SinusRhythmLabels.svg/1920px-SinusRhythmLabels.svg.png")),
    (["atherosclerosis", "atheromatous plaque", "plaque"], 
     ("Atherosclerosis and Arterial Plaque Formation", "https://upload.wikimedia.org/wikipedia/commons/d/d1/Blausen_0257_CoronaryArtery_Plaque.png")),
    (["myocardial infarction", "heart attack", "ischemic heart"], 
     ("Myocardial Infarction Pathology and Coronary Occlusion", "https://upload.wikimedia.org/wikipedia/commons/f/fb/Blausen_0463_HeartAttack.png")),
    (["stroke", "cerebral infarct", "mca infarct"], 
     ("Cerebral Infarction and Middle Cerebral Artery Territory", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/49/MCA_Territory_Infarct.png/1920px-MCA_Territory_Infarct.png")),

    # Obstetrics & Gynaecology (500L/600L)
    (["menstrual cycle", "ovarian cycle", "uterine cycle", "follicular phase", "luteal phase", "lh surge"], 
     ("Menstrual Cycle Hormonal Fluctuation & Endometrial Phases", "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/MenstrualCycle2_en.svg/1920px-MenstrualCycle2_en.svg.png")),

    # Surgery & Anatomy (500L/600L)
    (["calot", "calots triangle", "cystohepatic triangle", "cholecystectomy"], 
     ("Calot's Triangle (Cystohepatic Triangle Anatomy for Cholecystectomy)", "https://upload.wikimedia.org/wikipedia/commons/6/65/Gray532.png")),
    (["brachial plexus"], 
     ("Brachial Plexus (Roots, Trunks, Divisions, Cords, Branches)", "https://upload.wikimedia.org/wikipedia/commons/3/3a/Gray808.png")),
    (["femoral triangle", "scarpa triangle"], 
     ("Femoral Triangle Anatomy and Neurovascular Contents", "https://upload.wikimedia.org/wikipedia/commons/4/47/Femoral-triangle-diagram.jpg")),
    (["inguinal canal", "inguinal ring"], 
     ("Inguinal Canal Anatomy (Walls, Rings & Spermatic Cord)", "https://upload.wikimedia.org/wikipedia/commons/b/b4/Gray1227.png")),
    (["spinal cord", "spinal cord cross", "spinothalamic", "corticospinal"], 
     ("Spinal Cord Cross-Section and Ascending/Descending Tracts", "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Spinal_cord_tracts_-_English.svg/1920px-Spinal_cord_tracts_-_English.svg.png")),
    (["meninges", "dura mater", "arachnoid", "pia mater"], 
     ("Cranial Meninges (Dura, Arachnoid, and Pia Mater Layers)", "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Meninges-en.svg/1920px-Meninges-en.svg.png")),

    # Pediatrics (500L/600L)
    (["tetralogy of fallot", "tof", "fallot"], 
     ("Tetralogy of Fallot (4 Cardinal Anatomical Defects)", "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Tetralogy_of_Fallot.svg/1920px-Tetralogy_of_Fallot.svg.png")),

    # Pathology & Histology
    (["granuloma", "granulom", "tubercul", "tubercol", "caseat", "caseous", "langhans"], 
     ("Caseating Tubercular Granuloma (H&E Micrograph)", "https://upload.wikimedia.org/wikipedia/commons/1/10/Granuloma_mac.jpg")),
    (["epidermis", "skin layer", "layer skin", "stratum", "keratinocyte", "dermis"], 
     ("Epidermis Strata & Cellular Layers (Histological Cross-Section)", "https://upload.wikimedia.org/wikipedia/commons/f/f9/502_Layers_of_epidermis.jpg")),
    (["nephron", "renal tubule", "glomerul", "bowman"], 
     ("Nephron Structure and Renal Tubule Anatomy", "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1c/Kidney_Nephron.svg/1920px-Kidney_Nephron.svg.png")),
    (["sarcomere", "muscle fiber", "muscle structur", "myofibril", "sliding filament"], 
     ("Sarcomere Sliding Filament Mechanism (Actin-Myosin Cross-Bridges)", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Sarcomere_diagram.svg/1920px-Sarcomere_diagram.svg.png")),
    (["alveol", "alveolus", "lung unit", "respiratory membrane"], 
     ("Pulmonary Alveolus and Gas Exchange Barrier", "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Alveolus_diagram.svg/1920px-Alveolus_diagram.svg.png")),
    (["synapse", "synaptic", "neuromuscular junction", "acetylcholine receptor"], 
     ("Chemical Synapse and Neurotransmitter Transmission", "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Chemical_synapse_schema_cropped.jpg/1920px-Chemical_synapse_schema_cropped.jpg")),
    (["action potential", "nerve impulse", "depolarization"], 
     ("Neuronal Action Potential Phases (Depolarization / Repolarization)", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Action_potential.svg/1920px-Action_potential.svg.png")),

    # Biochemistry & Physiology
    (["glycolys", "glucose breakdown", "embden meyerhof"], 
     ("Glycolysis Metabolic Pathway (10 Enzymatic Steps & Energy Yields)", "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Glycolysis.svg/1920px-Glycolysis.svg.png")),
    (["krebs", "citric acid cycle", "tca cycle"], 
     ("Citric Acid Cycle (Krebs TCA Cycle Steps & Enzymes)", "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Citric_acid_cycle_with_aconitate_2.svg/1920px-Citric_acid_cycle_with_aconitate_2.svg.png")),
    (["diabetes", "insulin pathway", "glucose transporter", "glut4"], 
     ("Insulin Signaling and Glucose Regulation Mechanism", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Insulin_glucose_metabolism_ZP.svg/1920px-Insulin_glucose_metabolism_ZP.svg.png")),
]

# In-memory LRU/dict cache for dynamic queries
DIAGRAM_CACHE = {}

# Concurrency semaphore to throttle outbound Wikipedia requests to max 5 simultaneous
_WIKI_SEMAPHORE = asyncio.Semaphore(5)

async def retrieve_real_medical_diagram(topic_or_candidates):
    """
    1. Fast-Path: Instant lookup against verified medical atlas (0ms, 100% uptime, zero rate limits).
    2. Cache-Path: Check in-memory diagram cache (0ms).
    3. Dynamic Fallback: Query Wikipedia with connection pooling, semaphore rate-limiting, and retries.
    """
    if isinstance(topic_or_candidates, list):
        input_candidates = topic_or_candidates
        raw_topic = " ".join(topic_or_candidates[:3]).lower()
    else:
        input_candidates = [topic_or_candidates]
        raw_topic = (topic_or_candidates or "").lower()

    # 1. Check Verified Atlas (Deterministic & Instant 0ms)
    for patterns, (title, img_url) in VERIFIED_MEDICAL_ATLAS:
        if any(p in raw_topic for p in patterns):
            return img_url, title

    # 2. Check In-Memory Dynamic Cache
    cache_key = raw_topic.strip()
    if cache_key in DIAGRAM_CACHE:
        return DIAGRAM_CACHE[cache_key]

    # 3. Dynamic Lookup via Wikipedia API with Semaphore Throttling
    import urllib.parse
    headers = {"User-Agent": "NeuraAI-MBBS-Bot/2.0 (contact: medical.support@neura.ai)"}
    
    seen = set()
    search_candidates = []
    for c in input_candidates:
        c = c.strip() if c else ""
        if not c or len(c) < 2: continue
        if c.lower() not in seen:
            seen.add(c.lower())
            search_candidates.append(c)
        clean = _DIAGRAM_FILLER_PATTERN.sub(' ', c)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if clean and clean.lower() not in seen and len(clean) > 2:
            seen.add(clean.lower())
            search_candidates.append(clean)

    search_url = "https://en.wikipedia.org/w/api.php"
    
    try:
        async with _WIKI_SEMAPHORE:
            async with httpx.AsyncClient(timeout=8.0, limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)) as client:
                for cand in search_candidates[:5]:
                    if not cand or len(cand) < 2: continue
                    search_params = {"action": "opensearch", "search": cand, "limit": 5, "namespace": 0, "format": "json"}
                    s_res = await client.get(search_url, params=search_params, headers=headers)
                    if s_res.status_code == 200:
                        titles = s_res.json()[1] if len(s_res.json()) > 1 else []
                        for title in titles:
                            encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                            sum_res = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}", headers=headers)
                            if sum_res.status_code == 200:
                                data = sum_res.json()
                                img_url = data.get("originalimage", {}).get("source") or data.get("thumbnail", {}).get("source")
                                if img_url and any(img_url.lower().endswith(ext) or ext in img_url.lower() for ext in [".jpg", ".jpeg", ".png", ".svg"]):
                                    if not any(bad in img_url.lower() for bad in ["symbol", "icon", "stub", "question_mark", "disambig"]):
                                        DIAGRAM_CACHE[cache_key] = (img_url, title)
                                        return img_url, title
    except Exception as e:
        print(f"⚠️ Dynamic diagram fetch fallback: {e}")
        
    return None, None

async def send_commands_menu(sender_phone: str):
    """Sends an interactive WhatsApp List containing all available slash commands with 1-tap execution"""
    body_text = (
        "📋 *NEURA AI Commands Menu*\n\n"
        "Tap a command below to execute it instantly, or type any of them directly into the chat:"
    )
    options = [
        {"id": "/profile", "title": "👤 /profile", "description": "View your class, level & active textbooks"},
        {"id": "/update books", "title": "📚 /update books", "description": "Change or add your preferred medical textbooks"},
        {"id": "/update level", "title": "🎓 /update level", "description": "Update your current class/level (e.g. 400L)"},
        {"id": "/update name", "title": "✏️ /update name", "description": "Update your student display name"},
        {"id": "/reset", "title": "🔄 /reset", "description": "Reset full profile & chat history to start over"},
        {"id": "/feedback", "title": "📝 /feedback", "description": "Share anonymous feedback on NEURA AI"},
    ]
    await send_whatsapp_interactive_list(sender_phone, body_text, "View Commands", options)


# Filler words to strip for search (NOT removed from the AI prompt — only from Qdrant search)
SEARCH_STOP_WORDS = {
    "explain", "what", "is", "are", "the", "of", "in", "simple", "words", "terms",
    "tell", "me", "about", "can", "you", "how", "does", "do", "like", "a", "baby",
    "use", "analogies", "give", "an", "example", "simplify", "this", "that", "for",
    "with", "to", "it", "please", "help", "understand", "very", "briefly", "detail",
    "detailed", "describe", "define", "compare", "contrast", "difference", "between",
    "why", "when", "where", "which", "who", "whom", "would", "could", "should",
    "i", "my", "we", "our", "they", "their", "its", "has", "have", "had", "was",
    "were", "be", "been", "being", "am", "will", "shall", "may", "might", "must",
    "need", "want", "know", "think", "say", "said", "get", "got", "make", "made",
    "go", "going", "come", "take", "see", "look", "also", "just", "more", "some",
    "any", "all", "each", "every", "both", "few", "many", "much", "no", "not",
    "only", "own", "same", "so", "than", "too", "very", "really", "mean", "means",
    "meaning", "meant", "using", "used", "work", "works", "working", "way",
    "thing", "things", "something", "anything", "everything", "nothing",
    "talk", "talking", "teach", "show", "break", "down", "breakdown",
    "diagram", "diagrams", "illustration", "illustrations", "picture", "pictures",
    "image", "images", "draw", "drawing", "drawings", "photo", "photos", "pic", "pics",
    "sketch", "visual", "visualize", "view"
}

SPECIAL_SHORT_MEDICAL = {"b", "t", "nk", "av", "sa", "ph", "c3", "c4", "c5", "k", "na", "ca", "fe", "mg", "ig"}

FOLLOWUP_PHRASES = [
    "tell me more", "tell me more about this", "is this all", "is that all",
    "more details", "explain further", "elaborate", "what else", "give me more",
    "continue", "explain simpler", "can you explain more", "anything else",
    "tell me further", "more on this", "go on", "keep going", "is there more",
    "summarize more", "more info", "expand on this", "details", "are you sure",
    "what about this", "what of this", "further explanation"
]

def check_is_followup_query(msg: str) -> bool:
    """Detects if a user message is a conversational follow-up without standalone medical terms."""
    msg_clean = re.sub(r'[^\w\s]', '', msg.lower().strip())
    if any(phrase in msg_clean for phrase in FOLLOWUP_PHRASES):
        return True
    
    words = msg_clean.split()
    if len(words) <= 4:
        terms = extract_medical_terms(msg)
        if not terms or (len(terms) == 1 and terms[0].lower() == msg_clean):
            if all(w.lower() in SEARCH_STOP_WORDS for w in words):
                return True
    return False

def get_explicit_book_override(user_msg: str, preferred_books: list) -> list:
    """If the user explicitly mentions a subject or book name in their prompt, restrict the search to that book."""
    msg_lower = user_msg.lower()
    override_books = []
    for b in preferred_books or []:
        if not b or b.startswith("Skip"): continue
        b_lower = b.lower()
        if "pharmacology" in msg_lower and "pharmacology" in b_lower:
            override_books.append(b)
        elif "pathology" in msg_lower and ("pathology" in b_lower or "robbins" in b_lower):
            override_books.append(b)
        elif "anatomy" in msg_lower and "anatomy" in b_lower:
            override_books.append(b)
        elif "physiology" in msg_lower and ("physiology" in b_lower or "sembulingam" in b_lower):
            override_books.append(b)
        elif "biochemistry" in msg_lower and "biochemistry" in b_lower:
            override_books.append(b)
        elif "haematology" in msg_lower and ("haematology" in b_lower or "hoffbrand" in b_lower):
            override_books.append(b)
        elif "microbiology" in msg_lower and ("microbiology" in b_lower or "jawetz" in b_lower):
            override_books.append(b)
        elif "lippincott" in msg_lower and "lippincott" in b_lower:
            override_books.append(b)
        elif "robbins" in msg_lower and "robbins" in b_lower:
            override_books.append(b)
        elif "sembulingam" in msg_lower and "sembulingam" in b_lower:
            override_books.append(b)
            
    return override_books if override_books else preferred_books

def extract_medical_terms(user_msg: str) -> list:
    """Instantly extract clean medical keywords by normalizing typos, preserving short medical terms (B-cell, T-cell), and stripping filler words."""
    # 1. Normalize common student typos (e.g., 'b cel' -> 'b cell', 't cel' -> 't cell')
    msg = re.sub(r'\b([bt])\s*cel\b', r'\1 cell', user_msg, flags=re.IGNORECASE)
    msg = re.sub(r'\bcel\b', 'cell', msg, flags=re.IGNORECASE)
    
    msg_cleaned = re.sub(r'[^\w\s]', ' ', msg)
    words = msg_cleaned.split()
    
    # Check lowercase for stop words, but preserve essential short medical abbreviations
    meaningful_words = [
        w for w in words 
        if (w.lower() in SPECIAL_SHORT_MEDICAL or len(w) > 2) and w.lower() not in SEARCH_STOP_WORDS
    ]
    
    if not meaningful_words:
        return [user_msg]
        
    phrases = []
    current_phrase = []
    
    for w in meaningful_words:
        if w.lower() in ["and", "or", "vs", "versus"]:
            if current_phrase:
                phrases.append(" ".join(current_phrase))
                current_phrase = []
        else:
            current_phrase.append(w)
            
    if current_phrase:
        phrases.append(" ".join(current_phrase))
    
    # Add the unified cleaned query at the start to maximize semantic relevance
    joined_query = " ".join(meaningful_words)
    if joined_query and joined_query not in phrases:
        phrases.insert(0, joined_query)
        
    print(f"🔍 Extracted search keywords: {phrases} (from: '{user_msg}')")
    return phrases

async def normalize_medical_query(user_msg: str) -> dict:
    """Fast micro-LLM pass that resolves all typos, medical slang, and extracts authoritative textbook search phrases and diagram topics."""
    if not OPENROUTER_API_KEY:
        return {"search_keywords": extract_medical_terms(user_msg), "diagram_topic": None}
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neura-ai.org",
        "X-Title": "NEURA AI Medical Assistant"
    }
    system_prompt = (
        "You are an expert MBBS medical query normalizer. The student may send questions with typos, shorthand, Nigerian medical student slang, or visual requests (e.g. 'can I get diagram of b cel dev in body').\n"
        "Output ONLY a JSON object with two fields:\n"
        "1. 'search_keywords': A list of 1 to 3 authoritative medical textbook search phrases. Fix all typos and expand abbreviations (e.g. ['B-cell development and maturation', 'B-lymphocyte lymphopoiesis in bone marrow']). Strip visual words like 'diagram', 'picture', 'show me'.\n"
        "2. 'diagram_topic': The clean medical anatomical/pathological subject if a visual/diagram was requested or appropriate (e.g. 'B-cell development'), else null.\n"
        "Output ONLY valid JSON (no markdown, no ```json)."
    )
    payload = {
        "model": "deepseek/deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.0,
        "max_tokens": 140
    }
    try:
        async with httpx.AsyncClient(timeout=4.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            if res.status_code == 200:
                text = res.json()["choices"][0]["message"]["content"]
                text = text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(text)
                if isinstance(parsed, dict) and "search_keywords" in parsed:
                    return parsed
    except Exception as e:
        print(f"⚠️ Micro-LLM normalizer error (using fallback): {e}")
        
    return {"search_keywords": extract_medical_terms(user_msg), "diagram_topic": None}

def extract_book_keywords(preferred_books: list) -> list:
    """Extract core textbook keywords for matching (e.g., 'lippincott', 'robbins', 'moore', 'hoffbrand')"""
    keywords = []
    for b in preferred_books or []:
        if not b or not isinstance(b, str) or b.startswith("Skip"):
            continue
        b_lower = b.lower()
        if "lippincott" in b_lower:
            keywords.append("lippincott")
        elif "robbins" in b_lower:
            keywords.append("robbins")
        elif "moore" in b_lower:
            keywords.append("moore")
        elif "hoffbrand" in b_lower or "haematology" in b_lower:
            keywords.append("hoffbrand")
            keywords.append("haematology")
        else:
            words = [w.lower() for w in re.sub(r'[^\w\s]', '', b).split() if len(w) > 3]
            keywords.extend(words)
    return keywords

async def search_qdrant(query_text: str, limit: int = 4, preferred_books: list = None) -> list:
    """Search Qdrant securely per textbook to guarantee every selected book gets equal representation."""
    try:
        loop = asyncio.get_running_loop()
        query_vector = await loop.run_in_executor(embedding_pool, get_embedding_sync, query_text)
        
        all_points = []
        
        # If no preferred books are selected, fall back to a generic global search
        if not preferred_books:
            res = await qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit
            )
            return res.points
            
        # Guarantee equal representation by querying Qdrant for EACH book
        for book in preferred_books:
            if not book or not isinstance(book, str) or book.startswith("Skip"):
                continue
                
            hits = []
            # Attempt 1: Exact Match
            try:
                res = await qdrant.query_points(
                    collection_name=COLLECTION_NAME,
                    query=query_vector,
                    query_filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="book_title",
                                match=models.MatchValue(value=book)
                            )
                        ]
                    ),
                    limit=limit
                )
                hits = res.points
            except Exception as e:
                hits = []

            # Attempt 2: Fuzzy keyword match if exact string match returned 0 hits
            if not hits:
                book_kw = ""
                b_lower = book.lower()
                if "lippincott" in b_lower:
                    book_kw = "lippincott"
                elif "robbins" in b_lower:
                    book_kw = "robbins"
                elif "haematology" in b_lower or "hoffbrand" in b_lower:
                    book_kw = "haematology"
                elif "microbiology" in b_lower or "jawetz" in b_lower:
                    book_kw = "microbiology"
                elif "sembulingam" in b_lower:
                    book_kw = "sembulingam"
                elif "moore" in b_lower or "anatomy" in b_lower:
                    book_kw = "moore"
                
                if book_kw:
                    try:
                        res = await qdrant.query_points(
                            collection_name=COLLECTION_NAME,
                            query=query_vector,
                            query_filter=models.Filter(
                                must=[
                                    models.FieldCondition(
                                        key="book_title",
                                        match=models.MatchText(text=book_kw)
                                    )
                                ]
                            ),
                            limit=limit
                        )
                        hits = res.points
                    except Exception as fuzzy_e:
                        hits = []
                        
            all_points.extend(hits)
                
        # Sort the combined hits from all books by score
        all_points.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        return all_points

    except Exception as outer_e:
        print(f"❌ Error in search_qdrant: {outer_e}")
        return []

async def multi_search_qdrant(search_terms: list, preferred_books: list = None) -> list:
    """Run separate Qdrant searches for each extracted medical keyword CONCURRENTLY, then deduplicate"""
    seen_texts = set()
    all_results = []
    
    # Run all searches concurrently!
    tasks = [search_qdrant(term, limit=4, preferred_books=preferred_books) for term in search_terms]
    results_list = await asyncio.gather(*tasks)
    
    for results in results_list:
        for point in results:
            text_key = point.payload.get("text", "")[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_results.append(point)
    
    # Cap at 10 results max to optimize prompt processing speed while keeping 100% medical depth
    all_results = all_results[:10]
    print(f"📚 Multi-search returned {len(all_results)} unique chunks from {len(search_terms)} keyword(s) with filter {preferred_books}")
    return all_results

async def extract_name_with_llm(user_msg: str) -> str:
    prompt = """Extract the person's first name from this message. 
If they just say a greeting, or "why do you need it", or if it is random gibberish (e.g., "asdfgh"), or if there is clearly no name, return NONE.
Return ONLY the name, nothing else.
Examples:
- "I am Samuel" -> Samuel
- "Samuel" -> Samuel
- "Hi my name is John" -> John
- "Why do you want to know?" -> NONE
- "Hello" -> NONE
- "dhjdsf" -> NONE"""
    try:
        res = await call_openrouter_llm(prompt, user_msg)
        return res.strip() if res.strip().upper() != "NONE" else None
    except:
        return None

async def send_subject_book_menu(sender_phone: str, level: str, subject: str) -> bool:
    """Sends the interactive book selection menu for a subject using the Live Checklist Dropdown UI."""
    user_doc = await users_col.find_one({"user_id": sender_phone}) if users_col is not None else None
    preferred_books = user_doc.get("preferred_books_list", []) if user_doc else []
    
    all_books = AVAILABLE_BOOKS.get(subject, [])
    if not all_books:
        body_text = f"No textbooks currently indexed for *{subject}*."
        await send_whatsapp_interactive_list(
            sender_phone, 
            body_text, 
            "Select Option", 
            [{"id": "SKIP_SUBJECT", "title": "⏭️ Skip this subject", "description": "Continue to next subject"}]
        )
    elif len(all_books) == 1:
        # Single-book subject: clean 1-tap selection
        single_book = all_books[0]
        body_text = f"Please select your preferred textbook for *{subject}*:"
        options = [{
            "id": single_book,
            "title": single_book[:24].strip(),
            "description": single_book[:72].strip()
        }]
        await send_whatsapp_interactive_list(sender_phone, body_text, "Select Textbook", options)
    else:
        # Multi-book subject: Live Checklist Dropdown
        selected_for_subject = [b for b in all_books if b in preferred_books]
        
        # Build visual checklist lines
        checklist_lines = []
        for b in all_books:
            b_display = b.split(":")[0][:40]
            if b in selected_for_subject:
                checklist_lines.append(f"• [✓] *{b_display}*")
            else:
                checklist_lines.append(f"• [  ] {b_display}")
        checklist_str = "\n".join(checklist_lines)
        
        if selected_for_subject:
            body_text = (
                f"📚 *{subject} Textbooks* ({len(selected_for_subject)}/{len(all_books)} Selected):\n"
                f"{checklist_str}\n\n"
                f"Tap a book below to add/remove, or tap Finish when you're done!"
            )
        else:
            body_text = (
                f"📚 *{subject} Textbooks* (0/{len(all_books)} Selected):\n"
                f"{checklist_str}\n\n"
                f"Tap a textbook below to select it:"
            )
            
        options = []
        # Add Finish option if at least 1 book is selected
        if selected_for_subject:
            options.append({
                "id": f"FINISH_SUBJECT_{subject}",
                "title": "✅ Finish & Next Subject",
                "description": f"Proceed with {len(selected_for_subject)} selected book(s)"
            })
            
        # List all books with Add / Remove action indicators
        for b in all_books:
            if b in selected_for_subject:
                options.append({
                    "id": f"TOGGLE_{b}",
                    "title": f"❌ Remove: {b}"[:24].strip(),
                    "description": f"Remove {b}"[:72].strip()
                })
            else:
                options.append({
                    "id": f"TOGGLE_{b}",
                    "title": f"➕ Add: {b}"[:24].strip(),
                    "description": f"Add {b}"[:72].strip()
                })
                
        # If no book selected yet, offer skip option
        if not selected_for_subject:
            options.append({
                "id": "SKIP_SUBJECT",
                "title": "⏭️ Skip this subject",
                "description": "Do not select any textbook for this subject"
            })
            
        await send_whatsapp_interactive_list(sender_phone, body_text, "Select / Toggle", options)
        
    await users_col.update_one(
        {"user_id": sender_phone}, 
        {"$set": {"onboarding_step": f"ASK_BOOK_{subject}"}}
    )
    return True

async def send_next_subject_menu(sender_phone: str, level: str, current_subject: str = None) -> bool:
    """Finds the next subject for the level and sends the menu. Returns False if all done."""
    subjects = CURRICULUM.get(level, [])
    if not subjects:
        return False
        
    next_subject = None
    if current_subject is None:
        next_subject = subjects[0]
    else:
        try:
            idx = subjects.index(current_subject)
            if idx + 1 < len(subjects):
                next_subject = subjects[idx + 1]
        except ValueError:
            next_subject = subjects[0]
            
    if not next_subject:
        return False # We finished all subjects
        
    return await send_subject_book_menu(sender_phone, level, next_subject)

async def complete_onboarding(sender_phone: str):
    user_doc = await users_col.find_one({"user_id": sender_phone})
    name = user_doc.get("name", "Student") if user_doc else "Student"
    level = user_doc.get("level", "") if user_doc else ""
    preferred_books = user_doc.get("preferred_books_list", []) if user_doc else []
    
    await users_col.update_one({"user_id": sender_phone}, {"$set": {"onboarding_step": "COMPLETED"}})
    
    if preferred_books:
        books_summary = "\n".join(f"• {b}" for b in preferred_books)
    else:
        books_summary = "• None selected (searching general medical knowledge)"
        
    final_msg = (
        f"🎉 Awesome, {name}! Your profile is all set up for *{level}*.\n\n"
        f"📚 *Your Selected Textbooks:*\n"
        f"{books_summary}\n\n"
        f"You can now start asking me medical questions directly from these textbooks! 🧠⚡\n\n"
        f"⚙️ *Quick Commands:*\n"
        f"• Type */feedback* to share quick feedback\n"
        f"• Type */profile* to view your profile\n"
        f"• Type */update name* to change your name\n"
        f"• Type */update level* to change your level\n"
        f"• Type */update books* to change your textbooks\n\n"
        f"💬 _Help us improve! Share 2-min anonymous beta feedback anytime: https://forms.gle/dNr7SV5EUiqiFySx5_"
    )
    await send_whatsapp_cloud_msg(sender_phone, final_msg)

async def handle_onboarding(sender_phone: str, user_msg: str) -> bool:
    """Returns True if message was swallowed by onboarding, False if normal RAG should proceed."""
    if users_col is None:
        return False
        
    user_doc = await users_col.find_one({"user_id": sender_phone})
    
    # 1. New user
    if not user_doc:
        await users_col.insert_one({
            "user_id": sender_phone,
            "onboarding_step": "ASK_NAME"
        })
        welcome_msg = (
            "Hello! 👋 I'm *NEURA AI*, your elite medical study assistant.\n\n"
            "I can answer medical questions directly from your textbooks with exact citations, or generate practice MCQs for your MBBS exams!\n\n"
            "To give you the best personalized study experience, what is your first name?"
        )
        await send_whatsapp_cloud_msg(sender_phone, welcome_msg)
        return True
        
    step = user_doc.get("onboarding_step")
    name = user_doc.get("name", "Student")
    level = user_doc.get("level", "")
    
    if step == "COMPLETED":
        # Block users from reusing old menu selections or level buttons after completing setup
        all_book_options = [b for books in AVAILABLE_BOOKS.values() for b in books] + [
            "Skip (None available yet)", "SKIP_SUBJECT", "⏭️ Skip this subject", "✅ Finish & Next Subject", "➡️ Next Subject"
        ]
        is_menu_tap = (
            user_msg in ["200L", "300L", "400L", "500L", "600L"] or 
            user_msg in all_book_options or 
            user_msg in ["START_ONBOARDING", "🚀 Start Setup"] or
            user_msg.startswith("TOGGLE_") or
            user_msg.startswith("FINISH_SUBJECT_") or
            user_msg.startswith("DONE_SUBJECT_")
        )
        
        if is_menu_tap:
            await send_whatsapp_cloud_msg(
                sender_phone,
                "⚠️ Your profile is already completed!\n\n"
                "To change your level or textbooks, please use the profile commands:\n"
                "• Type */update level* to change your level\n"
                "• Type */update books* to change your textbooks\n"
                "• Type */reset* to start over"
            )
            return True # Swallow message so RAG doesn't search for "400L"
        return False
        
    # 2. Extract Name
    if step == "ASK_NAME":
        extracted_name = await extract_name_with_llm(user_msg)
        if not extracted_name:
            await send_whatsapp_cloud_msg(sender_phone, "I didn't quite catch that, or it didn't look like a real name! Please type your real first name so I know what to call you. 😊")
            return True
            
        await users_col.update_one({"user_id": sender_phone}, {"$set": {"name": extracted_name, "onboarding_step": "ASK_LEVEL"}})
        await send_whatsapp_interactive_list(
            sender_phone, 
            f"Nice to meet you, {extracted_name}! What is your current medical class/level?",
            "Select Level",
            ["200L", "300L", "400L", "500L", "600L"]
        )
        return True
        
    # 3. Extract Level
    if step == "ASK_LEVEL":
        if user_msg not in ["200L", "300L", "400L", "500L", "600L"]:
            await send_whatsapp_cloud_msg(sender_phone, "Please use the menu button to select your level.")
            return True
            
        new_level = user_msg
        await users_col.update_one({"user_id": sender_phone}, {"$set": {"level": new_level, "preferred_books_list": []}})
        
        # Start subject loop
        has_subjects = await send_next_subject_menu(sender_phone, new_level)
        if not has_subjects:
            await complete_onboarding(sender_phone)
        return True
        
    # 4. Extract Books (Dynamic Subject Loop with Live Checklist Dropdown)
    if step.startswith("ASK_BOOK_"):
        current_subject = step.replace("ASK_BOOK_", "")
        all_subject_books = AVAILABLE_BOOKS.get(current_subject, [])
        
        user_doc = await users_col.find_one({"user_id": sender_phone})
        preferred_books = user_doc.get("preferred_books_list", []) if user_doc else []
        
        # A. Finish / Skip / Next Subject
        is_finish = (
            user_msg in [
                "SKIP_SUBJECT", "Skip (None available yet)", "⏭️ Skip this subject",
                f"FINISH_SUBJECT_{current_subject}", "✅ Finish & Next Subject",
                f"DONE_SUBJECT_{current_subject}", "➡️ Next Subject"
            ] or
            user_msg.startswith("FINISH_SUBJECT_") or
            user_msg.startswith("DONE_SUBJECT_")
        )
        
        if is_finish:
            has_more = await send_next_subject_menu(sender_phone, level, current_subject)
            if not has_more:
                await complete_onboarding(sender_phone)
            return True
            
        # B. Handle Toggle / Book Selection
        raw_book = user_msg
        if raw_book.startswith("TOGGLE_"):
            raw_book = raw_book.replace("TOGGLE_", "", 1)
            
        matched_book = None
        for b in all_subject_books:
            if raw_book == b or raw_book == b[:24].strip() or raw_book.endswith(b[:20]) or b.startswith(raw_book):
                matched_book = b
                break
                
        if not matched_book:
            for b in [bk for books in AVAILABLE_BOOKS.values() for bk in books]:
                if raw_book == b or raw_book == b[:24].strip() or b.startswith(raw_book):
                    matched_book = b
                    break
                    
        if not matched_book:
            await send_whatsapp_cloud_msg(
                sender_phone, 
                "Please use the menu button to select/toggle your textbook, or tap Finish."
            )
            return True
            
        # Single-book subject: 1-tap select & auto-advance
        if len(all_subject_books) <= 1:
            if matched_book not in preferred_books:
                await users_col.update_one(
                    {"user_id": sender_phone},
                    {"$push": {"preferred_books_list": matched_book}}
                )
            has_more = await send_next_subject_menu(sender_phone, level, current_subject)
            if not has_more:
                await complete_onboarding(sender_phone)
            return True
            
        # Multi-book subject: Toggle selection
        if matched_book in preferred_books:
            # Remove it (toggle off)
            await users_col.update_one(
                {"user_id": sender_phone},
                {"$pull": {"preferred_books_list": matched_book}}
            )
        else:
            # Add it (toggle on)
            await users_col.update_one(
                {"user_id": sender_phone},
                {"$push": {"preferred_books_list": matched_book}}
            )
            
        # Re-send updated checklist menu for the same subject
        await send_subject_book_menu(sender_phone, level, current_subject)
        return True
        
async def start_interactive_quiz(sender_phone: str, topic: str, search_res: list):
    """Generates 5 structured MCQs as JSON and starts the 1-by-1 interactive quiz flow"""
    context_blocks = []
    for idx, point in enumerate(search_res[:10], 1):
        p = point.payload
        book_str = p.get('book_title', 'Textbook')
        text_str = p.get('text', '')
        context_blocks.append(f"[Chunk {idx} | Book: {book_str}]\n{text_str}")

    formatted_context = "\n\n".join(context_blocks)
    user_prompt = f"RETRIEVED TEXTBOOK CONTEXT:\n{formatted_context}\n\nTOPIC TO TEST: {topic}"

    try:
        json_raw = await call_openrouter_llm(SYSTEM_INTERACTIVE_QUIZ_PROMPT, user_prompt)
        cleaned_json = re.sub(r'```json\s*', '', json_raw)
        cleaned_json = re.sub(r'```\s*$', '', cleaned_json).strip()
        
        quiz_questions = json.loads(cleaned_json)
        
        if not isinstance(quiz_questions, list) or len(quiz_questions) == 0:
            raise ValueError("LLM did not return a valid list of questions")

        quiz_state = {
            "topic": topic,
            "questions": quiz_questions,
            "current_idx": 0,
            "score": 0
        }
        await users_col.update_one(
            {"user_id": sender_phone},
            {"$set": {"active_quiz": quiz_state}}
        )

        await send_quiz_question(sender_phone, quiz_state)

    except Exception as e:
        print(f"❌ Error starting interactive quiz: {e}")
        await send_whatsapp_cloud_msg(sender_phone, "Sorry, I had trouble creating the interactive quiz questions. Please try tapping Generate MCQs again!")

async def send_quiz_question(sender_phone: str, quiz_state: dict):
    """Sends the current question with a WhatsApp Interactive List dropdown for options A, B, C, D"""
    questions = quiz_state.get("questions", [])
    idx = quiz_state.get("current_idx", 0)
    total = len(questions)
    
    if idx >= total:
        score = quiz_state.get("score", 0)
        topic = quiz_state.get("topic", "Medical Quiz")
        percentage = int((score / total) * 100) if total > 0 else 0
        
        result_msg = (
            f"🎉 *QUIZ COMPLETE!*\n\n"
            f"📌 *Topic:* {topic}\n"
            f"📊 *Final Score:* {score}/{total} ({percentage}%)\n\n"
        )
        if percentage >= 80:
            result_msg += "🌟 Outstanding performance! You have mastered this concept."
        elif percentage >= 60:
            result_msg += "👍 Good effort! Review the citations to sharpen your knowledge."
        else:
            result_msg += "📖 Keep practicing! Ask NEURA AI to explain the topic again to strengthen your core concepts."

        result_msg += "\n\n💬 _Help us improve NEURA AI! Share quick anonymous feedback: https://forms.gle/dNr7SV5EUiqiFySx5_"

        await send_whatsapp_cloud_msg(sender_phone, result_msg)
        await users_col.update_one({"user_id": sender_phone}, {"$unset": {"active_quiz": ""}})
        return

    q = questions[idx]
    q_num = idx + 1
    book_source = q.get("book_source", "Textbook")
    vignette = q.get("vignette", "")

    question_text = (
        f"🏥 *NEURA AI MBBS Exam Quiz* (Q{q_num}/{total})\n"
        f"📚 *Source:* {book_source}\n\n"
        f"{vignette}\n\n"
        f"A) {q.get('option_a')}\n"
        f"B) {q.get('option_b')}\n"
        f"C) {q.get('option_c')}\n"
        f"D) {q.get('option_d')}"
    )

    options_list = [
        {
            "id": f"Q{q_num}_A",
            "title": "Option A",
            "description": q.get('option_a', '')[:72]
        },
        {
            "id": f"Q{q_num}_B",
            "title": "Option B",
            "description": q.get('option_b', '')[:72]
        },
        {
            "id": f"Q{q_num}_C",
            "title": "Option C",
            "description": q.get('option_c', '')[:72]
        },
        {
            "id": f"Q{q_num}_D",
            "title": "Option D",
            "description": q.get('option_d', '')[:72]
        },
        {
            "id": "EXIT_QUIZ",
            "title": "🛑 Exit Quiz",
            "description": "Stop this practice quiz and return to normal chat"
        }
    ]

    await send_whatsapp_interactive_list(
        sender_phone,
        question_text,
        "Select Option",
        options_list
    )

async def handle_quiz_answer(sender_phone: str, selected_option: str, user_doc: dict):
    """Processes the student's selected option (A, B, C, or D), provides textbook rationale, and advances to next question"""
    opt_upper = selected_option.strip().upper()
    active_quiz = user_doc.get("active_quiz") if user_doc else None

    # Handle explicit quiz exit
    if opt_upper in ["EXIT_QUIZ", "EXIT", "STOP", "QUIT", "CANCEL", "/EXIT", "STOP QUIZ", "EXIT QUIZ", "END QUIZ"]:
        if users_col is not None:
            await users_col.update_one({"user_id": sender_phone}, {"$unset": {"active_quiz": ""}})
        await send_whatsapp_cloud_msg(
            sender_phone,
            "🛑 *Practice Quiz Ended.*\n\nFeel free to ask any medical question or explore another topic whenever you're ready! 🧠⚡"
        )
        return True

    q_match = re.search(r'Q(\d+)_([A-D])', opt_upper)

    # If student taps an MCQ option dropdown after the quiz is finished/cleared
    if not active_quiz:
        if q_match:
            await send_whatsapp_cloud_msg(
                sender_phone,
                "⚠️ This quiz session has already ended! To start a new practice quiz, tap '📝 Practice MCQs' under any medical answer!"
            )
            return True
        return False

    # Check if the input is a valid MCQ choice (Q#_A or raw A, B, C, D, 1, 2, 3, 4, Option A)
    # If the user typed a new question (e.g. "What causes acute pancreatitis?"), AUTO-EXIT the quiz cleanly
    valid_raw_choices = {"A": "A", "B": "B", "C": "C", "D": "D", "OPTION A": "A", "OPTION B": "B", "OPTION C": "C", "OPTION D": "D", "1": "A", "2": "B", "3": "C", "4": "D"}
    
    choice = None
    if q_match:
        tapped_q_num = int(q_match.group(1))
        choice = q_match.group(2)
        current_q_num = active_quiz.get("current_idx", 0) + 1
        
        if tapped_q_num != current_q_num:
            await send_whatsapp_cloud_msg(
                sender_phone,
                f"⚠️ You have already answered Question {tapped_q_num}! Please select your answer for Question {current_q_num} below."
            )
            return True
    elif opt_upper in valid_raw_choices:
        choice = valid_raw_choices[opt_upper]
    else:
        # Non-option message: Student is moving on or asking a new question -> auto-exit quiz and let message process normally!
        if users_col is not None:
            await users_col.update_one({"user_id": sender_phone}, {"$unset": {"active_quiz": ""}})
        print(f"ℹ️ User {sender_phone} sent non-MCQ input during quiz: '{selected_option}'. Auto-exiting quiz session.")
        return False

    questions = active_quiz.get("questions", [])
    idx = active_quiz.get("current_idx", 0)
    score = active_quiz.get("score", 0)

    if idx >= len(questions):
        if q_match:
            await send_whatsapp_cloud_msg(
                sender_phone,
                "⚠️ This quiz session has already ended! To start a new practice quiz, tap '📝 Practice MCQs' under any medical answer!"
            )
            return True
        return False

    q = questions[idx]
    correct = q.get("correct_option", "A").upper().strip()
    explanation = q.get("explanation", "")
    book_source = q.get("book_source", "Textbook")

    is_correct = (choice == correct)
    if is_correct:
        score += 1
        feedback_header = f"✅ *CORRECT!* (Option {correct})"
    else:
        feedback_header = f"❌ *INCORRECT!* (Your Choice: {choice} | Correct Answer: Option {correct})"

    feedback_msg = (
        f"{feedback_header}\n\n"
        f"📖 *Textbook Rationale ({book_source}):*\n{explanation}"
    )

    await send_whatsapp_cloud_msg(sender_phone, feedback_msg)

    active_quiz["current_idx"] = idx + 1
    active_quiz["score"] = score

    await users_col.update_one(
        {"user_id": sender_phone},
        {"$set": {"active_quiz": active_quiz}}
    )

    await send_quiz_question(sender_phone, active_quiz)
    return True

# User-level sequential lock to prevent race conditions on simultaneous messages from the same user
_user_locks: dict[str, asyncio.Lock] = {}

def get_user_lock(user_id: str) -> asyncio.Lock:
    if user_id not in _user_locks:
        _user_locks[user_id] = asyncio.Lock()
    return _user_locks[user_id]

async def process_whatsapp_message(sender_phone: str, user_msg: str, is_tagged_reply: bool = False):
    """Background task wrapper to process messages sequentially per user lock"""
    lock = get_user_lock(sender_phone)
    async with lock:
        await _process_whatsapp_message_internal(sender_phone, user_msg, is_tagged_reply)

async def _process_whatsapp_message_internal(sender_phone: str, user_msg: str, is_tagged_reply: bool = False):
    """Internal task to run RAG & OpenRouter LLM and send WhatsApp reply"""
    try:
        # Fetch user doc early to get preferences
        user_doc = None
        preferred_books_list = []
        name = "Student"
        level = "Unknown Level"
        if users_col is not None:
            user_doc = await users_col.find_one({"user_id": sender_phone})
            if user_doc:
                name = user_doc.get("name", "Student")
                level = user_doc.get("level", "Unknown Level")
                preferred_books_list = user_doc.get("preferred_books_list", [])

        # Check for profile commands or menu first
        msg_lower = user_msg.strip().lower()
        if msg_lower in ["/", "/help", "help", "menu", "commands", "/menu", "/commands", "/start"]:
            await send_commands_menu(sender_phone)
            return

        if msg_lower.startswith("/") and users_col is not None:
            if msg_lower == "/reset":
                await users_col.delete_one({"user_id": sender_phone})
                if chat_history_col is not None:
                    await chat_history_col.delete_one({"user_id": sender_phone})
                await send_whatsapp_interactive_button(
                    sender_phone,
                    "✅ Your profile and chat history have been completely reset!\n\nTap the button below to set up your profile:",
                    [{"id": "START_ONBOARDING", "title": "🚀 Start Setup"}]
                )
                return
            elif msg_lower == "/profile":
                books_str = "\n  - ".join(preferred_books_list) if preferred_books_list else "None"
                await send_whatsapp_cloud_msg(
                    sender_phone, 
                    f"👤 *Your Profile*\n• Name: {name}\n• Level: {level}\n• Books:\n  - {books_str}\n\n"
                    f"📝 *Feedback Survey:* https://forms.gle/dNr7SV5EUiqiFySx5"
                )
                return
            elif msg_lower == "/feedback":
                feedback_msg = (
                    "📝 *NEURA AI Beta Feedback Survey*\n\n"
                    "Your feedback helps us make NEURA AI 10x better for medical students!\n\n"
                    "This survey is 100% anonymous (takes under 2 minutes):\n"
                    "👉 https://forms.gle/dNr7SV5EUiqiFySx5\n\n"
                    "Thank you for beta testing NEURA AI! 🧠⚡"
                )
                await send_whatsapp_cloud_msg(sender_phone, feedback_msg)
                return
            elif msg_lower == "/update name":
                await users_col.update_one({"user_id": sender_phone}, {"$set": {"onboarding_step": "ASK_NAME"}})
                await send_whatsapp_cloud_msg(sender_phone, "What would you like to change your name to?")
                return
            elif msg_lower == "/update level":
                await users_col.update_one({"user_id": sender_phone}, {"$set": {"onboarding_step": "ASK_LEVEL"}})
                await send_whatsapp_interactive_list(
                    sender_phone, 
                    "What is your new medical class/level?",
                    "Select Level",
                    ["200L", "300L", "400L", "500L", "600L"]
                )
                return
            elif msg_lower == "/update books":
                await users_col.update_one({"user_id": sender_phone}, {"$set": {"preferred_books_list": []}})
                has_subjects = await send_next_subject_menu(sender_phone, level)
                if not has_subjects:
                    await complete_onboarding(sender_phone)
                return

        # Handle active interactive quiz answer if student is answering an MCQ
        if user_doc and "active_quiz" in user_doc:
            handled = await handle_quiz_answer(sender_phone, user_msg, user_doc)
            if handled:
                return

        # Handle onboarding state machine
        is_onboarding = await handle_onboarding(sender_phone, user_msg)
        if is_onboarding:
            return

        query_to_search = user_msg
        is_button_quiz = user_msg.startswith("GENERATE_QUIZ:")
        if is_button_quiz:
            query_to_search = user_msg.replace("GENERATE_QUIZ:", "").strip()
            intent = "QUIZ"
        else:
            intent = classify_intent(user_msg)
        
        if intent == "GREETING":
            books_formatted = "\n• ".join(preferred_books_list) if preferred_books_list else "Your selected medical textbooks"
            greeting_msg = (
                f"Welcome to *NEURA AI* — Your Personal Medical Co-Pilot! 🧠⚡\n\n"
                f"I am engineered to transform your medical textbooks into high-yield exam insights, clinical breakdowns, and interactive MBBS practice quizzes!\n\n"
                f"📚 *Your Active Study Library:*\n• {books_formatted}\n\n"
                f"What medical topic, clinical case, or concept are we mastering today?"
            )
            await send_whatsapp_cloud_msg(sender_phone, greeting_msg)
            return

        # Check if the user query is a tagged reply OR a conversational follow-up
        is_followup = is_tagged_reply or check_is_followup_query(query_to_search)
        
        last_topic = None
        last_assistant_msg = None
        if is_followup and chat_history_col is not None:
            user_hist = await chat_history_col.find_one({"user_id": sender_phone})
            if user_hist and "messages" in user_hist:
                msgs = user_hist["messages"]
                for msg_item in reversed(msgs):
                    if msg_item.get("role") == "assistant" and not last_assistant_msg:
                        last_assistant_msg = msg_item.get("content")
                    if msg_item.get("role") == "user" and not last_topic:
                        content = msg_item.get("content", "")
                        if not check_is_followup_query(content) and not content.startswith("GENERATE_QUIZ"):
                            last_topic = content

        if is_followup:
            if last_topic or last_assistant_msg:
                search_term = last_topic if last_topic else query_to_search
                print(f"[Follow-up Router] (Tagged={is_tagged_reply}) Resolved query '{query_to_search}' to topic: '{search_term}'")
            else:
                prompt_msg = (
                    "What medical topic, clinical case, or concept would you like to learn more about?\n\n"
                    "Type a specific subject or drug (e.g., *Prazosin*, *MEN1A*, *Antibiotics*) and I'll pull exact details from your textbooks!"
                )
                await send_whatsapp_cloud_msg(sender_phone, prompt_msg)
                return
        else:
            search_term = query_to_search

        # Step 1: Zero-shot Micro-LLM Query Normalization & Typo Resolution
        normalized_data = await normalize_medical_query(search_term)
        medical_terms = normalized_data.get("search_keywords", [])
        diagram_target = normalized_data.get("diagram_topic")
        if not medical_terms:
            medical_terms = extract_medical_terms(search_term)
        # If LLM gave no clean diagram topic, use medical_terms as diagram candidates
        diagram_candidates = ([diagram_target] if diagram_target else []) + medical_terms
        
        # Step 1.5: Check for explicit book overrides (e.g. if user says "Use pharmacology")
        active_books = get_explicit_book_override(search_term, preferred_books_list)
        
        # Step 2: Multi-search Qdrant with clean medical terms, filtered by active books
        search_res = await multi_search_qdrant(medical_terms, preferred_books=active_books)

        if not search_res:
            await send_whatsapp_cloud_msg(sender_phone, "I couldn't find relevant textbook material for your question in your selected textbooks. Try rephrasing or updating your preferred books using /update books!")
            return

        # If button click [ 📝 Generate MCQs ] was tapped, launch the 1-by-1 interactive quiz!
        if is_button_quiz:
            extracted_terms = extract_medical_terms(query_to_search)
            clean_topic = ", ".join(extracted_terms) if extracted_terms else query_to_search
            clean_topic = clean_topic.title()
            await start_interactive_quiz(sender_phone, clean_topic, search_res)
            return

        context_blocks = []
        for idx, point in enumerate(search_res, 1):
            p = point.payload
            page_str = p.get('page_number') or p.get('chunk_index', 'N/A')
            book_str = p.get('book_title', 'Textbook')
            text_str = p.get('text', '')
            block = f"[Context {idx} | Book: {book_str}, Page/Chunk: {page_str}]\n{text_str}"
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)
        
        if is_tagged_reply and last_assistant_msg:
            tagged_snippet = last_assistant_msg[:400]
            user_prompt = (
                f"THE USER EXPLICITLY TAGGED/QUOTED YOUR PREVIOUS WHATSAPP MESSAGE BELOW:\n\"\"\"{tagged_snippet}\"\"\"\n\n"
                f"RETRIEVED TEXTBOOK CONTEXT:\n{formatted_context}\n\n"
                f"USER'S QUESTION/INSTRUCTION REGARDING THE TAGGED MESSAGE:\n{query_to_search}\n\n"
                f"CRITICAL INSTRUCTION: Jump straight into the answer. Do NOT start your response with 'Based on the retrieved context', 'According to', or any similar robotic phrasing. Just speak naturally."
            )
        else:
            user_prompt = (
                f"RETRIEVED TEXTBOOK CONTEXT:\n{formatted_context}\n\n"
                f"STUDENT QUESTION:\n{query_to_search}\n\n"
                f"CRITICAL INSTRUCTION: Jump straight into the answer. Do NOT start your response with 'Based on the retrieved context', 'According to', or any similar robotic phrasing. Just speak naturally."
            )
        
        # Build dynamic user context
        books_str = ", ".join(preferred_books_list) if preferred_books_list else "None"
        user_context_str = f"The student asking this question is {name}, a {level} medical student. Their preferred textbooks are: {books_str}. Tailor your explanation to their level.\n\n"

        prompt_to_use = SYSTEM_QUIZ_PROMPT if intent == "QUIZ" else SYSTEM_MEDICAL_PROMPT
        prompt_to_use = prompt_to_use.replace("{user_context}", user_context_str)

        # Chat memory
        chat_history = []
        if chat_history_col is not None:
            user_doc = await chat_history_col.find_one({"user_id": sender_phone})
            if user_doc and "messages" in user_doc:
                chat_history = user_doc["messages"][-6:]

        if intent == "QUIZ":
            ai_answer = await call_openrouter_llm(prompt_to_use, user_prompt, chat_history)
            await send_whatsapp_cloud_msg(sender_phone, ai_answer)
        else:
            ai_answer = await stream_openrouter_llm_to_whatsapp(prompt_to_use, user_prompt, sender_phone, chat_history)

        if chat_history_col is not None:
            new_msgs = [
                {"role": "user", "content": query_to_search},
                {"role": "assistant", "content": ai_answer}
            ]
            await chat_history_col.update_one(
                {"user_id": sender_phone},
                {"$push": {"messages": {"$each": new_msgs}}},
                upsert=True
            )

        # Check if the answer indicates information is missing from textbooks
        ai_lower = ai_answer.lower()
        is_not_covered = ("not covered" in ai_lower or "sorry" in ai_lower[:30] or "not found" in ai_lower)

        # Retrieve and send authentic peer-reviewed medical diagram / histology slide if topic is visual or requested
        if intent != "QUIZ" and not is_not_covered and should_generate_medical_illustration(query_to_search, ai_answer):
            try:
                img_url, img_title = await retrieve_real_medical_diagram(diagram_candidates)
                if img_url:
                    display_topic = (diagram_candidates[0] if diagram_candidates else query_to_search)[:60]
                    img_caption = f"🔬 *Authentic Medical Figure:* _{img_title or display_topic}_\n📚 _Peer-Reviewed Scientific & Textbook Archive_"
                    await send_whatsapp_image_url(sender_phone, img_url, img_caption)
            except Exception as img_err:
                print(f"⚠️ Non-critical error sending medical illustration: {img_err}")

        # Attach interactive follow-up button for quick MCQ generation ONLY if it was a valid medical answer
        if intent != "QUIZ" and not user_msg.startswith("GENERATE_QUIZ") and not is_not_covered:
            try:
                topic_snippet = query_to_search[:50]
                await send_whatsapp_interactive_button(
                    sender_phone,
                    "Ready to practice MCQs on this topic?",
                    [
                        {"id": f"GENERATE_QUIZ:{topic_snippet}", "title": "📝 Practice MCQs"}
                    ]
                )
            except Exception as btn_err:
                print(f"⚠️ Non-critical error sending interactive button: {btn_err}")

    except Exception as e:
        print(f"ERROR in process_whatsapp_message: {str(e)}")
        print(traceback.format_exc())
        await send_whatsapp_cloud_msg(sender_phone, "Sorry, NEURA AI experienced a temporary connection delay. Please try asking your medical question again!")

# ==========================================
# 3. ENDPOINTS
# ==========================================
@app.get("/")
@app.head("/")
def root():
    return {
        "status": "online",
        "system": "NEURA AI Official WhatsApp Cloud API Backend v2.0",
        "phone_number_id": PHONE_NUMBER_ID,
        "openrouter_configured": bool(OPENROUTER_API_KEY)
    }

@app.get("/api/books")
def get_books():
    """Debug endpoint to list all unique book_title payloads in Qdrant"""
    try:
        records, _ = qdrant.scroll(
            collection_name=COLLECTION_NAME,
            limit=250,
            with_payload=True,
            with_vectors=False
        )
        titles = list(set(r.payload.get("book_title") for r in records if r.payload and "book_title" in r.payload))
        return {"books_found": len(titles), "titles": sorted(titles)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/webhook")
async def verify_webhook(request: Request):
    """Meta Webhook Verification Endpoint"""
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        print("✅ Meta Webhook Verification Successful!")
        return Response(content=challenge, media_type="text/plain")
    else:
        print("❌ Meta Webhook Verification Failed: Invalid Token")
        raise HTTPException(status_code=403, detail="Verification token mismatch")

@app.post("/webhook")
async def handle_whatsapp_webhook(request: Request):
    """Incoming WhatsApp Message Webhook Endpoint from Meta"""
    try:
        body = await request.json()
        print(f"📩 Webhook Payload: {json.dumps(body)}")

        # Extract message payload
        entries = body.get("entry", [])
        for entry in entries:
            changes = entry.get("changes", [])
            for change in changes:
                value = change.get("value", {})
                messages = value.get("messages", [])
                for msg in messages:
                    sender_phone = msg.get("from")
                    msg_type = msg.get("type")
                    msg_id = msg.get("id")
                    
                    if msg_id:
                        asyncio.create_task(mark_message_as_read(msg_id))
                    
                    context_obj = msg.get("context", {})
                    is_tagged_reply = bool(context_obj.get("id"))
                    
                    text_body = ""
                    if msg_type == "text":
                        text_body = msg.get("text", {}).get("body", "")
                    elif msg_type == "interactive":
                        interactive_obj = msg.get("interactive", {})
                        interactive_type = interactive_obj.get("type")
                        if interactive_type == "list_reply":
                            text_body = interactive_obj.get("list_reply", {}).get("id") or interactive_obj.get("list_reply", {}).get("title", "")
                        elif interactive_type == "button_reply":
                            text_body = interactive_obj.get("button_reply", {}).get("id") or interactive_obj.get("button_reply", {}).get("title", "")

                    if text_body:
                        print(f"📩 Received msg ({msg_type}) from {sender_phone} (Tagged Reply: {is_tagged_reply}): '{text_body}'")
                        # Process in background task to respond to Meta immediately (prevents timeout)
                        task = BackgroundTask(process_whatsapp_message, sender_phone, text_body, is_tagged_reply)
                        return Response(content=json.dumps({"status": "processing"}), media_type="application/json", background=task)
                    else:
                        print(f"⚠️ Received unsupported message type '{msg_type}' from {sender_phone}")
                        task = BackgroundTask(send_whatsapp_cloud_msg, sender_phone, "I only read text messages right now! Please type out your medical question. 🤖📚")
                        return Response(content=json.dumps({"status": "unsupported_media"}), media_type="application/json", background=task)

        return Response(content=json.dumps({"status": "ignored"}), media_type="application/json")
    except Exception as e:
        print(f"Error handling webhook: {e}")
        return Response(content=json.dumps({"status": "error"}), media_type="application/json")

@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    """API endpoint for direct HTTP queries (e.g. testing or web frontends)"""
    try:
        user_msg = req.message.strip()
        intent = classify_intent(user_msg)
        
        if intent == "GREETING":
            return {
                "response": "Hello! 👋 I'm *NEURA AI*, your medical study assistant.\n\nI can answer medical questions directly from your textbooks (*Lippincott Pharmacology*, *Hoffbrand's Haematology*, etc.) with exact citations, or generate practice MCQs for your MBBS exams!\n\nWhat concept are we studying today?"
            }
        
        # Step 1: Extract medical terms from the user's message
        medical_terms = extract_medical_terms(user_msg)
        
        # Step 2: Multi-search Qdrant with extracted terms + original query
        if medical_terms:
            search_res = multi_search_qdrant(medical_terms)
        else:
            search_res = search_qdrant(user_msg, limit=4)
        
        if not search_res:
            return {
                "response": "I couldn't find relevant textbook material for your question. Please try asking a specific medical topic!"
            }
        
        context_blocks = []
        for idx, point in enumerate(search_res, 1):
            p = point.payload
            block = f"[Context {idx} | Book: {p['book_title']}, Page {p['page_number']}]\n{p['text']}"
            context_blocks.append(block)
        
        formatted_context = "\n\n".join(context_blocks)
        user_prompt = f"RETRIEVED TEXTBOOK CONTEXT:\n{formatted_context}\n\nSTUDENT QUESTION:\n{user_msg}"
        
        # Build dynamic user context
        user_context_str = ""
        if users_col is not None:
            user_doc = await users_col.find_one({"user_id": req.user_id})
            if user_doc:
                name = user_doc.get("name", "Student")
                level = user_doc.get("level", "Unknown Level")
                books = user_doc.get("preferred_books", "Unknown")
                user_context_str = f"The student asking this question is {name}, a {level} medical student. Their preferred textbooks are: {books}. Tailor your explanation to their level.\n\n"

        prompt_to_use = SYSTEM_QUIZ_PROMPT if intent == "QUIZ" else SYSTEM_MEDICAL_PROMPT
        prompt_to_use = prompt_to_use.replace("{user_context}", user_context_str)
        
        chat_history = []
        if chat_history_col is not None:
            user_doc = await chat_history_col.find_one({"user_id": req.user_id})
            if user_doc and "messages" in user_doc:
                chat_history = user_doc["messages"][-6:]
        
        ai_answer = await call_openrouter_llm(prompt_to_use, user_prompt, chat_history)
        
        if chat_history_col is not None:
            new_msgs = [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ai_answer}
            ]
            await chat_history_col.update_one(
                {"user_id": req.user_id},
                {"$push": {"messages": {"$each": new_msgs}}},
                upsert=True
            )
        
        return {
            "intent": intent,
            "response": ai_answer
        }
    except Exception as e:
        print(f"ERROR in chat_endpoint: {str(e)}")
        print(traceback.format_exc())
        return {
            "response": f"NEURA AI encountered an error processing your query: {str(e)}. Please check backend API configuration!"
        }
