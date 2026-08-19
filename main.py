import os
import re
import json
import logging
import traceback
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.background import BackgroundTask
from pydantic import BaseModel
from fastembed import TextEmbedding

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
2. NO ROBOT TALK & NO PREAMBLES: Never use opening filler, greetings, or announcements (e.g., "Certainly Samuel!", "Certainly!", "Absolutely!", "Sure thing!", "Here is the figure you requested", "I have attached the authentic textbook figure below", "Based on the retrieved context...", "According to this textbook..."). Jump DIRECTLY into the medical explanation starting immediately with 📖 *IN-DEPTH EXPLANATION*. Zero conversational filler.
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
6. ZERO FABRICATED FIGURE CITATIONS & CLEAN CITATIONS: Absolutely NEVER invent, cite, or mention specific figure numbers, diagram numbers, plate numbers, or table numbers (e.g., NEVER write "Figure 46-9", "Fig 12.8", "Figure 43.5", "Plate 3-1", "Table 14.2", "Robbins p. 787"). The attached diagrams and flowcharts are automatically delivered by the system; do NOT fabricate or cite figure numbers in your response text. All textbook citations MUST appear strictly at the very end under 📚 CITATIONS listing ONLY the authentic textbook title (e.g. "- Robbins Basic Pathology" or "- Lippincott Illustrated Reviews: Pharmacology"). Never include page numbers, figure numbers, or conversational commentary in citations.
7. NO SMASHED WORDS: Ensure flawless spacing between words, punctuation, and hyphens. Always put a space after colons, periods, and bold words (e.g. write "*Prazosin* is" instead of "*Prazosin*is").
8. Structure responses into clear sections separated by blank lines:
   📖 *IN-DEPTH EXPLANATION*
   
   💡 *KEY CLINICAL PEARLS*
   
   📚 *CITATIONS*
9. NO RAW MARKDOWN TABLES: WhatsApp does NOT render markdown tables. NEVER output pipes or table headers (| Col 1 | Col 2 | or |---|---|). Always structure comparisons and summary tables as clean bulleted list cards (e.g. - *Category:* followed by indented • *Detail:* bullets).
10. WHEN ASKED FOR DIAGRAMS/ILLUSTRATIONS: NEVER apologize or say 'I cannot generate or display diagrams' or 'I am only a text AI'. You ARE fully equipped with a real medical diagram and flowchart retrieval system that automatically delivers the authentic textbook schematic below your explanation. Confidently provide the structured textbook breakdown with clear headings and bullet cards. Do NOT announce or refer to figure numbers (e.g. do NOT say "I've attached Figure X-Y below" or "refer to the figure below") — the system delivers the visual asset seamlessly.
"""

SYSTEM_QUIZ_PROMPT = """{user_context}You are NEURA AI. Based ONLY on the retrieved medical textbook context, generate exactly 7 rigorous, medical-school standard (MBBS / USMLE Step 1 & 2 style) Multiple Choice Questions (MCQs).

RULES FOR MCQs:
1. NO PREAMBLES & NO CONVERSATIONAL FILLER: Start immediately with Question 1. Never include introductory conversational chatter, greetings, or announcements.
2. Each question must present a realistic clinical vignette, physiological mechanism, or pharmacological scenario appropriate for medical students. Never cite fabricated figure or table numbers in vignettes.
3. Provide 4 distinct options (A, B, C, D) for each of the 7 questions.
4. Structure your response clearly using WhatsApp Markdown:
   - List Question 1 through 7 with their options (A, B, C, D).
   - Provide a separate 🔑 *ANSWER KEY & DETAILED EXPLANATIONS* section at the bottom.
   - For every answer, explain why the correct option is right AND why the key distractor options are wrong, citing the specific textbook title. Never include fabricated figure numbers or page numbers in explanations or citations.
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
                headers = [h.strip('*_# ') for h in table_rows[0]]
                data_rows = table_rows[1:] if len(table_rows) > 1 else []
                output_lines.append("")
                for r in data_rows:
                    if not r or not any(r): continue
                    primary_title = r[0].strip('*_# ')
                    output_lines.append(f"- *{primary_title}*")
                    for c_idx in range(1, min(len(headers), len(r))):
                        col_name = headers[c_idx] if c_idx < len(headers) and headers[c_idx] else f"Detail {c_idx}"
                        val = r[c_idx].strip()
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

def strip_conversational_preambles(text: str) -> str:
    """Deterministically strips opening conversational filler, greetings, and robotic figure announcements."""
    if not text:
        return text

    # Remove markdown header hashes first from the text
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)

    preamble_patterns = [
        # Greeting + announcement (e.g. "Hello! I am attaching the diagram", "### *Certainly!* As requested, below is...", "**Certainly Samuel!** Here is...")
        r'^\s*(?:[#*_\s]*)(?:certainly|absolutely|sure|sure thing|of course|hello|hi|hey|greetings)[\s\w,!.*#_:\'-]*?(?:here is|here are|i(?:\x27ve|\x20have|\x20am|\x27m)\s+(?:attached|attaching|providing)|attached is|let(?:\x27s|\x20us)\s+(?:dive into|examine|look at|explore)|below is|as requested|the breakdown)[^\n]*(?:\n+|\:\s*|\.\s*)',
        # Standalone opening greeting or filler (e.g. "**Certainly, Samuel!**", "*Hello Samuel!*", "### Certainly!", "_Certainly!_", "the answer is:", "here is the answer:")
        r'^\s*(?:[#*_\s]*)(?:certainly|absolutely|sure|sure thing|of course|hello|hi|hey|greetings|the answer is|here is the answer)[,\s!\*#_.:]+(?:\w+[,\s!\*#_.:]+){0,3}(?:\n+|$)',
        # Direct figure / diagram / explanation announcement (e.g. "**Here is the requested diagram...**", "I've attached the authentic textbook figure below...")
        r'^\s*(?:[#*_\s]*)(?:here is|here are|below is|i have attached|i\x27ve attached|i am attaching|i\x27m attaching|attached is)\s+[^\n]*(?:\n+|\:\s*|\.\s*)',
        # Context grounding opener (e.g. "**Based on the retrieved context from Lippincott...**", "*According to the textbook context:*")
        r'^\s*(?:[#*_\s]*)(?:based on (?:the )?(?:retrieved )?(?:textbook )?(?:context|material|information)[^\n,.]*|according to (?:the )?(?:retrieved )?(?:textbook )?(?:context|material|information)[^\n,.]*)[\*#_\s:,.]*(?:\n+|$)?',
    ]

    modified = True
    while modified:
        modified = False
        for pat in preamble_patterns:
            new_text = re.sub(pat, '', text, flags=re.IGNORECASE).lstrip()
            if new_text != text:
                text = new_text
                modified = True
                break

    return text

def strip_figure_citations(text: str) -> str:
    """Eradicates hallucinated figure numbers, plate numbers, and table citations across sentences and citations."""
    if not text:
        return text

    # 1. Trailing figure/page tags in citations section (e.g. "- Robbins Pathology, p. 612", "- Robbins Pathology, Figure 12.8")
    text = re.sub(
        r'(?mi)^(-\s+[^\n]+?)(?:,\s*(?:figure|fig\.?|table|plate|chart|p\.|page|pp\.)\s*\d+[^\r\n]*)$',
        r'\1',
        text
    )

    # 2. Parenthetical citations: (see Figure 46-9 from Jawetz), (refer to Fig. 12.8 for details), (Figure 12.8), (Table 14.2), (Plate 3-1)
    text = re.sub(
        r'(?i)\(\s*(?:see\s+|refer to\s+|as shown in\s+|as seen in\s+|shown in\s+)?(?:figure|fig\.?|table|plate|chart)\s+\d+[-.:]\d+[a-z]?(?:\s+(?:from|in|of)\s+[A-Za-z\s]+)?(?:\s+(?:below|above))?(?:\s+for\s+[^)\n]*)?\s*\)',
        '',
        text
    )

    # 3. Introductory / transitional clauses mentioning figures:
    text = re.sub(
        r'(?i)\b(?:as\s+(?:shown|illustrated|seen|depicted|noted)\s+(?:in|on)\s+)?(?:figure|fig\.?|table|plate|chart)\s+\d+[-.:]\d+[a-z]?(?:\s+(?:from|in|of)\s+[A-Za-z\s]+)?(?:\s+(?:below|above))?(?:,\s*|\s+demonstrates\s+|\s+shows\s+|\s+illustrates\s+)',
        ' ',
        text
    )

    # 4. Dangling "as depicted in.", "as shown in."
    text = re.sub(r'(?i)\b(?:as\s+(?:shown|illustrated|seen|depicted|noted)\s+(?:in|on))\s*(?=[.,;:!?\n]|\s*$)', '', text)

    # 5. Directive phrases / sentences: "refer to Figure 12.8 for details", "see Fig 43-5"
    text = re.sub(
        r'(?i)\b(?:refer to|see)\s+(?:figure|fig\.?|table|plate|chart)\s+\d+[-.:]\d+[a-z]?(?:\s+(?:from|in|of)\s+[A-Za-z\s]+)?(?:\s+(?:below|above))?(?:\s+for\s+[^\n.]*)?',
        '',
        text
    )

    # 6. Standalone figure mentions: "Figure 46-9 from Jawetz", "Figure 12.8", "Fig. 43-5", "Table 14.2", "Plate 3-1"
    text = re.sub(
        r'(?i)\b(?:figure|fig\.?|table|plate|chart)\s+\d+[-.:]\d+[a-z]?(?:\s+(?:from|in|of)\s+[A-Za-z]+)?',
        '',
        text
    )

    # Clean up artifacts (only collapse multiple spaces inside line, preserving leading indentation)
    text = re.sub(r'(?<=\S) {2,}', ' ', text)
    text = re.sub(r' \.', '.', text)
    text = re.sub(r' ,', ',', text)
    text = re.sub(r'\(\s*\)', '', text)

    cleaned_lines = []
    for l in text.split('\n'):
        if re.match(r'^\s*[.,:;!?\- ]*\s*$', l) and not l.strip().startswith('-'):
            continue
        cleaned_lines.append(l)

    return '\n'.join(cleaned_lines)

def format_whatsapp_text(text: str) -> str:
    """Master WhatsApp text sanitizer. Fixes layout, tables, preambles, figure citations, and bolding without destroying text."""
    if not text:
        return text

    # 0. Convert raw markdown tables to readable bullet cards & remove --- lines
    text = convert_markdown_tables_to_whatsapp_cards(text)

    # 1. Deterministically strip opening preambles & conversational greetings
    text = strip_conversational_preambles(text)

    # 2. Deterministically strip fabricated figure/table citations
    text = strip_figure_citations(text)

    # 3. Remove markdown hashes (e.g. ### Header -> Header)
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = text.replace('###', '').replace('##', '')

    # 3.5. Ensure section headers with emojis are formatted with bold tags (e.g. 📖 IN-DEPTH EXPLANATION -> 📖 *IN-DEPTH EXPLANATION*)
    text = re.sub(r'(?m)^([📖💡📚🎯🔑])\s*([^*\n]+?)\s*$', r'\1 *\2*', text)

    # 4. Fix double asterisks **text** -> *text*
    text = re.sub(r'\*\*(.*?)\*\*', r'*\1*', text)

    # 5. Add space around bold text if attached to words, without corrupting inside bold
    text = re.sub(r'(?<!\*|\s)([a-zA-Z0-9,;:.!?])(\*[^\s\*][^\*\n]*\*)', r'\1 \2', text)
    text = re.sub(r'(\*[^\s\*][^\*\n]*\*)(?!\*|\s|[:;,!?])([a-zA-Z0-9])', r'\1 \2', text)

    # 6. Fix missing space after colons/commas inside bold or right after bold
    text = re.sub(r'(\*[^\s\*][^\*\n]*\*[:;,])([^\s\n])', r'\1 \2', text)

    # 7. Ensure space after bullet hyphens
    text = re.sub(r'^-([a-zA-Z0-9\*])', r'- \1', text, flags=re.MULTILINE)

    # 8. Ensure double-line spacing before list items
    text = re.sub(r'([^\n])\n(-|\*|[0-9]+\.)\s+', r'\1\n\n\2 ', text)

    # 9. Add double newline before section header emojis
    text = re.sub(r'([^\n])\s*([📖💡📚🎯🔑])', r'\1\n\n\2', text)

    # 10. FINAL PASS: Clean up orphan asterisks safely
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

    # 11. Normalize multiple blank lines
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
    """Sends an image to WhatsApp via Meta Cloud API using direct binary media upload (guaranteeing zero 404/429 hotlink failures)"""
    if not image_url or not WHATSAPP_TOKEN:
        return

    meta_media_url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/media"
    messages_url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    auth_headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}"}

    # Step 1: Download image bytes server-side with custom User-Agent to bypass Wikimedia bot blocks
    image_bytes = None
    content_type = "image/jpeg"
    
    try:
        req_headers = {
            "User-Agent": "NeuraAI-MedicalBot/2.0 (contact: info@neura.ai; MBBS study assistant)",
            "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8"
        }
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True, headers=req_headers) as downloader:
            r = await downloader.get(image_url)
            if r.status_code == 200 and len(r.content) > 1000:
                image_bytes = r.content
                content_type = r.headers.get("content-type", "image/jpeg").split(";")[0]
                if "svg" in content_type:
                    content_type = "image/png"
            else:
                print(f"⚠️ Image download failed with HTTP {r.status_code} for {image_url}")
    except Exception as fetch_err:
        print(f"⚠️ Error downloading image server-side: {fetch_err}")

    # Step 2: If downloaded, upload directly to WhatsApp's media endpoint to obtain a robust media_id
    if image_bytes:
        try:
            filename = image_url.split("/")[-1].split("?")[0] or "medical_diagram.jpg"
            if not any(filename.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"]):
                filename += ".jpg"
                
            files = {
                "file": (filename, image_bytes, content_type)
            }
            data = {
                "messaging_product": "whatsapp",
                "type": content_type
            }
            async with httpx.AsyncClient(timeout=25.0) as uploader:
                upload_res = await uploader.post(meta_media_url, headers=auth_headers, files=files, data=data)
                if upload_res.status_code == 200:
                    media_id = upload_res.json().get("id")
                    if media_id:
                        # Send using media_id (100% reliable, no 404/403/429 hotlink failures on WhatsApp)
                        payload = {
                            "messaging_product": "whatsapp",
                            "recipient_type": "individual",
                            "to": to_number,
                            "type": "image",
                            "image": {
                                "id": media_id,
                                "caption": caption[:1024] if caption else ""
                            }
                        }
                        async with httpx.AsyncClient(timeout=20.0) as sender:
                            send_res = await sender.post(messages_url, headers={"Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}", "Content-Type": "application/json"}, json=payload)
                            print(f"Meta Media-ID Send Status {send_res.status_code}: {send_res.text}")
                            return
        except Exception as upload_err:
            print(f"⚠️ Media upload to Meta failed: {upload_err}")

    # Step 3: Fallback to link payload only if server-side upload could not be performed
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
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(messages_url, headers={"Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}", "Content-Type": "application/json"}, json=payload)
            print(f"Meta Image Link Send Status {res.status_code}: {res.text}")
    except Exception as e:
        print(f"⚠️ Error sending WhatsApp image fallback: {e}")

REJECT_MICROGRAPH_REGEX = re.compile(
    r'(?i)(micrograph|photomicrograph|histolog|histopatholog|biopsy|'
    r'h&e|h[_\s\-]and[_\s\-]e|h\s*and\s*e|he[_\s\-]stain|hematoxylin|\beosin\b|eosin[_\s\-]stain|giemsa|gram[_\s\-]stain|'
    r'smear|blood[_\s\-]film|thin[_\s\-]film|thick[_\s\-]film|blood[_\s\-]smear|thin[_\s\-]smear|thick[_\s\-]smear|'
    r'pap[_\s\-]smear|cytolog|frozen[_\s\-]section|immunohistochem|\bihc\b|ihc[_\s\-]stain|'
    r'pathology[_\s\-]slide|microscopic[_\s\-]slide|'
    r'light[_\s\-]microscop|electron[_\s\-]microscop|'
    r'\b(sem|tem)\b|[_\-](sem|tem)[_\-]|400x|100x|1000x|\b\d{2,4}x\b|magnification|specimen|gross[_\s\-]pathology|'
    r'macroscopic|autopsy|cadaver|clinical[_\s\-]photo|patient[_\s\-]photo|endoscopy|colonoscopy|'
    r'single[_\s\-]cell|cell[_\s\-]crop|slide[_\s\-]stain|stained[_\s\-]slide|leishman[_\s\-]stain|wright[_\s\-]stain|'
    r'fluorescence[_\s\-]microscop|confocal)'
)

def _reject_micrograph_candidate(title: str, url: str, modality: str) -> bool:
    """
    Enforces strict rejection of micrographs, histology stains, blood smears, biopsy slides,
    single-cell crops, and macroscopic specimen photos when requesting flowcharts, schematics, or anatomical maps.
    When modality == 'HISTOLOGY_MICROSCOPY', accepts genuine histology/microscopy images.
    """
    if modality == "HISTOLOGY_MICROSCOPY":
        return False
    
    text_to_check = f"{title or ''} {url or ''}".lower()
    if REJECT_MICROGRAPH_REGEX.search(text_to_check):
        return True
        
    return False

def detect_visual_intent_modality(user_msg: str, ai_answer: str = "") -> str:
    """
    Detects the visual intent modality from user message (and optional AI answer):
    Returns one of:
      - "HISTOLOGY_MICROSCOPY": Biopsy slides, tissue histology, blood films/smears, H&E stains, light/electron micrographs.
      - "FLOWCHART_SCHEMATIC": Multi-stage life cycles, biochemical pathways, signaling cascades, physiological mechanisms, clinical algorithms, decision trees, scoring systems.
      - "ANATOMICAL_MAP": Gross anatomical structures, relationships, neurovascular bundles, cross-sections.
      - "NONE": Non-visual factual/conversational queries.
    """
    msg_low = (user_msg or "").lower()

    # 0. Non-visual fast exit: conversational, clinical dosing, definitions, calculations, criteria
    # (unless an explicit visual keyword like diagram, flowchart, illustration, image is present)
    visual_explicit = bool(re.search(r'\b(diagram|diagrams|flowchart|flowcharts|illustration|illustrations|schematic|picture|pictures|image|images|draw|drawings|visualize|figure)\b', msg_low))
    if not visual_explicit:
        non_visual_regex = (
            r'\b(dose|dosage|dosing|half[_\s\-]life|side[_\s\-]effect|side[_\s\-]effects|'
            r'adverse[_\s\-]effect|adverse[_\s\-]effects|contraindication|contraindications|'
            r'define|definition|definitions|criteria|criterion|history[_\s\-]taking|'
            r'physical[_\s\-]exam|differential[_\s\-]diagnosis|calculate|calculation|'
            r'formula|anion[_\s\-]gap|cockcroft|maintenance[_\s\-]fluid)\b'
        )
        if re.search(non_visual_regex, msg_low):
            return "NONE"

        conversational_meta = [
            "hello", "good morning", "good evening", "hi neura", "hey neura",
            "thank you", "thanks", "who created you", "what can you do", "can you help me study",
            "/profile", "/update", "/reset", "/balance", "/wallet", "/deposit"
        ]
        if any(msg_low.strip().startswith(c) for c in conversational_meta):
            return "NONE"

    # 1. Check HISTOLOGY_MICROSCOPY (explicit histology / microscopy / smear / biopsy / stain triggers)
    histology_pattern = re.compile(
        r'(?i)\b(histolog\w*|histopatholog\w*|microscop\w*|biopsy|biopsies|'
        r'h&e|h[_\s\-]and[_\s\-]e|h\s*and\s*e|he[_\s\-]stain|hematoxylin|\beosin\b|giemsa|gram[_\s\-]stain|'
        r'blood[_\s\-]smear|blood[_\s\-]film|thin[_\s\-]smear|thick[_\s\-]smear|pathology[_\s\-]slide|microscopic[_\s\-]slide|'
        r'photomicrograph|frozen[_\s\-]section|pap[_\s\-]smear|immunohistochem\w*|ihc[_\s\-]stain|'
        r'micrograph|stained[_\s\-]slide)\b'
    )
    if histology_pattern.search(msg_low):
        return "HISTOLOGY_MICROSCOPY"

    # 2. Check ANATOMICAL_MAP triggers & topics (gross anatomy, triangles, fossae, relations, tracts)
    anatomical_triggers = [
        "anatomy", "anatomical", "cross section", "blood supply", "arterial supply",
        "venous drainage", "innervation", "lymphatic drainage", "relations of", "parts of",
        "structure of", "structures of", "circle of willis", "brachial plexus", "inguinal canal",
        "femoral triangle", "calot", "cystohepatic triangle", "hesselbach", "carotid triangle",
        "cubital fossa", "popliteal fossa", "cranial nerve", "foramen", "fossa", "sulcus",
        "spinal cord tracts", "meninges", "tetralogy of fallot", "epidermis layers",
        "nephron structure", "alveolus structure", "sarcomere structure", "boundaries", "neurovascular"
    ]
    if any(t in msg_low for t in anatomical_triggers):
        return "ANATOMICAL_MAP"

    # 3. Check explicit Flowchart / Pathway / Cycle / Algorithm triggers
    flowchart_triggers = [
        "diagram", "illustration", "flowchart", "pathway", "cycle", "cycles",
        "cascade", "algorithm", "decision tree", "triage", "steps of", "stages of",
        "mechanism of action", "mechanism of", "moa", "schematic", "draw", "visualize", "chart",
        "synthesis", "breakdown", "degradation", "replication", "regulation",
        "signaling", "signal transduction", "action potential", "metabolism",
        "differentiation", "developmental stages", "life cycle", "lifecycles",
        "nomogram", "scoring", "resuscitation", "protocol", "guidelines",
        "sepsis bundle", "resuscitation bundle", "care bundle", "feedback loop", "axis", "transport", "wiggers", "partograph"
    ]
    if any(t in msg_low for t in flowchart_triggers):
        return "FLOWCHART_SCHEMATIC"

    # 4. Check high-yield preclinical & clinical pathway/cycle/mechanism topics across all 11 disciplines
    flowchart_topics = [
        # Parasitology & Microbiology
        "plasmodium", "malaria", "schistosoma", "bilharzia", "leishmania", "kala-azar",
        "entamoeba", "amoebiasis", "trypanosoma", "sleeping sickness", "chagas",
        "ascaris", "taenia", "cysticercosis", "echinococcus", "hydatid", "wuchereria",
        "filariasis", "elephantiasis", "giardia", "giardiasis", "toxoplasma", "toxoplasmosis",
        "hookworm", "ancylostoma", "necator", "strongyloides", "enterobius", "pinworm",
        "dracunculus", "guinea worm", "sporulation", "endospore", "spore formation",
        "viral replication", "hiv replication", "hepatitis b replication", "hbv replication",
        "bacteriophage", "lytic cycle", "lysogenic cycle", "influenza replication",
        "cholera toxin", "tetanospasmin", "botulinum toxin", "bacterial conjugation",
        # Biochemistry & Metabolism
        "glycolysis", "glycolytic", "krebs", "citric acid cycle", "tca cycle",
        "gluconeogenesis", "pentose phosphate", "hmp shunt", "urea cycle",
        "beta-oxidation", "beta oxidation", "fatty acid oxidation", "carnitine shuttle",
        "electron transport chain", "oxidative phosphorylation", "oxphos", "purine synthesis",
        "purine salvage", "hgprt", "lesch-nyhan", "pyrimidine synthesis", "glycogenolysis",
        "glycogenesis", "glycogen metabolism", "cholesterol synthesis", "mevalonate",
        "steroidogenesis", "cori cycle", "glucose-alanine", "alanine cycle", "cahill cycle",
        "ethanol metabolism", "heme synthesis", "porphyria",
        # Physiology & Pharmacology
        "cardiac action potential", "pacemaker action potential", "sa node action potential",
        "ventricular action potential", "wiggers diagram", "cardiac cycle", "cardiac conduction",
        "raas", "renin angiotensin", "renin-angiotensin", "aldosterone cascade",
        "neuromuscular junction", "excitation-contraction", "countercurrent multiplier",
        "countercurrent exchange", "oxyhemoglobin dissociation", "bohr effect",
        "neuronal action potential", "glomerular filtration", "tubular transport",
        "baroreceptor reflex", "respiratory control", "gpcr", "g-protein", "second messenger",
        "tyrosine kinase", "mapk", "ras-raf", "jak-stat", "jak stat", "coagulation cascade",
        "clotting cascade", "hemostasis", "autonomic receptor", "beta-blocker mechanism",
        "diuretic mechanism", "proton pump inhibitor", "parietal cell acid",
        "insulin signaling", "nitric oxide signaling", "local anesthetic mechanism",
        # Immunology & Hematology
        "hematopoiesis", "haematopoiesis", "b cell development", "b-cell development",
        "b cell maturation", "t cell development", "t-cell development", "thymic selection",
        "th1 th2", "th17", "treg", "helper t cell differentiation", "complement cascade",
        "complement activation", "classical pathway", "alternative pathway", "lectin pathway",
        "hypersensitivity reaction", "gell and coombs", "mhc class i", "mhc class ii",
        "antigen presentation", "tcr activation", "immunological synapse", "mast cell degranulation",
        "respiratory burst", "cytokine storm", "primary hemostasis", "platelet aggregation",
        "fibrinolysis", "hemoglobin synthesis", "iron metabolism", "hepcidin",
        "bilirubin metabolism", "erythropoiesis", "hemolytic anemia classification",
        "sickle cell pathophysiology",
        # Surgery, O&G, Pediatrics, Internal Medicine
        "atls", "primary survey", "rule of nines", "parkland formula", "burns resuscitation",
        "glasgow coma", "gcs", "acute abdomen triage", "alvarado score", "sepsis bundle",
        "menstrual cycle", "cardinal movements of labor", "stages of labor", "apgar",
        "bishop score", "postpartum hemorrhage", "pph algorithm", "preeclampsia management",
        "fetal circulation", "partograph", "rh isoimmunization", "imci", "developmental milestones",
        "neonatal resuscitation", "nrp", "dehydration plan", "bhutani nomogram",
        "congenital heart defect", "pals algorithm", "febrile seizure", "pediatric immunization",
        "meningitis csf", "jvp waveform", "dka protocol", "acid-base disturbance",
        "acls cardiac arrest", "acls tachycardia", "acls bradycardia", "shock classification",
        "portacaval anastomosis", "portal hypertension", "granuloma cascade", "atherosclerosis pathogenesis",
        "acute coronary syndrome triage"
    ]
    if any(topic in msg_low for topic in flowchart_topics):
        return "FLOWCHART_SCHEMATIC"

    # 5. Check generic visual triggers (default to FLOWCHART_SCHEMATIC if general image requested)
    generic_triggers = [
        "picture", "image", "show me", "illustrate", "get a", "can i get", "view"
    ]
    if any(t in msg_low for t in generic_triggers):
        return "FLOWCHART_SCHEMATIC"

    return "NONE"

def should_generate_medical_illustration(user_msg: str, ai_answer: str = "") -> bool:
    """Detects if query or medical topic warrants a visual anatomical, histological or clinical illustration (Pre-clinical & Clinical 200L-600L)"""
    return detect_visual_intent_modality(user_msg, ai_answer) != "NONE"

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
    # ==========================================
    # 1. PARASITOLOGY (Multi-Stage Life Cycles)
    # ==========================================
    (
        ["falciparum", "plasmodium falciparum", "p falciparum", "blackwater fever"],
        ("Plasmodium falciparum Life Cycle & Erythrocytic Schizogony",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/CDC_Malaria_LifeCycle.png/1920px-CDC_Malaria_LifeCycle.png")
    ),
    (
        ["vivax", "plasmodium vivax", "ovale", "plasmodium ovale", "relapse malaria"],
        ("Plasmodium vivax and ovale Hepatic Hypnozoite Dormancy Cycle",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/CDC_Malaria_LifeCycle.png/1920px-CDC_Malaria_LifeCycle.png")
    ),
    (
        ["malaria", "plasmodium", "sporogony", "schizogony", "sporozoite", "merozoite", "malaria trophozoite", "hypnozoite", "gametocyte", "ring form"],
        ("Malaria Plasmodium Life Cycle (Hepatic & Erythrocytic Schizogony, Mosquito Sporogony)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/CDC_Malaria_LifeCycle.png/1920px-CDC_Malaria_LifeCycle.png")
    ),
    (
        ["schistosoma", "schistosomiasis", "bilharzia", "miracidia", "cercariae", "snail host", "biomphalaria", "bulinus", "haematobium", "mansoni", "japonicum"],
        ("Schistosoma Multi-Host Life Cycle (Miracidia, Snail Intermediate, Cercariae)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Schistosoma_life_cycle.svg/1920px-Schistosoma_life_cycle.svg.png")
    ),
    (
        ["leishmania", "leishmaniasis", "kala-azar", "kala azar", "kalaazar", "promastigote", "amastigote", "sandfly", "phlebotomus", "donovani"],
        ("Leishmania Life Cycle (Sandfly Promastigote & Macrophage Amastigote Stages)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/b/be/Leishmania_LifeCycle.svg/1920px-Leishmania_LifeCycle.svg.png")
    ),
    (
        ["entamoeba", "amoebiasis", "amebiasis", "histolytica", "entamoeba cyst", "amebic trophozoite", "liver abscess amoebic", "flask-shaped"],
        ("Entamoeba histolytica Life Cycle (Trophozoite, Cyst & Colonic/Hepatic Invasion)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Entamoeba_histolytica_life_cycle.svg/1920px-Entamoeba_histolytica_life_cycle.svg.png")
    ),
    (
        ["trypanosoma brucei", "sleeping sickness", "tsetse", "trypanosomiasis african", "procyclic", "epimastigote", "trypomastigote"],
        ("Trypanosoma brucei Life Cycle (African Sleeping Sickness & Tsetse Fly Vector)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Trypanosoma_brucei_LifeCycle.svg/1920px-Trypanosoma_brucei_LifeCycle.svg.png")
    ),
    (
        ["trypanosoma cruzi", "chagas", "reduviid", "kissing bug", "triatomine", "romana sign"],
        ("Trypanosoma cruzi Life Cycle (Chagas Disease & Reduviid Vector)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Trypanosoma_cruzi_LifeCycle.svg/1920px-Trypanosoma_cruzi_LifeCycle.svg.png")
    ),
    (
        ["ascaris", "ascariasis", "lumbricoides", "roundworm", "pulmonary migration ascaris", "loeffler"],
        ("Ascaris lumbricoides Life Cycle (Tracheal & Pulmonary Migration Pathway)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/ae/Ascaris_Lumbricoides_Life_Cycle.svg/1920px-Ascaris_Lumbricoides_Life_Cycle.svg.png")
    ),
    (
        ["taenia solium", "pork tapeworm", "cysticercosis", "neurocysticercosis", "oncosphere", "proglottid"],
        ("Taenia solium Life Cycle (Pork Tapeworm & Cysticercosis Stages)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/72/Taenia_solium_Life_Cycle.svg/1920px-Taenia_solium_Life_Cycle.svg.png")
    ),
    (
        ["taenia saginata", "beef tapeworm", "cysticercus bovis"],
        ("Taenia saginata Life Cycle (Beef Tapeworm Intermediate Host Cycle)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d3/Taenia_saginata_Life_Cycle.svg/1920px-Taenia_saginata_Life_Cycle.svg.png")
    ),
    (
        ["echinococcus", "hydatid", "echinococcosis", "granulosus", "hydatid sand", "protoscolex"],
        ("Echinococcus granulosus Life Cycle (Hydatid Cyst Disease & Host Cycle)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f9/Echinococcus_LifeCycle.svg/1920px-Echinococcus_LifeCycle.svg.png")
    ),
    (
        ["wuchereria", "bancrofti", "filariasis", "elephantiasis", "microfilaria", "culex", "brugia"],
        ("Wuchereria bancrofti Life Cycle (Lymphatic Filariasis & Mosquito Transmission)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Wuchereria_bancrofti_LifeCycle.svg/1920px-Wuchereria_bancrofti_LifeCycle.svg.png")
    ),
    (
        ["giardia", "giardiasis", "lamblia", "duodenalis", "falling leaf motility", "steatorrhea"],
        ("Giardia lamblia Life Cycle (Flagellated Trophozoite & Cyst Fecal-Oral Cycle)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/23/Giardia_LifeCycle.svg/1920px-Giardia_LifeCycle.svg.png")
    ),
    (
        ["toxoplasma", "toxoplasmosis", "gondii", "bradyzoite", "tachyzoite", "cat definitive", "oocyst"],
        ("Toxoplasma gondii Life Cycle (Feline Definitive Host & Intermediate Tachyzoite Cycle)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Toxoplasma_gondii_life_cycle.svg/1920px-Toxoplasma_gondii_life_cycle.svg.png")
    ),
    (
        ["hookworm", "ancylostoma", "necator", "ground itch", "filariform", "rhabditiform hookworm"],
        ("Hookworm Life Cycle (Ancylostoma duodenale & Necator americanus Skin Penetration)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/69/Hookworm_LifeCycle.svg/1920px-Hookworm_LifeCycle.svg.png")
    ),
    (
        ["strongyloides", "strongyloidiasis", "stercoralis", "autoinfection", "hyperinfection"],
        ("Strongyloides stercoralis Life Cycle (Autoinfection & Filariform Larval Cycle)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Strongyloides_stercoralis_Life_Cycle.svg/1920px-Strongyloides_stercoralis_Life_Cycle.svg.png")
    ),
    (
        ["enterobius", "vermicularis", "pinworm", "scotch tape test", "perianal pruritus"],
        ("Enterobius vermicularis Life Cycle (Pinworm Retroinfection & Perianal Ovum Cycle)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6a/Enterobius_vermicularis_LifeCycle.svg/1920px-Enterobius_vermicularis_LifeCycle.svg.png")
    ),
    (
        ["dracunculus", "medinensis", "guinea worm", "copepod", "cyclops water"],
        ("Dracunculus medinensis Life Cycle (Guinea Worm Disease & Copepod Vector)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Dracunculus_medinensis_Life_Cycle.svg/1920px-Dracunculus_medinensis_Life_Cycle.svg.png")
    ),

    # ==========================================
    # 2. MICROBIOLOGY & VIROLOGY
    # ==========================================
    (
        ["sporulation", "endospore", "spore formation", "dipicolinic acid", "bacterial spore"],
        ("Bacterial Endospore Formation (7-Stage Sporulation Cycle)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/Bacterial_sporulation_diagram.svg/1920px-Bacterial_sporulation_diagram.svg.png")
    ),
    (
        ["cell wall", "gram positive", "gram negative", "gram positive cell", "gram negative cell", "gram-positive", "gram-negative", "peptidoglycan", "lipopolysaccharide", "periplasmic", "teichoic acid"],
        ("Gram-Positive vs. Gram-Negative Bacterial Cell Wall Architecture",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Gram_positive_and_negative_cell_wall.svg/1920px-Gram_positive_and_negative_cell_wall.svg.png")
    ),
    (
        ["influenza", "influenza replication", "hemagglutinin", "neuraminidase", "influenza life cycle", "sialic acid", "orthomyxovirus"],
        ("Influenza Virus Replication Cycle & Antiviral Targets (Oseltamivir/Zanamivir)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/61/Influenza_virus_replication_cycle.svg/1920px-Influenza_virus_replication_cycle.svg.png")
    ),
    (
        ["hiv replication", "hiv life cycle", "reverse transcriptase", "integrase", "protease inhibitor", "cd4 gp120", "ccr5"],
        ("HIV Replication Cycle and Antiretroviral Drug Targets",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b6/HIV_replication_cycle.svg/1920px-HIV_replication_cycle.svg.png")
    ),
    (
        ["hepatitis b replication", "hbv replication", "cccdna", "pgrna", "reverse transcription hbv"],
        ("Hepatitis B Virus (HBV) Replication Cycle & cccDNA Formation",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Hepatitis_B_virus_replication_cycle.png/1920px-Hepatitis_B_virus_replication_cycle.png")
    ),
    (
        ["bacteriophage", "lytic cycle", "lysogenic cycle", "prophage", "phage replication"],
        ("Bacteriophage Lytic vs. Lysogenic Replication Cycles",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/b/ba/Phage_lytic_and_lysogenic_cycles.svg/1920px-Phage_lytic_and_lysogenic_cycles.svg.png")
    ),
    (
        ["viral replication", "virus replication", "virus life cycle", "viral life cycle", "uncoating", "budding viral"],
        ("General Viral Replication Cycle (Adsorption, Penetration, Biosynthesis & Release)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Virus_Replication_Cycle.svg/1920px-Virus_Replication_Cycle.svg.png")
    ),
    (
        ["cholera toxin", "vibrio cholerae", "adp-ribosylation", "g protein cholera", "cftr secretion"],
        ("Cholera Toxin Mechanism of Action (Gs ADP-Ribosylation & Hypersecretion)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/64/Mechanism_of_Cholera_toxin.svg/1920px-Mechanism_of_Cholera_toxin.svg.png")
    ),
    (
        ["conjugation", "transformation", "transduction", "horizontal gene transfer", "sex pilus", "f plasmid"],
        ("Bacterial Genetic Exchange (Conjugation, Transformation & Transduction)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Horizontal_gene_transfer.svg/1920px-Horizontal_gene_transfer.svg.png")
    ),
    (
        ["tuberculosis pathogenesis", "granuloma cascade", "mycobacterium tuberculosis cascade", "caseating cascade", "cord factor", "tubercular granuloma"],
        ("Tuberculosis Pathogenesis & Tubercular Granuloma Immunological Cascade",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Granuloma_formation_cascade.svg/1920px-Granuloma_formation_cascade.svg.png")
    ),
    (
        ["tetanospasmin", "tetanus toxin", "renshaw cell", "gaba glycine cleavage", "clostridium tetani mechanism"],
        ("Tetanus Toxin Mechanism (Retrograde Axonal Transport & Synaptobrevin Cleavage)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4f/Mechanism_of_tetanus_toxin.svg/1920px-Mechanism_of_tetanus_toxin.svg.png")
    ),
    (
        ["botulinum toxin", "botulism", "snare protein", "acetylcholine release block", "flaccid paralysis toxin"],
        ("Botulinum Toxin Mechanism of Action (SNARE Protein Cleavage & ACh Blockade)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/df/Botulinum_toxin_mechanism.svg/1920px-Botulinum_toxin_mechanism.svg.png")
    ),

    # ==========================================
    # 3. BIOCHEMISTRY & METABOLISM
    # ==========================================
    (
        ["glycolysis", "glycolytic", "glucose breakdown", "embden meyerhof", "phosphofructokinase", "pfk-1", "pyruvate kinase"],
        ("Glycolysis Metabolic Pathway (10 Enzymatic Steps & Net Energy Yields)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Glycolysis.svg/1920px-Glycolysis.svg.png")
    ),
    (
        ["krebs", "citric acid cycle", "tca cycle", "isocitrate dehydrogenase", "alpha-ketoglutarate", "succinyl-coa"],
        ("Citric Acid Cycle (Krebs TCA Cycle Steps, Cofactors & ATP Yields)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0b/Citric_acid_cycle_with_aconitate_2.svg/1920px-Citric_acid_cycle_with_aconitate_2.svg.png")
    ),
    (
        ["cori cycle", "cori", "lactic acid cycle", "glucose-lactate cycle"],
        ("Cori Cycle (Muscle Glycolytic Lactate & Hepatic Gluconeogenesis Exchange)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Cori_cycle_integ.svg/1920px-Cori_cycle_integ.svg.png")
    ),
    (
        ["gluconeogenesis", "pepck", "pyruvate carboxylase", "fructose-1,6-bisphosphatase", "glucose-6-phosphatase"],
        ("Gluconeogenesis Pathway (4 Bypass Reactions & Substrate Flow)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Gluconeogenesis_pathway.svg/1920px-Gluconeogenesis_pathway.svg.png")
    ),
    (
        ["pentose phosphate", "hmp shunt", "hexose monophosphate", "g6pd", "transketolase", "transaldolase"],
        ("Pentose Phosphate Pathway / HMP Shunt (Oxidative & Non-Oxidative Phases)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Pentose_phosphate_pathway.svg/1920px-Pentose_phosphate_pathway.svg.png")
    ),
    (
        ["urea cycle", "ornithine", "citrulline", "argininosuccinate", "carbamoyl phosphate synthetase", "cps-1", "ammonia detox"],
        ("Urea Cycle (Mitochondrial & Cytosolic Ammonia Detoxification Pathway)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Urea_cycle.svg/1920px-Urea_cycle.svg.png")
    ),
    (
        ["beta-oxidation", "beta oxidation", "fatty acid degradation", "carnitine shuttle", "cpt-1", "acyl-coa dehydrogenase"],
        ("Beta-Oxidation of Fatty Acids (Carnitine Shuttle & Spiral Degradation)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/Beta-oxidation.svg/1920px-Beta-oxidation.svg.png")
    ),
    (
        ["electron transport chain", "oxidative phosphorylation", "oxphos", "atp synthase", "complex i", "complex iv", "cytochrome c"],
        ("Electron Transport Chain & Oxidative Phosphorylation (Complexes I-IV & Chemiosmosis)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Electron_transport_chain.svg/1920px-Electron_transport_chain.svg.png")
    ),
    (
        ["purine", "purine metabolism", "purine synthesis", "purine salvage", "hgprt", "lesch-nyhan", "xanthine oxidase", "uric acid"],
        ("Purine De Novo Synthesis & Salvage Pathway (HGPRT & Gout Cascade)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/60/Purine_metabolism.svg/1920px-Purine_metabolism.svg.png")
    ),
    (
        ["pyrimidine", "pyrimidine synthesis", "orotic aciduria", "ump synthase", "dihydroorotate", "thymidylate synthase"],
        ("Pyrimidine Biosynthesis Pathway (CAD Complex & Ribonucleotide Reductase)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fa/Pyrimidine_metabolism.svg/1920px-Pyrimidine_metabolism.svg.png")
    ),
    (
        ["glycogen", "glycogenolysis", "glycogenesis", "glycogen metabolism", "glycogen phosphorylase", "glycogen synthase", "debranching enzyme"],
        ("Glycogen Metabolism (Reciprocal Regulation of Glycogenolysis & Glycogenesis)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Glycogen_metabolism.svg/1920px-Glycogen_metabolism.svg.png")
    ),
    (
        ["cholesterol synthesis", "mevalonate", "hmg-coa reductase", "squalene", "statin pathway"],
        ("Cholesterol Biosynthesis (Mevalonate Pathway & HMG-CoA Reductase Regulation)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d7/Cholesterol_synthesis.svg/1920px-Cholesterol_synthesis.svg.png")
    ),
    (
        ["steroidogenesis", "adrenal steroid", "21-hydroxylase", "11-beta-hydroxylase", "cortisol synthesis", "congenital adrenal hyperplasia"],
        ("Adrenal Steroidogenesis Pathway (Mineralocorticoid, Glucocorticoid & Androgen Synthesis)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Steroidogenesis.svg/1920px-Steroidogenesis.svg.png")
    ),
    (
        ["glucose-alanine", "alanine cycle", "cahill cycle", "muscle nitrogen transport"],
        ("Glucose-Alanine Cycle (Muscle Amino Acid Transamination & Hepatic Urea Synthesis)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/20/Glucose-alanine_cycle.svg/1920px-Glucose-alanine_cycle.svg.png")
    ),
    (
        ["ethanol metabolism", "alcohol dehydrogenase", "acetaldehyde dehydrogenase", "cyp2e1", "disulfiram"],
        ("Ethanol Metabolism Pathway (Alcohol Dehydrogenase, CYP2E1 & ALDH)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c5/Ethanol_metabolism.svg/1920px-Ethanol_metabolism.svg.png")
    ),
    (
        ["heme synthesis", "ala synthase", "porphyria", "porphobilinogen", "ferrochelatase", "lead poisoning heme"],
        ("Heme Biosynthesis Pathway (Enzymatic Steps & Porphyria Blocks)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Heme_synthesis.svg/1920px-Heme_synthesis.svg.png")
    ),

    # ==========================================
    # 4. PHYSIOLOGY
    # ==========================================
    (
        ["ventricular action potential", "cardiac action potential", "myocyte action potential", "phase 0 na", "phase 2 ca"],
        ("Ventricular Myocyte Cardiac Action Potential (Phases 0 to 4 Ion Dynamics)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/46/Action_potential_myocyte.svg/1920px-Action_potential_myocyte.svg.png")
    ),
    (
        ["pacemaker action potential", "sa node action potential", "funny current", "phase 4 depolarization"],
        ("Cardiac Pacemaker (SA Node) Action Potential & Automaticity",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/27/Cardiac_pacemaker_action_potential.svg/1920px-Cardiac_pacemaker_action_potential.svg.png")
    ),
    (
        ["wiggers diagram", "cardiac cycle", "heart cycle", "ventricular volume pressure", "heart sounds wiggers"],
        ("Cardiac Cycle Wiggers Diagram (Pressures, Volumes, ECG & Heart Sounds)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Wiggers_Diagram_2.svg/1920px-Wiggers_Diagram_2.svg.png")
    ),
    (
        ["cardiac conduction", "cardiac electrical conduction", "electrical conduction", "heart conduction", "sa node", "av node", "bundle of his", "purkinje fibers", "purkinje"],
        ("Cardiac Electrical Conduction Pathway",
         "https://upload.wikimedia.org/wikipedia/commons/0/0f/Cardiac_Conduction_System.jpg")
    ),
    (
        ["raas inhibitors", "raas pharmacology", "ace inhibitor", "acei", "arb", "losartan", "aliskiren", "pharmacological inhibition of raas"],
        ("Pharmacological Inhibition of RAAS (ACE Inhibitors, ARBs & Renin Inhibitors)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/RAAS_pharmacology_targets.svg/1920px-RAAS_pharmacology_targets.svg.png")
    ),
    (
        ["raas", "raas cascade", "renin-angiotensin-aldosterone", "renin", "angiotensin", "aldosterone", "juxtaglomerular"],
        ("Renin-Angiotensin-Aldosterone System (RAAS Cascade & Hemodynamic Control)",
         "https://upload.wikimedia.org/wikipedia/commons/a/a7/2117_Renin_Angiotensin_Aldosterone_Pathway.jpg")
    ),
    (
        ["neuromuscular junction", "endplate potential", "excitation-contraction", "acetylcholine receptor nmj", "dhp ryanodine"],
        ("Neuromuscular Junction & Excitation-Contraction Coupling",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/Chemical_synapse_schema_cropped.jpg/1920px-Chemical_synapse_schema_cropped.jpg")
    ),
    (
        ["countercurrent multiplier", "countercurrent exchange", "loop of henle gradient", "medullary osmolarity", "vasa recta"],
        ("Renal Countercurrent Multiplier & Exchange System in the Loop of Henle",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Countercurrent_multiplier.svg/1920px-Countercurrent_multiplier.svg.png")
    ),
    (
        ["oxyhemoglobin dissociation", "oxygen-hemoglobin curve", "bohr effect", "2,3-bpg", "p50 shift"],
        ("Oxyhemoglobin Dissociation Curve & Physiological Modulators (Bohr Effect)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Oxyhaemoglobin_dissociation_curve.svg/1920px-Oxyhaemoglobin_dissociation_curve.svg.png")
    ),
    (
        ["action potential", "action potential phases", "neuronal action potential", "nerve action potential", "depolarization repolarization", "voltage-gated sodium"],
        ("Neuronal Action Potential Phases (Depolarization, Repolarization & Refractory Periods)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4a/Action_potential.svg/1920px-Action_potential.svg.png")
    ),
    (
        ["glomerular filtration", "tubular transport", "nephron transport", "pct reabsorption", "collecting duct transport"],
        ("Glomerular Filtration & Segmental Renal Tubular Solute Transport",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Physiology_of_Nephron.png/1920px-Physiology_of_Nephron.png")
    ),
    (
        ["baroreceptor", "baroreflex", "carotid sinus reflex", "aortic arch baroreceptor", "blood pressure reflex"],
        ("Baroreceptor Reflex Arc & Autonomic Blood Pressure Regulation",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Baroreceptor_reflex.svg/1920px-Baroreceptor_reflex.svg.png")
    ),
    (
        ["respiratory control", "chemoreceptor reflex", "medullary respiratory group", "central chemoreceptor", "pco2 control"],
        ("Brainstem Respiratory Control Centers & Chemoreceptor Feedback Loop",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/Respiratory_control_centers.svg/1920px-Respiratory_control_centers.svg.png")
    ),
    (
        ["sarcomere", "sliding filament", "cross-bridge cycle", "actin myosin", "tropomyosin troponin"],
        ("Sarcomere Sliding Filament Mechanism (Actin-Myosin Cross-Bridge Cycle)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Sarcomere_diagram.svg/1920px-Sarcomere_diagram.svg.png")
    ),

    # ==========================================
    # 5. PHARMACOLOGY
    # ==========================================
    (
        ["gpcr", "g protein coupled", "gs pathway", "gi pathway", "gq pathway", "adenylyl cyclase", "phospholipase c", "ip3 dag"],
        ("G-Protein Coupled Receptor (GPCR Gs, Gi, Gq) Second Messenger Cascades",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/GPCR_signaling_mechanism.svg/1920px-GPCR_signaling_mechanism.svg.png")
    ),
    (
        ["tyrosine kinase", "rtk signaling", "mapk pathway", "ras raf mek erk", "growth factor receptor"],
        ("Receptor Tyrosine Kinase (RTK) & Ras-Raf-MEK-ERK (MAPK) Signaling Cascade",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/MAPK_pathway.svg/1920px-MAPK_pathway.svg.png")
    ),
    (
        ["jak stat", "jak-stat", "janus kinase", "stat signaling", "cytokine receptor signaling"],
        ("JAK-STAT Cytokine Signaling Transduction Pathway",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/db/JAK-STAT_pathway.svg/1920px-JAK-STAT_pathway.svg.png")
    ),
    (
        ["anticoagulant targets", "anticoagulant", "anticoagulants", "heparin", "warfarin", "heparin mechanism", "warfarin mechanism", "doac mechanism", "antithrombin iii target", "warfarin vs heparin"],
        ("Coagulation Cascade with Anticoagulant Pharmacological Targets (Heparin, Warfarin, DOACs)",
         "https://upload.wikimedia.org/wikipedia/commons/4/4b/Coagulation_Cascade_with_Tests.jpg")
    ),
    (
        ["beta-blocker", "beta-blockers", "beta blocker", "beta blockers", "beta blocker mechanism", "propranolol mechanism", "metoprolol mechanism", "beta-1 blockade"],
        ("Beta-Blockers (Beta-Adrenergic Blockers) Mechanism of Action & Cardiovascular Effects",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/67/Beta_blocker_mechanism.svg/1920px-Beta_blocker_mechanism.svg.png")
    ),
    (
        ["autonomic", "autonomic nervous system", "autonomic receptor", "adrenergic", "cholinergic", "adrenergic receptor", "cholinergic receptor", "sympathetic vs parasympathetic receptors", "alpha beta receptors", "sympathetic", "parasympathetic"],
        ("Autonomic Nervous System Receptor Pathways & Neuroeffector Targets",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3d/Autonomic_nervous_system_actions.svg/1920px-Autonomic_nervous_system_actions.svg.png")
    ),
    (
        ["diuretic", "diuretics", "diuretic mechanism", "diuretic sites", "nephron sites", "loop diuretic", "thiazide", "nkcc2 inhibitor", "spironolactone moa"],
        ("Diuretics Nephron Sites of Action & Electrolyte Transport Mechanisms",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/Diuretic_sites_of_action.svg/1920px-Diuretic_sites_of_action.svg.png")
    ),
    (
        ["proton pump inhibitor", "ppi mechanism", "gastric acid secretion", "parietal cell mechanism", "h+/k+ atpase"],
        ("Proton Pump Inhibitors (PPIs) & Parietal Cell Gastric Acid Secretion Pathway",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Parietal_cell_acid_secretion.svg/1920px-Parietal_cell_acid_secretion.svg.png")
    ),
    (
        ["insulin signaling", "glut4 translocation", "pi3k akt insulin", "irs-1 pathway"],
        ("Insulin Receptor Signaling & PI3K-Akt GLUT4 Translocation Pathway",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/90/Insulin_glucose_metabolism_ZP.svg/1920px-Insulin_glucose_metabolism_ZP.svg.png")
    ),
    (
        ["nitric oxide signaling", "cgmp pathway", "pde5 inhibitor", "sildenafil mechanism", "smooth muscle relaxation no"],
        ("Nitric Oxide (NO) / cGMP / PDE5 Inhibitor Vasodilation Pathway",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Nitric_oxide_cGMP_signaling.svg/1920px-Nitric_oxide_cGMP_signaling.svg.png")
    ),
    (
        ["local anesthetic", "local anesthetics", "local anesthetic mechanism", "lidocaine", "lidocaine mechanism", "sodium channel block anesthetic", "use-dependent block"],
        ("Local Anesthetics Voltage-Gated Sodium Channel Blockade Mechanism",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/89/Local_anesthetic_mechanism.svg/1920px-Local_anesthetic_mechanism.svg.png")
    ),
    # ==========================================
    # 6. IMMUNOLOGY
    # ==========================================
    (
        ["hematopoiesis", "haematopoiesis", "blood cell lineage", "stem cell differentiation", "myeloid lymphoid", "hsc"],
        ("Hematopoiesis and Blood Cell Lineage Differentiation Tree",
         "https://upload.wikimedia.org/wikipedia/commons/6/69/Hematopoiesis_%28human%29_diagram.png")
    ),
    (
        ["b cell development", "b-cell development", "b cell", "b-cell", "b cell maturation", "b cell differentiation", "b and t cell", "vdj recombination", "pro-b pre-b", "b cell activation"],
        ("B-Cell Maturation, V(D)J Recombination & Activation Stages",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a8/B_cell_development_stages.svg/1920px-B_cell_development_stages.svg.png")
    ),
    (
        ["helper t cell", "th1 th2", "th17", "treg", "t-helper differentiation", "foxp3", "ror-gamma-t", "t-bet", "gata3"],
        ("Helper T-Cell (Th1, Th2, Th17, Treg) Lineage Differentiation Network",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c9/Th_cell_differentiation.svg/1920px-Th_cell_differentiation.svg.png")
    ),
    (
        ["t cell selection", "t cell", "t-cell", "t cell differentiation", "thymic selection", "positive selection", "negative selection thymus", "cd4 cd8 commitment", "aire gene"],
        ("T-Cell Thymic Selection (Cortex Positive & Medulla Negative Selection)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/7b/T_cell_selection_thymus.svg/1920px-T_cell_selection_thymus.svg.png")
    ),
    (
        ["complement", "complement system", "complement cascade", "complement activation", "classical pathway", "alternative pathway", "lectin pathway", "membrane attack complex", "mac c5b-9"],
        ("Complement System Cascades (Classical, Lectin & Alternative Pathways to MAC)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/98/Complement_pathway.svg/1920px-Complement_pathway.svg.png")
    ),
    (
        ["hypersensitivity", "gell and coombs", "type i hypersensitivity", "type ii hypersensitivity", "type iii hypersensitivity", "type iv hypersensitivity"],
        ("Gell and Coombs Hypersensitivity Reactions (Types I, II, III & IV Pathways)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Hypersensitivity_reactions_types_I_IV.svg/1920px-Hypersensitivity_reactions_types_I_IV.svg.png")
    ),
    (
        ["mhc class i", "mhc class ii", "antigen presentation", "antigen processing", "tap transporter", "invariant chain", "clip peptide"],
        ("MHC Class I vs. MHC Class II Antigen Processing & Presentation Pathways",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/d4/Antigen_presentation_MHC_I_II.svg/1920px-Antigen_presentation_MHC_I_II.svg.png")
    ),
    (
        ["tcr", "tcr mhc", "tcr activation", "t-cell receptor signaling", "immunological synapse", "antigen presentation synapse", "costimulation cd28", "cd80 cd86"],
        ("T-Cell Receptor (TCR) Immunological Synapse & Dual-Signal Activation",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Immunological_synapse_TCR.svg/1920px-Immunological_synapse_TCR.svg.png")
    ),
    (
        ["clonal selection", "somatic hypermutation", "affinity maturation", "germinal center b cell", "isotype switching"],
        ("Clonal Selection & Somatic Hypermutation in Germinal Centers",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/c/cb/Clonal_selection.svg/1920px-Clonal_selection.svg.png")
    ),
    (
        ["immunoglobulin", "immunoglobulin structure", "antibody", "antibody structure", "igg", "igg structure", "fab fc fragment", "heavy light chain"],
        ("Immunoglobulin (IgG) Molecular Architecture & Fab/Fc Domains",
         "https://upload.wikimedia.org/wikipedia/commons/3/3a/Antibody_IgG1_surface.png")
    ),
    (
        ["mast cell", "mast cells", "mast cell degranulation", "ige degranulation", "fceri", "ige cross-linking", "histamine release", "anaphylaxis mechanism"],
        ("Mast Cell IgE Receptor Cross-Linking & Degranulation Cascade",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/Mast_cell_activation_IgE.svg/1920px-Mast_cell_activation_IgE.svg.png")
    ),
    (
        ["respiratory burst", "phagocytosis cascade", "nadph oxidase", "superoxide", "myeloperoxidase", "chronic granulomatous disease"],
        ("Phagocytic Respiratory Burst & Reactive Oxygen Species Cascade",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/Respiratory_burst_pathway.svg/1920px-Respiratory_burst_pathway.svg.png")
    ),
    (
        ["cytokine storm", "systemic inflammatory response", "sirs cascade", "il-6 storm", "tnf-alpha cascade"],
        ("Cytokine Storm & Systemic Inflammatory Cascade",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9d/Cytokine_storm_cascade.svg/1920px-Cytokine_storm_cascade.svg.png")
    ),

    # ==========================================
    # 7. HEMATOLOGY
    # ==========================================
    (
        ["primary hemostasis", "platelet adhesion", "platelet aggregation", "vwf", "gpib", "gpiib/iiia", "platelet plug"],
        ("Primary Hemostasis (Endothelial Injury, Platelet Adhesion & Plug Formation)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/74/Platelet_adhesion_aggregation.svg/1920px-Platelet_adhesion_aggregation.svg.png")
    ),
    (
        ["coagulation", "clotting", "secondary hemostasis", "coagulation waterfall", "coagulation cascade", "extrinsic pathway", "intrinsic pathway", "thrombin generation", "fibrin mesh"],
        ("Secondary Hemostasis Waterfall (Intrinsic, Extrinsic & Common Coagulation Pathways)",
         "https://upload.wikimedia.org/wikipedia/commons/4/4b/Coagulation_Cascade_with_Tests.jpg")
    ),
    (
        ["fibrinolysis", "clot breakdown", "plasminogen", "tpa", "plasmin", "d-dimer formation"],
        ("Fibrinolysis Cascade & Fibrin Degradation (tPA, Plasmin & D-Dimer)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5b/Fibrinolysis_pathway.svg/1920px-Fibrinolysis_pathway.svg.png")
    ),
    (
        ["hemoglobin structure", "globin chains", "heme binding", "taut relaxed state", "cooperative binding oxygen"],
        ("Hemoglobin Tetrameric Structure & Allosteric Oxygen Binding",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Hemoglobin_structure.svg/1920px-Hemoglobin_structure.svg.png")
    ),
    (
        ["iron metabolism", "iron absorption", "ferroportin", "transferrin", "ferritin", "hepcidin", "dmt1"],
        ("Iron Metabolism (Enterocyte Absorption, Hepcidin Regulation & Storage)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3c/Iron_metabolism.svg/1920px-Iron_metabolism.svg.png")
    ),
    (
        ["bilirubin", "bilirubin metabolism", "bilirubin degradation", "jaundice pathway", "jaundice classification", "unconjugated bilirubin", "conjugated bilirubin", "ugt1a1", "urobilinogen"],
        ("Bilirubin Degradation Pathway & Diagnostic Classification of Jaundice",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/78/Bilirubin_metabolism.svg/1920px-Bilirubin_metabolism.svg.png")
    ),
    (
        ["abo blood group", "rh blood group", "blood typing", "agglutinogen", "h antigen", "rhesus factor"],
        ("ABO and Rh Blood Group Antigens & Transfusion Compatibility Matrix",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b3/ABO_blood_group_diagram.svg/1920px-ABO_blood_group_diagram.svg.png")
    ),
    (
        ["erythropoiesis", "red blood cell maturation", "reticulocyte", "proerythroblast", "erythropoietin cascade"],
        ("Erythropoiesis Stages (Proerythroblast to Mature Erythrocyte)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/Erythropoiesis_stages.svg/1920px-Erythropoiesis_stages.svg.png")
    ),
    (
        ["hemolytic anemia", "hemolytic anemias", "hemolytic anemia algorithm", "coombs test algorithm", "intravascular vs extravascular hemolysis", "anemia decision tree"],
        ("Diagnostic Algorithm for Hemolytic Anemias (Immune vs. Non-Immune Etiologies)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/26/Hemolytic_anemia_classification.svg/1920px-Hemolytic_anemia_classification.svg.png")
    ),
    (
        ["sickle cell pathophysiology", "hbs polymerization", "sickling crisis", "vaso-occlusion", "sickle cell anemia cascade"],
        ("Sickle Cell Disease Pathophysiology & Vaso-Occlusive Cascade",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Sickle_cell_pathophysiology.svg/1920px-Sickle_cell_pathophysiology.svg.png")
    ),

    # ==========================================
    # 8. SURGERY & TRAUMA
    # ==========================================
    (
        ["atls", "primary survey", "abcde trauma", "trauma resuscitation", "airway breathing circulation disability"],
        ("ATLS Primary Survey (ABCDE Life Support & Resuscitation Algorithm)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5e/ATLS_Primary_Survey_Algorithm.svg/1920px-ATLS_Primary_Survey_Algorithm.svg.png")
    ),
    (
        ["rule of nines", "parkland formula", "burn percentage", "burn fluid resuscitation", "wallace rule of nines"],
        ("Burns Wallace Rule of Nines & Parkland Fluid Resuscitation Formula",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b7/Rule_of_nines_burn_percentage.svg/1920px-Rule_of_nines_burn_percentage.svg.png")
    ),
    (
        ["glasgow coma scale", "gcs", "gcs score", "coma scale", "eye verbal motor"],
        ("Glasgow Coma Scale (GCS Scoring & Head Injury Triage Algorithm)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4e/Glasgow_Coma_Scale_detailed.svg/1920px-Glasgow_Coma_Scale_detailed.svg.png")
    ),
    (
        ["calot", "calots triangle", "cystohepatic triangle", "cholecystectomy anatomy", "cystic duct artery"],
        ("Calot's Triangle (Cystohepatic Triangle Anatomy for Safe Cholecystectomy)",
         "https://upload.wikimedia.org/wikipedia/commons/6/65/Gray532.png")
    ),
    (
        ["brachial plexus", "brachial plexus roots", "trunks cords branches", "erbs palsy", "klumpke"],
        ("Brachial Plexus (Roots, Trunks, Divisions, Cords & Peripheral Terminal Branches)",
         "https://upload.wikimedia.org/wikipedia/commons/3/3a/Gray808.png")
    ),
    (
        ["inguinal canal", "direct inguinal hernia", "indirect inguinal hernia", "hesselbach triangle", "deep inguinal ring"],
        ("Inguinal Canal Architecture & Direct vs. Indirect Hernia Anatomy",
         "https://upload.wikimedia.org/wikipedia/commons/b/b4/Gray1227.png")
    ),
    (
        ["femoral triangle", "navel", "femoral sheath", "femoral canal", "scarpa triangle"],
        ("Femoral Triangle Anatomy & NAVEL Neurovascular Bundle",
         "https://upload.wikimedia.org/wikipedia/commons/4/47/Femoral-triangle-diagram.jpg")
    ),
    (
        ["acute abdomen", "abdominal triage", "peritonitis algorithm", "acute surgical abdomen"],
        ("Acute Abdomen Diagnostic Triage & Surgical Decision Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Acute_abdomen_triage_algorithm.svg/1920px-Acute_abdomen_triage_algorithm.svg.png")
    ),
    (
        ["alvarado", "alvarado score", "alvarado scoring", "appendicitis", "acute appendicitis", "appendicitis score", "mantrels", "appendicitis algorithm"],
        ("Acute Appendicitis Alvarado Scoring & Management Decision Tree",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Alvarado_score_flowchart.svg/1920px-Alvarado_score_flowchart.svg.png")
    ),
    (
        ["sepsis", "septic shock", "sepsis bundle", "surviving sepsis", "septic shock management", "hour-1 bundle", "sepsis resuscitation"],
        ("Surviving Sepsis Campaign Hour-1 Resuscitation Bundle Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/48/Sepsis_bundle_flowchart.svg/1920px-Sepsis_bundle_flowchart.svg.png")
    ),

    # ==========================================
    # 9. OBSTETRICS & GYNECOLOGY
    # ==========================================
    (
        ["menstrual cycle", "ovarian cycle", "uterine cycle", "follicular phase", "luteal phase", "lh surge", "endometrial cycle"],
        ("Menstrual Cycle Hormonal Fluctuation & Endometrial Phases",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2a/MenstrualCycle2_en.svg/1920px-MenstrualCycle2_en.svg.png")
    ),
    (
        ["cardinal movements", "mechanism of labor", "fetal descent", "internal rotation labor", "restitution labor"],
        ("Cardinal Movements of Normal Labor & Fetal Delivery Mechanism",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Mechanisms_of_normal_labor.svg/1920px-Mechanisms_of_normal_labor.svg.png")
    ),
    (
        ["stages of labor", "labor stages", "first stage of labor", "second stage of labor", "third stage of labor"],
        ("Stages of Labor Progress (Cervical Dilation, Expulsion & Placental Delivery)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Stages_of_labor_progress.svg/1920px-Stages_of_labor_progress.svg.png")
    ),
    (
        ["apgar", "apgar score", "apgar scoring", "newborn apgar", "apgar assessment"],
        ("APGAR Score Clinical Assessment Matrix for Neonatal Vitality",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/APGAR_scoring_matrix.svg/1920px-APGAR_scoring_matrix.svg.png")
    ),
    (
        ["bishop score", "cervical ripening", "induction of labor score", "favorable cervix"],
        ("Bishop Score Assessment Algorithm for Cervical Ripening and Labor Induction",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Bishop_score_flowchart.svg/1920px-Bishop_score_flowchart.svg.png")
    ),
    (
        ["postpartum hemorrhage", "pph", "4 ts of pph", "uterine atony management", "postpartum bleeding algorithm"],
        ("Postpartum Hemorrhage (PPH) 4 Ts Clinical Management Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8f/PPH_management_algorithm.svg/1920px-PPH_management_algorithm.svg.png")
    ),
    (
        ["preeclampsia", "eclampsia", "magnesium sulfate protocol", "gestational hypertension algorithm", "severe preeclampsia"],
        ("Preeclampsia & Eclampsia Clinical Evaluation and Magnesium Sulfate Protocol",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/Preeclampsia_management_algorithm.svg/1920px-Preeclampsia_management_algorithm.svg.png")
    ),
    (
        ["fetal circulation", "ductus venosus", "foramen ovale", "ductus arteriosus", "transitional neonatal circulation"],
        ("Fetal Circulation & Neonatal Transitional Shunt Closures",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5a/Fetal_circulation_diagram.svg/1920px-Fetal_circulation_diagram.svg.png")
    ),
    (
        ["partograph", "partogram", "who partograph", "alert line action line", "cervicograph"],
        ("WHO Modified Partograph Clinical Labor Monitoring Flowchart",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2b/Partograph_clinical_schematic.svg/1920px-Partograph_clinical_schematic.svg.png")
    ),
    (
        ["rh isoimmunization", "rh incompatibility", "anti-d immunoglobulin", "hydrops fetalis", "hemolytic disease of newborn"],
        ("Rhesus Isoimmunization & Hemolytic Disease of the Fetus/Newborn Pathway",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4b/Rh_isoimmunization_pathway.svg/1920px-Rh_isoimmunization_pathway.svg.png")
    ),

    # ==========================================
    # 10. PEDIATRICS
    # ==========================================
    (
        ["imci", "integrated management of childhood", "imci algorithm", "childhood triage", "danger signs child"],
        ("IMCI (Integrated Management of Childhood Illness) Clinical Triage Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a2/IMCI_pediatric_triage_flowchart.svg/1920px-IMCI_pediatric_triage_flowchart.svg.png")
    ),
    (
        ["tetralogy of fallot", "tof", "fallot", "boot-shaped heart", "tet spell"],
        ("Tetralogy of Fallot (4 Cardinal Anatomical & Hemodynamic Defects)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e6/Tetralogy_of_Fallot.svg/1920px-Tetralogy_of_Fallot.svg.png")
    ),
    (
        ["developmental milestones", "milestones timeline", "pediatric milestones", "gross motor fine motor", "red flags development"],
        ("Pediatric Developmental Milestones Timeline (0-5 Years)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/88/Developmental_milestones_timeline.svg/1920px-Developmental_milestones_timeline.svg.png")
    ),
    (
        ["neonatal resuscitation", "nrp", "neonatal resuscitation program", "golden minute newborn", "ppv newborn"],
        ("Neonatal Resuscitation Program (NRP) Stepwise Resuscitation Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/34/Neonatal_resuscitation_algorithm.svg/1920px-Neonatal_resuscitation_algorithm.svg.png")
    ),
    (
        ["dehydration", "pediatric dehydration", "dehydration plan", "who dehydration", "plan a plan b plan c", "diarrhea rehydration", "ors protocol"],
        ("Pediatric Dehydration Assessment & WHO Rehydration Plan A/B/C Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/7/75/WHO_dehydration_plan_ABC.svg/1920px-WHO_dehydration_plan_ABC.svg.png")
    ),
    (
        ["bhutani", "bhutani nomogram", "neonatal jaundice nomogram", "phototherapy", "phototherapy nomogram", "hyperbilirubinemia newborn", "exchange transfusion nomogram"],
        ("Bhutani Hour-Specific Bilirubin Nomogram & Phototherapy Triage",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/25/Bhutani_phototherapy_nomogram.svg/1920px-Bhutani_phototherapy_nomogram.svg.png")
    ),
    (
        ["congenital heart", "congenital heart defect", "congenital heart defects", "chd classification", "cyanotic heart disease", "acyanotic shunt", "5 ts congenital", "cyanotic", "acyanotic"],
        ("Congenital Heart Defects Classification Tree (Cyanotic vs. Acyanotic Lesions)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/1/1a/Congenital_heart_defects_classification.svg/1920px-Congenital_heart_defects_classification.svg.png")
    ),
    (
        ["pals", "pediatric advanced life support", "pediatric cardiac arrest", "pals algorithm"],
        ("Pediatric Advanced Life Support (PALS) Cardiac Arrest Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/82/PALS_cardiac_arrest_algorithm.svg/1920px-PALS_cardiac_arrest_algorithm.svg.png")
    ),
    (
        ["febrile seizure", "febrile convulsion", "simple febrile seizure", "complex febrile seizure algorithm"],
        ("Febrile Seizure Clinical Assessment & Triage Flowchart",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6f/Febrile_convulsion_algorithm.svg/1920px-Febrile_convulsion_algorithm.svg.png")
    ),
    (
        ["immunization schedule", "vaccination schedule", "epi schedule", "pediatric vaccines", "bcg opv pentavalent"],
        ("Expanded Programme on Immunization (EPI) Childhood Vaccination Schedule",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Pediatric_vaccination_schedule.svg/1920px-Pediatric_vaccination_schedule.svg.png")
    ),

    # ==========================================
    # 11. INTERNAL MEDICINE & PATHOLOGY
    # ==========================================
    (
        ["meningitis csf", "lumbar puncture csf", "csf interpretation", "csf analysis algorithm", "bacterial vs viral csf"],
        ("Meningitis CSF Diagnostic Interpretation Decision Tree",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/d/de/Meningitis_CSF_interpretation_algorithm.svg/1920px-Meningitis_CSF_interpretation_algorithm.svg.png")
    ),
    (
        ["jvp", "jugular venous pressure", "jvp waveform", "cannon a wave", "kussmaul sign", "a c v waves"],
        ("Jugular Venous Pressure (JVP) Waveform Analysis & Physiological Correlates",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/JVP_waveform_clinical.svg/1920px-JVP_waveform_clinical.svg.png")
    ),
    (
        ["dka", "diabetic ketoacidosis", "dka protocol", "ketoacidosis management", "dka insulin protocol"],
        ("Diabetic Ketoacidosis (DKA) Clinical Management & Fluid Resuscitation Protocol",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f6/DKA_clinical_management_flowchart.svg/1920px-DKA_clinical_management_flowchart.svg.png")
    ),
    (
        ["acid-base", "acid base disorder", "anion gap algorithm", "metabolic acidosis flowchart", "winter formula"],
        ("Stepwise Acid-Base Disturbance Diagnostic Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/3/30/Acid_base_disorder_algorithm.svg/1920px-Acid_base_disorder_algorithm.svg.png")
    ),
    (
        ["acls cardiac arrest", "acls algorithm", "vf pvt algorithm", "asystole pea algorithm", "adult cardiac arrest"],
        ("ACLS Adult Cardiac Arrest Algorithm (Shockable vs. Non-Shockable Pathways)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/5/52/ACLS_cardiac_arrest_algorithm.svg/1920px-ACLS_cardiac_arrest_algorithm.svg.png")
    ),
    (
        ["acls tachycardia", "tachycardia algorithm", "wide complex tachycardia", "narrow complex tachycardia", "svt algorithm"],
        ("ACLS Tachycardia with Pulse Clinical Management Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0d/ACLS_tachycardia_algorithm.svg/1920px-ACLS_tachycardia_algorithm.svg.png")
    ),
    (
        ["acls bradycardia", "bradycardia algorithm", "symptomatic bradycardia", "atropine pacing"],
        ("ACLS Bradycardia with Pulse Management Algorithm",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/ACLS_bradycardia_algorithm.svg/1920px-ACLS_bradycardia_algorithm.svg.png")
    ),
    (
        ["shock classification", "hypovolemic cardiogenic septic", "distributive shock", "hemodynamic profile shock"],
        ("Classification of Shock & Hemodynamic Diagnostic Flowchart",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/Shock_classification_algorithm.svg/1920px-Shock_classification_algorithm.svg.png")
    ),
    (
        ["portacaval", "portal hypertension", "portosystemic", "esophageal varices", "caput medusae"],
        ("Portacaval Anastomoses & Portal Hypertension Collateral Pathways",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/87/Portacaval_anastomoses.svg/1920px-Portacaval_anastomoses.svg.png")
    ),
    (
        ["granuloma", "tubercular granuloma", "granuloma formation", "granuloma cascade", "tuberculosis immunology", "langhans giant cell", "caseating necrosis", "caseous necrosis", "epithelioid macrophage"],
        ("Tubercular Granuloma Immunological Cascade (Macrophage-T-Cell Interaction)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Granuloma_formation_cascade.svg/1920px-Granuloma_formation_cascade.svg.png")
    ),
    (
        ["atherosclerosis", "atheroma", "foam cell", "fatty streak", "plaque rupture", "atherogenesis"],
        ("Atherosclerosis Pathogenesis & Arterial Plaque Formation Cascade",
         "https://upload.wikimedia.org/wikipedia/commons/d/d1/Blausen_0257_CoronaryArtery_Plaque.png")
    ),
    (
        ["acute coronary syndrome", "acs triage", "stemi nstemi algorithm", "myocardial infarction triage", "timi risk score"],
        ("Acute Coronary Syndrome (ACS / STEMI / NSTEMI) Clinical Triage Flowchart",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8d/ACS_triage_flowchart.svg/1920px-ACS_triage_flowchart.svg.png")
    ),
    (
        ["circle of willis", "circle willis", "cerebral arterial circle", "willis arterial"],
        ("Circle of Willis (Cerebral Arterial Network Architecture)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2e/Circle_of_Willis_en.svg/1920px-Circle_of_Willis_en.svg.png")
    ),
    (
        ["spinal cord tracts", "corticospinal", "spinothalamic", "dorsal columns", "spinal cord cross"],
        ("Spinal Cord Cross-Section & Ascending/Descending Tracts",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/b/b2/Spinal_cord_tracts_-_English.svg/1920px-Spinal_cord_tracts_-_English.svg.png")
    ),
    (
        ["meninges", "dura mater", "arachnoid mater", "pia mater", "meningeal layers"],
        ("Cranial Meninges (Dura, Arachnoid, and Pia Mater Layers)",
         "https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Meninges-en.svg/1920px-Meninges-en.svg.png")
    ),
    (
        ["epidermis", "skin layer", "stratum corneum", "stratum basale", "stratum spinosum"],
        ("Epidermis Strata & Cellular Layers (Anatomical Schematic Cross-Section)",
         "https://upload.wikimedia.org/wikipedia/commons/f/f9/502_Layers_of_epidermis.jpg")
    ),
    (
        ["granuloma slide", "granuloma histology", "caseating granuloma h&e", "tuberculous granuloma slide", "tubercular granuloma biopsy"],
        ("Caseating Tubercular Granuloma (H&E Histology Slide)",
         "https://upload.wikimedia.org/wikipedia/commons/1/10/Granuloma_mac.jpg")
    ),
]

# Pre-compiled regexes for ultra-fast (sub-millisecond) atlas lookups
ATLAS_COMPILED = [
    (re.compile(rf"\b(?:{'|'.join(re.escape(p) for p in patterns)})\b", re.IGNORECASE), title, img_url)
    for patterns, (title, img_url) in VERIFIED_MEDICAL_ATLAS
]

ATLAS_EXACT_MAP = {}
for patterns, (title, img_url) in VERIFIED_MEDICAL_ATLAS:
    for p in patterns:
        ATLAS_EXACT_MAP[p.lower().strip()] = (title, img_url)

# In-memory LRU/dict cache for dynamic queries
DIAGRAM_CACHE = {}

# Concurrency semaphore to throttle outbound Wikipedia requests to max 5 simultaneous
_WIKI_SEMAPHORE = asyncio.Semaphore(5)

async def retrieve_real_medical_diagram(topic_or_candidates, modality: str = "FLOWCHART_SCHEMATIC"):
    """
    1. Fast-Path: Instant lookup against verified medical atlas (0ms, 100% uptime, zero rate limits),
       with modality-aware candidate selection and micrograph rejection.
    2. Cache-Path: Check in-memory diagram cache (0ms).
    3. Dynamic Fallback: Query Wikipedia/Wikimedia with positive schematic decorators,
       connection pooling, semaphore rate-limiting, and micrograph rejection filtering.
    """
    if isinstance(topic_or_candidates, list):
        input_candidates = topic_or_candidates
        raw_topic = " ".join(str(c) for c in topic_or_candidates[:3]).lower()
    else:
        input_candidates = [topic_or_candidates]
        raw_topic = (topic_or_candidates or "").lower()

    # 0. O(1) Exact Match Fast-Path
    stripped_topic = raw_topic.strip()
    if modality != "HISTOLOGY_MICROSCOPY" and stripped_topic in ATLAS_EXACT_MAP:
        title, img_url = ATLAS_EXACT_MAP[stripped_topic]
        if not _reject_micrograph_candidate(title, img_url, modality):
            return img_url, title

    # 1. Check Verified Atlas (Deterministic & Instant 0ms)
    search_texts = [raw_topic] + [str(c).lower() for c in input_candidates if c]

    # Prioritize genuine histology entries when in HISTOLOGY_MICROSCOPY mode
    if modality == "HISTOLOGY_MICROSCOPY":
        for compiled_rx, title, img_url in ATLAS_COMPILED:
            if any(h in title.lower() for h in ["histology", "slide", "biopsy", "smear", "stain", "micrograph"]):
                if any(compiled_rx.search(s) for s in search_texts):
                    return img_url, title

    for compiled_rx, title, img_url in ATLAS_COMPILED:
        if any(compiled_rx.search(s) for s in search_texts):
            if not _reject_micrograph_candidate(title, img_url, modality):
                return img_url, title

    # 2. Check In-Memory Dynamic Cache
    cache_key = f"{modality}:{raw_topic.strip()}"
    if cache_key in DIAGRAM_CACHE:
        return DIAGRAM_CACHE[cache_key]

    # 3. Dynamic Lookup via Wikipedia API with Semaphore Throttling & Schematic Decoration
    import urllib.parse
    headers = {"User-Agent": "NeuraAI-MBBS-Bot/2.0 (contact: medical.support@neura.ai)"}
    
    seen = set()
    search_candidates = []
    for c in input_candidates:
        c = str(c).strip() if c else ""
        if not c or len(c) < 2: continue
        if c.lower() not in seen:
            seen.add(c.lower())
            search_candidates.append(c)
        clean = _DIAGRAM_FILLER_PATTERN.sub(' ', c)
        clean = re.sub(r'\s+', ' ', clean).strip()
        if clean and clean.lower() not in seen and len(clean) > 2:
            seen.add(clean.lower())
            search_candidates.append(clean)

    # Modality decoration for Wikimedia search
    decorated_queries = []
    for cand in search_candidates[:4]:
        if modality == "FLOWCHART_SCHEMATIC":
            decorated_queries.append(f"{cand} diagram")
            decorated_queries.append(f"{cand} flowchart")
            decorated_queries.append(cand)
        elif modality == "HISTOLOGY_MICROSCOPY":
            decorated_queries.append(f"{cand} histology")
            decorated_queries.append(cand)
        elif modality == "ANATOMICAL_MAP":
            decorated_queries.append(f"{cand} anatomy")
            decorated_queries.append(cand)
        else:
            decorated_queries.append(cand)

    search_url = "https://en.wikipedia.org/w/api.php"
    
    try:
        async with _WIKI_SEMAPHORE:
            async with httpx.AsyncClient(timeout=8.0, limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)) as client:
                for cand in decorated_queries[:6]:
                    if not cand or len(cand) < 2: continue
                    search_params = {"action": "opensearch", "search": cand, "limit": 5, "namespace": 0, "format": "json"}
                    s_res = await client.get(search_url, params=search_params, headers=headers)
                    if s_res.status_code == 200:
                        s_data = s_res.json()
                        if asyncio.iscoroutine(s_data):
                            s_data = await s_data
                        titles = s_data[1] if isinstance(s_data, (list, tuple)) and len(s_data) > 1 else []
                        for title in titles:
                            if _reject_micrograph_candidate(title, "", modality):
                                continue
                            encoded_title = urllib.parse.quote(title.replace(" ", "_"))
                            sum_res = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{encoded_title}", headers=headers)
                            if sum_res.status_code == 200:
                                data = sum_res.json()
                                if asyncio.iscoroutine(data):
                                    data = await data
                                img_url = data.get("originalimage", {}).get("source") or data.get("thumbnail", {}).get("source")
                                if img_url and any(img_url.lower().endswith(ext) or ext in img_url.lower() for ext in [".jpg", ".jpeg", ".png", ".svg"]):
                                    if not any(bad in img_url.lower() for bad in ["symbol", "icon", "stub", "question_mark", "disambig"]):
                                        if not _reject_micrograph_candidate(title, img_url, modality):
                                            DIAGRAM_CACHE[cache_key] = (img_url, title)
                                            return img_url, title
    except Exception as e:
        logger.warning(f"Dynamic diagram fetch fallback: {e}")
        
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
                f"CRITICAL INSTRUCTION: Jump straight into the answer starting directly with 📖 *IN-DEPTH EXPLANATION*. Do NOT start your response with 'Based on the retrieved context', 'According to', 'Certainly', 'Here is', 'I have attached', or any similar robotic preamble or conversational filler. Absolutely NEVER cite fabricated figure numbers (e.g. 'Figure X-Y'). Just provide the structured medical explanation directly."
            )
        else:
            user_prompt = (
                f"RETRIEVED TEXTBOOK CONTEXT:\n{formatted_context}\n\n"
                f"STUDENT QUESTION:\n{query_to_search}\n\n"
                f"CRITICAL INSTRUCTION: Jump straight into the answer starting directly with 📖 *IN-DEPTH EXPLANATION*. Do NOT start your response with 'Based on the retrieved context', 'According to', 'Certainly', 'Here is', 'I have attached', or any similar robotic preamble or conversational filler. Absolutely NEVER cite fabricated figure numbers (e.g. 'Figure X-Y'). Just provide the structured medical explanation directly."
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
        if intent != "QUIZ" and not is_not_covered:
            visual_modality = detect_visual_intent_modality(query_to_search, ai_answer)
            if visual_modality != "NONE":
                try:
                    img_url, img_title = await retrieve_real_medical_diagram(diagram_candidates, modality=visual_modality)
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
        user_prompt = (
            f"RETRIEVED TEXTBOOK CONTEXT:\n{formatted_context}\n\n"
            f"STUDENT QUESTION:\n{user_msg}\n\n"
            f"CRITICAL INSTRUCTION: Jump straight into the answer starting directly with 📖 *IN-DEPTH EXPLANATION*. Do NOT start your response with 'Based on the retrieved context', 'According to', 'Certainly', 'Here is', 'I have attached', or any similar robotic preamble or conversational filler. Absolutely NEVER cite fabricated figure numbers (e.g. 'Figure X-Y'). Just provide the structured medical explanation directly."
        )
        
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
