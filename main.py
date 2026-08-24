import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import os
import re
import json
import logging
import traceback
import httpx
import hmac
import hashlib
from datetime import datetime, timedelta
import time
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.background import BackgroundTask
from pydantic import BaseModel
import numpy as np
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
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY", "")
FLUTTERWAVE_SECRET_KEY = os.getenv("FLUTTERWAVE_SECRET_KEY", "")
FLUTTERWAVE_SECRET_HASH = os.getenv("FLUTTERWAVE_SECRET_HASH", "neura_flw_hash_2026")
BASE_URL = os.getenv("BASE_URL", "https://neura-ai-df6q.onrender.com")

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

from collections import OrderedDict
import time

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
shared_http_client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_keepalive_connections=30, max_connections=60))
embedding_pool = ThreadPoolExecutor(max_workers=4)

def get_embedding_sync(text: str):
    return list(embedder.embed(text))[0]

print(f"MONGO_URI Present: {bool(MONGO_URI)}")
mongo_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = mongo_client.neura_db if mongo_client else None
chat_history_col = db.chat_history if db is not None else None
users_col = db.users if db is not None else None

async def update_user_study_streak(user_id: str) -> int:
    """Updates user's daily study streak based on calendar date (WAT / UTC+1)."""
    if users_col is None:
        return 1
    
    try:
        now_wat = datetime.utcnow() + timedelta(hours=1)
        today_str = now_wat.strftime("%Y-%m-%d")
        yesterday_str = (now_wat - timedelta(days=1)).strftime("%Y-%m-%d")
        
        user = await users_col.find_one({"user_id": user_id})
        if not user:
            await users_col.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "study_streak_days": 1,
                        "last_study_date": today_str,
                        "last_active_timestamp": time.time()
                    }
                },
                upsert=True
            )
            return 1
            
        last_study_date = user.get("last_study_date", "")
        current_streak = user.get("study_streak_days", 0)
        
        if last_study_date == today_str:
            await users_col.update_one(
                {"user_id": user_id},
                {"$set": {"last_active_timestamp": time.time()}}
            )
            return current_streak or 1
        elif last_study_date == yesterday_str:
            new_streak = current_streak + 1
        else:
            new_streak = 1
            
        await users_col.update_one(
            {"user_id": user_id},
            {
                "$set": {
                    "study_streak_days": new_streak,
                    "last_study_date": today_str,
                    "last_active_timestamp": time.time()
                }
            },
            upsert=True
        )
        return new_streak
    except Exception as e:
        print(f"Error updating study streak: {e}")
        return 1

async def check_and_send_inactivity_reminders(force_ignore_quiet_hours: bool = False, simulated_hour: int = None):
    """Background worker to send Duolingo-style study streak reminders to users inactive for 8-12 hours between 6am and 11pm WAT."""
    if users_col is None:
        return
        
    try:
        now_wat = datetime.utcnow() + timedelta(hours=1)
        current_hour = simulated_hour if simulated_hour is not None else now_wat.hour
        
        # Only send between 6:00 AM and 11:00 PM WAT unless force_ignore_quiet_hours is True
        if not force_ignore_quiet_hours and (current_hour < 6 or current_hour >= 23):
            return
            
        today_str = now_wat.strftime("%Y-%m-%d")
        now_ts = time.time()
        min_inactive_sec = 8 * 3600   # 8 hours
        max_inactive_sec = 13 * 3600  # 13 hours (safely within 24h Meta service window)
        
        cursor = users_col.find({
            "last_active_timestamp": {
                "$gte": now_ts - max_inactive_sec,
                "$lte": now_ts - min_inactive_sec
            },
            "last_reminder_sent_date": {"$ne": today_str},
            "reminders_enabled": {"$ne": False}
        })
        
        async for user in cursor:
            phone = user.get("user_id")
            if not phone:
                continue
                
            name = user.get("name", "Student")
            streak = user.get("study_streak_days", 0)
            last_study = user.get("last_study_date", "")
            last_topic = user.get("last_medical_topic", "High-Yield Clinical Concepts")
            
            # Formulate streak nudge message
            if streak > 1 and last_study != today_str:
                streak_msg = (
                    f"🔥 *{streak}-DAY STUDY STREAK AT RISK!* 🧊\n\n"
                    f"Hey *{name}*, don't let your study streak freeze today! Complete a quick 2-minute review or MCQ drill to keep your momentum alive for MBBS exams.\n\n"
                    f"Ready to conquer your next clinical topic?"
                )
            elif streak == 1 and last_study != today_str:
                streak_msg = (
                    f"🔥 *KEEP YOUR 1-DAY STREAK ALIVE!* ⚡\n\n"
                    f"Hey *{name}*, consistency is what turns good students into top clinicians! Review 1 concept today to grow your streak to *2 Days*.\n\n"
                    f"What medical topic are we breaking down right now?"
                )
            else:
                streak_msg = (
                    f"🔥 *START YOUR STUDY STREAK TODAY!* 🩺\n\n"
                    f"Hey *{name}*, time for your quick daily study check-in! Ask a question or complete a practice quiz to kickstart your Daily Streak.\n\n"
                    f"What medical topic from your textbooks should we tackle?"
                )
            
            topic_snippet = last_topic[:100] if last_topic else "High-Yield Clinical Concepts"
            
            # Send message with 1-tap interactive practice buttons
            await send_whatsapp_interactive_button(
                phone,
                streak_msg,
                [
                    {"id": f"GENERATE_QUIZ:{topic_snippet}", "title": "📝 Practice Daily MCQs"},
                    {"id": "START_STUDY_SESSION", "title": "📚 Start Study Session"}
                ]
            )
            
            # Mark reminder sent date to ensure strict 1-per-day cap
            await users_col.update_one(
                {"user_id": phone},
                {"$set": {"last_reminder_sent_date": today_str}}
            )
            print(f"[NUDGE] Sent streak reminder to {phone} (Streak: {streak} days)")
            
    except Exception as e:
        print(f"Error in inactivity reminder worker: {e}")

async def start_inactivity_reminder_loop():
    """Background periodic loop running every 30 minutes."""
    while True:
        try:
            await asyncio.sleep(1800) # 30 minutes
            await check_and_send_inactivity_reminders()
        except asyncio.CancelledError:
            break
        except Exception as loop_err:
            print(f"Error in reminder loop: {loop_err}")
            await asyncio.sleep(60)

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
        
    # Launch inactivity streak reminder worker in the background
    asyncio.create_task(start_inactivity_reminder_loop())

class LRUTopicCache:
    """High-speed in-memory 24-hour LRU cache for authoritative textbook explanations (~4KB per topic, max 1000 topics = ~4MB RAM)."""
    def __init__(self, maxsize=1000, ttl_seconds=86400):
        self.cache = OrderedDict()
        self.maxsize = maxsize
        self.ttl = ttl_seconds

    def _make_key(self, query: str, preferred_books: list = None) -> str:
        clean_q = re.sub(r'[^\w\s]', '', query.strip().lower())
        books_key = "_".join(sorted([b.lower()[:12] for b in (preferred_books or []) if b and not b.startswith("Skip")]))
        return f"{clean_q}::{books_key}"

    def get(self, query: str, preferred_books: list = None):
        key = self._make_key(query, preferred_books)
        if key in self.cache:
            entry = self.cache[key]
            if time.time() - entry["timestamp"] < self.ttl:
                self.cache.move_to_end(key)
                return entry["answer"], entry.get("context", "")
            else:
                del self.cache[key]
        return None, None

    def set(self, query: str, answer: str, context: str = "", preferred_books: list = None):
        if not answer or len(answer) < 50:
            return
        key = self._make_key(query, preferred_books)
        if key in self.cache:
            self.cache.move_to_end(key)
        elif len(self.cache) >= self.maxsize:
            self.cache.popitem(last=False)
        self.cache[key] = {
            "answer": answer,
            "context": context,
            "timestamp": time.time()
        }

TOPIC_CACHE = LRUTopicCache(maxsize=1000, ttl_seconds=86400)

class QueryRequest(BaseModel):
    user_id: str
    message: str

# ==========================================
# 2. SYSTEM PROMPTS & INTENT ROUTER
# ==========================================
SYSTEM_MEDICAL_PROMPT = """{user_context}You are NEURA AI, an elite medical expert, clinical professor, and co-pilot designed for Nigerian medical students.
Your goal is to explain clinical concepts, pathophysiological mechanisms, diagnostic criteria, and pharmacotherapies authoritatively, clearly, and naturally.

CLINICAL EXPLANATION & EXPERT VOICE RULES:
1. AUTHORITATIVE CLINICAL EXPERT VOICE: Respond like a top-tier medical professor and expert AI assistant. Deliver factual, comprehensive, and accurate explanations directly from the knowledge provided.
2. TOPIC-ANCHORED HEADING: Every response MUST start with a clear, topic-anchored heading formatted as:
   📖 *[TOPIC NAME: CLINICAL FOCUS]*
   (e.g., 📖 *SHIGELLA: PATHOPHYSIOLOGY & CLINICAL MANIFESTATIONS*, 📖 *SODIUM CHANNEL BLOCKERS: DISSOCIATION KINETICS*, 📖 *LOBAR PNEUMONIA: PHASES & PATHOLOGY*).
3. CLINICAL OVERVIEW FIRST: Immediately after the topic heading, provide a concise 1-2 sentence high-level overview or definition so anyone reading instantly understands what pathogen, condition, drug class, or physiological mechanism is being discussed before diving into detailed sub-sections. Never start abruptly with a random isolated complication or single bullet point.
4. STRUCTURED SUB-SECTIONS: Organize into clear, logically ordered sections separated by double line breaks:
   - *Core Pathophysiology / Mechanism of Action / Etiology:* Detailed step-by-step breakdown.
   - *Clinical Manifestations / Signs & Symptoms / Diagnostic Findings:* Key clinical presentations.
   - *Complications & High-Yield Associations:* Important consequences.
5. ZERO TEXTBOOK META-TALK & ZERO SOURCE REFERENCES: Absolutely NEVER use phrases like "your textbook explains", "your textbook also lists", "the textbook states", "according to the text", "the retrieved context mentions", "in your selected textbooks", or refer to textbooks/sources in your explanation. Never write meta-commentary about where facts come from. Jump straight into the clinical breakdown and state the facts directly with expert authority.
6. NO PREAMBLES & NO FILLER: Never use opening filler, greetings, or announcements (e.g., "Certainly Samuel!", "Certainly!", "Absolutely!", "Sure thing!", "Here is the breakdown you requested", "Based on the retrieved context..."). Jump DIRECTLY into the medical explanation starting with your topic header: 📖 *[TOPIC NAME: CLINICAL FOCUS]*. Zero conversational filler.
7. ZERO CITATION BLOCKS & ZERO FOOTERS: Absolutely NEVER output citation footers, '📚 CITATIONS', or list textbook names at the end. Explain the core medical facts directly in your own structured words.
8. SIMPLIFY COMPLEX MECHANISMS: Whenever explaining complex biochemistry, pharmacology kinetics, or high-level pathology, break down the mechanisms step-by-step with clear, intuitive reasoning so students can grasp the concepts effortlessly.
9. HIGHLIGHT IMPORTANT TERMS: Bold key clinical terms, drug names, and diagnostic criteria (*Drug Name*, *Phase 0*, *5-HIAA*) so the text is visually crisp and easy to scan.
10. CONVERSATIONAL SOCRATIC CO-PILOT: Engage students naturally. When they ask hypothetical "what if" questions or follow-ups, synthesize clinical principles with common-sense medical reasoning.

CRITICAL FORMATTING & LAYOUT RULES FOR WHATSAPP:
1. DOUBLE-LINE SPACING: Every section heading, sub-heading, bullet item, and paragraph MUST be separated by a full blank line (`\n\n`). Never stack bullet items or headers back-to-back on consecutive single lines.
2. SHORT PARAGRAPHS: Keep text blocks short (maximum 2 to 3 sentences per paragraph) so the message feels spacious and easy to read on mobile screens.
3. NO HASHTAGS OR RAW SYMBOLS: Never use `#`, `##`, or `###` headers. Never output raw visible asterisks.
4. BOLD HEADINGS & KEY TERMS: Use valid WhatsApp bold syntax (*Heading:* or *Key Term*) for all section headings, sub-headings, and key concepts so WhatsApp renders them in clean bold text.
5. PROPERLY INDENTED LISTS: Format bullet lists neatly using hyphens and proper double-line spacing:
   
   - *Main Section:* Clear explanation.
   
     - *Sub-detail:* Properly indented supporting detail.
6. ZERO FABRICATED FIGURE CITATIONS: Absolutely NEVER invent, cite, or mention specific figure numbers, diagram numbers, plate numbers, or table numbers (e.g., NEVER write "Figure 46-9", "Fig 12.8", "Figure 43.5", "Plate 3-1", "Table 14.2", "Robbins p. 787").
7. NO SMASHED WORDS: Ensure flawless spacing between words, punctuation, and hyphens. Always put a space after colons, periods, and bold words (e.g. write "*Prazosin* is" instead of "*Prazosin*is").
8. Structure responses into clear sections separated by blank lines:
   📖 *[TOPIC NAME: CLINICAL FOCUS]*
   
   [1-2 sentence high-level clinical overview / definition]
   
   💡 *KEY CLINICAL PEARLS*
9. NO RAW MARKDOWN TABLES: WhatsApp does NOT render markdown tables. NEVER output pipes or table headers (| Col 1 | Col 2 | or |---|---|). Always structure comparisons and summary tables as clean bulleted list cards (e.g. - *Category:* followed by indented • *Detail:* bullets).
10. WHEN ASKED FOR DIAGRAMS/ILLUSTRATIONS: NEVER apologize or say 'I cannot generate or display diagrams' or 'I am only a text AI'. You ARE fully equipped with a real medical diagram and flowchart retrieval system that automatically delivers the authentic textbook schematic below your explanation. Confidently provide the structured breakdown with clear headings and bullet cards. Do NOT announce or refer to figure numbers — the system delivers the visual asset seamlessly.
"""

SYSTEM_QUIZ_PROMPT = """{user_context}You are NEURA AI. Based on the retrieved medical context, generate exactly 7 rigorous, medical-school standard (MBBS / USMLE Step 1 & 2 style) Multiple Choice Questions (MCQs).

RULES FOR MCQs:
1. NO PREAMBLES & NO CONVERSATIONAL FILLER: Start immediately with Question 1. Never include introductory conversational chatter, greetings, or announcements.
2. Each question must present a realistic clinical vignette, physiological mechanism, or pharmacological scenario appropriate for medical students. Never cite fabricated figure or table numbers in vignettes.
3. Provide 4 distinct options (A, B, C, D) for each of the 7 questions.
4. Structure your response clearly using WhatsApp Markdown:
   - List Question 1 through 7 with their options (A, B, C, D).
   - Provide a separate 🔑 *ANSWER KEY & DETAILED EXPLANATIONS* section at the bottom.
   - For every answer, explain why the correct option is right AND why the key distractor options are wrong. State the clinical rationale directly without meta-commentary about textbooks or sources.
"""

SYSTEM_INTERACTIVE_QUIZ_PROMPT = """You are NEURA AI, an elite medical study assistant and co-pilot for MBBS students.
Your task is to generate exactly 5 rigorous, high-yield, medical-school standard (MBBS / USMLE Step 1 & 2 style) Multiple Choice Questions that test the student DIRECTLY and EXCLUSIVELY on the MEDICAL EXPLANATION AND CLINICAL CONCEPTS provided in the prompt.

CRITICAL RULES:
1. STRICT LESSON COHESION: Every single question (all 5) MUST test core concepts, pathophysiology, clinical presentations, diagnostic criteria, or treatments directly discussed in the medical explanation just taught to the student. Never ask questions about unrelated topics.
2. VIGNETTES & MECHANISMS: Provide realistic clinical vignettes or mechanism-of-action questions appropriate for medical students based on the lesson material.
3. 4 DISTINCT OPTIONS: Provide 4 options (A, B, C, D) with exactly 1 unambiguous correct answer and 3 clinically plausible distractors.
4. CLINICAL RATIONALE: Provide a clear, authoritative explanation of why the correct option is right and why key distractors are wrong based on the provided explanation.
5. STRICT JSON OUTPUT ONLY: Output ONLY a valid JSON array of 5 question objects. No markdown formatting, no code block backticks (no ```json), and no introductory or concluding commentary.

JSON Schema format:
[
  {
    "q_num": 1,
    "vignette": "A clinical vignette or mechanism question specifically testing the concepts taught in the explanation...",
    "option_a": "First option",
    "option_b": "Second option",
    "option_c": "Third option",
    "option_d": "Fourth option",
    "correct_option": "A",
    "explanation": "Clear clinical explanation of why A is correct based on the lesson and why distractors are incorrect.",
    "book_source": "High-Yield Clinical Concepts"
  }
]
"""

async def classify_intent(message: str) -> str:
    msg_clean = message.strip().lower()
    
    # 1. Quizzes & Exam Testing Intent (Instant Fast-Path)
    if any(k in msg_clean for k in ["mcq", "quiz", "practice question", "test me", "exam question", "generate_quiz", "practice mcqs"]):
        return "QUIZ"

    # 2. Obvious Gratitude / Acknowledgment / Nigerian Slang Fast-Path (<0.05ms)
    gratitude_patterns = [
        r"\b(thank\s*(you|u)|thanks|thx|ty|tysm|appreciate|god\s*bless(\s*you)?|nice\s*one|well\s*done|welldone|good\s*job|great\s*job)\b",
        r"\byou('re|\s*are)?\s*(smart|good|the\s*best|great|awesome|helpful)\b"
    ]
    if any(re.search(pat, msg_clean) for pat in gratitude_patterns):
        return "GRATITUDE"

    if msg_clean in ["ok", "okay", "k", "alright", "cool", "noted", "got it", "understood", "makes sense", "i see", "nice", "great", "awesome", "perfect", "good", "fine", "correct", "yes", "yep", "yeah"]:
        return "ACKNOWLEDGMENT"

    greeting_patterns = [
        r"\b(hi|hello|hey|heya|yo|sup|wassup|what'?s\s*up|good\s*(morning|afternoon|evening|day)|greetings)\b",
        r"\b(how\s*far|wetin\s*dey|how\s*body|how\s*you\s*dey|how\s*things|kedu|bawo|sannu)\b",
        r"\b(boss\s*man|senior\s*man|chief|my\s*guy|boss)\b",
        r"\b(how\s*(are\s*you|r\s*u|is\s*it\s*going|you\s*doing|are\s*you\s*doing|everything))\b",
        r"\b(who\s*are\s*you|what\s*is\s*neura(\s*ai)?|what\s*can\s*you\s*do|introduce\s*yourself)\b",
        r"\b(hallo|wie\s*geht'?s|guten\s*tag|servus|moin|bonjour|salut|cava|comment\s*ca\s*va|hola|buenos\s*dias)\b"
    ]
    if any(re.search(pat, msg_clean) for pat in greeting_patterns) and len(msg_clean.split()) <= 4:
        return "GREETING"

    # 3. Unambiguous Medical Questions (Fast-Path to vector search)
    terms = extract_medical_terms(message)
    known_medical_indicators = ["syndrome", "disease", "treatment", "pathology", "pharmacology", "anatomy", "physiology", "symptoms", "diagnosis", "mechanism", "pathophysiology", "infection", "bacteria", "virus", "artery", "nerve", "muscle", "bone", "cell", "receptor", "drug", "inhibitor", "agonist", "antagonist", "furosemide", "prazosin", "malaria", "pneumonia", "diabetes", "hypertension", "anemia", "carcinoid", "hypersensitivity"]
    if any(ind in msg_clean for ind in known_medical_indicators) and len(msg_clean.split()) >= 2:
        return "MEDICAL"

    # 4. Deterministic Gibberish & Noise Filter (y is treated as vowel)
    clean_alpha = re.sub(r'[^a-zA-Z]', '', msg_clean)
    has_long_consonants = bool(re.search(r'[bcdfghjklmnpqrstvwxz]{5,}', msg_clean))
    is_gibberish_pattern = bool(
        not clean_alpha or
        re.match(r'^[^a-zA-Z0-9\s]+$', msg_clean) or
        re.match(r'^[0-9\s.,]+$', msg_clean) or
        (len(clean_alpha) >= 4 and not any(v in clean_alpha for v in "aeiouy")) or
        has_long_consonants or
        re.search(r'(.)\1{3,}', msg_clean)
    )
    if msg_clean and is_gibberish_pattern:
        return "GIBBERISH"

    # 4. LLM Universal Intent Classifier (Handles ANY language: German, French, Arabic, slang, gibberish, vague chatter)
    if OPENROUTER_API_KEY:
        try:
            router_prompt = (
                "You are an intent classifier for NEURA AI, a medical study co-pilot.\n"
                "Analyze the user's message (which could be English, Nigerian Pidgin/slang, German, French, Arabic, Yoruba, Igbo, Hausa, or any language) and classify it into EXACTLY ONE label:\n"
                "- GREETING: Greetings, hello, how are you, Nigerian slang (e.g. 'how far', 'boss man', 'wetin dey'), foreign greetings (e.g. German 'wie gehts', French 'bonjour', 'kedu'), introductions ('who are you', 'what can you do').\n"
                "- GRATITUDE: Thank you, thanks, nice one, well done, praise, appreciation in any language.\n"
                "- ACKNOWLEDGMENT: Short confirmations (ok, cool, noted, got it, understood, alright).\n"
                "- GIBBERISH: Random keyboard mash, nonsense characters (e.g. 'asdfgh', '12345', '????'), meaningless noise.\n"
                "- QUIZ: Explicit requests for MCQs, practice questions, quizzes, tests.\n"
                "- MEDICAL: Any question, concept, disease, pharmacology, physiology, anatomy, or medical topic query in any language.\n\n"
                "Output ONLY the category name in uppercase with no punctuation."
            )
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://neura-ai.org",
                "X-Title": "NEURA AI Intent Router"
            }
            payload = {
                "model": "deepseek/deepseek-v4-flash",
                "messages": [
                    {"role": "system", "content": router_prompt},
                    {"role": "user", "content": message}
                ],
                "temperature": 0.0,
                "max_tokens": 10
            }
            resp = await shared_http_client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                cat = resp.json()["choices"][0]["message"]["content"].strip().upper()
                for valid in ["GREETING", "GRATITUDE", "ACKNOWLEDGMENT", "GIBBERISH", "QUIZ", "MEDICAL"]:
                    if valid in cat:
                        return valid
        except Exception as e:
            print(f"LLM Intent Classifier fallback error: {e}")

    # Default fallback
    if len(msg_clean) <= 4 or not terms:
        return "GIBBERISH"
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
        "max_tokens": 900,
        "provider": {
            "order": ["DeepSeek", "Together", "Fireworks", "Hyperbolic", "Novita"],
            "allow_fallbacks": True
        }
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
                                    
                                    # Fast first-chunk dispatch (<120 chars on newline) so student gets instant answer in <1.5s
                                    if chunk_sent_count == 0:
                                        if ("\n" in current_chunk and len(current_chunk) > 90) or len(current_chunk) > 180:
                                            split_idx = current_chunk.rfind("\n")
                                            if split_idx == -1: split_idx = len(current_chunk)
                                            send_part = current_chunk[:split_idx].strip()
                                            if send_part:
                                                await send_whatsapp_cloud_msg(sender_phone, send_part)
                                                chunk_sent_count += 1
                                                current_chunk = current_chunk[split_idx:].strip()
                                    else:
                                        if "\n\n" in current_chunk and len(current_chunk) > 650:
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

def strip_textbook_meta_talk(text: str) -> str:
    """Removes meta-commentary mentioning 'your textbook explains', 'the textbook states', 'according to your textbook', etc., making the AI speak directly and authoritatively."""
    if not text:
        return text

    patterns = [
        # "Your textbook explains that ", "The textbook states that ", "The textbook notes that "
        r'(?i)\b(?:your|the)\s+(?:selected\s+)?(?:textbooks?|texts?|contexts?)\s+(?:explains?|states?|notes?|highlights?|describes?|indicates?|mentions?|shows?)\s+that\s+',
        # "Your textbook explains, ", "The textbook states, "
        r'(?i)\b(?:your|the)\s+(?:selected\s+)?(?:textbooks?|texts?|contexts?)\s+(?:explains?|states?|notes?|highlights?|describes?|indicates?|mentions?|shows?)[,\s:]+',
        # "Your textbook also lists ", "Your textbook also mentions "
        r'(?i)\b(?:your|the)\s+(?:selected\s+)?(?:textbooks?|texts?|contexts?)\s+also\s+(?:lists?|mentions?|describes?|notes?|highlights?)\s+',
        # "According to your textbook, ", "According to the textbook, ", "Based on your textbook, "
        r'(?i)\b(?:according to|based on|as stated in|as described in|as explained in|as noted in)\s+(?:your|the)\s+(?:selected\s+)?(?:textbooks?|texts?|contexts?)[,\s:]*',
        # "In your textbook, ", "In your selected textbooks, "
        r'(?i)\bin\s+(?:your|the)\s+(?:selected\s+)?(?:textbooks?|texts?)[,\s:]*',
        # "The retrieved context from [Book] explains that "
        r'(?i)\b(?:the\s+)?retrieved\s+(?:contexts?|materials?|informations?)(?:\s+from\s+[A-Za-z\s]+?)?\s+(?:explains?|states?|describes?|indicates?|shows?|mentions?)\s+(?:that\s+)?',
        r'(?i)\b(?:the\s+)?retrieved\s+(?:contexts?|materials?|informations?)(?:\s+from\s+[A-Za-z\s]+?)?[,\s:]+',
    ]

    for pat in patterns:
        text = re.sub(pat, '', text)

    # Capitalize the start of sentences if the stripped phrase left a lowercase letter at the start of a line or after punctuation
    def capitalize_match(m):
        prefix = m.group(1)
        char = m.group(2)
        return prefix + char.upper()

    # Start of line
    text = re.sub(r'(?m)^([ \t]*)([a-z])', capitalize_match, text)
    # After period/question/exclamation and space
    text = re.sub(r'([.!?]\s+)([a-z])', capitalize_match, text)
    # After bullet hyphen/asterisk
    text = re.sub(r'(?m)^([ \t]*[-*]\s+)([a-z])', capitalize_match, text)

    return text

def format_whatsapp_text(text: str) -> str:
    """Master WhatsApp text sanitizer. Fixes layout, tables, preambles, figure citations, textbook meta-talk, and bolding without destroying text."""
    if not text:
        return text

    # 0. Convert raw markdown tables to readable bullet cards & remove --- lines
    text = convert_markdown_tables_to_whatsapp_cards(text)

    # 1. Deterministically strip opening preambles & conversational greetings
    text = strip_conversational_preambles(text)

    # 2. Deterministically strip fabricated figure/table citations
    text = strip_figure_citations(text)

    # 2.2. Deterministically strip textbook meta-talk ("your textbook explains that", "according to your textbook", etc.)
    text = strip_textbook_meta_talk(text)

    # 2.5. Deterministically strip any citation section or textbook footnote lists
    text = re.sub(r'(?mi)^\s*📚\s*\*?CITATIONS\*?[\s\S]*$', '', text)
    text = re.sub(r'(?mi)^\s*\*?CITATIONS:\*?[\s\S]*$', '', text)
    text = re.sub(r'(?mi)^\s*📚\s*\*?TEXTBOOK CITATIONS\*?[\s\S]*$', '', text)
    text = re.sub(r'(?mi)^\s*-\s+(?:Robbins|Lippincott|Guyton|Moore|Hoffbrand|Jawetz|Sembulingam|Katzung|Ganong|Kumar)[^\n]*$', '', text)

    # 3. Remove markdown hashes (e.g. ### Header -> Header)
    text = re.sub(r'^\s*#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = text.replace('###', '').replace('##', '')

    # 3.5. Ensure section headers with emojis are formatted with bold tags (e.g. 📖 IN-DEPTH EXPLANATION -> 📖 *IN-DEPTH EXPLANATION*)
    text = re.sub(r'(?m)^([📖💡🎯🔑])\s*([^*\n]+?)\s*$', r'\1 *\2*', text)

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
    """Sends an Interactive CTA URL Button that opens directly in WhatsApp's in-app webview"""
    body_text = format_whatsapp_text(body_text)
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
    """Marks incoming message as read and activates WhatsApp native floating typing indicator bubble (<50ms)"""
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
        "message_id": message_id,
        "typing_indicator": {
            "type": "text"
        }
    }
    try:
        res = await shared_http_client.post(url, headers=headers, json=payload)
        if res.status_code != 200:
            # Fallback if typing_indicator is rejected by older API version
            fallback_payload = {
                "messaging_product": "whatsapp",
                "status": "read",
                "message_id": message_id
            }
            await shared_http_client.post(url, headers=headers, json=fallback_payload)
    except Exception as e:
        print(f"⚠️ Read/Typing status error: {e}")

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

    # Step 3: If image bytes could not be downloaded server-side, abort sending media to prevent Meta 131053 error
    print(f"⚠️ Image could not be fetched server-side from {image_url}. Aborting image delivery to protect WhatsApp delivery status.")
    return



async def initialize_flutterwave_transaction(amount_ngn: int, email: str, phone: str, name: str = "Student"):
    """Initializes a Flutterwave payment and returns the hosted checkout URL"""
    import uuid
    url = "https://api.flutterwave.com/v3/payments"
    headers = {
        "Authorization": f"Bearer {FLUTTERWAVE_SECRET_KEY.strip()}",
        "Content-Type": "application/json"
    }
    reference = f"NEURA_{phone}_{uuid.uuid4().hex[:8]}"
    payload = {
        "tx_ref": reference,
        "amount": str(amount_ngn),
        "currency": "NGN",
        "payment_options": "banktransfer,card",
        "redirect_url": f"{BASE_URL}/api/payment-complete",
        "customer": {
            "email": email,
            "phonenumber": phone,
            "name": name
        },
        "customizations": {
            "title": "NEURA AI Wallet Top-Up",
            "description": f"NEURA AI MBBS Study Assistant (₦{amount_ngn:,})",
            "logo": "https://raw.githubusercontent.com/Samueldmh/Neura-AI/main/assets/logo.png"
        }
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            data = res.json()
            if data.get("status") == "success" and "data" in data and "link" in data["data"]:
                return data["data"]["link"]
            print(f"Flutterwave init response: {data}")
    except Exception as e:
        print(f"Error initializing Flutterwave: {e}")
    return None

async def initialize_paystack_transaction(amount_ngn: int, email: str, phone: str):
    """Initializes a Paystack transaction and returns the checkout URL (Fallback)"""
    import uuid
    url = "https://api.paystack.co/transaction/initialize"
    headers = {
        "Authorization": f"Bearer {PAYSTACK_SECRET_KEY.strip()}",
        "Content-Type": "application/json"
    }
    reference = f"NEURA_{phone}_{uuid.uuid4().hex[:8]}"
    payload = {
        "amount": amount_ngn * 100,
        "email": email,
        "reference": reference,
        "callback_url": f"{BASE_URL}/api/payment-complete",
        "metadata": {
            "phone_number": phone,
            "custom_fields": [
                {
                    "display_name": "Phone Number",
                    "variable_name": "phone_number",
                    "value": phone
                },
                {
                    "display_name": "Product",
                    "variable_name": "NEURA AI Wallet Credit"
                }
            ]
        }
    }
    try:
        async with httpx.AsyncClient(timeout=20.0) as client:
            res = await client.post(url, headers=headers, json=payload)
            data = res.json()
            if data.get("status"):
                return data["data"]["authorization_url"]
            print(f"Paystack init failed: {data}")
    except Exception as e:
        print(f"Error initializing Paystack: {e}")
    return None

async def send_deposit_menu(sender_phone: str, current_balance: float):
    """Sends the hybrid deposit menu with quick presets and custom amount prompt"""
    body_text = (
        f"💳 *NEURA AI Wallet Top-Up*\n\n"
        f"• Current Balance: *₦{current_balance:.2f}*\n"
        f"• Minimum Deposit: *₦500*\n\n"
        f"Tap a quick tier below, or reply with any custom amount (e.g. *500*, *1000*, *1500*, or */deposit 500*):"
    )
    options = [
        {"id": "DEPOSIT_500", "title": "₦500 Deposit", "description": "~180 In-Depth Medical Explanations"},
        {"id": "DEPOSIT_1500", "title": "₦1,500 Deposit", "description": "~550 In-Depth Medical Explanations"},
        {"id": "DEPOSIT_3000", "title": "₦3,000 Deposit", "description": "~1,200 In-Depth Medical Explanations"},
    ]
    await send_whatsapp_interactive_list(sender_phone, body_text, "Select Deposit", options)

async def handle_deposit_request(sender_phone: str, user_msg: str) -> bool:
    """Parses deposit selection or custom typed amount and sends an In-App Flutterwave CTA button"""
    msg_trim = user_msg.strip().upper()
    amount_ngn = None
    if msg_trim == "DEPOSIT_500":
        amount_ngn = 500
    elif msg_trim == "DEPOSIT_1500":
        amount_ngn = 1500
    elif msg_trim == "DEPOSIT_3000":
        amount_ngn = 3000
    elif msg_trim == "DEPOSIT_5000":
        amount_ngn = 5000
    elif msg_trim == "DEPOSIT_10000":
        amount_ngn = 10000
    elif msg_trim == "DEPOSIT_20000":
        amount_ngn = 20000
    else:
        m = re.match(r'^(?:/deposit\s+)?(?:₦|NGN\s*)?(\d{1,3}(?:,\d{3})*|\d+)(?:\.00)?$', user_msg.strip(), re.IGNORECASE)
        if m:
            try:
                amount_ngn = int(m.group(1).replace(',', ''))
            except:
                pass

    if amount_ngn is None:
        return False

    if amount_ngn < 500:
        await send_whatsapp_cloud_msg(
            sender_phone,
            "⚠️ Minimum deposit amount is *₦500*.\n\nPlease choose ₦500 or more (e.g. *500*, *1000*, *2000*, *5000*)."
        )
        return True

    email = f"user_{sender_phone.replace('+', '')}@neura.ai"
    
    # Try Flutterwave first, fallback to Paystack if configured
    auth_url = None
    if FLUTTERWAVE_SECRET_KEY:
        auth_url = await initialize_flutterwave_transaction(amount_ngn, email, sender_phone)
    elif PAYSTACK_SECRET_KEY:
        auth_url = await initialize_paystack_transaction(amount_ngn, email, sender_phone)

    if auth_url:
        card_body = (
            f"💳 *NEURA AI In-App Checkout*\n\n"
            f"• Amount: *₦{amount_ngn:,}*\n"
            f"• Payment Gateway: *Flutterwave*\n"
            f"• Status: *Ready*\n\n"
            f"Tap the button below to complete your deposit directly inside WhatsApp (Supports all Nigerian Cards, Bank Transfer & USSD):"
        )
        await send_whatsapp_cta_url_button(sender_phone, card_body, f"Pay ₦{amount_ngn:,} Now", auth_url)
    else:
        await send_whatsapp_cloud_msg(
            sender_phone,
            "Sorry, we couldn't generate the payment link right now. Please verify your payment gateway setup or try again in a moment!"
        )
    return True

async def send_commands_menu(sender_phone: str):
    """Sends an interactive WhatsApp List containing all available slash commands with 1-tap execution"""
    body_text = (
        "📋 *NEURA AI Commands Menu*\n\n"
        "Tap a command below to execute it instantly, or type any of them directly into the chat:"
    )
    options = [
        {"id": "/wallet", "title": "💳 /wallet", "description": "Check balance, total spent & queries remaining"},
        {"id": "/deposit", "title": "💰 /deposit", "description": "Top up your study wallet (min ₦500)"},
        {"id": "/profile", "title": "👤 /profile", "description": "View study streak, class, balance & textbooks"},
        {"id": "/reminders on", "title": "🔔 /reminders on", "description": "Toggle daily study streak reminders on/off"},
        {"id": "/update books", "title": "📚 /update books", "description": "Change or add your preferred medical textbooks"},
        {"id": "/update level", "title": "🎓 /update level", "description": "Update your current class/level (e.g. 400L)"},
        {"id": "/update name", "title": "✏️ /update name", "description": "Update your student display name"},
        {"id": "/clearwallet", "title": "🗑️ /clearwallet", "description": "Reset wallet balance to ₦0.00 (for testing)"},
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

MEDICAL_TYPOS_MAP = {
    "disassociation": "dissociation",
    "disassociate": "dissociate",
    "metabolsim": "metabolism",
    "pharamcology": "pharmacology",
    "pharmaclogy": "pharmacology",
    "pathyphysiology": "pathophysiology",
    "pathophyisology": "pathophysiology",
    "hypetension": "hypertension",
    "arrythmia": "arrhythmia",
    "arythmia": "arrhythmia",
    "arrhthmia": "arrhythmia",
    "antiarrhytmic": "antiarrhythmic",
    "antiarrythmic": "antiarrhythmic",
    "antiarrthymic": "antiarrhythmic",
    "glomeular": "glomerular",
    "glomerlar": "glomerular",
    "heamatology": "haematology",
    "haemtology": "haematology",
    "mycardial": "myocardial",
    "infarction": "infarction",
    "pnuemonia": "pneumonia",
    "pnemonia": "pneumonia",
    "pathoiof": "pathophysiology of",
    "pathoof": "pathophysiology of",
    "pathologyof": "pathology of",
    "pathophys": "pathophysiology",
    "patho": "pathology",
    "shigela": "shigella",
    "salmonela": "salmonella",
    "falx cerebrii": "falx cerebri"
}

MEDICAL_ACRONYMS_MAP = {
    r'\ball\b': 'acute lymphoblastic leukemia (ALL)',
    r'\baml\b': 'acute myeloid leukemia (AML)',
    r'\bcll\b': 'chronic lymphocytic leukemia (CLL)',
    r'\bcml\b': 'chronic myeloid leukemia (CML)',
    r'\bdka\b': 'diabetic ketoacidosis (DKA)',
    r'\bgerd\b': 'gastroesophageal reflux disease (GERD)',
    r'\bdvt\b': 'deep vein thrombosis (DVT)',
    r'\bpe\b': 'pulmonary embolism (PE)',
    r'\bards\b': 'acute respiratory distress syndrome (ARDS)',
    r'\bdic\b': 'disseminated intravascular coagulation (DIC)',
    r'\bsle\b': 'systemic lupus erythematosus (SLE)',
    r'\bmen1\b': 'multiple endocrine neoplasia type 1 (MEN1)',
    r'\bmen2\b': 'multiple endocrine neoplasia type 2 (MEN2)',
    r'\bmen2a\b': 'multiple endocrine neoplasia type 2A (MEN2A)',
    r'\bmen2b\b': 'multiple endocrine neoplasia type 2B (MEN2B)',
    r'\braas\b': 'renin angiotensin aldosterone system (RAAS)',
    r'\bcopd\b': 'chronic obstructive pulmonary disease (COPD)',
}

def extract_json_from_llm(raw_text: str):
    """Robust JSON extractor that handles markdown fences, leading conversational text, trailing notes, and single quotes."""
    if not raw_text:
        return None
    cleaned = re.sub(r'```json\s*', '', raw_text)
    cleaned = re.sub(r'```\s*', '', cleaned).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Extract JSON Array [...]
    m_list = re.search(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL)
    if m_list:
        try:
            return json.loads(m_list.group(0))
        except Exception:
            pass

    # Extract JSON Object {...}
    m_dict = re.search(r'\{.*\}', raw_text, re.DOTALL)
    if m_dict:
        try:
            return json.loads(m_dict.group(0))
        except Exception:
            pass

    return None

def clean_medical_topic_title(raw_query: str, corrected_topic: str = "") -> str:
    """Extracts a clean, authoritative, title-cased medical topic name from raw student queries."""
    if corrected_topic and len(corrected_topic.strip()) >= 3:
        topic = corrected_topic.strip().strip('"\'')
        topic = re.sub(r'^(?:Medical Topic:\s*|Topic:\s*)', '', topic, flags=re.IGNORECASE).strip()
        for typo, correction in MEDICAL_TYPOS_MAP.items():
            topic = re.sub(rf'\b{re.escape(typo)}\b', correction, topic, flags=re.IGNORECASE)
        if topic and len(topic) >= 3:
            return topic.title()

    text = raw_query.strip()
    # 0. Expand medical acronyms (e.g. "ALL" -> "Acute Lymphoblastic Leukemia")
    for pat, repl in MEDICAL_ACRONYMS_MAP.items():
        if re.search(pat, text, flags=re.IGNORECASE):
            text = re.sub(pat, repl, text, flags=re.IGNORECASE)
            break

    # 1. Normalize typos
    for typo, correction in MEDICAL_TYPOS_MAP.items():
        text = re.sub(rf'\b{re.escape(typo)}\b', correction, text, flags=re.IGNORECASE)

    # 2. Strip conversational query framing
    framing_patterns = [
        r'^(?:can you\s+)?(?:tell me about|explain|what is|what are|describe|discuss|give me|how does|outline|overview of|notes on|details on|summary of)\s+',
        r'^(?:i want to know about|let us talk about|let\'s discuss|briefly explain)\s+',
    ]
    for pat in framing_patterns:
        text = re.sub(pat, '', text, flags=re.IGNORECASE).strip()

    # 3. Clean up leading/trailing punctuation
    text = re.sub(r'[?!.,:;]+$', '', text).strip()

    if not text:
        text = "High-Yield Clinical Concepts"

    return text.title()

def extract_medical_terms(user_msg: str) -> list:
    """Instantly extract clean medical keywords by normalizing typos, expanding clinical synonyms/subclasses, preserving short medical terms (B-cell, T-cell), and stripping filler words."""
    msg = user_msg
    # 0. Expand clinical acronyms before stop-word removal (prevents "ALL", "AML", "DKA" from being stripped)
    for pat, repl in MEDICAL_ACRONYMS_MAP.items():
        if re.search(pat, msg, flags=re.IGNORECASE):
            msg = re.sub(pat, repl, msg, flags=re.IGNORECASE)
            break

    # 1. Normalize common student typos
    for typo, correction in MEDICAL_TYPOS_MAP.items():
        msg = re.sub(rf'\b{re.escape(typo)}\b', correction, msg, flags=re.IGNORECASE)
        
    msg = re.sub(r'\b([bt])\s*cel\b', r'\1 cell', msg, flags=re.IGNORECASE)
    msg = re.sub(r'\bcel\b', 'cell', msg, flags=re.IGNORECASE)
    
    msg_cleaned = re.sub(r'[^\w\s]', ' ', msg)
    words = msg_cleaned.split()
    
    # Check lowercase for stop words, but preserve essential short medical abbreviations
    meaningful_words = [
        w for w in words 
        if (w.lower() in SPECIAL_SHORT_MEDICAL or len(w) > 2) and w.lower() not in SEARCH_STOP_WORDS
    ]
    
    if not meaningful_words:
        return [msg.strip()]
        
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

    # 2. Clinical Domain Multi-Angle Expansion
    joined_lower = joined_query.lower()
    if "sodium channel" in joined_lower and ("block" in joined_lower or "kinetic" in joined_lower or "dissociation" in joined_lower or "rate" in joined_lower):
        for exp in ["class I antiarrhythmics sodium channel kinetics", "class IA IB IC rate of dissociation recovery", "sodium channel blockers use dependence unbinding"]:
            if exp not in phrases:
                phrases.append(exp)
    elif "action potential" in joined_lower:
        for exp in ["cardiac action potential phases ion currents", "ventricular action potential phase 0 1 2 3 4"]:
            if exp not in phrases:
                phrases.append(exp)
    elif "pneumonia" in joined_lower:
        for exp in ["lobar pneumonia stages congestion red grey hepatization", "bronchopneumonia pathology histology"]:
            if exp not in phrases:
                phrases.append(exp)
    elif "carcinoid" in joined_lower:
        for exp in ["carcinoid syndrome serotonin 5-HIAA flushing diarrhea", "neuroendocrine tumor carcinoid heart disease"]:
            if exp not in phrases:
                phrases.append(exp)
        
    print(f"[SEARCH] Extracted search keywords: {phrases} (from: '{user_msg}')")
    return phrases

async def normalize_medical_query(user_msg: str) -> dict:
    """Upfront Micro-LLM normalizer: resolves typos, abbreviations, and expands clinical concepts into standard textbook search queries."""
    fallback_result = {"search_keywords": extract_medical_terms(user_msg), "corrected_topic": user_msg}
    if not OPENROUTER_API_KEY:
        return fallback_result

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neura-ai.org",
        "X-Title": "NEURA AI Medical Assistant"
    }
    system_prompt = (
        "You are an expert MBBS medical query normalizer. Medical students frequently send questions with medical acronyms (e.g. 'ALL' for Acute Lymphoblastic Leukemia, 'AML', 'CLL', 'CML', 'DKA', 'GERD', 'DVT', 'PE', 'ARDS', 'DIC', 'SLE', 'MEN1', 'RAAS', 'COPD', 'ITP', 'TTP'), shorthand, slang, typos (e.g. 'disassociation', 'pnuemonia', 'arrythmia'), or in other languages.\n"
        "1. Dynamically recognize any medical acronyms or abbreviations, fix any typos, and resolve the query to proper clinical terminology.\n"
        "2. Generate 2 to 4 authoritative medical textbook search queries (including pharmacological classes, anatomical names, or physiological processes).\n"
        "Output ONLY a valid JSON object in this exact schema:\n"
        "{\n"
        '  "corrected_topic": "Acute Lymphoblastic Leukemia Symptoms",\n'
        '  "search_keywords": ["acute lymphoblastic leukemia symptoms", "ALL clinical features presentation", "lymphoblast bone marrow failure"]\n'
        "}\n"
        "Output ONLY valid JSON."
    )
    payload = {
        "model": "deepseek/deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "provider": {
            "order": ["DeepSeek", "Together", "Fireworks", "Hyperbolic", "Novita"],
            "allow_fallbacks": True
        }
    }
    try:
        res = await shared_http_client.post(url, headers=headers, json=payload)
        if res.status_code == 200:
            text = res.json()["choices"][0]["message"]["content"]
            parsed = extract_json_from_llm(text)
            if isinstance(parsed, dict) and "search_keywords" in parsed:
                return parsed
    except Exception as e:
        print(f"⚠️ Micro-LLM normalizer error: {e}")

    return fallback_result

async def evaluate_retrieval_adequacy(user_msg: str, retrieved_points: list) -> dict:
    """Evaluates whether retrieved textbook chunks sufficiently cover the student's question before declaring a topic missing.
    If inadequate due to synonym mismatch, narrow search, or hierarchical difference, returns re-anchored queries for second-pass scan.
    """
    if not OPENROUTER_API_KEY or not retrieved_points:
        return {"is_adequate": bool(retrieved_points), "is_genuinely_absent": not bool(retrieved_points), "re_anchored_queries": []}

    # Prepare compact context summary (first 180 chars of each chunk)
    context_summaries = []
    for idx, p in enumerate(retrieved_points[:8], 1):
        payload = p.payload
        b_title = payload.get("book_title", "Textbook")
        snippet = payload.get("text", "")[:200].replace("\n", " ")
        context_summaries.append(f"[{idx}. {b_title}]: {snippet}...")
    
    combined_context_summary = "\n".join(context_summaries)

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neura-ai.org",
        "X-Title": "NEURA AI Medical Assistant"
    }
    system_prompt = (
        "You are an expert MBBS medical retrieval evaluator. A student asked a clinical question, and the vector database returned initial textbook snippets.\n"
        "Determine if the retrieved text sufficiently covers the core question or if the retrieval missed the specific topic due to terminology mismatch, subclass distinctions (e.g. Class IA/IB/IC antiarrhythmics), anatomical hierarchy, or phrasing differences.\n\n"
        "Output ONLY a valid JSON object in this schema:\n"
        "{\n"
        '  "is_adequate": true,\n'
        '  "is_genuinely_absent": false,\n'
        '  "re_anchored_queries": ["query 1", "query 2", "query 3"]\n'
        "}\n"
        "If 'is_adequate' is false and it is a valid medical concept, provide 2 to 3 broader or alternative authoritative textbook search queries (e.g. parent chapter titles, drug class mechanisms, anatomical systems). Only set 'is_genuinely_absent' to true if the question is genuinely non-medical or completely nonexistent in medical curricula.\n"
        "Output ONLY valid JSON."
    )
    user_payload_text = (
        f"STUDENT QUESTION: {user_msg}\n\n"
        f"RETRIEVED TEXTBOOK CONTEXT SNIPPETS:\n{combined_context_summary}"
    )
    payload = {
        "model": "deepseek/deepseek-v4-flash",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload_text}
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "provider": {
            "order": ["DeepSeek", "Together", "Fireworks", "Hyperbolic", "Novita"],
            "allow_fallbacks": True
        }
    }
    try:
        res = await shared_http_client.post(url, headers=headers, json=payload)
        if res.status_code == 200:
            text = res.json()["choices"][0]["message"]["content"]
            parsed = extract_json_from_llm(text)
            if isinstance(parsed, dict) and "is_adequate" in parsed:
                return parsed
    except Exception as e:
        print(f"⚠️ Micro-LLM retrieval evaluator error: {e}")

    return {"is_adequate": True, "is_genuinely_absent": False, "re_anchored_queries": []}

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

async def search_single_book(query_vector: list, book: str, limit: int = 4) -> list:
    if not book or not isinstance(book, str) or book.startswith("Skip"):
        return []
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
        if res.points:
            return res.points
    except Exception:
        pass

    # Fuzzy keyword fallback
    book_kw = ""
    b_lower = book.lower()
    if "lippincott" in b_lower: book_kw = "lippincott"
    elif "robbins" in b_lower: book_kw = "robbins"
    elif "haematology" in b_lower or "hoffbrand" in b_lower: book_kw = "haematology"
    elif "microbiology" in b_lower or "jawetz" in b_lower: book_kw = "microbiology"
    elif "sembulingam" in b_lower: book_kw = "sembulingam"
    elif "moore" in b_lower or "anatomy" in b_lower: book_kw = "moore"

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
            return res.points
        except Exception:
            pass
    return []

async def search_qdrant(query_text: str, limit: int = 8, preferred_books: list = None) -> list:
    """Search Qdrant in PARALLEL across all selected textbooks for sub-second retrieval."""
    try:
        loop = asyncio.get_running_loop()
        query_vector = await loop.run_in_executor(embedding_pool, get_embedding_sync, query_text)

        if not preferred_books:
            res = await qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit
            )
            return res.points

        # Query all selected textbooks concurrently in parallel!
        tasks = [search_single_book(query_vector, b, limit=limit) for b in preferred_books if b and not b.startswith("Skip")]
        book_results = await asyncio.gather(*tasks)
        all_points = [p for sub in book_results for p in sub]
        all_points.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        return all_points

    except Exception as outer_e:
        print(f"❌ Error in search_qdrant: {outer_e}")
        return []

async def multi_search_qdrant(search_terms: list, preferred_books: list = None) -> list:
    """Run separate Qdrant searches for each extracted medical keyword CONCURRENTLY, with automatic cross-textbook safety net if single book context is sparse."""
    seen_texts = set()
    all_results = []
    
    # Run all searches concurrently across preferred books with limit=8
    tasks = [search_qdrant(term, limit=8, preferred_books=preferred_books) for term in search_terms]
    results_list = await asyncio.gather(*tasks)
    
    for results in results_list:
        for point in results:
            text_key = point.payload.get("text", "")[:120]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_results.append(point)
    
    # Cross-Textbook Safety Net: If preferred book returned < 5 chunks, also search across all textbooks in parallel!
    if len(all_results) < 5 and preferred_books:
        print(f"[CROSS-BOOK SAFETY NET] Preferred books returned only {len(all_results)} chunks. Searching across full medical library...")
        fallback_tasks = [search_qdrant(term, limit=8, preferred_books=None) for term in search_terms[:3]]
        fallback_results_list = await asyncio.gather(*fallback_tasks)
        for results in fallback_results_list:
            for point in results:
                text_key = point.payload.get("text", "")[:120]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    all_results.append(point)
    
    # Sort points by score descending and cap at 15 points
    all_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
    all_results = all_results[:15]
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
        
async def start_interactive_quiz(sender_phone: str, topic: str, search_res: list = None, context_text: str = "", explanation_text: str = ""):
    """Generates 5 structured MCQs as JSON strictly grounded in the medical explanation just given to the student."""
    source_material = ""
    if explanation_text and len(explanation_text.strip()) > 50:
        source_material = explanation_text.strip()
    elif context_text and len(context_text.strip()) > 50:
        source_material = context_text.strip()
    elif chat_history_col is not None:
        try:
            user_hist = await chat_history_col.find_one({"user_id": sender_phone})
            if user_hist and "messages" in user_hist:
                for m in reversed(user_hist["messages"]):
                    if m.get("role") == "assistant" and len(m.get("content", "")) > 80:
                        source_material = m.get("content")
                        break
        except Exception:
            pass

    if not source_material:
        context_blocks = []
        for idx, point in enumerate((search_res or [])[:10], 1):
            p = point.payload
            book_str = p.get('book_title', 'Textbook')
            text_str = p.get('text', '')
            context_blocks.append(f"[Chunk {idx} | Book: {book_str}]\n{text_str}")
        source_material = "\n\n".join(context_blocks)

    if not source_material or len(source_material.strip()) < 50:
        # Fallback search if context is sparse
        try:
            terms = extract_medical_terms(topic)
            fallback_pts = await multi_search_qdrant(terms)
            context_blocks = [f"[Chunk {i+1} | Book: {p.payload.get('book_title', 'Textbook')}]\n{p.payload.get('text', '')}" for i, p in enumerate(fallback_pts[:8])]
            source_material = "\n\n".join(context_blocks)
        except Exception:
            pass

    user_prompt = (
        f"TARGET MEDICAL TOPIC:\n{topic}\n\n"
        f"MEDICAL EXPLANATION PROVIDED TO STUDENT:\n{source_material}\n\n"
        f"CRITICAL INSTRUCTION: Generate exactly 5 medical-school standard MCQs (with options A, B, C, D) that test the student DIRECTLY and EXCLUSIVELY on the concepts, mechanisms, signs/symptoms, and clinical pearls taught in the medical explanation above. Do not ask questions about unrelated topics. Return ONLY the valid JSON array of 5 objects."
    )

    try:
        json_raw = await call_openrouter_llm(SYSTEM_INTERACTIVE_QUIZ_PROMPT, user_prompt)
        quiz_questions = extract_json_from_llm(json_raw)
        
        if not isinstance(quiz_questions, list) or len(quiz_questions) == 0:
            raise ValueError(f"LLM did not return a valid list of questions. Raw output: {json_raw[:200]}")

        quiz_state = {
            "topic": topic,
            "questions": quiz_questions,
            "current_idx": 0,
            "score": 0
        }
        if users_col is not None:
            await users_col.update_one(
                {"user_id": sender_phone},
                {"$set": {"active_quiz": quiz_state}}
            )

        await send_quiz_question(sender_phone, quiz_state)

    except Exception as e:
        print(f"❌ Error starting interactive quiz: {e}")
        await send_whatsapp_cloud_msg(sender_phone, "Sorry, I had trouble creating the interactive practice questions. Please tap *📝 Practice MCQs* again!")

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

        # Update daily study streak and activity timestamp
        streak = await update_user_study_streak(sender_phone)

        # Handle Start Study Session button from reminder
        if user_msg == "START_STUDY_SESSION":
            study_prompt = (
                f"Welcome back to your study session, *{name}*! 🧠⚡\n\n"
                f"What medical topic, clinical case, or drug mechanism from your textbooks are we mastering right now?"
            )
            await send_whatsapp_cloud_msg(sender_phone, study_prompt)
            return

        # Check for profile and wallet commands first
        msg_lower = user_msg.strip().lower()
        if (msg_lower.startswith("/") or msg_lower in ["topup_wallet", "start_deposit", "clearwallet", "deposit", "wallet", "balance", "clear wallet", "help", "menu", "commands", "reminders on", "reminders off"]):
            if msg_lower in ["/", "/help", "help", "menu", "commands", "/menu", "/commands", "/start"]:
                await send_commands_menu(sender_phone)
                return

            if msg_lower in ["/clearwallet", "/clear_wallet", "/clear wallet", "/resetwallet", "/reset_wallet", "/emptywallet", "/empty_wallet", "clearwallet"]:
                if users_col is not None:
                    await users_col.update_one({"user_id": sender_phone}, {"$set": {"wallet_balance_ngn": 0.0}}, upsert=True)
                await send_whatsapp_cloud_msg(
                    sender_phone,
                    "🗑️ *Wallet Cleared!*\n\nYour wallet balance has been reset to *₦0.00* for testing.\n\nType */deposit* to test depositing funds again, or ask a question to test the low-balance prompt!"
                )
                return

            if msg_lower in ["/wallet", "/balance", "wallet", "balance"]:
                balance = user_doc.get("wallet_balance_ngn", 0.0) if user_doc else 0.0
                spent = user_doc.get("total_spent_ngn", 0.0) if user_doc else 0.0
                est_queries = int(balance // 2.75)
                wallet_msg = (
                    f"💳 *NEURA AI Wallet*\n\n"
                    f"• Available Balance: *₦{balance:.2f}*\n"
                    f"• Total Spent: *₦{spent:.2f}*\n"
                    f"• Estimated Queries Remaining: *~{est_queries}*\n\n"
                    f"Type */deposit* to top up your wallet with any custom amount (min ₦500)!"
                )
                await send_whatsapp_interactive_button(
                    sender_phone,
                    wallet_msg,
                    [{"id": "TOPUP_WALLET", "title": "💳 Deposit ₦500+"}]
                )
                return

            if msg_lower in ["/deposit", "/topup", "topup_wallet", "start_deposit", "deposit", "topup"]:
                balance = user_doc.get("wallet_balance_ngn", 0.0) if user_doc else 0.0
                await send_deposit_menu(sender_phone, balance)
                return

            if msg_lower.startswith("/deposit ") or msg_lower.startswith("deposit "):
                handled = await handle_deposit_request(sender_phone, user_msg)
                if handled:
                    return

            if users_col is not None:
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
                    balance = user_doc.get("wallet_balance_ngn", 0.0) if user_doc else 0.0
                    streak_count = user_doc.get("study_streak_days", streak) if user_doc else streak
                    reminders_status = "Enabled 🔔" if (user_doc and user_doc.get("reminders_enabled", True)) else "Disabled 🔕"
                    await send_whatsapp_cloud_msg(
                        sender_phone, 
                        f"👤 *Your Profile*\n• Name: {name}\n• Level: {level}\n• Study Streak: 🔥 {streak_count} Days\n• Reminders: {reminders_status}\n• Wallet Balance: ₦{balance:.2f}\n• Books:\n  - {books_str}\n\n"
                        f"📝 *Feedback Survey:* https://forms.gle/dNr7SV5EUiqiFySx5"
                    )
                    return
                elif msg_lower in ["/reminders on", "/reminder on", "reminders on"]:
                    await users_col.update_one({"user_id": sender_phone}, {"$set": {"reminders_enabled": True}}, upsert=True)
                    await send_whatsapp_cloud_msg(
                        sender_phone,
                        "🔔 *Study Streak Reminders Enabled!*\n\nNEURA AI will gently safeguard your streak if you're inactive for 8–12 hours (between 6:00 AM and 11:00 PM WAT)."
                    )
                    return
                elif msg_lower in ["/reminders off", "/reminder off", "reminders off"]:
                    await users_col.update_one({"user_id": sender_phone}, {"$set": {"reminders_enabled": False}}, upsert=True)
                    await send_whatsapp_cloud_msg(
                        sender_phone,
                        "🔕 *Study Streak Reminders Paused.*\n\nYou can re-enable them anytime by typing */reminders on*."
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

        # Handle deposit menu selection (e.g. DEPOSIT_500) or custom amount entry
        handled_deposit = await handle_deposit_request(sender_phone, user_msg)
        if handled_deposit:
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

        # Low balance guard (< ₦20)
        wallet_balance = user_doc.get("wallet_balance_ngn", 0.0) if user_doc else 0.0
        if wallet_balance < 20.0:
            low_bal_card = (
                f"⚠️ *Insufficient Wallet Balance (₦{wallet_balance:.2f})*\n\n"
                f"To continue asking clinical questions and practicing MBBS MCQs, please top up your wallet (minimum deposit is ₦500).\n\n"
                f"Tap below to deposit:"
            )
            await send_whatsapp_interactive_button(
                sender_phone,
                low_bal_card,
                [{"id": "TOPUP_WALLET", "title": "💳 Deposit ₦500+"}]
            )
            return

        msg_clean_for_quiz = user_msg.strip()
        msg_lower_quiz = msg_clean_for_quiz.lower()
        
        is_quiz_trigger = (
            msg_clean_for_quiz.startswith("GENERATE_QUIZ") or
            msg_lower_quiz in [
                "📝 practice mcqs", "practice mcqs", "practice mcq", "start quiz", "test me", 
                "quiz on this", "quiz me on this", "quiz me", "mcqs", "practice questions", "generate mcqs"
            ]
        )

        if is_quiz_trigger:
            quiz_topic = ""
            if msg_clean_for_quiz.startswith("GENERATE_QUIZ:"):
                quiz_topic = msg_clean_for_quiz.replace("GENERATE_QUIZ:", "").strip()
            
            # If quiz_topic is missing, vague, or a button label, retrieve from user_doc["last_medical_topic"]
            if (not quiz_topic or 
                check_is_followup_query(quiz_topic) or 
                len(quiz_topic) < 3 or 
                quiz_topic.lower() in ["practice mcqs", "quiz", "test me", "mcqs"]):
                quiz_topic = user_doc.get("last_medical_topic", "") if user_doc else ""
                
            if not quiz_topic:
                # Check chat history for last user medical message
                if chat_history_col is not None:
                    user_hist = await chat_history_col.find_one({"user_id": sender_phone})
                    if user_hist and "messages" in user_hist:
                        for m in reversed(user_hist["messages"]):
                            content = m.get("content", "")
                            if m.get("role") == "user" and not content.startswith("GENERATE_QUIZ") and not check_is_followup_query(content):
                                quiz_topic = content
                                break
                                
            if not quiz_topic:
                quiz_topic = "High-Yield Clinical Concepts"

            # Pull the exact explanation the student just received!
            last_explanation = user_doc.get("last_assistant_answer", "") if user_doc else ""
            if not last_explanation and user_doc:
                last_explanation = user_doc.get("last_context_text", "")

            await start_interactive_quiz(sender_phone, quiz_topic.title(), explanation_text=last_explanation)
            return

        query_to_search = user_msg
        intent = await classify_intent(user_msg)
        
        if intent == "GREETING":
            clean_msg = user_msg.strip().lower()
            is_intro = any(w in clean_msg for w in ["who are you", "what is neura", "what can you do", "introduce yourself", "wer bist du", "qui es-tu"])
            is_german = any(w in clean_msg for w in ["wie geht", "hallo", "guten tag", "servus", "moin", "alles gut"])
            is_french = any(w in clean_msg for w in ["bonjour", "salut", "ca va", "comment ca va"])
            is_slang = any(w in clean_msg for w in ["how far", "boss man", "wetin", "my guy", "chief", "senior man", "yo", "wassup", "sup", "boss", "kedu", "bawo", "sannu"])
            
            if is_intro:
                intro_msg = (
                    f"Hello *{name}*! 👋 I am *NEURA AI* 🧠⚡ — your elite medical study co-pilot engineered specifically for MBBS students!\n\n"
                    f"🩺 *What I do:*\n"
                    f"• Deliver instant, high-yield clinical breakdowns and in-depth pathophysiological explanations.\n"
                    f"• Drill you with interactive 1-by-1 USMLE/MBBS practice MCQs with instant feedback.\n"
                    f"• Simplify complex biochemical and drug mechanisms with crystal clarity.\n\n"
                    f"What medical topic or clinical case are we mastering today?"
                )
                await send_whatsapp_cloud_msg(sender_phone, intro_msg)
                return
            elif is_german:
                greeting_msg = (
                    f"Hallo *{name}*! 👋 Ich bin *NEURA AI* 🧠⚡ — dein medizinischer Lernassistent und Co-Pilot für dein Medizinstudium!\n\n"
                    f"Welches medizinische Thema oder welchen klinischen Fall möchtest du heute durchgehen?"
                )
                await send_whatsapp_cloud_msg(sender_phone, greeting_msg)
                return
            elif is_french:
                greeting_msg = (
                    f"Bonjour *{name}*! 👋 Je suis *NEURA AI* 🧠⚡ — ton assistant d'études médicales et co-pilote pour tes études de médecine!\n\n"
                    f"Quel sujet médical ou cas clinique veux-tu explorer aujourd'hui?"
                )
                await send_whatsapp_cloud_msg(sender_phone, greeting_msg)
                return
            elif is_slang:
                greeting_msg = (
                    f"Boss man! I dey sharp and ready. 🧠⚡\n\n"
                    f"How is study / clinical postings going today, *{name}*?\n\n"
                    f"What medical topic, clinical case, or drug mechanism are we breaking down right now?"
                )
                await send_whatsapp_cloud_msg(sender_phone, greeting_msg)
                return
            else:
                greeting_msg = (
                    f"Hello *{name}*! 👋 Welcome to *NEURA AI* — Your Personal Medical Co-Pilot! 🧠⚡\n\n"
                    f"What medical topic, clinical case, or concept are we mastering today?"
                )
                await send_whatsapp_cloud_msg(sender_phone, greeting_msg)
                return

        if intent == "GRATITUDE":
            gratitude_msg = (
                f"You're very welcome, *{name}*! 🩺 Happy to help you master this concept.\n\n"
                f"Whenever you're ready for the next topic, clinical scenario, or MCQ drill, just drop it here!"
            )
            await send_whatsapp_cloud_msg(sender_phone, gratitude_msg)
            return

        if intent == "ACKNOWLEDGMENT":
            ack_msg = (
                f"Awesome, *{name}*! 💡\n\n"
                f"Whenever you want to explore the next topic, ask a follow-up, or practice some MCQs, I'm right here."
            )
            await send_whatsapp_cloud_msg(sender_phone, ack_msg)
            return

        if intent == "GIBBERISH":
            gibberish_msg = (
                f"I didn't quite catch that, *{name}*! 🧐\n\n"
                f"Type a medical condition, drug mechanism, anatomical structure, or clinical case (e.g. *Carcinoid Syndrome*, *Prazosin*, *Lobar Pneumonia*, *Tetralogy of Fallot*), and I will dive straight into an in-depth breakdown!"
            )
            await send_whatsapp_cloud_msg(sender_phone, gibberish_msg)
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

        # ⚡ Step 0: Instant In-Memory Cache Check (<1ms lookup for repeat high-yield questions)
        if intent != "QUIZ" and not is_followup:
            cached_answer, cached_context = TOPIC_CACHE.get(search_term, preferred_books=preferred_books_list)
            if cached_answer:
                clean_topic = clean_medical_topic_title(search_term)
                print(f"[CACHE HIT ⚡] Returning instant cached explanation for '{clean_topic}'")
                await send_whatsapp_cloud_msg(sender_phone, cached_answer)
                
                if chat_history_col is not None:
                    new_msgs = [
                        {"role": "user", "content": query_to_search},
                        {"role": "assistant", "content": cached_answer}
                    ]
                    await chat_history_col.update_one(
                        {"user_id": sender_phone},
                        {"$push": {"messages": {"$each": new_msgs}}},
                        upsert=True
                    )
                if users_col is not None:
                    await users_col.update_one(
                        {"user_id": sender_phone},
                        {
                            "$set": {
                                "last_medical_topic": clean_topic,
                                "last_context_text": cached_context or cached_answer[:2000],
                                "last_assistant_answer": cached_answer
                            }
                        }
                    )
                
                clean_topic_label = clean_topic
                if len(clean_topic_label) > 45:
                    clean_topic_label = clean_topic_label[:42] + "..."
                topic_snippet = clean_topic[:100]
                await send_whatsapp_interactive_button(
                    sender_phone,
                    f"Ready to practice MCQs on *{clean_topic_label}*?",
                    [
                        {"id": f"GENERATE_QUIZ:{topic_snippet}", "title": "📝 Practice MCQs"}
                    ]
                )
                return

        # ⚡ Step 1: Speculative Parallel Execution (Concurrent Vector Retrieval + Typo Normalization)
        local_terms = extract_medical_terms(search_term)
        active_books = get_explicit_book_override(search_term, preferred_books_list)

        # Launch vector search and micro-LLM normalizer simultaneously in parallel
        task_norm = normalize_medical_query(search_term)
        task_search = multi_search_qdrant(local_terms, preferred_books=active_books)

        normalized_data, search_res = await asyncio.gather(task_norm, task_search)

        clean_topic = clean_medical_topic_title(search_term, normalized_data.get("corrected_topic", ""))
        medical_terms = normalized_data.get("search_keywords") or local_terms

        # ⚡ Fast-Path Check: If local search returned >= 3 high-confidence chunks, bypass evaluator completely!
        is_high_confidence = (
            len(search_res) >= 3 and 
            any(getattr(p, 'score', 0) >= 0.70 for p in search_res)
        )

        eval_result = {"is_adequate": True, "is_genuinely_absent": False}

        if not is_high_confidence:
            # Fallback path for ambiguous or 0-chunk queries
            if not search_res:
                print(f"[SEARCH FALLBACK] 0 chunks with local terms. Re-querying with normalized terms: {medical_terms}")
                search_res = await multi_search_qdrant(medical_terms, preferred_books=active_books)

            eval_result = await evaluate_retrieval_adequacy(search_term, search_res)
            
            if not eval_result.get("is_adequate", True) and not eval_result.get("is_genuinely_absent", False):
                re_anchored = eval_result.get("re_anchored_queries", [])
                if re_anchored:
                    print(f"[SELF-CORRECTING RETRIEVAL] Context judged inadequate for '{search_term}'. Re-querying full medical library with: {re_anchored}")
                    second_pass_res = await multi_search_qdrant(re_anchored, preferred_books=None)
                    if second_pass_res:
                        # Merge and deduplicate with initial search results
                        seen_p_keys = {p.payload.get("text", "")[:120] for p in search_res}
                        for p in second_pass_res:
                            p_key = p.payload.get("text", "")[:120]
                            if p_key not in seen_p_keys:
                                seen_p_keys.add(p_key)
                                search_res.append(p)
                        search_res.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
                        search_res = search_res[:15]

        # Step 4: If still 0 chunks found even after fallback, emergency scan across full library
        if not search_res:
            print(f"[SEARCH FALLBACK] Re-querying full library across all books for: '{clean_topic}'")
            search_res = await multi_search_qdrant(medical_terms, preferred_books=None)

        if not search_res or (eval_result.get("is_genuinely_absent", False) and not search_res):
            await send_whatsapp_cloud_msg(
                sender_phone, 
                f"I've thoroughly checked your medical textbooks for *{clean_topic}*, but couldn't find a dedicated chapter or section on this specific concept in the indexed library.\n\n"
                f"Try rephrasing with related clinical terms, or explore other subjects using */update books*!"
            )
            return

        # If user explicitly asked for a quiz on a topic via text, launch the interactive quiz directly!
        if intent == "QUIZ":
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
                f"CLEAN MEDICAL TOPIC: {clean_topic}\n"
                f"USER'S QUESTION/INSTRUCTION REGARDING THE TAGGED MESSAGE:\n{query_to_search}\n\n"
                f"RETRIEVED MEDICAL KNOWLEDGE CONTEXT:\n{formatted_context}\n\n"
                f"CRITICAL INSTRUCTION: Start your response with the topic-anchored header: 📖 *{clean_topic.upper()}*. Immediately follow the header with a 1-2 sentence high-level clinical overview/definition so anyone reading immediately understands what condition, pathogen, or pharmacological concept is being explained. Then provide the structured clinical breakdown. Speak authoritatively like an elite medical professor. Absolutely NEVER use textbook meta-talk (e.g., 'your textbook explains', 'the text states') and NEVER cite fabricated figure numbers."
            )
        else:
            user_prompt = (
                f"CLEAN MEDICAL TOPIC: {clean_topic}\n"
                f"STUDENT QUESTION:\n{query_to_search}\n\n"
                f"RETRIEVED MEDICAL KNOWLEDGE CONTEXT:\n{formatted_context}\n\n"
                f"CRITICAL INSTRUCTION: Start your response with the topic-anchored header: 📖 *{clean_topic.upper()}*. Immediately follow the header with a 1-2 sentence high-level clinical overview/definition so anyone reading immediately understands what condition, pathogen, or pharmacological concept is being explained. Then provide the structured clinical breakdown. Speak authoritatively like an elite medical professor. Absolutely NEVER use textbook meta-talk (e.g., 'your textbook explains', 'the text states') and NEVER cite fabricated figure numbers."
            )
        
        # Build dynamic user context
        books_str = ", ".join(preferred_books_list) if preferred_books_list else "None"
        user_context_str = f"The student asking this question is {name}, a {level} medical student. Their preferred textbooks are: {books_str}. Tailor your explanation to their level.\n\n"

        prompt_to_use = SYSTEM_MEDICAL_PROMPT
        prompt_to_use = prompt_to_use.replace("{user_context}", user_context_str)

        # Chat memory
        chat_history = []
        if chat_history_col is not None:
            user_doc = await chat_history_col.find_one({"user_id": sender_phone})
            if user_doc and "messages" in user_doc:
                chat_history = user_doc["messages"][-6:]

        ai_answer = await call_openrouter_llm(prompt_to_use, user_prompt, chat_history)
        await send_whatsapp_cloud_msg(sender_phone, ai_answer)

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

        # Dynamic Token Billing Deduction (2.5x Markup ~ 60% Margin) & Topic Persistence
        if users_col is not None:
            try:
                est_prompt_tokens = 1500 + len(user_prompt) // 4
                est_compl_tokens = len(ai_answer) // 4
                raw_cost_usd = (est_prompt_tokens * 0.00000014) + (est_compl_tokens * 0.00000028)
                cost_ngn = max(2.00, raw_cost_usd * 1550.0 * 2.5)
                await users_col.update_one(
                    {"user_id": sender_phone},
                    {
                        "$inc": {"wallet_balance_ngn": -cost_ngn, "total_spent_ngn": cost_ngn},
                        "$set": {
                            "last_medical_topic": clean_topic,
                            "last_context_text": formatted_context,
                            "last_assistant_answer": ai_answer
                        },
                        "$push": {"transactions": {
                            "amount_ngn": cost_ngn,
                            "type": "query_deduction",
                            "description": "Medical Query / RAG Explanation",
                            "timestamp": datetime.utcnow().isoformat()
                        }}
                    }
                )
            except Exception as bill_err:
                print(f"⚠️ Billing deduction error: {bill_err}")

        # Check if the answer indicates information is missing from textbooks
        ai_lower = ai_answer.lower()
        is_not_covered = ("not covered" in ai_lower or "sorry" in ai_lower[:30] or "not found" in ai_lower)

        # Save in 24-hour LRU Cache for instant 0.05s delivery for other students
        if not is_not_covered and len(ai_answer) > 100:
            TOPIC_CACHE.set(search_term, ai_answer, formatted_context, preferred_books=preferred_books_list)

        # Attach interactive follow-up button for quick MCQ generation ONLY if it was a valid medical answer
        if not user_msg.startswith("GENERATE_QUIZ") and not is_not_covered:
            try:
                clean_topic_label = clean_topic
                if len(clean_topic_label) > 45:
                    clean_topic_label = clean_topic_label[:42] + "..."
                topic_snippet = clean_topic[:100]
                await send_whatsapp_interactive_button(
                    sender_phone,
                    f"Ready to practice MCQs on *{clean_topic_label}*?",
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
        "openrouter_configured": bool(OPENROUTER_API_KEY),
        "billing": "Flutterwave In-App WebView + Dynamic Token Multiplier"
    }

@app.post("/webhook/flutterwave")
async def flutterwave_webhook(request: Request):
    """Flutterwave Webhook Endpoint to credit student wallets upon successful charge"""
    try:
        signature = request.headers.get("verif-hash", "")
        if FLUTTERWAVE_SECRET_HASH and signature != FLUTTERWAVE_SECRET_HASH:
            print("❌ Flutterwave secret hash mismatch!")
            raise HTTPException(status_code=401, detail="Invalid signature")

        body = await request.json()
        print(f"💳 Flutterwave Webhook: {json.dumps(body)}")
        
        event = body.get("event")
        data = body.get("data", {})
        status = data.get("status")
        
        if event == "charge.completed" and status == "successful":
            amount_ngn = float(data.get("amount", 0.0))
            tx_ref = data.get("tx_ref", "")
            flw_ref = str(data.get("flw_ref") or tx_ref)
            customer = data.get("customer", {})
            phone = customer.get("phone_number") or customer.get("phonenumber")
            
            if not phone and "NEURA_" in tx_ref:
                parts = tx_ref.split("_")
                if len(parts) >= 2:
                    phone = parts[1]

            if phone and amount_ngn > 0 and users_col is not None:
                # Idempotent credit
                res = await users_col.update_one(
                    {"user_id": phone, "transactions.reference": {"$ne": flw_ref}},
                    {
                        "$inc": {"wallet_balance_ngn": amount_ngn},
                        "$push": {"transactions": {
                            "amount_ngn": amount_ngn,
                            "reference": flw_ref,
                            "tx_ref": tx_ref,
                            "type": "deposit",
                            "description": f"Flutterwave Wallet Deposit (₦{amount_ngn:,.2f})",
                            "timestamp": datetime.utcnow().isoformat()
                        }}
                    },
                    upsert=True
                )
                if res.modified_count > 0 or res.upserted_id:
                    u = await users_col.find_one({"user_id": phone})
                    bal = u.get("wallet_balance_ngn", amount_ngn) if u else amount_ngn
                    receipt = (
                        f"🎉 *PAYMENT RECEIVED!*\n\n"
                        f"• Amount Credited: *₦{amount_ngn:,.2f}*\n"
                        f"• New Wallet Balance: *₦{bal:,.2f}*\n"
                        f"• Ref: _{flw_ref}_\n\n"
                        f"You can now continue asking medical questions with full textbook grounding! 🧠⚡"
                    )
                    await send_whatsapp_cloud_msg(phone, receipt)
        return {"status": "success"}
    except Exception as e:
        print(f"Error handling Flutterwave webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/webhook/paystack")
async def paystack_webhook(request: Request):
    """Paystack Webhook Endpoint to credit student wallets upon successful charge (Fallback)"""
    try:
        body_bytes = await request.body()
        signature = request.headers.get("x-paystack-signature", "")
        if PAYSTACK_SECRET_KEY:
            expected = hmac.new(PAYSTACK_SECRET_KEY.strip().encode(), body_bytes, hashlib.sha512).hexdigest()
            if expected.lower() != signature.lower():
                print("❌ Paystack signature mismatch!")
                raise HTTPException(status_code=401, detail="Invalid signature")

        data = json.loads(body_bytes.decode())
        if data.get("event") == "charge.success":
            tx_data = data.get("data", {})
            amount_kobo = tx_data.get("amount", 0)
            ref = tx_data.get("reference", "")
            metadata = tx_data.get("metadata", {})
            phone = metadata.get("phone_number") or tx_data.get("customer", {}).get("phone")

            if phone and amount_kobo > 0 and users_col is not None:
                amount_ngn = amount_kobo / 100.0
                # Idempotent credit
                res = await users_col.update_one(
                    {"user_id": phone, "transactions.reference": {"$ne": ref}},
                    {
                        "$inc": {"wallet_balance_ngn": amount_ngn},
                        "$push": {"transactions": {
                            "amount_ngn": amount_ngn,
                            "reference": ref,
                            "type": "deposit",
                            "description": f"Paystack Wallet Deposit (₦{amount_ngn:,.2f})",
                            "timestamp": datetime.utcnow().isoformat()
                        }}
                    },
                    upsert=True
                )
                if res.modified_count > 0 or res.upserted_id:
                    u = await users_col.find_one({"user_id": phone})
                    bal = u.get("wallet_balance_ngn", amount_ngn) if u else amount_ngn
                    receipt = (
                        f"🎉 *PAYMENT RECEIVED!*\n\n"
                        f"• Amount Credited: *₦{amount_ngn:,.2f}*\n"
                        f"• New Wallet Balance: *₦{bal:,.2f}*\n"
                        f"• Ref: _{ref}_\n\n"
                        f"You can now continue asking medical questions with full textbook grounding! 🧠⚡"
                    )
                    await send_whatsapp_cloud_msg(phone, receipt)
        return {"status": "success"}
    except Exception as e:
        print(f"Error handling Paystack webhook: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/api/payment-complete")
async def payment_complete_page(request: Request):
    status = request.query_params.get("status", "").lower()
    
    is_successful = status in ["successful", "success", "completed"]
    
    if is_successful:
        html_content = """
        <!DOCTYPE html>
        <html><head><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Payment Confirmed - NEURA AI</title>
        <style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#f8fafc;color:#0f172a;text-align:center;}
        .card{background:white;padding:32px;border-radius:16px;box-shadow:0 4px 6px -1px rgb(0 0 0/0.1);max-width:400px;margin:20px;}
        .icon{font-size:48px;margin-bottom:16px;}h1{font-size:24px;margin:0 0 8px;color:#16a34a;}p{color:#64748b;font-size:16px;line-height:1.5;}</style></head>
        <body><div class="card"><div class="icon">✅</div><h1>Payment Confirmed!</h1><p>Your NEURA AI wallet has been successfully credited. You can return to WhatsApp.</p></div></body></html>
        """
    else:
        html_content = """
        <!DOCTYPE html>
        <html><head><meta name="viewport" content="width=device-width, initial-scale=1.0"><title>Payment Cancelled - NEURA AI</title>
        <style>body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;display:flex;align-items:center;justify-content:center;height:100vh;margin:0;background:#f8fafc;color:#0f172a;text-align:center;}
        .card{background:white;padding:32px;border-radius:16px;box-shadow:0 4px 6px -1px rgb(0 0 0/0.1);max-width:400px;margin:20px;}
        .icon{font-size:48px;margin-bottom:16px;}h1{font-size:24px;margin:0 0 8px;color:#dc2626;}p{color:#64748b;font-size:16px;line-height:1.5;}</style></head>
        <body><div class="card"><div class="icon">❌</div><h1>Payment Cancelled</h1><p>The transaction was not completed and your wallet was not charged. You can return to WhatsApp and try again anytime with <b>/deposit</b>.</p></div></body></html>
        """
    return Response(content=html_content, media_type="text/html")


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
        intent = await classify_intent(user_msg)
        
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
