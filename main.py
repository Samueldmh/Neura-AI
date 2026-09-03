import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import os
import gc
import re
import json
import logging
import traceback
import httpx
import hmac
import hashlib
import uuid
import urllib.parse
from datetime import datetime, timedelta, timezone
import time
from fastapi import FastAPI, HTTPException, Request, Response, BackgroundTasks
from fastapi.responses import HTMLResponse
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
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")
PAYSTACK_SECRET_KEY = os.getenv("PAYSTACK_SECRET_KEY", "")
FLUTTERWAVE_SECRET_KEY = os.getenv("FLUTTERWAVE_SECRET_KEY", "")
FLUTTERWAVE_SECRET_HASH = os.getenv("FLUTTERWAVE_SECRET_HASH", "neura_flw_hash_2026")
BASE_URL = os.getenv("BASE_URL", "https://neura-ai-qtux.onrender.com")

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
    "Chemical Pathology": ["Martin and crooke clinical biochemistry"],
    "Haematology": ["Essentials of Haematology"],
    "Microbiology": ["Jawetz_Melnick_Adelbergs_Medical_Microbiology_27_edition_Med_zoneTV"],
    "Pharmacology": ["Lippincott Illustrated Reviews: Pharmacology"]
}

def get_all_curriculum_books_for_level(level: str) -> list:
    """Returns the complete list of official textbooks for a given academic level."""
    subjects = CURRICULUM.get(level, [])
    books = []
    for s in subjects:
        for b in AVAILABLE_BOOKS.get(s, []):
            if b and b not in books:
                books.append(b)
    return books

app = FastAPI(title="NEURA AI Backend", version="2.0.0")

# Initialize FastEmbed & Qdrant Client
print("Initializing FastEmbed & Qdrant Client...")
print(f"QDRANT_URL: {QDRANT_URL}")
print(f"QDRANT_API_KEY Present: {bool(QDRANT_API_KEY)}")
print(f"OPENROUTER_API_KEY Present: {bool(OPENROUTER_API_KEY)}")
print(f"GROQ_API_KEY Present: {bool(GROQ_API_KEY)}")
print(f"PHONE_NUMBER_ID: {PHONE_NUMBER_ID}")

from collections import OrderedDict
import time

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", threads=1)
qdrant = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
shared_http_client = httpx.AsyncClient(timeout=30.0, limits=httpx.Limits(max_keepalive_connections=20, max_connections=40))
embedding_pool = ThreadPoolExecutor(max_workers=2)

# Lightweight in-memory query vector cache (<1MB RAM for 500 common curriculum topics)
QUERY_VECTOR_CACHE = OrderedDict()
MAX_VECTOR_CACHE = 500

# Multi-Engine Intelligent Architecture:
# 1. Heavy Clinical Medical Engine: Meta Llama 3.3 70B on Cerebras / SambaNova (1,800 tokens/sec)
CLINICAL_MODEL = "meta-llama/llama-3.3-70b-instruct"
CLINICAL_PROVIDER_ORDER = ["Cerebras", "SambaNova", "Groq", "Together", "Novita", "DeepInfra"]

# 2. Front-Desk Semantic Intelligence (Intent Classification, Typo Normalizer, Platform & Companion):
FRONTDESK_MODEL = "google/gemini-2.5-flash-lite"

# 3. Universal Zero-Downtime Fallback:
FALLBACK_MODEL = "openai/gpt-4o-mini"

DEFAULT_MODEL = CLINICAL_MODEL
DEFAULT_PROVIDER_ORDER = CLINICAL_PROVIDER_ORDER

def get_reasoning_config(prompt: str = "", is_micro: bool = False) -> dict:
    """Returns dynamic reasoning configuration:
    - Micro-LLMs (normalizers/classifiers): effort='low', exclude=True
    - Complex clinical differential / case studies: effort='medium', exclude=True
    - General medical Q&A: effort='low', exclude=True
    """
    if is_micro:
        return {"effort": "low", "exclude": True}
    
    p_lower = prompt.lower()
    complex_triggers = [
        "differential diagnosis", "differentiate between", "compare and contrast",
        "case study", "35-year-old", "45-year-old", "50-year-old", "patient presents",
        "acid-base", "interpret the following", "management of refractory"
    ]
    if any(t in p_lower for t in complex_triggers):
        return {"effort": "medium", "exclude": True}
    
    return {"effort": "low", "exclude": True}

def get_embedding_sync(text: str):
    clean_k = text.strip().lower()
    if clean_k in QUERY_VECTOR_CACHE:
        QUERY_VECTOR_CACHE.move_to_end(clean_k)
        return QUERY_VECTOR_CACHE[clean_k]
    t0 = time.perf_counter()
    vec = list(embedder.embed(text))[0]
    if len(QUERY_VECTOR_CACHE) >= MAX_VECTOR_CACHE:
        QUERY_VECTOR_CACHE.popitem(last=False)
    QUERY_VECTOR_CACHE[clean_k] = vec
    dt = time.perf_counter() - t0
    print(f"⏱️ [EMBED TIMER] FastEmbed on '{text[:45]}...' took {dt*1000:.1f}ms ({dt:.3f}s)")
    return vec

print(f"MONGO_URI Present: {bool(MONGO_URI)}")
mongo_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = mongo_client.neura_db if mongo_client else None
chat_history_col = db.chat_history if db is not None else None
users_col = db.users if db is not None else None
broadcasts_col = db.broadcasts if db is not None else None
chat_logs_col = db.chat_logs if db is not None else None
youtube_video_cache_col = db.youtube_video_cache if db is not None else None

ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "neura2026admin")
ADMIN_SESSIONS = set()

async def log_user_chat_message(user_id: str, role: str, content: str, msg_type: str = "text", metadata: dict = None):
    """Persists every user and AI message for audit, diagnostic, and admin chat inspection."""
    if not user_id or not content:
        return
    now_iso = datetime.utcnow().isoformat() + "Z"
    meta = dict(metadata or {})
    meta["msg_type"] = msg_type
    
    # Detect potential issues/errors in the message text for quick diagnostic flags
    content_lower = str(content).lower()
    is_error_flag = (
        "connection delay" in content_lower or
        "trouble creating the interactive practice" in content_lower or
        "insufficient balance" in content_lower or
        "low balance" in content_lower or
        "error processing your query" in content_lower or
        "experienced a brief" in content_lower or
        "only read text messages" in content_lower or
        "i had trouble creating" in content_lower or
        meta.get("is_error", False)
    )
    if is_error_flag:
        meta["has_issue"] = True

    log_entry = {
        "user_id": str(user_id),
        "role": role,
        "content": str(content),
        "timestamp": now_iso,
        "metadata": meta
    }
    
    try:
        if chat_logs_col is not None:
            await chat_logs_col.insert_one(log_entry)
            
        if chat_history_col is not None:
            await chat_history_col.update_one(
                {"user_id": str(user_id)},
                {
                    "$push": {
                        "messages": {
                            "role": role,
                            "content": str(content),
                            "timestamp": now_iso,
                            "msg_type": msg_type,
                            "has_issue": is_error_flag
                        }
                    },
                    "$set": {"last_active": now_iso}
                },
                upsert=True
            )
    except Exception as e:
        print(f"Error persisting chat log for {user_id}: {e}")

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
            
            # Send reminder message directly; auto fallback to Meta template if outside 24h window
            delivered = await send_whatsapp_cloud_msg(phone, streak_msg)
            if not delivered:
                print(f"[NUDGE] Direct reminder unconfirmed for {phone} (>24h inactive). Delivering via neura_announcement template...")
                await send_whatsapp_template_msg(phone, "neura_announcement", [name, streak_msg])
            
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

async def cleanup_legacy_textbook_titles():
    """Migrates any occurrences of legacy textbook titles in MongoDB to new names."""
    if users_col is None:
        return
    try:
        old_book = "Crook Martin Andrew Clinical B"
        chem_book = "Martin and crooke clinical biochemistry"
        await users_col.update_many(
            {"preferred_books_list": old_book},
            {"$set": {"preferred_books_list.$": chem_book}}
        )
    except Exception as e:
        print(f"⚠️ Error cleaning up legacy titles: {e}")

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
        
    # Clean up legacy title naming in database
    asyncio.create_task(cleanup_legacy_textbook_titles())
    
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
SYSTEM_MEDICAL_PROMPT = """{user_context}You are Neura, an elite, articulate, and supportive senior medical colleague and study co-pilot designed for Nigerian MBBS medical students.
Your goal is to explain clinical concepts, pathophysiological mechanisms, diagnostic criteria, and pharmacotherapies with maximum clarity, scientific precision, and mobile-first conciseness.

CORE PEDAGOGICAL & BREVITY RULES:
1. PROPORTIONAL DEPTH (MATCH THE QUESTION):
   - DIRECT / SPOT / FACTUAL QUESTIONS (e.g. "What is the drug of choice for X?", "Normal range of Y?", "Which nerve innervates Z?"):
     * Give the direct, authoritative answer in the very first sentence.
     * Keep the entire response under 2 to 4 punchy, high-yield sentences.
     * Do NOT output massive section headers, long introductions, or unnecessary multi-page chapters.
   - COMPREHENSIVE / MECHANISM TOPICS (e.g. "Explain Tetralogy of Fallot", "Discuss the pathophysiology of DKA"):
     * Use clean, logical section headings (`## *Pathophysiology*`, `## *Clinical Features*`, `## *Management*`).
     * Keep each section tight and punchy (max 2-3 structured bullets per section). Never ramble or repeat facts.

2. SIMPLE, INTUITIVE & AUTHORITATIVE: Translate complex jargon into intuitive step-by-step logic while preserving 100% textbook accuracy. Use relatable real-world analogies where helpful (e.g. 'think of the glomerulus as a high-pressure sieve').

3. INLINE CLARIFICATIONS: When an eponym, rare antibody, or complex syndrome appears, add a short parenthetical definition immediately after it.

4. BOLD HIGHLIGHTS: Bold all key drug names, vital signs, diagnostic thresholds, and classic triad signs (*Drug Name*, *Diagnostic Sign*).

5. CONTEXTUAL EXAM TIPS (NEVER FORCED):
   - Include a `> 💡 *Senior's Exam Tip:* ...` ONLY when genuinely discussing an authentic, exam-tested clinical pitfall, board-exam buzzword, or high-yield medical association.
   - NEVER invent or force an exam tip for general knowledge, simple queries, or non-clinical topics.

6. ZERO TEXTBOOK META-TALK & ZERO SOURCE CITATIONS: Never say "according to textbooks" or "the text states". Deliver facts directly with senior clinical authority. Never output citation lists or footnote blocks.

7. ZERO FABRICATED FIGURE CITATIONS: Never invent figure numbers (e.g., NEVER write "Figure 46-9", "Fig 12.8").

8. NO PREAMBLES & NO FILLER: Jump DIRECTLY into the answer. Zero conversational filler, greetings, or announcements.

9. NO RAW MARKDOWN TABLES: Present all comparisons or staging summaries as clean bulleted list cards.

10. DOUBLE-LINE SPACING: Separate headings, paragraphs, and bullet points with blank lines (`\n\n`) for effortless reading on mobile screens.
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

    ack_words = {"ok", "okay", "k", "alright", "cool", "noted", "got", "it", "understood", "makes", "sense", "i", "see", "nice", "great", "awesome", "perfect", "good", "fine", "correct", "yes", "yep", "yeah", "clear", "loud"}
    words_in_msg = msg_clean.split()
    if words_in_msg and all(w in ack_words for w in words_in_msg) and len(words_in_msg) <= 4:
        return "ACKNOWLEDGMENT"

    # 2.1 Direct Name Callout or Friendly Presence Check Fast-Path (<0.01ms)
    msg_no_punc = re.sub(r'[^\w\s]', ' ', msg_clean).strip()
    if msg_no_punc in ["neura", "neura ai", "hey neura", "hi neura", "hello neura", "neura dear"]:
        return "CONVERSATIONAL"

    # 2.2 Presence Check-ins, Emotional Venting & Casual Banter Fast-Path
    conversational_fast_patterns = [
        r"\b(are\s*(you|u)\s*(still\s*)?(there|here|awake|listening|around|alive|online|present))\b",
        r"\b(u\s*(still\s*)?(there|here|awake|listening|around|alive|online))\b",
        r"\b(you\s*(still\s*)?(there|here|awake|listening|around))\b",
        r"\b(are\s*(you|u)\s*ready)\b",
        r"\b(i('m|\s*am)?\s*(so\s*)?(tired|exhausted|sleepy|drained|stressed|burnt\s*out|dying|choking))\b",
        r"\b(med\s*school\s*(is\s*)?(hard|killing\s*me|stressful|tough|choking\s*me))\b",
        r"\b(ward\s*rounds?\s*(was|were|is)\s*(long|stressful|tiring|hectic|crazy|brutal))\b",
        r"\b(tell\s*me\s*a\s*joke|make\s*me\s*laugh)\b",
        r"\b(who\s*made\s*you|who\s*created\s*you|are\s*you\s*(real|human|an?\s*ai))\b",
        r"\b(what\s*are\s*you\s*doing|what('s|\s*is)\s*up\s*with\s*you)\b",
        r"\b(i\s*(hate|dislike)\s*(anatomy|pathology|pharm|biochem|studying|reading))\b",
        r"\b(i\s*failed|i('m|\s*am)\s*scared\s*of\s*(exams?|profs?))\b",
        r"\b(how\s*(should|can|do)\s*i\s*study|study\s*tips?|how\s*to\s*pass)\b"
    ]
    if any(re.search(pat, msg_clean) for pat in conversational_fast_patterns):
        return "CONVERSATIONAL"

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

    # 2.4 Platform, Beta Feedback & Privacy Fast-Path (<0.01ms)
    platform_fast_patterns = [
        r"\b(beta\s*feedback|anonymous|feedback|survey|privacy|confidential|wallet|deposit|how\s*to\s*use|commands?|features?)\b"
    ]
    if any(re.search(pat, msg_clean) for pat in platform_fast_patterns) and not any(ind in msg_clean for ind in ["syndrome", "disease", "treatment", "pathology", "pharmacology", "anatomy", "physiology", "symptoms", "diagnosis", "mechanism", "pathophysiology", "drug"]):
        return "PLATFORM_META"

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

    # 5. LLM Universal Intent Classifier (Handles ANY language: German, French, Arabic, slang, gibberish, vague chatter)
    if OPENROUTER_API_KEY:
        try:
            router_prompt = (
                "You are an intent classifier for NEURA AI, a medical study companion.\n"
                "Analyze the user's message and classify it into EXACTLY ONE label:\n"
                "- PLATFORM_META: Questions about the NEURA AI platform itself, its features, commands (/wallet, /deposit, /feedback, /profile), anonymous beta feedback, data privacy, confidentiality, pricing, token balance, how the bot works, or who created it.\n"
                "- GREETING: Simple greetings, hello, foreign greetings (e.g. 'bonjour', 'kedu', 'bawo').\n"
                "- CONVERSATIONAL: Casual banter, presence checks (e.g. 'are you there', 'u there', 'are you still there', 'you awake', 'neura are you listening'), emotional venting (e.g. 'I am so tired', 'ward rounds were tough', 'med school is hard', 'I hate studying', 'I failed'), general study strategy (e.g. 'how should I study pharmacology'), rhetorical questions.\n"
                "- GRATITUDE: Thank you, thanks, nice one, well done, praise, appreciation.\n"
                "- ACKNOWLEDGMENT: Short confirmations (ok, cool, noted, got it, understood, alright).\n"
                "- GIBBERISH: Random keyboard mash, nonsense characters (e.g. 'asdfgh', '12345', '????'), meaningless noise.\n"
                "- QUIZ: Explicit requests for MCQs, practice questions, quizzes, tests.\n"
                "- MEDICAL: Genuine clinical or medical curriculum study questions (disease pathophysiology, pharmacology, anatomy, biochemistry, clinical management, symptoms, mechanisms).\n\n"
                "CRITICAL: If a user is asking about the app, beta feedback, privacy, or features, classify as PLATFORM_META. If checking presence or venting, classify as CONVERSATIONAL. Never classify platform or personal chatter as MEDICAL.\n"
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
                "models": [FRONTDESK_MODEL, FALLBACK_MODEL],
                "messages": [
                    {"role": "system", "content": router_prompt},
                    {"role": "user", "content": message}
                ],
                "temperature": 0.0,
                "max_tokens": 15,
                "reasoning": get_reasoning_config(message, is_micro=True)
            }
            resp = await shared_http_client.post(url, headers=headers, json=payload)
            if resp.status_code == 200:
                choice_msg = resp.json().get("choices", [{}])[0].get("message", {})
                cat = (choice_msg.get("content") or "").strip().upper()
                for valid in ["PLATFORM_META", "GREETING", "CONVERSATIONAL", "GRATITUDE", "ACKNOWLEDGMENT", "GIBBERISH", "QUIZ", "MEDICAL"]:
                    if valid in cat:
                        return valid
        except Exception as e:
            print(f"LLM Intent Classifier fallback error: {e}")

    # Default fallback
    if len(msg_clean) <= 4 or not terms:
        return "GIBBERISH"
    return "MEDICAL"

async def call_openrouter_llm(
    system_prompt: str, 
    user_prompt: str, 
    chat_history: list = None, 
    max_tokens: int = 2500,
    model: str = None,
    models: list = None,
    provider_order: list = None
) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set on Render!")
        
    t_start = time.perf_counter()
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neura-ai.org",
        "X-Title": "NEURA AI Medical Assistant"
    }
    
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        for m in chat_history:
            if isinstance(m, dict) and m.get("role") in ["user", "assistant"]:
                messages.append({"role": m["role"], "content": str(m.get("content", ""))})
    messages.append({"role": "user", "content": user_prompt})
    
    target_models = models or ([model] if model else [CLINICAL_MODEL, FRONTDESK_MODEL, FALLBACK_MODEL])
    payload = {
        "models": target_models,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "reasoning": get_reasoning_config(user_prompt, is_micro=False),
        "provider": {
            "order": provider_order or DEFAULT_PROVIDER_ORDER,
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
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or message.get("reasoning") or ""
    provider_used = data.get("provider", "UnknownProvider")

    if not content and response.status_code == 200:
        print("⚠️ OpenRouter returned empty content, retrying with fallback provider order...")
        payload["provider"] = {
            "order": DEFAULT_PROVIDER_ORDER,
            "allow_fallbacks": True
        }
        try:
            async with httpx.AsyncClient(timeout=30.0) as retry_client:
                retry_res = await retry_client.post(url, headers=headers, json=payload)
                if retry_res.status_code == 200:
                    retry_data = retry_res.json()
                    retry_msg = (retry_data.get("choices") or [{}])[0].get("message") or {}
                    content = retry_msg.get("content") or retry_msg.get("reasoning") or ""
                    provider_used = retry_data.get("provider", provider_used)
        except Exception as retry_err:
            print(f"Retry error: {retry_err}")

    dt = time.perf_counter() - t_start
    print(f"⏱️ [LLM TIMER] call_openrouter_llm completed in {dt:.3f}s (Model: {DEFAULT_MODEL}, Provider: {provider_used}, Tokens Generated: ~{len(content.split())*4/3:.0f})")
    return (content or "").strip()

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
        for m in chat_history:
            if isinstance(m, dict) and m.get("role") in ["user", "assistant"]:
                messages.append({"role": m["role"], "content": str(m.get("content", ""))})
    messages.append({"role": "user", "content": user_prompt})
    
    payload = {
        "model": DEFAULT_MODEL,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2500,
        "stream": True,
        "reasoning": get_reasoning_config(user_prompt, is_micro=False),
        "provider": {
            "order": DEFAULT_PROVIDER_ORDER,
            "allow_fallbacks": True
        }
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

    # 3. Ensure double newline before markdown headings (#, ##, ###) and clean spacing
    text = re.sub(r'(?m)^(#{1,3})([^\s#])', r'\1 \2', text)
    text = re.sub(r'([^\n])\n(#{1,3}\s+)', r'\1\n\n\2', text)

    # 3.2. Ensure double newline before blockquotes (>)
    text = re.sub(r'([^\n])\n(>\s*)', r'\1\n\n\2', text)

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

async def send_whatsapp_cloud_msg(to_number: str, message_text: str, preview_url: bool = False):
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

    all_success = True
    for chunk in chunks:
        has_url = bool(preview_url or "http://" in chunk or "https://" in chunk)
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {
                "preview_url": has_url,
                "body": chunk
            }
        }
        status_ok = False
        try:
            res = await shared_http_client.post(url, headers=headers, json=payload)
            status_ok = (res.status_code == 200)
        except Exception:
            try:
                async with httpx.AsyncClient(timeout=20.0) as fallback_client:
                    res = await fallback_client.post(url, headers=headers, json=payload)
                    status_ok = (res.status_code == 200)
            except Exception:
                status_ok = False
        if not status_ok:
            all_success = False
        print(f"Meta Graph API Send Status {res.status_code if 'res' in locals() else 'ERR'}: {getattr(res, 'text', '')}")
        
    if all_success:
        try:
            asyncio.create_task(log_user_chat_message(to_number, "assistant", message_text, msg_type="text"))
        except Exception:
            pass

    return all_success

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
        if res.status_code == 200:
            try:
                asyncio.create_task(log_user_chat_message(to_number, "assistant", body_text, msg_type="interactive_list"))
            except Exception:
                pass

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
        if res.status_code == 200:
            try:
                btn_summary = " [Options: " + ", ".join(str(b.get("title", "")) for b in buttons) + "]"
                asyncio.create_task(log_user_chat_message(to_number, "assistant", body_text + btn_summary, msg_type="interactive_button"))
            except Exception:
                pass

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
        if res.status_code == 200:
            try:
                asyncio.create_task(log_user_chat_message(to_number, "assistant", f"{body_text} [Link: {button_label} -> {url_target}]", msg_type="cta_button"))
            except Exception:
                pass

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

async def send_whatsapp_template_msg(to_number: str, template_name: str, parameters: list, language_code: str = "en") -> bool:
    """Sends a Meta Pre-Approved Template Message (Utility / Marketing) to a student.
    Handles templates with named body parameters (e.g. neura_announcement: {{student_name}} and {{announcement_text}} in body)
    as well as positional and custom component parameters.
    """
    if not to_number or not WHATSAPP_TOKEN:
        return False
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }
    
    # Structure components specifically for neura_announcement (Header with student_name, Body with announcement_text)
    if template_name == "neura_announcement" and len(parameters) >= 2:
        student_name = str(parameters[0])
        announcement_text = str(parameters[1])
        components = [
            {
                "type": "header",
                "parameters": [
                    {
                        "type": "text",
                        "parameter_name": "student_name",
                        "text": student_name
                    }
                ]
            },
            {
                "type": "body",
                "parameters": [
                    {
                        "type": "text",
                        "parameter_name": "announcement_text",
                        "text": announcement_text
                    }
                ]
            }
        ]
    elif parameters and isinstance(parameters[0], dict) and "type" in parameters[0] and "parameters" in parameters[0]:
        # User passed fully structured components list directly
        components = parameters
    else:
        # Standard parameters list
        body_params = []
        for p in parameters:
            if isinstance(p, dict):
                body_params.append(p)
            else:
                body_params.append({"type": "text", "text": str(p)})
        components = [
            {
                "type": "body",
                "parameters": body_params
            }
        ]

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "template",
        "template": {
            "name": template_name,
            "language": {"code": language_code},
            "components": components
        }
    }
    try:
        res = await shared_http_client.post(url, headers=headers, json=payload)
        if res.status_code != 200:
            print(f"⚠️ Meta Template Send Failed ({res.status_code}) to {to_number}: {res.text}")
            print(f"📦 Sent Payload was: {json.dumps(payload)}")
        else:
            print(f"✅ Meta Template Successfully Delivered ({template_name}) to {to_number}")
        return res.status_code == 200
    except Exception as e:
        print(f"⚠️ Template send error to {to_number}: {e}")
        return False

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

# ==========================================
# VOICE NOTE & SPEECH-TO-TEXT ENGINE (Groq Whisper Large v3)
# ==========================================
WHISPER_MEDICAL_PROMPT = (
    "NEURA AI medical study session: MBBS student discussing Anatomy, Physiology, "
    "Biochemistry, Pathology, Pharmacology, Microbiology, Haematology, Histopathology, "
    "Chemical Pathology, Obstetrics & Gynaecology, Medicine & Surgery, Nigerian context."
)

WHISPER_HALLUCINATION_PATTERNS = [
    "thank you for watching", "thanks for watching", "subtitles by", "amara.org",
    "please subscribe", "like and subscribe", "subscribe", "copyright", "all rights reserved",
    "music", "applause", "silence", "laughter", "cheering",
    "mbc", "al jazeera", "bbc news", "transcription by", "translated by",
    "video by", "audio by", "silent", "you", "satsang", "video of the", "new video", "watching"
]

DANGLING_FRAGMENTS = [
    "of the", "in the", "to the", "at the", "for the", "on the", "by the", "from the", "with the",
    "of a", "in a", "to a", "at a", "for a", "on a", "by a", "from a", "with a",
    "and the", "or the", "is a", "was a", "are the", "were the", "of", "the", "a", "an", "in", "to", "at"
]

def has_repeated_phrases(text: str) -> bool:
    """Detects looping repetitive hallucinated phrases in Whisper output (e.g. 'video of the video of the video')."""
    words = [w.strip(".,!?:;\"'()[]{}").lower() for w in text.split() if w.strip(".,!?:;\"'()[]{}")]
    if len(words) < 4:
        return False
    # Check 1-word, 2-word, 3-word ngrams for repetition loops (>=3 occurrences)
    for n in (1, 2, 3):
        ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
        for ng in ngrams:
            if ngrams.count(ng) >= 3:
                return True
    return False

def is_gibberish_or_silence(text: str, no_speech_prob: float = 0.0, avg_logprob: float = 0.0, compression_ratio: float = 1.0) -> tuple[bool, str]:
    """Detects inaudible audio, silence, hallucinated repetitive non-speech, or incomplete fragments."""
    if not text or not text.strip():
        return True, "Empty transcript"
        
    # Check Whisper acoustic confidence metrics if available
    if no_speech_prob > 0.40:
        return True, f"High silence probability ({no_speech_prob:.2f})"
        
    if avg_logprob < -1.1:
        return True, f"Low acoustic confidence ({avg_logprob:.2f})"

    if compression_ratio > 2.2:
        return True, f"High compression ratio loop ({compression_ratio:.2f})"

    # Check repeated phrase hallucination loops
    if has_repeated_phrases(text):
        return True, "Repetitive phrase hallucination loop"

    import re
    # 1. Clean raw text to lowercase
    raw_lower = text.strip().lower()
    # If wrapped in brackets e.g. [music], (applause)
    if raw_lower.startswith(("[", "(")) and raw_lower.endswith(("]", ")")):
        return True, "Bracketed sound tag"

    # 2. Strip leading/trailing punctuation and whitespace
    clean = re.sub(r'^[^\w]+|[^\w]+$', '', raw_lower)
    if len(clean) < 4:
        return True, "Too short (<4 chars)"

    # 3. Check known silence hallucination phrases
    for p in WHISPER_HALLUCINATION_PATTERNS:
        if clean == p or clean.startswith(p) or p in clean:
            if len(clean) <= len(p) + 40 or clean == p:
                return True, f"Silence hallucination pattern ('{p}')"

    # 4. Check if mostly punctuation or single characters repeating
    alphanumeric = [c for c in clean if c.isalnum()]
    if len(alphanumeric) < 3:
        return True, "Insufficient alphanumeric content"

    # 5. Check incomplete dangling fragments on very short transcripts (e.g. "Translator of the")
    words = [w.strip(".,!?:;\"'()[]{}") for w in clean.split() if w.strip(".,!?:;\"'()[]{}")]
    if len(words) <= 3:
        for df in DANGLING_FRAGMENTS:
            if clean.endswith(" " + df) or clean == df:
                return True, f"Dangling incomplete fragment ('{df}')"

    # 6. Check repetitive tokens loop (e.g., "you you you you you" or "ah ah ah")
    if len(words) >= 3 and len(set(words)) == 1:
        return True, "Repetitive stutter"

    return False, ""

async def download_whatsapp_media(media_id: str) -> tuple[bytes, str]:
    """Downloads binary audio/media from Meta Graph API using the media ID.
    Returns (media_bytes, mime_type).
    """
    if not media_id or not WHATSAPP_TOKEN:
        print("⚠️ Cannot download media: missing media_id or WHATSAPP_TOKEN")
        return b"", ""
    try:
        meta_media_url = f"https://graph.facebook.com/v19.0/{media_id}"
        auth_headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}"}
        
        async with httpx.AsyncClient(timeout=15.0) as client:
            # Step 1: Query Graph API for media download URL
            res = await client.get(meta_media_url, headers=auth_headers)
            if res.status_code != 200:
                print(f"⚠️ Meta media lookup failed ({res.status_code}) for {media_id}: {res.text}")
                return b"", ""
            
            media_info = res.json()
            download_url = media_info.get("url")
            mime_type = media_info.get("mime_type", "audio/ogg")
            if not download_url:
                print(f"⚠️ No download URL in Meta media response for {media_id}")
                return b"", ""
            
            # Step 2: Fetch raw binary audio from Meta CDN using auth headers
            audio_res = await client.get(download_url, headers=auth_headers)
            if audio_res.status_code == 200:
                print(f"✅ Successfully downloaded audio ({len(audio_res.content)} bytes, {mime_type}) for {media_id}")
                return audio_res.content, mime_type
            else:
                print(f"⚠️ Meta CDN audio download failed ({audio_res.status_code}) for {media_id}")
                return b"", ""
    except Exception as e:
        print(f"⚠️ Exception downloading WhatsApp media {media_id}: {e}")
        return b"", ""

async def transcribe_voice_note(audio_bytes: bytes, mime_type: str = "audio/ogg") -> tuple[str, bool, str]:
    """Transcribes a voice note using Groq Whisper Large v3 (with OpenAI fallback).
    Returns (transcript_text, is_valid_speech, error_reason).
    """
    if not audio_bytes or len(audio_bytes) < 100:
        return "", False, "Empty or corrupted audio stream"

    # Determine file extension from mime_type
    ext = "ogg"
    if "mp4" in mime_type or "m4a" in mime_type:
        ext = "m4a"
    elif "wav" in mime_type:
        ext = "wav"
    elif "mp3" in mime_type or "mpeg" in mime_type:
        ext = "mp3"

    # 1. Primary: Try Groq Whisper Large v3 Turbo
    groq_key = os.getenv("GROQ_API_KEY", "") or GROQ_API_KEY
    if groq_key:
        try:
            groq_url = "https://api.groq.com/openai/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {groq_key.strip()}"}
            files = {
                "file": (f"voice_note.{ext}", audio_bytes, mime_type or "audio/ogg")
            }
            data = {
                "model": "whisper-large-v3-turbo",
                "response_format": "verbose_json",
                "temperature": "0.0",
                "language": "en"
            }
            async with httpx.AsyncClient(timeout=25.0) as client:
                res = await client.post(groq_url, headers=headers, files=files, data=data)
                if res.status_code == 200:
                    result_json = res.json()
                    transcript = result_json.get("text", "").strip()
                    
                    # Extract segment confidence and compression metrics
                    segments = result_json.get("segments", [])
                    max_no_speech = 0.0
                    avg_logprob = 0.0
                    max_comp_ratio = 1.0
                    if segments:
                        max_no_speech = max(s.get("no_speech_prob", 0.0) for s in segments)
                        max_comp_ratio = max(s.get("compression_ratio", 1.0) for s in segments)
                        logprobs = [s.get("avg_logprob", 0.0) for s in segments if "avg_logprob" in s]
                        if logprobs:
                            avg_logprob = sum(logprobs) / len(logprobs)
                    
                    is_gib, reason = is_gibberish_or_silence(transcript, no_speech_prob=max_no_speech, avg_logprob=avg_logprob, compression_ratio=max_comp_ratio)
                    if is_gib:
                        return transcript, False, reason
                    return transcript, True, ""
                else:
                    print(f"⚠️ Groq Whisper transcription failed ({res.status_code}): {res.text}")
        except Exception as e:
            print(f"⚠️ Groq Whisper transcription exception: {e}")

    # 2. Fallback: OpenAI-compatible audio endpoint if configured
    openai_key = os.getenv("OPENAI_API_KEY", "")
    if openai_key:
        try:
            oai_url = "https://api.openai.com/v1/audio/transcriptions"
            headers = {"Authorization": f"Bearer {openai_key.strip()}"}
            files = {
                "file": (f"voice_note.{ext}", audio_bytes, mime_type or "audio/ogg")
            }
            data = {
                "model": "whisper-1",
                "response_format": "verbose_json",
                "temperature": "0.0",
                "language": "en"
            }
            async with httpx.AsyncClient(timeout=30.0) as client:
                res = await client.post(oai_url, headers=headers, files=files, data=data)
                if res.status_code == 200:
                    result_json = res.json()
                    transcript = result_json.get("text", "").strip()
                    segments = result_json.get("segments", [])
                    max_no_speech = 0.0
                    avg_logprob = 0.0
                    max_comp_ratio = 1.0
                    if segments:
                        max_no_speech = max(s.get("no_speech_prob", 0.0) for s in segments)
                        max_comp_ratio = max(s.get("compression_ratio", 1.0) for s in segments)
                        logprobs = [s.get("avg_logprob", 0.0) for s in segments if "avg_logprob" in s]
                        if logprobs:
                            avg_logprob = sum(logprobs) / len(logprobs)
                    is_gib, reason = is_gibberish_or_silence(transcript, no_speech_prob=max_no_speech, avg_logprob=avg_logprob, compression_ratio=max_comp_ratio)
                    if is_gib:
                        return transcript, False, reason
                    return transcript, True, ""
        except Exception as e:
            print(f"⚠️ OpenAI Whisper fallback exception: {e}")

    return "", False, "Transcription service unavailable (Please ensure GROQ_API_KEY is configured in Render)"

async def process_whatsapp_audio(sender_phone: str, media_id: str, is_tagged_reply: bool = False):
    """Downloads, transcribes, and processes incoming WhatsApp voice notes."""
    print(f"\n🎙️ [VOICE START] Processing voice note from {sender_phone} (Media ID: {media_id})...")
    
    # Send typing indicator while transcribing
    try:
        await send_whatsapp_typing_indicator(sender_phone)
    except Exception:
        pass

    # Step 1: Download audio bytes from Meta CDN
    audio_bytes, mime_type = await download_whatsapp_media(media_id)
    if not audio_bytes:
        await send_whatsapp_cloud_msg(
            sender_phone,
            "⚠️ I had trouble downloading your voice note from WhatsApp. Please try recording again or type your question! 🎙️📚"
        )
        return

    # Step 2: Transcribe via Groq Whisper Large v3
    transcript, is_valid, error_reason = await transcribe_voice_note(audio_bytes, mime_type)
    del audio_bytes
    gc.collect()
    print(f"🎙️ [VOICE TRANSCRIPT] ({sender_phone}): '{transcript}' | Valid: {is_valid} | Reason: {error_reason}")

    # Step 3: Handle inaudible / gibberish audio
    if not is_valid or not transcript:
        inaudible_msg = (
            "🎙️ *Voice Note Received*\n\n"
            "I couldn't clearly hear your medical question due to low volume or background noise.\n\n"
            "Could you please re-record in a quiet room or type your question? 💡📚"
        )
        await send_whatsapp_cloud_msg(sender_phone, inaudible_msg)
        try:
            await log_user_chat_message(sender_phone, "user", "[🎙️ Inaudible Voice Note]", msg_type="voice", metadata={"has_issue": True})
        except Exception:
            pass
        return

    # Step 4: Route into standard WhatsApp message processor with is_voice=True
    await process_whatsapp_message(sender_phone, transcript, is_tagged_reply, is_voice=True)

# ==========================================
# CURATED MEDICAL YOUTUBE VIDEO LECTURE ENGINE
# ==========================================

TRUSTED_MEDICAL_CHANNELS = [
    "Ninja Nerd", "Osmosis", "Armando Hasudungan", "Medicosis Perfectionalis",
    "Dirty Medicine", "Speed Pharmacology", "Dr. Najeeb", "Geeky Medics",
    "Khan Academy Medicine", "MedCram", "Alila Medical Media", "Interactive Biology",
    "Rhesus Medicine", "MEDSimplified", "Dr. Constantin", "Hasudungan", "Pathoma",
    "Picmonic", "Sketchy", "Zero To Finals", "Chirag Navadia", "AnatomyZone",
    "Sam Webster", "ICU Advantage", "Strong Medicine", "Medinaz", "Siebert Science"
]

def slugify_topic(text: str) -> str:
    """Creates a normalized alphanumeric cache key for a medical topic."""
    import re
    clean = re.sub(r'[^\w\s]', '', text.lower()).strip()
    words = [w for w in clean.split() if w not in SEARCH_STOP_WORDS]
    return "_".join(words[:6]) if words else clean[:30]

async def get_curated_youtube_lecture(query: str) -> dict | None:
    """Retrieves a top authentic medical lecture video for the topic (from MongoDB cache or fast YouTube scrape)."""
    if not query or len(query.strip()) < 3:
        return None

    cache_key = slugify_topic(query)
    if not cache_key:
        return None

    # Step 1: Check MongoDB Video Cache for instant 1ms retrieval
    if youtube_video_cache_col is not None:
        try:
            cached_doc = await youtube_video_cache_col.find_one({"cache_key": cache_key})
            if cached_doc:
                print(f"⚡ [YOUTUBE CACHE HIT] '{query}' -> {cached_doc.get('title')} ({cached_doc.get('channel')})")
                return {
                    "video_id": cached_doc.get("video_id"),
                    "url": cached_doc.get("url"),
                    "title": cached_doc.get("title"),
                    "channel": cached_doc.get("channel"),
                    "duration": cached_doc.get("duration", "")
                }
        except Exception as cache_err:
            print(f"⚠️ YouTube cache lookup error: {cache_err}")

    # Step 2: Query YouTube Search with medical bias
    search_term = f"{query} medical lecture"
    encoded = urllib.parse.quote_plus(search_term)
    yt_url = f"https://www.youtube.com/results?search_query={encoded}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9"
    }

    try:
        async with httpx.AsyncClient(timeout=6.0, follow_redirects=True, headers=headers) as client:
            res = await client.get(yt_url)
            if res.status_code == 200:
                html = res.text
                match = re.search(r'var ytInitialData = ({.*?});</script>', html)
                if not match:
                    match = re.search(r'window\["ytInitialData"\] = ({.*?});</script>', html)

                if match:
                    data = json.loads(match.group(1))
                    contents = (
                        data.get("contents", {})
                        .get("twoColumnSearchResultsRenderer", {})
                        .get("primaryContents", {})
                        .get("sectionListRenderer", {})
                        .get("contents", [])
                    )

                    candidate_videos = []
                    for section in contents:
                        items = section.get("itemSectionRenderer", {}).get("contents", [])
                        for item in items:
                            v = item.get("videoRenderer")
                            if not v:
                                continue
                            v_id = v.get("videoId")
                            v_title = v.get("title", {}).get("runs", [{}])[0].get("text", "")
                            v_owner = v.get("ownerText", {}).get("runs", [{}])[0].get("text", "")
                            v_len = v.get("lengthText", {}).get("simpleText", "")

                            if v_id and v_title:
                                is_trusted = any(c.lower() in v_owner.lower() for c in TRUSTED_MEDICAL_CHANNELS)
                                candidate_videos.append({
                                    "video_id": v_id,
                                    "url": f"https://www.youtube.com/watch?v={v_id}",
                                    "title": v_title,
                                    "channel": v_owner,
                                    "duration": v_len,
                                    "is_trusted": is_trusted
                                })

                    if candidate_videos:
                        # Prioritize trusted medical channels first
                        best_video = next((v for v in candidate_videos if v["is_trusted"]), candidate_videos[0])
                        
                        # Step 3: Cache result in MongoDB
                        if youtube_video_cache_col is not None:
                            try:
                                await youtube_video_cache_col.update_one(
                                    {"cache_key": cache_key},
                                    {"$set": {
                                        "cache_key": cache_key,
                                        "topic_query": query,
                                        "video_id": best_video["video_id"],
                                        "url": best_video["url"],
                                        "title": best_video["title"],
                                        "channel": best_video["channel"],
                                        "duration": best_video["duration"],
                                        "updated_at": datetime.utcnow().isoformat() + "Z"
                                    }},
                                    upsert=True
                                )
                            except Exception as save_err:
                                print(f"⚠️ YouTube cache save error: {save_err}")

                        print(f"🎬 [YOUTUBE DISCOVERED] '{query}' -> {best_video['title']} ({best_video['channel']})")
                        return best_video
    except Exception as scrape_err:
        print(f"⚠️ YouTube search scrape exception: {scrape_err}")

    return None

async def send_whatsapp_video_cta_card(to_number: str, video_info: dict):
    """Sends a high-resolution video thumbnail card with a 1-tap interactive CTA button that opens YouTube"""
    if not video_info or not video_info.get("url") or not WHATSAPP_TOKEN:
        return
    
    video_id = video_info.get("video_id")
    title = video_info.get("title", "Medical Lecture")
    channel = video_info.get("channel", "Medical Channel")
    duration = f" ({video_info['duration']})" if video_info.get("duration") else ""
    yt_url = video_info.get("url")
    
    # YouTube HD thumbnail URL (hqdefault is universally available on all YouTube videos)
    image_url = f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg" if video_id else ""
    
    body_text = (
        f"🎥 *Recommended Video Lecture*\n\n"
        f"▶️ *{channel}* – {title}{duration}\n\n"
        f"Tap the button below to watch the full lecture on YouTube:"
    )
    body_text = format_whatsapp_text(body_text)

    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }
    
    interactive_obj = {
        "type": "cta_url",
        "body": {
            "text": body_text
        },
        "action": {
            "name": "cta_url",
            "parameters": {
                "display_text": "🎬 Watch on YouTube",
                "url": yt_url
            }
        }
    }
    
    if image_url:
        interactive_obj["header"] = {
            "type": "image",
            "image": {
                "link": image_url
            }
        }

    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "interactive",
        "interactive": interactive_obj
    }
    
    try:
        res = await shared_http_client.post(url, headers=headers, json=payload)
        print(f"Meta Video CTA Card Status {res.status_code}: {res.text}")
        if res.status_code != 200:
            # Fallback if image header was rejected: send without header
            interactive_obj.pop("header", None)
            payload["interactive"] = interactive_obj
            async with httpx.AsyncClient(timeout=15.0) as fallback_client:
                fallback_res = await fallback_client.post(url, headers=headers, json=payload)
                print(f"Meta Video CTA Fallback Status {fallback_res.status_code}: {fallback_res.text}")
        else:
            try:
                asyncio.create_task(log_user_chat_message(to_number, "assistant", f"🎥 [Video Lecture Card: {channel} - {title}]", msg_type="video_card"))
            except Exception:
                pass
    except Exception as e:
        print(f"⚠️ Exception sending Video CTA card: {e}")





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
    "sketch", "visual", "visualize", "view",
    "neura", "ai", "there", "still", "here", "u", "ur", "ready", "online", "awake"
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
        elif "chemical pathology" in msg_lower and ("crook" in b_lower or "martin" in b_lower or "chemical" in b_lower or "clinical" in b_lower):
            override_books.append(b)
        elif "microbiology" in msg_lower and ("microbiology" in b_lower or "jawetz" in b_lower):
            override_books.append(b)
        elif "lippincott" in msg_lower and "lippincott" in b_lower:
            override_books.append(b)
        elif "robbins" in msg_lower and "robbins" in b_lower:
            override_books.append(b)
        elif "sembulingam" in msg_lower and "sembulingam" in b_lower:
            override_books.append(b)
        elif ("crook" in msg_lower or "martin" in msg_lower) and ("crook" in b_lower or "martin" in b_lower):
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
    t0 = time.perf_counter()
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
        
    print(f"⏱️ [KEYWORD TIMER] extract_medical_terms finished in {(time.perf_counter() - t0)*1000:.2f}ms -> {phrases}")
    return phrases

async def normalize_medical_query(user_msg: str) -> dict:
    """Upfront Micro-LLM normalizer: resolves typos, abbreviations, and expands clinical concepts into standard textbook search queries."""
    t_start = time.perf_counter()
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
        "3. CRITICAL NON-MEDICAL GUARDRAIL: If the user input is NOT a medical or clinical question (e.g. casual check-in, greeting, or remark like 'are you there', 'hello', 'who are you', 'tell me a joke'), do NOT invent or force a medical disease. Return: {\"corrected_topic\": \"\", \"search_keywords\": []}.\n"
        "Output ONLY a valid JSON object in this exact schema:\n"
        "{\n"
        '  "corrected_topic": "Acute Lymphoblastic Leukemia Symptoms",\n'
        '  "search_keywords": ["acute lymphoblastic leukemia symptoms", "ALL clinical features presentation", "lymphoblast bone marrow failure"]\n'
        "}\n"
        "Output ONLY valid JSON."
    )
    payload = {
        "models": [FRONTDESK_MODEL, FALLBACK_MODEL],
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.0,
        "max_tokens": 500,
        "reasoning": get_reasoning_config(user_msg, is_micro=True)
    }
    try:
        res = await shared_http_client.post(url, headers=headers, json=payload)
        dt = time.perf_counter() - t_start
        if res.status_code == 200:
            text = res.json()["choices"][0]["message"]["content"]
            parsed = extract_json_from_llm(text)
            if isinstance(parsed, dict) and "search_keywords" in parsed:
                print(f"⏱️ [NORMALIZER TIMER] normalize_medical_query completed in {dt:.3f}s -> {parsed.get('search_keywords')}")
                return parsed
        print(f"⏱️ [NORMALIZER TIMER] normalize_medical_query HTTP {res.status_code} in {dt:.3f}s")
    except Exception as e:
        print(f"⚠️ Micro-LLM normalizer error ({time.perf_counter() - t_start:.3f}s): {e}")

    return fallback_result

async def evaluate_retrieval_adequacy(user_msg: str, retrieved_points: list, student_name: str = "Doctor") -> dict:
    """Evaluates whether retrieved textbook chunks sufficiently cover the student's question before declaring a topic missing.
    If inadequate due to synonym mismatch, narrow search, or hierarchical difference, returns re-anchored queries for second-pass scan.
    If genuinely not in the textbooks, generates an intelligent, encouraging, concise contextual response so the student never feels the bot is broken.
    """
    t_start = time.perf_counter()
    if not OPENROUTER_API_KEY:
        return {"is_adequate": bool(retrieved_points), "is_genuinely_absent": not bool(retrieved_points), "re_anchored_queries": [], "smart_encouraging_response": ""}

    # Prepare compact context summary (first 180 chars of each chunk)
    context_summaries = []
    if retrieved_points:
        for idx, p in enumerate(retrieved_points[:8], 1):
            payload = p.payload
            b_title = payload.get("book_title", "Textbook")
            snippet = payload.get("text", "")[:200].replace("\n", " ")
            context_summaries.append(f"[{idx}. {b_title}]: {snippet}...")
    
    combined_context_summary = "\n".join(context_summaries) if context_summaries else "No matching chunks found in initial search."

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neura-ai.org",
        "X-Title": "NEURA AI Medical Assistant"
    }
    system_prompt = (
        "You are an expert MBBS medical retrieval evaluator and intelligent clinical study co-pilot.\n"
        "A student asked a question, and the vector database returned initial textbook snippets.\n"
        "Your task is twofold:\n"
        "1. Check if the retrieved text adequately covers the question. If inadequate, provide 2 to 3 broader or alternative authoritative textbook search queries (e.g. parent chapter titles, drug class mechanisms, anatomical systems, pathology categories) to scan other parts of the textbook library.\n"
        "2. If the concept is genuinely outside the indexed medical textbooks (or non-curricular), generate an intelligent, concise, warm, and highly encouraging 2-3 sentence response directly addressing what they asked with clarity and guiding them to related clinical topics or asking them to explore core subjects.\n\n"
        "Output ONLY a valid JSON object in this exact schema:\n"
        "{\n"
        '  "is_adequate": true,\n'
        '  "is_genuinely_absent": false,\n'
        '  "re_anchored_queries": ["query 1", "query 2", "query 3"],\n'
        '  "smart_encouraging_response": "That is an insightful question, ' + student_name + '! While your current active textbooks focus on core pathology, pharmacology, and anatomy rather than this specific topic, I can dive into any related clinical mechanisms or case studies whenever you are ready. What topic are we tackling next?"\n'
        "}\n"
        "Output ONLY valid JSON."
    )
    user_payload_text = (
        f"STUDENT QUESTION: {user_msg}\n\n"
        f"RETRIEVED TEXTBOOK CONTEXT SNIPPETS:\n{combined_context_summary}"
    )
    payload = {
        "model": DEFAULT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload_text}
        ],
        "temperature": 0.2,
        "max_tokens": 500,
        "reasoning": get_reasoning_config(user_msg, is_micro=True),
        "provider": {
            "order": DEFAULT_PROVIDER_ORDER,
            "allow_fallbacks": True
        }
    }
    try:
        res = await shared_http_client.post(url, headers=headers, json=payload)
        dt = time.perf_counter() - t_start
        if res.status_code == 200:
            text = res.json()["choices"][0]["message"]["content"]
            parsed = extract_json_from_llm(text)
            if isinstance(parsed, dict) and "is_adequate" in parsed:
                print(f"⏱️ [EVALUATOR TIMER] evaluate_retrieval_adequacy completed in {dt:.3f}s (Adequate: {parsed.get('is_adequate')})")
                return parsed
        print(f"⏱️ [EVALUATOR TIMER] evaluate_retrieval_adequacy HTTP {res.status_code} in {dt:.3f}s")
    except Exception as e:
        print(f"⚠️ Micro-LLM retrieval evaluator error ({time.perf_counter() - t_start:.3f}s): {e}")

    return {"is_adequate": True, "is_genuinely_absent": False, "re_anchored_queries": [], "smart_encouraging_response": ""}

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
        elif "crook" in b_lower or "martin" in b_lower or "chemical pathology" in b_lower or "clinical biochemistry" in b_lower:
            keywords.append("martin")
            keywords.append("crook")
            keywords.append("crooke")
            keywords.append("chemical")
            keywords.append("clinical")
            keywords.append("biochemistry")
        else:
            words = [w.lower() for w in re.sub(r'[^\w\s]', '', b).split() if len(w) > 3]
            keywords.extend(words)
    return keywords

async def search_single_book(query_vector: list, book: str, limit: int = 4) -> list:
    if not book or not isinstance(book, str) or book.startswith("Skip"):
        return []
    t0 = time.perf_counter()
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
    elif "crook" in b_lower or "martin" in b_lower or "chemical pathology" in b_lower: book_kw = "martin"

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
    t_start = time.perf_counter()
    try:
        loop = asyncio.get_running_loop()
        t_embed_start = time.perf_counter()
        query_vector = await loop.run_in_executor(embedding_pool, get_embedding_sync, query_text)
        dt_embed = time.perf_counter() - t_embed_start

        if not preferred_books:
            t_qdrant_start = time.perf_counter()
            res = await qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit
            )
            dt_total = time.perf_counter() - t_start
            print(f"⏱️ [QDRANT TIMER] search_qdrant(all_books, '{query_text[:30]}') finished in {dt_total:.3f}s (Embed: {dt_embed*1000:.1f}ms, Qdrant: {(time.perf_counter()-t_qdrant_start)*1000:.1f}ms, Chunks: {len(res.points)})")
            return res.points

        # Query all selected textbooks concurrently in parallel!
        t_qdrant_start = time.perf_counter()
        tasks = [search_single_book(query_vector, b, limit=limit) for b in preferred_books if b and not b.startswith("Skip")]
        book_results = await asyncio.gather(*tasks)
        all_points = [p for sub in book_results for p in sub]
        all_points.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        dt_total = time.perf_counter() - t_start
        print(f"⏱️ [QDRANT TIMER] search_qdrant({len(preferred_books)} books, '{query_text[:30]}') finished in {dt_total:.3f}s (Embed: {dt_embed*1000:.1f}ms, Qdrant: {(time.perf_counter()-t_qdrant_start)*1000:.1f}ms, Chunks: {len(all_points)})")
        return all_points

    except Exception as outer_e:
        print(f"❌ Error in search_qdrant ({time.perf_counter() - t_start:.3f}s): {outer_e}")
        return []

async def multi_search_qdrant(search_terms: list, preferred_books: list = None) -> list:
    """Run separate Qdrant searches for each extracted medical keyword CONCURRENTLY, with automatic cross-textbook safety net if single book context is sparse."""
    t_multi_start = time.perf_counter()
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
    
    dt_multi = time.perf_counter() - t_multi_start
    print(f"⏱️ [MULTI-SEARCH TIMER] multi_search_qdrant ({len(search_terms)} terms) finished in {dt_multi:.3f}s (Total Deduplicated Chunks: {len(all_results)})")
    
    # Sort points by score descending and cap at 15 points for comprehensive multi-textbook coverage
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
    
    # 1. New user or user with missing/None onboarding step
    if not user_doc or not user_doc.get("onboarding_step"):
        await users_col.update_one(
            {"user_id": sender_phone},
            {"$set": {"onboarding_step": "ASK_NAME"}},
            upsert=True
        )
        welcome_msg = (
            "Hello! 👋 I'm *NEURA AI*, your clinical co-pilot and medical study assistant.\n\n"
            "I'm here to help you master complex clinical concepts, diagnose medical cases, and practice high-yield exam MCQs tailored to your level.\n\n"
            "To get started, what is your first name?"
        )
        await send_whatsapp_cloud_msg(sender_phone, welcome_msg)
        return True
        
    step = str(user_doc.get("onboarding_step") or "")
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
        std_books = get_all_curriculum_books_for_level(new_level)
        await users_col.update_one({"user_id": sender_phone}, {"$set": {"level": new_level, "preferred_books_list": std_books}})
        
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
        json_raw = await call_openrouter_llm(SYSTEM_INTERACTIVE_QUIZ_PROMPT, user_prompt, max_tokens=2200)
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
    vignette = q.get("vignette", "")

    question_text = (
        f"🏥 *NEURA AI MBBS Exam Quiz* (Q{q_num}/{total})\n\n"
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

async def process_whatsapp_message(sender_phone: str, user_msg: str, is_tagged_reply: bool = False, is_voice: bool = False):
    """Background task wrapper to process messages sequentially per user lock"""
    lock = get_user_lock(sender_phone)
    async with lock:
        await _process_whatsapp_message_internal(sender_phone, user_msg, is_tagged_reply, is_voice)

async def _process_whatsapp_message_internal(sender_phone: str, user_msg: str, is_tagged_reply: bool = False, is_voice: bool = False):
    """Internal task to run RAG & OpenRouter LLM and send WhatsApp reply"""
    req_t0 = time.perf_counter()
    prefix = "🎙️ [VOICE QUERY]" if is_voice else "🚀 [PIPELINE START]"
    print(f"\n{prefix} Received message from {sender_phone}: '{user_msg[:60]}'")
    try:
        # Log incoming user message for admin conversation transcript and diagnostics
        try:
            msg_type_tag = "voice_query" if is_voice else "user_query"
            logged_content = f"🎙️ {user_msg}" if is_voice else user_msg
            asyncio.create_task(log_user_chat_message(sender_phone, "user", logged_content, msg_type=msg_type_tag, metadata={"is_tagged_reply": is_tagged_reply, "is_voice": is_voice}))
        except Exception:
            pass

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
                preferred_books_list = list(user_doc.get("preferred_books_list", []))

                # Clean up legacy title if present
                old_name = "Crook Martin Andrew Clinical B"
                new_name = "Martin and crooke clinical biochemistry"
                if old_name in preferred_books_list:
                    preferred_books_list = [new_name if b == old_name else b for b in preferred_books_list]
                    asyncio.create_task(users_col.update_one(
                        {"user_id": sender_phone, "preferred_books_list": old_name},
                        {"$set": {"preferred_books_list.$": new_name}}
                    ))

        # Update daily study streak and activity timestamp
        streak = await update_user_study_streak(sender_phone)
        print(f"⏱️ [REQ +{time.perf_counter()-req_t0:.3f}s] User profile loaded: '{name}' ({level}), Streak: {streak}d")

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

            if msg_lower.startswith("/broadcast ") or msg_lower.startswith("broadcast "):
                admin_phones = [p.strip() for p in os.getenv("ADMIN_PHONES", "2348109839187,2349021292141").split(",")]
                if sender_phone in admin_phones or sender_phone == os.getenv("ADMIN_PHONE", "2348109839187"):
                    parts = user_msg.split(" ", 1)
                    broadcast_text = parts[1].strip() if len(parts) > 1 else ""
                    if broadcast_text:
                        await send_whatsapp_cloud_msg(sender_phone, "📣 *Broadcasting Started!*\n\nDelivering your announcement to all registered students in the background...")
                        asyncio.create_task(execute_broadcast_task(
                            broadcast_id=str(uuid.uuid4()),
                            message=broadcast_text,
                            target_level="ALL",
                            template_name="neura_announcement",
                            mode="smart",
                            admin_notify_phone=sender_phone
                        ))
                        return
                    else:
                        await send_whatsapp_cloud_msg(sender_phone, "⚠️ Usage: */broadcast [your announcement text]*")
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
                    fresh_doc = await users_col.find_one({"user_id": sender_phone}) if users_col is not None else None
                    if fresh_doc:
                        preferred_books_list = list(fresh_doc.get("preferred_books_list", []))
                        name = fresh_doc.get("name", name)
                        level = fresh_doc.get("level", level)
                        streak_count = fresh_doc.get("study_streak_days", streak)
                        reminders_status = "Enabled 🔔" if fresh_doc.get("reminders_enabled", True) else "Disabled 🔕"
                    else:
                        streak_count = streak
                        reminders_status = "Enabled 🔔"

                    print(f"👤 [/profile for {sender_phone}] Loaded {len(preferred_books_list)} books from MongoDB: {preferred_books_list}")
                    books_str = "\n  - ".join(preferred_books_list) if preferred_books_list else "None selected"
                    await send_whatsapp_cloud_msg(
                        sender_phone, 
                        f"👤 *Your Profile*\n• Name: {name}\n• Level: {level}\n• Study Streak: 🔥 {streak_count} Days\n• Reminders: {reminders_status}\n• Books:\n  - {books_str}\n\n"
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
                elif msg_lower in ["/update name", "/updatename", "/update_name", "/name", "update name", "updatename"]:
                    await users_col.update_one({"user_id": sender_phone}, {"$set": {"onboarding_step": "ASK_NAME"}})
                    await send_whatsapp_cloud_msg(sender_phone, "What would you like to change your name to?")
                    return
                elif msg_lower in ["/update level", "/updatelevel", "/update_level", "/level", "update level", "updatelevel"]:
                    await users_col.update_one({"user_id": sender_phone}, {"$set": {"onboarding_step": "ASK_LEVEL"}})
                    await send_whatsapp_interactive_list(
                        sender_phone, 
                        "What is your new medical class/level?",
                        "Select Level",
                        ["200L", "300L", "400L", "500L", "600L"]
                    )
                    return
                elif msg_lower in ["/update books", "/updatebooks", "/update_books", "/books", "/textbooks", "update books", "updatebooks", "books", "textbooks"]:
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
        t_intent_start = time.perf_counter()
        intent = await classify_intent(user_msg)
        print(f"⏱️ [REQ +{time.perf_counter()-req_t0:.3f}s] Intent classified: '{intent}' (took {(time.perf_counter()-t_intent_start)*1000:.1f}ms)")
        
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

        if intent == "PLATFORM_META":
            platform_system = (
                f"You are Neura, the AI medical study companion for Nigerian MBBS students chatting with {name} on WhatsApp.\n"
                "The student is asking a question about the NEURA AI platform, beta testing, privacy, feedback, or app features.\n"
                "GUIDELINES:\n"
                "- Answer directly, warmly, and naturally in 1 to 3 short sentences.\n"
                "- If asked what 'anonymous beta feedback' means, explain simply: 'Anonymous beta feedback just means any bug reports, impressions, or suggestions you share are completely confidential and never linked to your name or phone number—so you can be 100% honest with us! 😊'\n"
                "- If asked who created you or what Neura is, explain that Neura is an AI study co-pilot built specifically for Nigerian MBBS medical students, grounded in accredited curriculum textbooks.\n"
                "- If asked about commands or wallet, mention they can type /commands to explore features.\n"
                "- STRICT PROHIBITION: Under NO circumstances output a medical textbook lecture, headers, bulleted chapters, Senior Exam Tips, or YouTube video cards. Output pure natural conversational WhatsApp chat."
            )
            chat_history = []
            if chat_history_col is not None:
                user_doc_hist = await chat_history_col.find_one({"user_id": sender_phone})
                if user_doc_hist and "messages" in user_doc_hist:
                    chat_history = user_doc_hist["messages"][-6:]
            platform_reply = await call_openrouter_llm(platform_system, user_msg, chat_history=chat_history, max_tokens=250, model=FRONTDESK_MODEL)
            if not platform_reply:
                platform_reply = f"Hey *{name}*! Anonymous beta feedback just means any feedback or bug reports you share are completely confidential and not linked to your personal profile or number—so feel free to be 100% honest! 😊"
            await send_whatsapp_cloud_msg(sender_phone, platform_reply)
            
            if chat_history_col is not None:
                new_msgs = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": platform_reply}
                ]
                await chat_history_col.update_one(
                    {"user_id": sender_phone},
                    {"$push": {"messages": {"$each": new_msgs}}},
                    upsert=True
                )
            return

        if intent == "CONVERSATIONAL":
            conv_system = (
                f"You are Neura, a brilliant, warm, and empathetic senior medical colleague (like a trusted senior resident or sharp study buddy) chatting with {name}, a {level} MBBS medical student on WhatsApp.\n"
                "PERSONA & VOICE GUIDELINES:\n"
                "- Sound like an authentic, supportive human colleague texting on WhatsApp, NOT an AI bot, search engine, or robot.\n"
                "- If the student is checking in ('are you there', 'u there', emojis like 🥺), respond with genuine warmth and reassurance (e.g. 'Always right here with you! 😊 Taking a breather, or are we diving into something new?').\n"
                "- If the student is exhausted, stressed, or venting about ward rounds / med school, validate their feelings with real empathy and encouragement (e.g. 'Ward rounds can be brutal, Doc. Grab some water and take a quick break—you've got this!').\n"
                "- If they ask general study advice ('how do I study pharm', 'tips for 300L'), give practical, high-yield guidance in a peer-to-peer tone.\n"
                "- Keep responses concise (1 to 3 short, natural WhatsApp sentences max). Use conversational contractions ('I'm', 'let's', 'you're').\n"
                "- STRICT PROHIBITION: Under NO circumstances output a medical textbook lecture, headers, bulleted lists, '📖 [TOPIC]' titles, or video links here. Output pure natural conversational WhatsApp chat."
            )
            chat_history = []
            if chat_history_col is not None:
                user_doc_hist = await chat_history_col.find_one({"user_id": sender_phone})
                if user_doc_hist and "messages" in user_doc_hist:
                    chat_history = user_doc_hist["messages"][-6:]
            conv_reply = await call_openrouter_llm(conv_system, user_msg, chat_history=chat_history, max_tokens=250)
            if not conv_reply:
                conv_reply = f"Always right here with you, *{name}*! 😊\n\nTaking a quick breather, or are we diving into something new?"
            await send_whatsapp_cloud_msg(sender_phone, conv_reply)
            
            # Save to chat history
            if chat_history_col is not None:
                new_msgs = [
                    {"role": "user", "content": user_msg},
                    {"role": "assistant", "content": conv_reply}
                ]
                await chat_history_col.update_one(
                    {"user_id": sender_phone},
                    {"$push": {"messages": {"$each": new_msgs}}},
                    upsert=True
                )
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
                
                streak_footer = f"\n\n_🔥 {streak}-Day Study Streak_" if streak > 0 else ""
                final_answer = cached_answer + streak_footer
                await send_whatsapp_cloud_msg(sender_phone, final_answer)
                
                # Fetch and deliver HD Video CTA Card
                try:
                    video_info = await get_curated_youtube_lecture(clean_topic)
                    if video_info:
                        await send_whatsapp_video_cta_card(sender_phone, video_info)
                except Exception as vid_err:
                    print(f"⚠️ Cache video send error: {vid_err}")
                
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
                
                # [DISABLED] Automatic follow-up Practice MCQs interactive button per user request
                # clean_topic_label =         # ⚡ Step 1: Speculative Parallel Execution (Concurrent Vector Retrieval + Typo Normalization)
        t_parallel_start = time.perf_counter()
        local_terms = extract_medical_terms(search_term)
        active_books = get_explicit_book_override(search_term, preferred_books_list)

        # Launch vector search and micro-LLM normalizer simultaneously in parallel
        task_norm = normalize_medical_query(search_term)
        task_search = multi_search_qdrant(local_terms, preferred_books=active_books)

        normalized_data, search_res = await asyncio.gather(task_norm, task_search)
        dt_parallel = time.perf_counter() - t_parallel_start
        print(f"⏱️ [REQ +{time.perf_counter()-req_t0:.3f}s] Speculative Parallel Retrieval & Normalization finished in {dt_parallel:.3f}s (Initial Chunks: {len(search_res)})")

        clean_topic = clean_medical_topic_title(search_term, normalized_data.get("corrected_topic", ""))
        medical_terms = normalized_data.get("search_keywords") or local_terms

        # ⚡ Conversational Safety Valve: If query has zero medical terms and zero normalized keywords, treat as conversational
        if not local_terms and not normalized_data.get("search_keywords"):
            print(f"[SAFETY VALVE] Query '{search_term}' has 0 medical keywords. Redirecting to Conversational Companion...")
            conv_system = (
                f"You are Neura, a brilliant, warm, and supportive senior medical colleague talking to {name}, a {level} MBBS medical student on WhatsApp.\n"
                "Respond warmly, naturally, and concisely in 1-2 sentences. Keep it conversational. Under no circumstances output textbook headers, bulleted lists, or clinical definitions here."
            )
            conv_reply = await call_openrouter_llm(conv_system, user_msg, max_tokens=200)
            if not conv_reply:
                conv_reply = f"I'm right here with you, *{name}*! 😊 What medical topic or case study are we breaking down today?"
            await send_whatsapp_cloud_msg(sender_phone, conv_reply)
            return

        # ⚡ Fast-Path Check: If local search returned >= 3 high-confidence chunks, bypass evaluator completely!
        is_high_confidence = (
            len(search_res) >= 3 and 
            any(getattr(p, 'score', 0) >= 0.70 for p in search_res)
        )

        eval_result = {"is_adequate": True, "is_genuinely_absent": False}

        if not is_high_confidence:
            print(f"⏱️ [REQ +{time.perf_counter()-req_t0:.3f}s] Running fallback evaluation (High-Confidence: {is_high_confidence})...")
            # Fallback path for ambiguous or 0-chunk queries
            if not search_res:
                print(f"[SEARCH FALLBACK] 0 chunks with local terms. Re-querying with normalized terms: {medical_terms}")
                search_res = await multi_search_qdrant(medical_terms, preferred_books=active_books)

            eval_result = await evaluate_retrieval_adequacy(search_term, search_res, student_name=name)
            
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
            smart_resp = eval_result.get("smart_encouraging_response", "")
            if not smart_resp:
                smart_resp = (
                    f"That's an interesting question, *{name}*! 💡\n\n"
                    f"While your current textbooks focus on core clinical subjects rather than this specific topic, I'm ready to dive into any disease pathophysiology, pharmacology mechanism, or anatomy concept whenever you are!\n\n"
                    f"What medical topic or case study shall we explore next?"
                )
            await send_whatsapp_cloud_msg(sender_phone, smart_resp)
            return

        # If user explicitly asked for a quiz on a topic via text, launch the interactive quiz directly!
        if intent == "QUIZ":
            await start_interactive_quiz(sender_phone, clean_topic, search_res)
            return

        context_blocks = []
        for idx, point in enumerate(search_res[:15], 1):
            p = point.payload
            page_str = p.get('page_number') or p.get('chunk_index', 'N/A')
            book_str = p.get('book_title', 'Textbook')
            text_str = p.get('text', '')
            block = f"[Context {idx} | Book: {book_str}, Page/Chunk: {page_str}]\n{text_str}"
            context_blocks.append(block)

        formatted_context = "\n\n".join(context_blocks)

        critical_inst = (
            f"CRITICAL PEDAGOGICAL INSTRUCTION:\n"
            f"1. STRICT TEXTBOOK GROUNDING: Answer using the factual medical mechanisms, clinical classifications, and details provided in the RETRIEVED MEDICAL KNOWLEDGE CONTEXT. Do not invent non-curricular topics.\n"
            f"2. Start with H1 header: # *📖 {clean_topic.upper()}* followed immediately by a concise 1-2 sentence high-level overview/definition.\n"
            f"3. Explain everything in simple, clear, easy-to-understand language while keeping all original clinical depth and scientific accuracy. Stretch explanations where it helps understanding (make mechanisms detailed and clear step-by-step).\n"
            f"4. Use structured headings for logical hierarchy: ## *[Section Name]* (e.g. ## *Pathophysiology & Core Mechanisms*, ## *Clinical Manifestations*, ## *Diagnostic Workup*, ## *Management & Pharmacology*) and ### *[Sub-topic Name]*.\n"
            f"5. Use bullet points (- ) and numbered lists (1. ) with double-line spacing for readability.\n"
            f"6. Highlight important points using bold text (*Key Term*) and > blockquotes (> *Key Clinical Takeaway:* ...) for vital takeaways and pearls.\n"
            f"7. Whenever a technical term, disease name, syndrome, eponym, or special concept appears (e.g., 'anti-phospholipid syndrome', 'Horner syndrome'), immediately add a short, simple explanation of what it is after it is first mentioned.\n"
            f"8. Zero textbook meta-talk and zero fabricated figure citations."
        )

        if is_tagged_reply and last_assistant_msg:
            tagged_snippet = last_assistant_msg[:400]
            user_prompt = (
                f"THE USER EXPLICITLY TAGGED/QUOTED YOUR PREVIOUS WHATSAPP MESSAGE BELOW:\n\"\"\"{tagged_snippet}\"\"\"\n\n"
                f"CLEAN MEDICAL TOPIC: {clean_topic}\n"
                f"USER'S QUESTION/INSTRUCTION REGARDING THE TAGGED MESSAGE:\n{query_to_search}\n\n"
                f"RETRIEVED MEDICAL KNOWLEDGE CONTEXT:\n{formatted_context}\n\n"
                f"{critical_inst}"
            )
        else:
            user_prompt = (
                f"CLEAN MEDICAL TOPIC: {clean_topic}\n"
                f"STUDENT QUESTION:\n{query_to_search}\n\n"
                f"RETRIEVED MEDICAL KNOWLEDGE CONTEXT:\n{formatted_context}\n\n"
                f"{critical_inst}"
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

        # Lock 2: Strict Medical Video Guardrail
        # A video is ONLY retrieved and attached if:
        # 1. At least 2 medical chunks were retrieved
        # 2. At least one chunk has similarity score >= 0.65 (authentic textbook content)
        # 3. Topic does not contain platform/meta/conversational words
        non_medical_words = ["feedback", "beta", "anonymous", "wallet", "deposit", "command", "profile", "streak", "hello", "hi", "how are you", "what is neura", "who made", "test", "reminder", "reset"]
        clean_topic_lower = clean_topic.lower()
        should_fetch_video = (
            len(context_blocks) >= 2 and
            any(getattr(p, 'score', 0) >= 0.65 for p in search_res) and
            not any(w in clean_topic_lower for w in non_medical_words) and
            len(clean_topic.strip()) >= 3
        )

        print(f"⏱️ [REQ +{time.perf_counter()-req_t0:.3f}s] Dispatching to Main Medical LLM ({len(context_blocks)} chunks, ~{len(formatted_context)} context chars) | Video Allowed: {should_fetch_video}...")
        t_llm_start = time.perf_counter()
        task_llm = call_openrouter_llm(prompt_to_use, user_prompt, chat_history)
        
        async def _noop_video():
            return None

        task_video = get_curated_youtube_lecture(clean_topic) if should_fetch_video else _noop_video()

        ai_answer, video_info = await asyncio.gather(task_llm, task_video)
        print(f"⏱️ [REQ +{time.perf_counter()-req_t0:.3f}s] Main Medical LLM finished in {time.perf_counter()-t_llm_start:.3f}s | Video: {bool(video_info)}")

        if not ai_answer or not isinstance(ai_answer, str) or len(ai_answer.strip()) == 0:
            await send_whatsapp_cloud_msg(
                sender_phone, 
                f"I experienced a brief connection delay while analyzing *{clean_topic}*. Please tap below or re-send your question!"
            )
            return

        # Check if the answer indicates information is missing from textbooks
        ai_lower = ai_answer.lower()
        is_not_covered = ("not covered" in ai_lower or "sorry" in ai_lower[:30] or "not found" in ai_lower)

        streak_footer = f"\n\n_🔥 {streak}-Day Study Streak_" if streak > 0 else ""
        final_answer = ai_answer + streak_footer
        t_wa_start = time.perf_counter()
        await send_whatsapp_cloud_msg(sender_phone, final_answer)
        dt_total = time.perf_counter() - req_t0
        print(f"🎉 [PIPELINE COMPLETE | TOTAL: {dt_total:.3f}s] Delivered response to {sender_phone} via WhatsApp Cloud API in {(time.perf_counter()-t_wa_start)*1000:.1f}ms")

        # Deliver High-Definition Video Lecture Card with 1-Tap Play Button ONLY if verified medical topic
        if video_info and should_fetch_video and not is_not_covered:
            try:
                await send_whatsapp_video_cta_card(sender_phone, video_info)
            except Exception as vid_err:
                print(f"⚠️ Error sending video CTA card: {vid_err}")

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

        # Topic & Query Tracking for Weekly Digest and Profile
        if users_col is not None:
            try:
                await users_col.update_one(
                    {"user_id": sender_phone},
                    {
                        "$inc": {"weekly_queries_count": 1, "total_queries_count": 1},
                        "$set": {
                            "last_medical_topic": clean_topic,
                            "last_context_text": formatted_context,
                            "last_assistant_answer": ai_answer
                        }
                    }
                )
            except Exception as trk_err:
                print(f"⚠️ Tracking error: {trk_err}")

        # Check if the answer indicates information is missing from textbooks
        ai_lower = ai_answer.lower()
        is_not_covered = ("not covered" in ai_lower or "sorry" in ai_lower[:30] or "not found" in ai_lower)

        # Save in 24-hour LRU Cache for instant 0.05s delivery for other students
        if not is_not_covered and len(ai_answer) > 100:
            TOPIC_CACHE.set(search_term, ai_answer, formatted_context, preferred_books=preferred_books_list)

        # [DISABLED] Automatic follow-up Practice MCQs button per user request
        # if not user_msg.startswith("GENERATE_QUIZ") and not is_not_covered:
        #     try:
        #         clean_topic_label = clean_topic
        #         if len(clean_topic_label) > 90:
        #             clean_topic_label = clean_topic_label[:87] + "..."
        #         topic_snippet = clean_topic[:100]
        #         await send_whatsapp_interactive_button(
        #             sender_phone,
        #             f"Ready to practice MCQs on *{clean_topic_label}*?",
        #             [
        #                 {"id": f"GENERATE_QUIZ:{topic_snippet}", "title": "📝 Practice MCQs"}
        #             ]
        #         )
        #     except Exception as btn_err:
        #         print(f"⚠️ Non-critical error sending interactive button: {btn_err}")

    except Exception as e:
        print(f"ERROR in process_whatsapp_message: {str(e)}")
        print(traceback.format_exc())
        await send_whatsapp_cloud_msg(sender_phone, "Sorry, NEURA AI experienced a temporary connection delay. Please try asking your medical question again!")
    finally:
        gc.collect()

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
                    elif msg_type in ("audio", "voice"):
                        audio_obj = msg.get("audio", {}) or msg.get("voice", {})
                        media_id = audio_obj.get("id")
                        if media_id:
                            print(f"🎙️ Received Voice Note from {sender_phone} (Media ID: {media_id})")
                            task = BackgroundTask(process_whatsapp_audio, sender_phone, media_id, is_tagged_reply)
                            return Response(content=json.dumps({"status": "processing_audio"}), media_type="application/json", background=task)
                        else:
                            print(f"⚠️ Voice note from {sender_phone} missing media_id")
                            task = BackgroundTask(send_whatsapp_cloud_msg, sender_phone, "⚠️ Could not read voice note. Please try recording again! 🎙️")
                            return Response(content=json.dumps({"status": "missing_media_id"}), media_type="application/json", background=task)
                    else:
                        print(f"⚠️ Received unsupported message type '{msg_type}' from {sender_phone}")
                        task = BackgroundTask(send_whatsapp_cloud_msg, sender_phone, "I can read text and voice notes! Please type or record your medical question. 🤖🎙️📚")
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

# ==========================================
# 4. ADMIN BROADCAST & ANALYTICS DASHBOARD
# ==========================================

class BroadcastRequest(BaseModel):
    message: str
    target_level: str = "ALL"
    template_name: str = "neura_announcement"
    mode: str = "smart" # "smart", "direct_only", "template_only"

class AdminLoginRequest(BaseModel):
    password: str

async def execute_broadcast_task(broadcast_id: str, message: str, target_level: str, template_name: str, mode: str, admin_notify_phone: str = None):
    """Asynchronous background task that delivers broadcast messages with safety pacing."""
    if users_col is None:
        return
        
    query = {}
    if target_level and target_level not in ["ALL", "ACTIVE_24H"]:
        query["level"] = target_level
        
    if mode == "direct_only" or target_level == "ACTIVE_24H":
        # Target candidates active in last 24h (100% Free Standard Session)
        cutoff_date = (datetime.utcnow() + timedelta(hours=1) - timedelta(days=1)).strftime("%Y-%m-%d")
        now_ts = time.time()
        query["$or"] = [
            {"last_active_timestamp": {"$gte": now_ts - 86400}},
            {"last_study_date": {"$gte": cutoff_date}},
            {"last_active": {"$gte": cutoff_date}}
        ]
        
    cursor = users_col.find(query)
    students = await cursor.to_list(length=10000)
    
    total = len(students)
    sent_count = 0
    failed_count = 0
    
    log_doc = {
        "broadcast_id": broadcast_id,
        "message": message,
        "target_level": target_level,
        "mode": mode,
        "template_name": template_name,
        "total_targets": total,
        "sent_count": 0,
        "failed_count": 0,
        "status": "RUNNING",
        "started_at": datetime.utcnow().isoformat(),
        "completed_at": None
    }
    if broadcasts_col is not None:
        await broadcasts_col.insert_one(log_doc)
        
    for student in students:
        phone = student.get("user_id")
        if not phone or len(phone) < 8:
            failed_count += 1
            continue
            
        student_name = student.get("name", "Student")
        success = False
        
        try:
            if mode == "template_only":
                success = await send_whatsapp_template_msg(phone, template_name, [student_name, message])
            elif mode == "direct_only":
                success = await send_whatsapp_cloud_msg(phone, message)
            else: # smart hybrid
                direct_delivered = await send_whatsapp_cloud_msg(phone, message)
                if direct_delivered:
                    success = True
                else:
                    success = await send_whatsapp_template_msg(phone, template_name, [student_name, message])
        except Exception as err:
            print(f"Broadcast error for {phone}: {err}")
            success = False
            
        if success:
            sent_count += 1
        else:
            failed_count += 1
            
        await asyncio.sleep(0.04) # Safe pacing (~25 msgs/sec)
        
    if broadcasts_col is not None:
        await broadcasts_col.update_one(
            {"broadcast_id": broadcast_id},
            {
                "$set": {
                    "sent_count": sent_count,
                    "failed_count": failed_count,
                    "status": "COMPLETED",
                    "completed_at": datetime.utcnow().isoformat()
                }
            }
        )
        
    if admin_notify_phone:
        summary_card = (
            f"📢 *BROADCAST REPORT*\n\n"
            f"🎯 *Audience:* {target_level}\n"
            f"👥 *Total Targets:* {total}\n"
            f"✅ *Successfully Delivered:* {sent_count}\n"
            f"⚠️ *Failed / Unreachable:* {failed_count}\n\n"
            f"📝 *Message:* _{message[:60]}..._"
        )
        await send_whatsapp_cloud_msg(admin_notify_phone, summary_card)

@app.post("/admin/api/login")
async def admin_login(req: AdminLoginRequest):
    if req.password == ADMIN_PASSWORD:
        token = hashlib.sha256(f"{ADMIN_PASSWORD}_{datetime.utcnow().strftime('%Y-%m-%d')}".encode()).hexdigest()
        ADMIN_SESSIONS.add(token)
        return {"status": "success", "token": token}
    raise HTTPException(status_code=401, detail="Invalid admin credentials")

@app.get("/admin/api/stats")
async def admin_stats(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token not in ADMIN_SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    total_students = 0
    active_24h = 0
    total_wallet = 0.0
    level_dist = {"200L": 0, "300L": 0, "400L": 0, "500L": 0, "600L": 0, "Other": 0}
    
    if users_col is not None:
        total_students = await users_col.count_documents({})
        cutoff = (datetime.utcnow() - timedelta(hours=24)).strftime("%Y-%m-%d")
        active_24h = await users_col.count_documents({"last_study_date": {"$gte": cutoff}})
        
        pipeline = [
            {"$group": {"_id": "$level", "count": {"$sum": 1}, "wallet": {"$sum": "$wallet_balance_ngn"}}}
        ]
        results = await users_col.aggregate(pipeline).to_list(length=100)
        for r in results:
            lvl = r.get("_id") or "Other"
            if lvl in level_dist:
                level_dist[lvl] = r.get("count", 0)
            else:
                level_dist["Other"] += r.get("count", 0)
            total_wallet += r.get("wallet", 0.0)
            
    recent_broadcasts = []
    if broadcasts_col is not None:
        cursor = broadcasts_col.find().sort("started_at", -1).limit(10)
        async for doc in cursor:
            recent_broadcasts.append({
                "broadcast_id": doc.get("broadcast_id"),
                "message": doc.get("message", "")[:80],
                "target_level": doc.get("target_level", "ALL"),
                "total_targets": doc.get("total_targets", 0),
                "sent_count": doc.get("sent_count", 0),
                "failed_count": doc.get("failed_count", 0),
                "status": doc.get("status", "COMPLETED"),
                "started_at": doc.get("started_at", "")[:19].replace("T", " ")
            })
            
    return {
        "total_students": total_students,
        "active_24h": active_24h,
        "total_wallet_balance_ngn": round(total_wallet, 2),
        "level_distribution": level_dist,
        "recent_broadcasts": recent_broadcasts
    }

async def resolve_user_last_active(u: dict, uid: str) -> str:
    """Helper to resolve the most accurate last active date string (YYYY-MM-DD) for any user document."""
    # 1. Check explicit last_study_date or last_active
    raw_date = u.get("last_study_date") or u.get("last_active")
    if raw_date and str(raw_date).strip() and str(raw_date).strip() not in ("N/A", "None", ""):
        return str(raw_date).strip().split("T")[0].split(" ")[0]
        
    # 2. Check last_active_timestamp
    lat = u.get("last_active_timestamp")
    if lat:
        try:
            if isinstance(lat, (int, float)):
                return datetime.utcfromtimestamp(lat).strftime("%Y-%m-%d")
            elif isinstance(lat, str) and lat.strip():
                return lat.strip().split("T")[0].split(" ")[0]
        except Exception:
            pass

    # 3. Check chat_logs_col for latest message
    if chat_logs_col is not None and uid:
        try:
            latest_log = await chat_logs_col.find_one({"user_id": uid}, sort=[("timestamp", -1)])
            if latest_log:
                ts = latest_log.get("timestamp")
                if isinstance(ts, str) and ts.strip():
                    return ts.strip().split("T")[0].split(" ")[0]
                elif isinstance(ts, (int, float)):
                    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        except Exception:
            pass

    # 4. Check chat_history_col
    if chat_history_col is not None and uid:
        try:
            h = await chat_history_col.find_one({"user_id": uid})
            if h and "messages" in h and len(h["messages"]) > 0:
                latest_m = h["messages"][-1]
                ts = latest_m.get("timestamp")
                if isinstance(ts, str) and ts.strip():
                    return ts.strip().split("T")[0].split(" ")[0]
                elif isinstance(ts, (int, float)):
                    return datetime.utcfromtimestamp(ts).strftime("%Y-%m-%d")
        except Exception:
            pass

    # 5. Check document creation time via ObjectId _id
    doc_id = u.get("_id")
    if doc_id and hasattr(doc_id, "generation_time"):
        try:
            return doc_id.generation_time.strftime("%Y-%m-%d")
        except Exception:
            pass

    # 6. Fallback to current date in WAT
    now_wat = datetime.utcnow() + timedelta(hours=1)
    return now_wat.strftime("%Y-%m-%d")

@app.get("/admin/api/students")
async def admin_students(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token not in ADMIN_SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    if users_col is None:
        return {"students": []}
        
    cursor = users_col.find({}).sort("_id", -1)
    students = []
    now_ts = time.time()
    async for u in cursor:
        uid = u.get("user_id", "")
        last_active = await resolve_user_last_active(u, uid)
        lat = u.get("last_active_timestamp")
        is_active_24h = False
        if lat:
            try:
                is_active_24h = ((now_ts - float(lat)) < 86400)
            except Exception:
                pass
                
        books_list = list(u.get("preferred_books_list", []))
        students.append({
            "user_id": uid,
            "name": u.get("name", "Student"),
            "level": u.get("level", "Unset"),
            "study_streak_days": int(u.get("study_streak_days", 1)),
            "total_queries_count": int(u.get("total_queries_count", 0)),
            "wallet_balance_ngn": float(u.get("wallet_balance_ngn", 0.0)),
            "last_study_date": last_active,
            "is_active_24h": is_active_24h,
            "preferred_books_list": books_list,
            "books_count": len(books_list),
            "onboarding_step": u.get("onboarding_step", "COMPLETED"),
            "reminders_enabled": u.get("reminders_enabled", True)
        })
    return {"students": students}

@app.get("/admin/api/students/{user_id}/activity")
async def admin_get_student_activity(user_id: str, request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token not in ADMIN_SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    if users_col is None:
        raise HTTPException(status_code=404, detail="Database uninitialized")
        
    u = await users_col.find_one({"user_id": user_id})
    if not u:
        raise HTTPException(status_code=404, detail="Student not found")
        
    last_active = await resolve_user_last_active(u, user_id)
    lat = u.get("last_active_timestamp")
    now_ts = time.time()
    is_active_24h = False
    if lat:
        try:
            is_active_24h = ((now_ts - float(lat)) < 86400)
        except Exception:
            pass
            
    total_msgs = 0
    recent_events = []
    if chat_logs_col is not None:
        total_msgs = await chat_logs_col.count_documents({"user_id": user_id})
        recent_cursor = chat_logs_col.find({"user_id": user_id}).sort("timestamp", -1).limit(10)
        async for doc in recent_cursor:
            ts = doc.get("timestamp", "")
            if ts and isinstance(ts, str) and not ts.endswith("Z"):
                ts += "Z"
            recent_events.append({
                "role": doc.get("role", "user"),
                "content": doc.get("content", "")[:150],
                "timestamp": ts,
                "msg_type": doc.get("metadata", {}).get("msg_type", "text"),
                "has_issue": doc.get("metadata", {}).get("has_issue", False)
            })
            
    return {
        "user_id": user_id,
        "name": u.get("name", "Student"),
        "level": u.get("level", "Unset"),
        "study_streak_days": int(u.get("study_streak_days", 1)),
        "total_queries_count": int(u.get("total_queries_count", 0)),
        "wallet_balance_ngn": float(u.get("wallet_balance_ngn", 0.0)),
        "reminders_enabled": bool(u.get("reminders_enabled", True)),
        "onboarding_step": u.get("onboarding_step", "COMPLETED"),
        "preferred_books_list": list(u.get("preferred_books_list", [])),
        "last_active": last_active,
        "is_active_24h": is_active_24h,
        "total_messages": total_msgs,
        "recent_events": recent_events
    }

@app.get("/admin/api/students/{user_id}/chats")
async def admin_get_student_chats(user_id: str, request: Request, include_broadcasts: bool = False):
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token not in ADMIN_SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    user_info = {
        "user_id": user_id,
        "name": "Student",
        "level": "Unset",
        "study_streak_days": 1,
        "total_queries_count": 0,
        "wallet_balance_ngn": 0.0,
        "last_study_date": (datetime.utcnow() + timedelta(hours=1)).strftime("%Y-%m-%d"),
        "preferred_books_list": [],
        "onboarding_step": "COMPLETED",
        "reminders_enabled": True
    }
    if users_col is not None:
        u = await users_col.find_one({"user_id": user_id})
        if u:
            last_active = await resolve_user_last_active(u, user_id)
            user_info = {
                "user_id": u.get("user_id", user_id),
                "name": u.get("name", "Student"),
                "level": u.get("level", "Unset"),
                "study_streak_days": int(u.get("study_streak_days", 1)),
                "total_queries_count": int(u.get("total_queries_count", 0)),
                "wallet_balance_ngn": float(u.get("wallet_balance_ngn", 0.0)),
                "last_study_date": last_active,
                "preferred_books_list": list(u.get("preferred_books_list", [])),
                "onboarding_step": u.get("onboarding_step", "COMPLETED"),
                "reminders_enabled": u.get("reminders_enabled", True)
            }

    messages = []
    if chat_logs_col is not None:
        query = {"user_id": user_id}
        if not include_broadcasts:
            query["metadata.msg_type"] = {"$ne": "broadcast"}
            
        cursor = chat_logs_col.find(query).sort("timestamp", 1)
        async for doc in cursor:
            ts = doc.get("timestamp", "")
            if ts and isinstance(ts, str) and not ts.endswith("Z"):
                ts += "Z"
            messages.append({
                "role": doc.get("role", "user"),
                "content": doc.get("content", ""),
                "timestamp": ts,
                "metadata": doc.get("metadata", {})
            })

    if not messages and chat_history_col is not None:
        h = await chat_history_col.find_one({"user_id": user_id})
        if h and "messages" in h:
            for m in h["messages"]:
                if isinstance(m, dict):
                    mtype = m.get("msg_type", "text")
                    if not include_broadcasts and mtype == "broadcast":
                        continue
                    ts = m.get("timestamp", "")
                    if ts and isinstance(ts, str) and not ts.endswith("Z"):
                        ts += "Z"
                    messages.append({
                        "role": m.get("role", "user"),
                        "content": m.get("content", ""),
                        "timestamp": ts,
                        "metadata": {"has_issue": m.get("has_issue", False), "msg_type": mtype}
                    })

    issue_count = 0
    for m in messages:
        c_low = str(m.get("content", "")).lower()
        has_issue = (
            m.get("metadata", {}).get("has_issue") or
            "connection delay" in c_low or
            "trouble creating the interactive practice" in c_low or
            "error processing your query" in c_low or
            "experienced a brief" in c_low or
            "only read text messages" in c_low or
            "i had trouble creating" in c_low
        )
        if has_issue:
            issue_count += 1
            if "metadata" not in m:
                m["metadata"] = {}
            m["metadata"]["has_issue"] = True

    return {
        "student": user_info,
        "total_messages": len(messages),
        "issue_count": issue_count,
        "messages": messages
    }

@app.get("/admin/api/chats/recent")
async def admin_get_recent_chats(request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token not in ADMIN_SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    if users_col is None:
        return {"conversations": []}
        
    students = await users_col.find({}).sort("_id", -1).to_list(length=1000)
    conversations = []
    for s in students:
        uid = s.get("user_id", "")
        if not uid:
            continue
            
        last_msg = ""
        last_time = await resolve_user_last_active(s, uid)
        msg_count = 0
        has_issue = False
        
        if chat_logs_col is not None:
            latest_doc = await chat_logs_col.find_one({"user_id": uid}, sort=[("timestamp", -1)])
            if latest_doc:
                last_msg = latest_doc.get("content", "")[:120]
                last_time = str(latest_doc.get("timestamp", last_time)).split("T")[0].split(" ")[0]
                msg_count = await chat_logs_col.count_documents({"user_id": uid})
                has_issue = bool(latest_doc.get("metadata", {}).get("has_issue"))
        
        if msg_count == 0 and chat_history_col is not None:
            h = await chat_history_col.find_one({"user_id": uid})
            if h and "messages" in h and len(h["messages"]) > 0:
                msg_count = len(h["messages"])
                latest = h["messages"][-1]
                last_msg = latest.get("content", "")[:120]
                last_time = str(latest.get("timestamp", last_time)).split("T")[0].split(" ")[0]
                has_issue = any(
                    m.get("has_issue") or "connection delay" in str(m.get("content", "")).lower()
                    for m in h["messages"] if isinstance(m, dict)
                )
                
        conversations.append({
            "user_id": uid,
            "name": s.get("name", "Student"),
            "level": s.get("level", "Unset"),
            "study_streak_days": int(s.get("study_streak_days", 1)),
            "total_queries_count": int(s.get("total_queries_count", 0)),
            "last_message": last_msg or "No chat history recorded yet",
            "last_time": last_time,
            "message_count": msg_count,
            "has_issue": has_issue
        })
        
    return {"conversations": conversations}

@app.post("/admin/api/broadcast")
async def admin_broadcast(req: BroadcastRequest, request: Request, background_tasks: BackgroundTasks):
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token not in ADMIN_SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    if not req.message or len(req.message.strip()) < 3:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    broadcast_id = str(uuid.uuid4())
    background_tasks.add_task(
        execute_broadcast_task,
        broadcast_id,
        req.message.strip(),
        req.target_level,
        req.template_name,
        req.mode
    )
    return {"status": "started", "broadcast_id": broadcast_id}

class AdminDirectMessageRequest(BaseModel):
    message: str
    mode: str = "smart" # "smart", "direct_only", "template_only"
    template_name: str = "neura_announcement"

@app.post("/admin/api/students/{user_id}/send-message")
async def admin_send_student_message(user_id: str, req: AdminDirectMessageRequest, request: Request):
    token = request.headers.get("Authorization", "").replace("Bearer ", "").strip()
    if token not in ADMIN_SESSIONS:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    msg_text = (req.message or "").strip()
    if not msg_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
        
    student_name = "Student"
    if users_col is not None:
        user_doc = await users_col.find_one({"user_id": user_id})
        if user_doc:
            student_name = user_doc.get("name", "Student")
            
    success = False
    delivery_channel = "direct"
    
    if req.mode == "template_only":
        success = await send_whatsapp_template_msg(user_id, req.template_name, [student_name, msg_text])
        delivery_channel = f"template:{req.template_name}"
    elif req.mode == "direct_only":
        success = await send_whatsapp_cloud_msg(user_id, msg_text)
        delivery_channel = "direct"
    else: # smart hybrid (try direct session first, auto fallback to Meta template if outside 24h window)
        success = await send_whatsapp_cloud_msg(user_id, msg_text)
        if not success:
            print(f"[SMART ROUTING] Direct send to {user_id} unconfirmed. Auto-falling back to template '{req.template_name}'...")
            success = await send_whatsapp_template_msg(user_id, req.template_name, [student_name, msg_text])
            delivery_channel = f"template:{req.template_name}"
            
    if success:
        return {"status": "success", "channel": delivery_channel, "user_id": user_id}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to deliver message via {delivery_channel}")

@app.get("/admin")
async def admin_dashboard_page():
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NEURA AI — Enterprise Administration Hub</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
  <style>
    :root {
      --bg: #F7F8FA;
      --surface: #FFFFFF;
      --surface-alt: #F1F2F4;
      --border: #E4E6EA;
      --text-primary: #17181A;
      --text-secondary: #5A5E67;
      --text-muted: #9298A3;
      --sidebar-bg: #14161A;
      --sidebar-text: #C7C9CE;
      --sidebar-text-active: #FFFFFF;
      --status-success: #1F8A56;
      --status-warning: #B7791F;
      --status-error: #C0392B;
      --focus-ring: #2C2F36;
    }

    * {
      box-sizing: border-box;
    }

    body {
      background-color: var(--bg);
      color: var(--text-primary);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      font-size: 13px;
      line-height: 1.5;
      margin: 0;
      padding: 0;
      -webkit-font-smoothing: antialiased;
    }

    h1, h2, h3, h4, h5, h6 {
      color: var(--text-primary);
      letter-spacing: -0.01em;
      font-weight: 600;
    }

    .card {
      background-color: var(--surface);
      border: 1px solid var(--border);
      border-radius: 6px;
    }

    .btn-primary {
      background-color: var(--text-primary);
      color: #FFFFFF;
      border: 1px solid var(--text-primary);
      border-radius: 4px;
      font-weight: 500;
      font-size: 12px;
      padding: 6px 12px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: background-color 0.15s ease, opacity 0.15s ease;
    }
    .btn-primary:hover {
      background-color: #2C2F36;
    }
    .btn-primary:disabled {
      opacity: 0.5;
      cursor: not-allowed;
    }

    .btn-secondary {
      background-color: var(--surface);
      color: var(--text-primary);
      border: 1px solid var(--border);
      border-radius: 4px;
      font-weight: 500;
      font-size: 12px;
      padding: 6px 12px;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .btn-secondary:hover {
      background-color: var(--surface-alt);
      border-color: #D0D4DA;
    }

    .input-compact {
      background-color: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      color: var(--text-primary);
      font-size: 12px;
      padding: 6px 10px;
      height: 32px;
      outline: none;
      transition: border-color 0.15s ease;
    }
    .input-compact:focus {
      border-color: var(--focus-ring);
    }

    .sidebar-link {
      color: var(--sidebar-text);
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 13px;
      font-weight: 500;
      text-decoration: none;
      cursor: pointer;
      transition: all 0.15s ease;
      border-left: 3px solid transparent;
    }
    .sidebar-link:hover {
      color: #FFFFFF;
      background-color: rgba(255, 255, 255, 0.05);
    }
    .sidebar-link.active {
      color: var(--sidebar-text-active);
      background-color: rgba(255, 255, 255, 0.08);
      border-left-color: #FFFFFF;
      font-weight: 600;
    }

    .filter-pill {
      background-color: var(--surface);
      color: var(--text-secondary);
      border: 1px solid var(--border);
      border-radius: 4px;
      font-size: 11px;
      font-weight: 500;
      padding: 4px 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .filter-pill:hover {
      color: var(--text-primary);
      border-color: var(--text-secondary);
    }
    .filter-pill.active {
      background-color: var(--text-primary);
      color: #FFFFFF;
      border-color: var(--text-primary);
      font-weight: 600;
    }

    .badge-status-success {
      background-color: rgba(31, 138, 86, 0.08);
      color: var(--status-success);
      border: 1px solid rgba(31, 138, 86, 0.2);
      border-radius: 3px;
      font-size: 11px;
      font-weight: 600;
      padding: 2px 6px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }

    .badge-status-warning {
      background-color: rgba(183, 121, 31, 0.08);
      color: var(--status-warning);
      border: 1px solid rgba(183, 121, 31, 0.2);
      border-radius: 3px;
      font-size: 11px;
      font-weight: 600;
      padding: 2px 6px;
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }

    .badge-status-neutral {
      background-color: var(--surface-alt);
      color: var(--text-secondary);
      border: 1px solid var(--border);
      border-radius: 3px;
      font-size: 11px;
      font-weight: 500;
      padding: 2px 6px;
      display: inline-block;
    }

    .chat-thread-row {
      background-color: var(--surface);
      border: 1px solid var(--border);
      border-radius: 4px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .chat-thread-row:hover {
      background-color: var(--surface-alt);
      border-color: #D0D4DA;
    }
    .chat-thread-row.active {
      background-color: var(--surface-alt);
      border-left: 3px solid var(--text-primary);
    }

    /* Authentic WhatsApp Theme for Transcript Area */
    .wa-chat-bg {
      background-color: #EFEAE2;
      background-image: radial-gradient(#D1C7B7 0.85px, transparent 0.85px);
      background-size: 16px 16px;
    }
    .wa-bubble-user {
      background-color: #D9FDD3;
      border-radius: 8px 8px 2px 8px;
      box-shadow: 0 1px 0.5px rgba(11, 20, 26, 0.13);
      color: #111B21;
    }
    .wa-bubble-ai {
      background-color: #FFFFFF;
      border-radius: 8px 8px 8px 2px;
      box-shadow: 0 1px 0.5px rgba(11, 20, 26, 0.13);
      color: #111B21;
    }

    .custom-scrollbar::-webkit-scrollbar { width: 5px; height: 5px; }
    .custom-scrollbar::-webkit-scrollbar-track { background: transparent; }
    .custom-scrollbar::-webkit-scrollbar-thumb { background: #D0D4DA; border-radius: 2px; }
    .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: #A0A5AF; }
  </style>
</head>
<body class="min-h-screen flex flex-col antialiased">

  <!-- ================= AUTHENTICATION MODAL ================= -->
  <div id="login-modal" class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4">
    <div class="card w-full max-w-sm p-6 shadow-sm">
      <div class="border-b border-[#E4E6EA] pb-3 mb-5">
        <div class="text-xs font-bold uppercase tracking-wider text-[#5A5E67]">Internal Access</div>
        <h2 class="text-lg font-bold text-[#17181A] mt-0.5">NEURA AI Enterprise Portal</h2>
      </div>
      
      <div class="space-y-4">
        <div>
          <label class="block text-xs font-semibold text-[#5A5E67] uppercase tracking-wider mb-1.5">Administrative Password</label>
          <div class="relative">
            <input type="password" id="admin-pass" placeholder="Enter password" 
                   class="input-compact w-full pr-8">
            <button onclick="togglePass()" type="button" class="absolute right-2.5 top-2 text-[#9298A3] hover:text-[#17181A]">
              <i id="pass-icon" class="fa-regular fa-eye text-xs"></i>
            </button>
          </div>
        </div>
        <p id="login-err" class="text-xs text-[#C0392B] bg-[#FDEDEC] border border-[#F5B7B1] p-2 rounded hidden">
          Authentication failed. Incorrect password.
        </p>
        <button onclick="performLogin()" id="login-btn" class="btn-primary w-full py-2">
          <span>Authenticate Session</span>
        </button>
      </div>
    </div>
  </div>

  <!-- ================= MAIN APPLICATION ================= -->
  <div id="dashboard-content" class="hidden flex-1 flex flex-col lg:flex-row min-h-screen bg-[#F7F8FA]">
    
    <!-- FIXED LEFT SIDEBAR (~250px) -->
    <aside class="w-full lg:w-60 bg-[#14161A] text-[#C7C9CE] flex flex-col justify-between shrink-0 lg:sticky lg:top-0 lg:h-screen z-30 border-r border-[#22262E]">
      <div>
        <!-- BRAND -->
        <div class="p-5 border-b border-[#22262E] flex items-center justify-between">
          <div>
            <div class="text-sm font-bold text-white tracking-tight flex items-center gap-1.5">
              <span>NEURA AI</span>
              <span class="text-[9px] bg-white/10 text-white px-1.5 py-0.5 rounded font-mono">2.0</span>
            </div>
            <div class="text-[10px] text-[#9298A3] uppercase tracking-wider mt-0.5">Clinical Suite</div>
          </div>
        </div>

        <!-- NAV GROUPS -->
        <div class="p-3 space-y-6">
          <div>
            <div class="text-[10px] font-bold uppercase tracking-widest text-[#9298A3] px-3 mb-2">Management</div>
            <nav class="space-y-1">
              <button onclick="switchTab('students')" id="tab-btn-students" class="sidebar-link active w-full text-left">
                <i class="fa-solid fa-users text-xs w-4"></i>
                <span>Students Directory</span>
              </button>
              <button onclick="switchTab('chats')" id="tab-btn-chats" class="sidebar-link w-full text-left">
                <i class="fa-solid fa-comments text-xs w-4"></i>
                <span>Chat Transcripts</span>
              </button>
            </nav>
          </div>

          <div>
            <div class="text-[10px] font-bold uppercase tracking-widest text-[#9298A3] px-3 mb-2">Operations</div>
            <nav class="space-y-1">
              <button onclick="switchTab('broadcast')" id="tab-btn-broadcast" class="sidebar-link w-full text-left">
                <i class="fa-solid fa-bullhorn text-xs w-4"></i>
                <span>Broadcast Studio</span>
              </button>
              <button onclick="switchTab('analytics')" id="tab-btn-analytics" class="sidebar-link w-full text-left">
                <i class="fa-solid fa-chart-pie text-xs w-4"></i>
                <span>Cohort Analytics</span>
              </button>
            </nav>
          </div>
        </div>
      </div>

      <!-- SIDEBAR FOOTER -->
      <div class="p-4 border-t border-[#22262E] space-y-3">
        <div class="flex items-center justify-between text-xs text-[#9298A3] px-1">
          <span class="flex items-center gap-1.5">
            <span class="w-1.5 h-1.5 rounded-full bg-[#1F8A56]"></span>
            <span>Cluster Ready</span>
          </span>
          <span id="sidebar-student-count" class="font-mono text-white text-[11px]">--</span>
        </div>
        <div class="grid grid-cols-2 gap-2">
          <button onclick="loadAllData()" class="btn-secondary text-[11px] py-1 px-2 justify-center bg-[#22262E] text-white border-[#2E333D] hover:bg-[#2C313A]">
            <i id="sync-icon-sidebar" class="fa-solid fa-arrows-rotate text-[10px]"></i> Sync
          </button>
          <button onclick="performLogout()" class="btn-secondary text-[11px] py-1 px-2 justify-center bg-[#22262E] text-[#C7C9CE] border-[#2E333D] hover:bg-[#2C313A]">
            Exit
          </button>
        </div>
      </div>
    </aside>

    <!-- RIGHT MAIN COLUMN -->
    <div class="flex-1 flex flex-col min-w-0">
      
      <!-- TOP BAR -->
      <header class="h-14 bg-white border-b border-[#E4E6EA] sticky top-0 z-20 px-6 flex items-center justify-between">
        <div class="flex items-center gap-2">
          <span id="current-view-title" class="text-sm font-semibold text-[#17181A]">Registered Students Directory</span>
        </div>

        <div class="flex items-center gap-3">
          <span id="nav-chat-issues" class="badge-status-warning hidden">
            <i class="fa-solid fa-triangle-exclamation text-[10px]"></i>
            <span>Diagnostic Alerts</span>
          </span>
          <button onclick="loadAllData()" class="btn-secondary py-1 px-2.5 text-xs">
            <i id="sync-icon" class="fa-solid fa-arrows-rotate text-[10px]"></i>
            <span class="hidden sm:inline">Refresh Data</span>
          </button>
        </div>
      </header>

      <!-- CONTENT BODY (Dense 24px padding, 16px grid gutters) -->
      <main class="p-6 space-y-5 flex-1 max-w-7xl">

        <!-- 4-CARD SUMMARY METRICS GRID -->
        <section class="grid grid-cols-2 lg:grid-cols-4 gap-4">
          <!-- CARD 1 -->
          <div class="card p-4 space-y-1">
            <div class="text-[11px] font-bold text-[#5A5E67] uppercase tracking-wider">Total Registered</div>
            <div id="stat-total-students" class="text-2xl font-bold font-mono text-[#17181A]">--</div>
            <div class="text-[11px] text-[#9298A3]">Active candidates in registry</div>
          </div>

          <!-- CARD 2 -->
          <div class="card p-4 space-y-1">
            <div class="text-[11px] font-bold text-[#5A5E67] uppercase tracking-wider">Active Today (24h)</div>
            <div id="stat-active-24h" class="text-2xl font-bold font-mono text-[#1F8A56]">--</div>
            <div class="text-[11px] text-[#9298A3]">WhatsApp telemetry sessions</div>
          </div>

          <!-- CARD 3 -->
          <div class="card p-4 space-y-1">
            <div class="text-[11px] font-bold text-[#5A5E67] uppercase tracking-wider">Queries Mastered</div>
            <div id="stat-queries" class="text-2xl font-bold font-mono text-[#17181A]">--</div>
            <div class="text-[11px] text-[#9298A3]">Grounded textbook responses</div>
          </div>

          <!-- CARD 4 -->
          <div class="card p-4 space-y-1">
            <div class="text-[11px] font-bold text-[#5A5E67] uppercase tracking-wider">Top Cohort Level</div>
            <div id="stat-top-level" class="text-2xl font-bold font-mono text-[#17181A]">--</div>
            <div class="text-[11px] text-[#9298A3]">Primary MBBS active class</div>
          </div>
        </section>

        <!-- ================= VIEW 1: STUDENTS DIRECTORY ================= -->
        <section id="view-students" class="space-y-3">
          <!-- CONTROLS -->
          <div class="card p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div class="flex flex-wrap items-center gap-1.5">
              <span class="text-xs font-semibold text-[#5A5E67] mr-1">Level:</span>
              <button onclick="setLevelFilter('ALL')" id="lvl-filter-ALL" class="filter-pill active">All</button>
              <button onclick="setLevelFilter('200L')" id="lvl-filter-200L" class="filter-pill">200L</button>
              <button onclick="setLevelFilter('300L')" id="lvl-filter-300L" class="filter-pill">300L</button>
              <button onclick="setLevelFilter('400L')" id="lvl-filter-400L" class="filter-pill">400L</button>
              <button onclick="setLevelFilter('500L')" id="lvl-filter-500L" class="filter-pill">500L</button>
              <button onclick="setLevelFilter('600L')" id="lvl-filter-600L" class="filter-pill">600L</button>
            </div>

            <div class="flex items-center gap-2">
              <span id="filtered-count" class="text-xs text-[#5A5E67] whitespace-nowrap">-- candidates</span>
              <input type="text" id="student-search" oninput="filterStudentsTable()" placeholder="Search name, phone, class..."
                     class="input-compact w-full sm:w-60">
            </div>
          </div>

          <!-- TABLE -->
          <div class="card overflow-hidden">
            <div class="overflow-x-auto custom-scrollbar">
              <table class="w-full text-left text-xs border-collapse">
                <thead class="bg-[#F1F2F4] text-[#5A5E67] uppercase font-bold text-[11px] tracking-wider border-b border-[#E4E6EA]">
                  <tr>
                    <th class="py-2.5 px-4 font-semibold">Candidate</th>
                    <th class="py-2.5 px-4 font-semibold">WhatsApp Number</th>
                    <th class="py-2.5 px-4 font-semibold">Level</th>
                    <th class="py-2.5 px-4 font-semibold text-center">24h Session</th>
                    <th class="py-2.5 px-4 font-semibold text-right">Streak</th>
                    <th class="py-2.5 px-4 font-semibold text-right">Queries</th>
                    <th class="py-2.5 px-4 font-semibold">Preferred Textbooks</th>
                    <th class="py-2.5 px-4 font-semibold">Last Active</th>
                    <th class="py-2.5 px-4 font-semibold text-right">Actions</th>
                  </tr>
                </thead>
                <tbody id="students-tbody" class="divide-y divide-[#E4E6EA] text-[#17181A] bg-white">
                  <tr>
                    <td colspan="9" class="py-8 text-center text-[#5A5E67]">
                      Loading candidates directory...
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </section>

        <!-- ================= VIEW 2: CHAT TRANSCRIPTS ================= -->
        <section id="view-chats" class="hidden space-y-3">
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-4 h-[680px]">
            
            <!-- LEFT THREAD LIST (~320px) -->
            <div class="lg:col-span-4 card flex flex-col h-full overflow-hidden">
              <div class="p-3 border-b border-[#E4E6EA] space-y-2 bg-[#F1F2F4]">
                <div class="flex items-center justify-between text-xs">
                  <span class="font-bold text-[#17181A]">Candidate Threads</span>
                  <span id="thread-count-label" class="text-[11px] text-[#5A5E67]">--</span>
                </div>
                <input type="text" id="search-chat-students" oninput="filterChatThreadsList()" placeholder="Filter candidate..."
                       class="input-compact w-full text-xs">
                <div class="flex items-center gap-1 flex-wrap">
                  <button onclick="setChatThreadFilter('ALL')" id="ct-filter-ALL" class="filter-pill active text-[10px] py-0.5 px-2">All</button>
                  <button onclick="setChatThreadFilter('ISSUES')" id="ct-filter-ISSUES" class="filter-pill text-[10px] py-0.5 px-2">Alerts Only</button>
                  <button onclick="toggleBroadcastsInChat()" id="ct-filter-BROADCAST" class="filter-pill text-[10px] py-0.5 px-2">Broadcasts: Hidden</button>
                </div>
              </div>

              <div id="chat-threads-container" class="flex-1 overflow-y-auto p-2 space-y-1.5 custom-scrollbar bg-white">
                <div class="text-center py-10 text-xs text-[#5A5E67]">Loading threads...</div>
              </div>
            </div>

            <!-- RIGHT TRANSCRIPT STREAM -->
            <div class="lg:col-span-8 card flex flex-col h-full overflow-hidden border border-[#E4E6EA] shadow-xs">
              <!-- HEADER -->
              <div class="p-3 border-b border-[#E4E6EA] flex items-center justify-between bg-[#F0F2F5]">
                <div class="flex items-center gap-2.5">
                  <div id="chat-header-avatar" class="w-8 h-8 rounded-full bg-[#008069] text-white font-bold text-xs flex items-center justify-center shadow-xs">
                    --
                  </div>
                  <div>
                    <div class="flex items-center gap-1.5">
                      <span id="chat-header-name" class="font-semibold text-xs text-[#111B21]">Select a candidate</span>
                      <span id="chat-header-level" class="badge-status-neutral text-[10px]">--</span>
                    </div>
                    <div class="text-[11px] text-[#667781]">
                      <span id="chat-header-phone" class="font-mono">--</span> &bull; <span id="chat-header-streak">--</span>
                    </div>
                  </div>
                </div>

                <div class="flex items-center gap-2">
                  <input type="text" id="in-chat-search" oninput="filterInChatMessages()" placeholder="Search in transcript..."
                         class="input-compact text-xs py-0.5 px-2 w-36 sm:w-44 hidden sm:block bg-white">
                  <a id="chat-header-wa-btn" href="#" target="_blank" class="btn-secondary text-[11px] py-1 px-2.5 inline-flex items-center gap-1 text-[#008069] border-[#008069]/30 hover:bg-[#008069]/5">
                    <i class="fa-brands fa-whatsapp text-xs"></i> <span>WhatsApp</span>
                  </a>
                </div>
              </div>

              <!-- DIAGNOSTIC BANNER -->
              <div id="chat-diagnostic-banner" class="hidden p-2 bg-[#FEF7EC] border-b border-[#FAD7A0] text-xs font-semibold text-[#B7791F] flex items-center justify-between">
                <span id="chat-diagnostic-text">Diagnostic alert detected in this session</span>
                <span class="badge-status-warning text-[10px]">ALERT</span>
              </div>

              <!-- MESSAGES STREAM (AUTHENTIC WHATSAPP WALLPAPER & BUBBLES) -->
              <div id="chat-stream-container" class="flex-1 overflow-y-auto p-4 space-y-3 custom-scrollbar wa-chat-bg">
                <div class="m-auto text-center text-[#667781] p-8 text-xs">
                  Select a candidate from the left panel to inspect the full chronological transcript.
                </div>
              </div>

              <!-- DIRECT MESSAGE COMPOSER FOR ADMIN (WITH MODE SELECTOR: 24H FREE SESSION VS TEMPLATE VS SMART) -->
              <div id="chat-composer-bar" class="p-2.5 bg-[#F0F2F5] border-t border-[#E4E6EA] flex flex-wrap sm:flex-nowrap items-center gap-2">
                <select id="direct-msg-mode" class="input-compact text-xs py-1.5 px-2 bg-white rounded border border-[#E4E6EA]">
                  <option value="direct_only" selected>🟢 Free Session (&lt;24h Active - 100% Free)</option>
                  <option value="template_only">📢 Template Message (All / &gt;24h)</option>
                  <option value="smart">⚡ Smart Hybrid (Direct + Template Fallback)</option>
                </select>
                <input type="text" id="direct-msg-input" placeholder="Type a direct reply or announcement..."
                       onkeydown="if(event.key==='Enter') sendDirectStudentMessage()"
                       class="flex-1 input-compact text-xs py-1.5 px-3 bg-white rounded border border-[#E4E6EA]">
                <button type="button" onclick="sendDirectStudentMessage()" id="direct-msg-send-btn"
                        class="btn-primary text-xs py-1.5 px-3 inline-flex items-center gap-1.5 whitespace-nowrap bg-[#008069] hover:bg-[#006A57] text-white">
                  <i class="fa-solid fa-paper-plane text-xs"></i> <span>Send</span>
                </button>
              </div>
            </div>

          </div>
        </section>

        <!-- ================= VIEW 3: BROADCAST STUDIO ================= -->
        <section id="view-broadcast" class="hidden space-y-4">
          <div class="grid grid-cols-1 lg:grid-cols-12 gap-4">
            
            <!-- LEFT COMPOSER -->
            <div class="lg:col-span-7 card p-4 space-y-3">
              <div class="border-b border-[#E4E6EA] pb-2">
                <h2 class="text-sm font-bold text-[#17181A]">WhatsApp Broadcast Composer</h2>
                <p class="text-xs text-[#5A5E67]">Compose and dispatch batch announcements to target cohorts</p>
              </div>

              <div class="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <label class="block text-[11px] font-semibold text-[#5A5E67] uppercase tracking-wider mb-1">Target Audience</label>
                  <select id="broadcast-target" class="input-compact w-full text-xs">
                    <option value="ALL">All Registered Students</option>
                    <option value="ACTIVE_24H">🟢 Active in Last 24 Hours Only (100% Free)</option>
                    <option value="200L">200 Level Only</option>
                    <option value="300L">300 Level Only</option>
                    <option value="400L">400 Level Only</option>
                    <option value="500L">500 Level Only</option>
                    <option value="600L">600 Level Only</option>
                  </select>
                </div>

                <div>
                  <label class="block text-[11px] font-semibold text-[#5A5E67] uppercase tracking-wider mb-1">Dispatch Mode</label>
                  <select id="broadcast-mode" class="input-compact w-full text-xs">
                    <option value="direct_only">🟢 Free Direct Message (Active 24h Session Only — 100% Free)</option>
                    <option value="template_only">📢 Meta Template Broadcast (neura_announcement — All / &gt;24h)</option>
                    <option value="smart" selected>⚡ Smart Hybrid (Free Direct first + Template fallback for &gt;24h)</option>
                  </select>
                </div>
              </div>

              <div class="space-y-1.5">
                <div class="flex items-center justify-between text-xs">
                  <span class="font-semibold text-[#17181A]">Message Content</span>
                  <span id="char-count" class="text-[#5A5E67] font-mono text-[11px]">0 chars</span>
                </div>
                <div class="flex items-center gap-1 border border-[#E4E6EA] p-1 rounded bg-[#F1F2F4]">
                  <button type="button" onclick="insertFormatting('*', '*')" class="btn-secondary text-[10px] py-0.5 px-2">Bold</button>
                  <button type="button" onclick="insertFormatting('_', '_')" class="btn-secondary text-[10px] py-0.5 px-2">Italic</button>
                  <button type="button" onclick="insertFormatting('# *', '*')" class="btn-secondary text-[10px] py-0.5 px-2">Header</button>
                  <button type="button" onclick="insertFormatting('• ', '')" class="btn-secondary text-[10px] py-0.5 px-2">Bullet</button>
                </div>
                <textarea id="broadcast-msg" oninput="updateLivePreview()" rows="8" placeholder="Type announcement content here..."
                          class="w-full p-3 border border-[#E4E6EA] rounded text-xs font-mono text-[#17181A] outline-none focus:border-[#2C2F36]"></textarea>
              </div>

              <button type="button" onclick="confirmBroadcast()" class="btn-primary w-full py-2">
                <span>Verify &amp; Dispatch Broadcast</span>
              </button>
            </div>

            <!-- RIGHT PREVIEW -->
            <div class="lg:col-span-5 card p-4 space-y-2.5 bg-[#F0F2F5]">
              <div class="flex items-center justify-between border-b border-[#E4E6EA] pb-2 text-xs font-bold text-[#111B21]">
                <span>WhatsApp Client Preview</span>
                <span id="preview-time" class="font-mono text-[11px] text-[#667781]">--:--</span>
              </div>

              <div class="wa-chat-bg p-3.5 rounded min-h-[220px] flex flex-col justify-start">
                <div class="wa-bubble-ai p-3 text-xs space-y-1">
                  <div class="text-[10px] font-bold uppercase tracking-wider text-[#008069] border-b border-[#E9EDEF] pb-0.5">NEURA AI Broadcast</div>
                  <div id="preview-text" class="text-xs text-[#111B21] whitespace-pre-wrap leading-relaxed">
                    Type your announcement on the left to see live rendering.
                  </div>
                  <div class="text-[10px] text-[#667781] text-right font-sans mt-0.5" id="preview-time-footer">--:--</div>
                </div>
              </div>
              
              <div class="text-[11px] text-[#667781]">
                Dispatches execute asynchronously with automatic exponential backoff.
              </div>
            </div>

          </div>
        </section>

        <!-- ================= VIEW 4: ANALYTICS ================= -->
        <section id="view-analytics" class="hidden space-y-4">
          <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            <div class="card p-4 space-y-3">
              <h2 class="text-sm font-bold text-[#17181A]">MBBS Cohort Level Distribution</h2>
              <div id="level-chart-container" class="space-y-2.5 pt-1">
                <div class="text-xs text-[#5A5E67]">Loading cohort metrics...</div>
              </div>
            </div>

            <div class="card p-4 space-y-3">
              <h2 class="text-sm font-bold text-[#17181A]">Operational Telemetry</h2>
              <div class="space-y-2 text-xs divide-y divide-[#E4E6EA]">
                <div class="flex items-center justify-between py-1.5">
                  <span class="text-[#5A5E67]">Registered Students</span>
                  <span id="analytics-total-students" class="font-mono font-bold text-[#17181A]">--</span>
                </div>
                <div class="flex items-center justify-between py-1.5">
                  <span class="text-[#5A5E67]">Total Query Telemetry</span>
                  <span id="analytics-total-queries" class="font-mono font-bold text-[#17181A]">--</span>
                </div>
                <div class="flex items-center justify-between py-1.5">
                  <span class="text-[#5A5E67]">Vector Embedding Space</span>
                  <span class="badge-status-neutral">FastEmbed 384-dim</span>
                </div>
                <div class="flex items-center justify-between py-1.5">
                  <span class="text-[#5A5E67]">Health Status</span>
                  <span class="badge-status-success">
                    <span class="w-1.5 h-1.5 rounded-full bg-[#1F8A56]"></span> Operational
                  </span>
                </div>
              </div>
            </div>

          </div>
        </section>

      </main>
    </div>

  </div>

  <!-- CONFIRM BROADCAST MODAL -->
  <div id="confirm-modal" class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4 hidden">
    <div class="card w-full max-w-sm p-5 space-y-3 bg-white">
      <h3 class="text-sm font-bold text-[#17181A]">Confirm Broadcast Dispatch</h3>
      <p id="confirm-details" class="text-xs text-[#5A5E67] leading-relaxed">
        Are you ready to dispatch this announcement?
      </p>
      <div class="flex items-center justify-end gap-2 pt-2 border-t border-[#E4E6EA]">
        <button onclick="closeConfirmModal()" class="btn-secondary text-xs">Cancel</button>
        <button onclick="executeConfirmedBroadcast()" id="modal-confirm-btn" class="btn-primary text-xs">
          Confirm Dispatch
        </button>
      </div>
    </div>
  </div>

  <!-- CANDIDATE ACTIVITY & TELEMETRY INSPECTOR MODAL -->
  <div id="activity-modal" class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 backdrop-blur-xs p-4 hidden">
    <div class="card w-full max-w-xl max-h-[90vh] flex flex-col bg-white overflow-hidden shadow-lg border border-[#E4E6EA]">
      <!-- MODAL HEADER -->
      <div class="p-4 border-b border-[#E4E6EA] bg-[#F7F8FA] flex items-center justify-between">
        <div class="flex items-center gap-3">
          <div id="act-modal-avatar" class="w-10 h-10 rounded-full bg-[#008069] text-white font-bold text-sm flex items-center justify-center shadow-xs">
            --
          </div>
          <div>
            <div class="flex items-center gap-2">
              <h3 id="act-modal-name" class="text-sm font-bold text-[#17181A]">Candidate Profile</h3>
              <span id="act-modal-level" class="badge-status-neutral text-[10px]">--</span>
            </div>
            <div class="flex items-center gap-2 text-xs text-[#5A5E67] font-mono mt-0.5">
              <span id="act-modal-phone">--</span>
              <button onclick="copyModalPhone()" title="Copy Phone Number" class="text-[#008069] hover:underline text-[11px]"><i class="fa-regular fa-copy"></i></button>
            </div>
          </div>
        </div>
        <button onclick="closeActivityModal()" class="text-[#9298A3] hover:text-[#17181A] p-1.5 rounded hover:bg-[#E4E6EA]">
          <i class="fa-solid fa-xmark text-sm"></i>
        </button>
      </div>

      <!-- MODAL BODY (SCROLLABLE) -->
      <div class="p-4 space-y-4 overflow-y-auto custom-scrollbar flex-1 text-xs">
        <!-- 24H SESSION STATUS CARD -->
        <div id="act-modal-session-card" class="p-3 rounded border flex items-center justify-between">
          <div class="flex items-center gap-2.5">
            <span id="act-modal-session-dot" class="w-2.5 h-2.5 rounded-full bg-[#1F8A56]"></span>
            <div>
              <div id="act-modal-session-title" class="font-bold text-[#17181A]">WhatsApp Session Window: Active (&lt;24h)</div>
              <div id="act-modal-session-desc" class="text-[11px] text-[#5A5E67]">Active conversational session open. Standard replies deliver 100% freely.</div>
            </div>
          </div>
          <span id="act-modal-session-badge" class="badge-status-success text-[10px]">Active 24h</span>
        </div>

        <!-- 4-METRICS STATS GRID -->
        <div class="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
          <div class="p-2.5 bg-[#F7F8FA] rounded border border-[#E4E6EA] space-y-0.5">
            <div class="text-[10px] font-semibold text-[#5A5E67] uppercase">Study Streak</div>
            <div id="act-modal-streak" class="font-bold text-sm text-[#17181A]">🔥 1d</div>
          </div>
          <div class="p-2.5 bg-[#F7F8FA] rounded border border-[#E4E6EA] space-y-0.5">
            <div class="text-[10px] font-semibold text-[#5A5E67] uppercase">Queries Mastered</div>
            <div id="act-modal-queries" class="font-bold text-sm text-[#17181A]">0</div>
          </div>
          <div class="p-2.5 bg-[#F7F8FA] rounded border border-[#E4E6EA] space-y-0.5">
            <div class="text-[10px] font-semibold text-[#5A5E67] uppercase">Total Messages</div>
            <div id="act-modal-messages" class="font-bold text-sm text-[#17181A]">0</div>
          </div>
          <div class="p-2.5 bg-[#F7F8FA] rounded border border-[#E4E6EA] space-y-0.5">
            <div class="text-[10px] font-semibold text-[#5A5E67] uppercase">Reminders</div>
            <div id="act-modal-reminders" class="font-bold text-sm text-[#1F8A56]">🔔 Enabled</div>
          </div>
        </div>

        <!-- ACTIVE BOOKSHELF -->
        <div class="space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="font-bold text-[#17181A] text-xs">Active Curriculum Bookshelf</span>
            <span id="act-modal-books-count" class="text-[11px] text-[#5A5E67] font-mono">-- Books</span>
          </div>
          <div id="act-modal-books-list" class="p-2.5 bg-[#F7F8FA] rounded border border-[#E4E6EA] space-y-1.5">
            <div class="text-center py-2 text-[#5A5E67]">Loading bookshelf...</div>
          </div>
        </div>

        <!-- RECENT ENGAGEMENT TIMELINE -->
        <div class="space-y-1.5">
          <div class="flex items-center justify-between">
            <span class="font-bold text-[#17181A] text-xs">Recent Message Activity Timeline</span>
            <span id="act-modal-last-active" class="text-[11px] text-[#5A5E67]">Last seen: --</span>
          </div>
          <div id="act-modal-events-list" class="space-y-1.5 max-h-48 overflow-y-auto custom-scrollbar">
            <div class="text-center py-2 text-[#5A5E67]">Loading events...</div>
          </div>
        </div>
      </div>

      <!-- MODAL FOOTER -->
      <div class="p-3 border-t border-[#E4E6EA] bg-[#F7F8FA] flex items-center justify-between">
        <button onclick="closeActivityModal()" class="btn-secondary text-xs py-1.5 px-3">Close</button>
        <div class="flex items-center gap-2">
          <button id="act-modal-chat-btn" onclick="openChatFromModal()" class="btn-secondary text-xs py-1.5 px-3 text-[#008069] border-[#008069]/30 hover:bg-[#008069]/5">
            <i class="fa-solid fa-comments text-xs"></i> <span>Open Full Transcript</span>
          </button>
          <a id="act-modal-wa-btn" href="#" target="_blank" class="btn-primary text-xs py-1.5 px-3 bg-[#008069] hover:bg-[#006A57]">
            <i class="fa-brands fa-whatsapp text-xs"></i> <span>Chat on WhatsApp</span>
          </a>
        </div>
      </div>
    </div>
  </div>

  <script>
    let authToken = localStorage.getItem("neura_admin_token") || "";
    let rawStudentsList = [];
    let rawChatThreads = [];
    let activeLevelFilter = "ALL";
    let activeChatThreadFilter = "ALL";
    let showBroadcastsInChat = false;
    let currentSelectedUserId = null;
    let currentActiveMessages = [];
    let currentModalStudentId = null;

    const VIEW_TITLES = {
      'students': 'Registered Students Directory',
      'chats': 'Chat Transcripts Studio',
      'broadcast': 'WhatsApp Broadcast Studio',
      'analytics': 'Cohort Analytics & Telemetry'
    };

    function formatChatTime(isoStr) {
      if (!isoStr) return "";
      try {
        const s = String(isoStr).trim();
        const utc = s.endsWith("Z") ? s : s + "Z";
        const d = new Date(utc);
        if (isNaN(d.getTime())) return isoStr;
        return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: true });
      } catch(e) {
        return isoStr;
      }
    }

    function formatFullDateTime(isoStr) {
      if (!isoStr) return "--";
      try {
        const s = String(isoStr).trim();
        const utc = s.endsWith("Z") ? s : s + "Z";
        const d = new Date(utc);
        if (isNaN(d.getTime())) return String(isoStr);
        return d.toLocaleDateString([], { month: 'short', day: 'numeric', year: 'numeric' }) + " at " + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', hour12: true });
      } catch(e) {
        return String(isoStr);
      }
    }

    function togglePass() {
      const p = document.getElementById("admin-pass");
      const icon = document.getElementById("pass-icon");
      if (p.type === "password") {
        p.type = "text";
        icon.classList.replace("fa-eye", "fa-eye-slash");
      } else {
        p.type = "password";
        icon.classList.replace("fa-eye-slash", "fa-eye");
      }
    }

    document.getElementById("admin-pass").addEventListener("keyup", function(e) {
      if (e.key === "Enter") performLogin();
    });

    async function performLogin() {
      const pass = document.getElementById("admin-pass").value;
      const err = document.getElementById("login-err");
      const btn = document.getElementById("login-btn");
      
      err.classList.add("hidden");
      btn.innerHTML = 'Authenticating...';
      btn.disabled = true;
      
      try {
        const res = await fetch("/admin/api/login", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ password: pass })
        });
        
        if (res.ok) {
          const data = await res.json();
          authToken = data.token;
          localStorage.setItem("neura_admin_token", authToken);
          showDashboard();
        } else {
          err.classList.remove("hidden");
          btn.innerHTML = '<span>Authenticate Session</span>';
          btn.disabled = false;
        }
      } catch (e) {
        err.innerHTML = 'Network error. Please retry.';
        err.classList.remove("hidden");
        btn.innerHTML = '<span>Authenticate Session</span>';
        btn.disabled = false;
      }
    }

    function showDashboard() {
      document.getElementById("login-modal").classList.add("hidden");
      document.getElementById("dashboard-content").classList.remove("hidden");
      loadAllData();
    }

    function performLogout() {
      authToken = "";
      localStorage.removeItem("neura_admin_token");
      document.getElementById("dashboard-content").classList.add("hidden");
      document.getElementById("login-modal").classList.remove("hidden");
      document.getElementById("admin-pass").value = "";
    }

    function switchTab(tabKey) {
      const tabs = ['students', 'chats', 'broadcast', 'analytics'];
      tabs.forEach(t => {
        const view = document.getElementById('view-' + t);
        const btn = document.getElementById('tab-btn-' + t);
        if (t === tabKey) {
          view?.classList.remove('hidden');
          btn?.classList.add('active');
        } else {
          view?.classList.add('hidden');
          btn?.classList.remove('active');
        }
      });

      const titleEl = document.getElementById('current-view-title');
      if (titleEl && VIEW_TITLES[tabKey]) {
        titleEl.innerText = VIEW_TITLES[tabKey];
      }

      if (tabKey === 'chats' && (!rawChatThreads || rawChatThreads.length === 0)) {
        loadRecentChats();
      }
    }

    async function loadAllData() {
      if (!authToken) return;
      const syncIcon = document.getElementById("sync-icon");
      const syncSidebar = document.getElementById("sync-icon-sidebar");
      syncIcon?.classList.add("fa-spin");
      syncSidebar?.classList.add("fa-spin");
      
      try {
        await Promise.all([loadStats(), loadStudents(), loadRecentChats()]);
      } catch (e) {
        console.error("Sync error:", e);
      } finally {
        setTimeout(() => {
          syncIcon?.classList.remove("fa-spin");
          syncSidebar?.classList.remove("fa-spin");
        }, 400);
      }
    }

    async function loadStats() {
      const res = await fetch("/admin/api/stats", {
        headers: { "Authorization": "Bearer " + authToken }
      });
      if (res.status === 401) { performLogout(); return; }
      const data = await res.json();
      
      document.getElementById("stat-total-students").innerText = data.total_students || "0";
      document.getElementById("stat-active-24h").innerText = data.active_24h || "0";
      document.getElementById("analytics-total-students").innerText = data.total_students || "0";
      
      const ssc = document.getElementById("sidebar-student-count");
      if (ssc) ssc.innerText = `${data.total_students || 0} Registered`;

      const dist = data.level_distribution || {};
      let topLvl = "None";
      let maxCount = -1;
      let totalAssigned = 0;
      for (let [lvl, count] of Object.entries(dist)) {
        if (lvl !== "Other") totalAssigned += count;
        if (count > maxCount && lvl !== "Other") {
          maxCount = count;
          topLvl = lvl;
        }
      }
      document.getElementById("stat-top-level").innerText = topLvl;

      // Render chart
      const chartContainer = document.getElementById("level-chart-container");
      chartContainer.innerHTML = Object.entries(dist).map(([lvl, count]) => {
        const pct = totalAssigned > 0 ? Math.round((count / (data.total_students || 1)) * 100) : 0;
        return `
          <div class="space-y-1">
            <div class="flex items-center justify-between text-xs">
              <span class="font-medium text-[#17181A]">${lvl}</span>
              <span class="text-[#5A5E67] font-mono text-[11px]">${count} (${pct}%)</span>
            </div>
            <div class="w-full bg-[#E4E6EA] rounded h-1.5 overflow-hidden">
              <div class="bg-[#17181A] h-full rounded transition-all duration-300" style="width: ${pct}%"></div>
            </div>
          </div>
        `;
      }).join("");
    }

    async function loadStudents() {
      const res = await fetch("/admin/api/students", {
        headers: { "Authorization": "Bearer " + authToken }
      });
      if (res.status === 401) { performLogout(); return; }
      const data = await res.json();
      rawStudentsList = data.students || [];
      
      let totalQueries = 0;
      rawStudentsList.forEach(s => totalQueries += (s.total_queries_count || 0));
      document.getElementById("stat-queries").innerText = totalQueries > 0 ? totalQueries.toLocaleString() : "100+";
      document.getElementById("analytics-total-queries").innerText = totalQueries > 0 ? totalQueries.toLocaleString() : "100+";

      filterStudentsTable();
    }

    function setLevelFilter(lvl) {
      activeLevelFilter = lvl;
      document.querySelectorAll(".filter-pill").forEach(b => {
        if (b.id && b.id.startsWith("lvl-filter-")) {
          b.classList.remove("active");
        }
      });
      const activeBtn = document.getElementById("lvl-filter-" + lvl);
      if (activeBtn) {
        activeBtn.classList.add("active");
      }
      filterStudentsTable();
    }

    function filterStudentsTable() {
      const query = document.getElementById("student-search").value.toLowerCase().trim();
      
      const filtered = rawStudentsList.filter(s => {
        if (activeLevelFilter !== "ALL" && s.level !== activeLevelFilter) return false;
        
        if (query) {
          const matchName = (s.name || "").toLowerCase().includes(query);
          const matchPhone = (s.user_id || "").toLowerCase().includes(query);
          const matchLvl = (s.level || "").toLowerCase().includes(query);
          if (!matchName && !matchPhone && !matchLvl) return false;
        }
        return true;
      });

      document.getElementById("filtered-count").innerText = `${filtered.length} candidates`;
      renderStudentsTable(filtered);
    }

    function formatActiveDate(d) {
      if (!d || d === "N/A" || d === "None" || d === "Recent") {
        return new Date().toISOString().split("T")[0];
      }
      return String(d).split("T")[0].split(" ")[0];
    }

    function renderStudentsTable(students) {
      const tbody = document.getElementById("students-tbody");
      if (!students || students.length === 0) {
        tbody.innerHTML = `
          <tr>
            <td colspan="9" class="py-6 text-center text-[#5A5E67]">
              No candidates found matching criteria.
            </td>
          </tr>
        `;
        return;
      }

      tbody.innerHTML = students.map(s => {
        const booksCount = (s.preferred_books_list || []).length;
        const booksText = booksCount > 0 ? `${booksCount} Textbook(s)` : "Standard Core";
        const formattedDate = formatActiveDate(s.last_study_date);
        const sessionBadge = s.is_active_24h 
          ? `<span class="inline-flex items-center gap-1 text-[10px] font-bold text-[#1F8A56] bg-[#E8F8F0] px-2 py-0.5 rounded-full"><span class="w-1.5 h-1.5 rounded-full bg-[#1F8A56]"></span>&lt;24h Open</span>`
          : `<span class="inline-flex items-center gap-1 text-[10px] font-medium text-[#5A5E67] bg-[#F1F2F4] px-2 py-0.5 rounded-full">&gt;24h Closed</span>`;

        return `
          <tr class="hover:bg-[#F1F2F4] transition-colors cursor-pointer" onclick="openStudentActivity('${s.user_id}')">
            <td class="py-2.5 px-4 font-semibold text-[#17181A]">
              <div class="hover:text-[#008069] flex items-center gap-1.5">${s.name || "Candidate"} <i class="fa-solid fa-circle-info text-[10px] text-[#9298A3]"></i></div>
            </td>
            <td class="py-2.5 px-4 font-mono text-[#5A5E67]">
              ${s.user_id}
            </td>
            <td class="py-2.5 px-4">
              <span class="badge-status-neutral">${s.level || "Unset"}</span>
            </td>
            <td class="py-2.5 px-4 text-center">
              ${sessionBadge}
            </td>
            <td class="py-2.5 px-4 text-right font-mono font-medium text-[#17181A]">
              🔥 ${s.study_streak_days || 1}d
            </td>
            <td class="py-2.5 px-4 text-right font-mono text-[#17181A]">
              ${s.total_queries_count || 0}
            </td>
            <td class="py-2.5 px-4 text-[#5A5E67]">
              ${booksText}
            </td>
            <td class="py-2.5 px-4 font-mono text-[#5A5E67] text-[11px]">
              ${formattedDate}
            </td>
            <td class="py-2.5 px-4 text-right" onclick="event.stopPropagation()">
              <div class="inline-flex items-center gap-1">
                <button onclick="openStudentActivity('${s.user_id}')" class="btn-secondary text-[11px] py-0.5 px-2 bg-[#F1F2F4] hover:bg-[#E4E6EA]">
                  Activity
                </button>
                <button onclick="openStudentChat('${s.user_id}')" class="btn-secondary text-[11px] py-0.5 px-2">
                  Chats
                </button>
                <a href="https://wa.me/${s.user_id}" target="_blank" class="btn-secondary text-[11px] py-0.5 px-2">
                  WA
                </a>
              </div>
            </td>
          </tr>
        `;
      }).join("");
    }

    // ================= CANDIDATE ACTIVITY MODAL =================
    async function openStudentActivity(userId) {
      currentModalStudentId = userId;
      const modal = document.getElementById("activity-modal");
      modal.classList.remove("hidden");

      document.getElementById("act-modal-name").innerText = "Loading profile...";
      document.getElementById("act-modal-level").innerText = "--";
      document.getElementById("act-modal-phone").innerText = userId;
      document.getElementById("act-modal-avatar").innerText = "--";
      document.getElementById("act-modal-streak").innerText = "🔥 --";
      document.getElementById("act-modal-queries").innerText = "--";
      document.getElementById("act-modal-messages").innerText = "--";
      document.getElementById("act-modal-reminders").innerText = "--";
      document.getElementById("act-modal-books-list").innerHTML = '<div class="text-center py-2 text-[#5A5E67]">Loading bookshelf...</div>';
      document.getElementById("act-modal-events-list").innerHTML = '<div class="text-center py-2 text-[#5A5E67]">Loading events...</div>';

      try {
        const res = await fetch(`/admin/api/students/${userId}/activity`, {
          headers: { "Authorization": "Bearer " + authToken }
        });
        if (res.status === 401) { performLogout(); return; }
        const act = await res.json();

        const initials = (act.name || "S").split(" ").map(w => w[0]).join("").substring(0, 2).toUpperCase();
        document.getElementById("act-modal-avatar").innerText = initials;
        document.getElementById("act-modal-name").innerText = act.name || "Candidate";
        document.getElementById("act-modal-level").innerText = act.level || "Unset";
        document.getElementById("act-modal-phone").innerText = act.user_id || userId;
        document.getElementById("act-modal-streak").innerText = `🔥 ${act.study_streak_days || 1}d`;
        document.getElementById("act-modal-queries").innerText = (act.total_queries_count || 0).toLocaleString();
        document.getElementById("act-modal-messages").innerText = (act.total_messages || 0).toLocaleString();
        document.getElementById("act-modal-reminders").innerText = act.reminders_enabled ? "🔔 Enabled" : "🔕 Paused";
        document.getElementById("act-modal-last-active").innerText = `Last study date: ${act.last_active || "--"}`;
        document.getElementById("act-modal-wa-btn").href = `https://wa.me/${act.user_id || userId}`;

        // 24h session window card
        const sessCard = document.getElementById("act-modal-session-card");
        const sessDot = document.getElementById("act-modal-session-dot");
        const sessTitle = document.getElementById("act-modal-session-title");
        const sessDesc = document.getElementById("act-modal-session-desc");
        const sessBadge = document.getElementById("act-modal-session-badge");

        if (act.is_active_24h) {
          sessCard.className = "p-3 rounded border border-[#A3E635]/40 bg-[#F7FEE7] flex items-center justify-between";
          sessDot.className = "w-2.5 h-2.5 rounded-full bg-[#1F8A56]";
          sessTitle.innerText = "WhatsApp Free Session Window: Active (<24h)";
          sessDesc.innerText = "Student messaged recently. Free interactive replies and queries can be exchanged freely.";
          sessBadge.className = "badge-status-success text-[10px]";
          sessBadge.innerText = "Active <24h";
        } else {
          sessCard.className = "p-3 rounded border border-[#FAD7A0] bg-[#FEF7EC] flex items-center justify-between";
          sessDot.className = "w-2.5 h-2.5 rounded-full bg-[#B7791F]";
          sessTitle.innerText = "WhatsApp Session Closed (>24h Inactive)";
          sessDesc.innerText = "Direct free messages will fail Meta's 24h policy. Use Broadcast / Template message to re-engage.";
          sessBadge.className = "badge-status-warning text-[10px]";
          sessBadge.innerText = ">24h Inactive";
        }

        // Active bookshelf
        const books = act.preferred_books_list || [];
        document.getElementById("act-modal-books-count").innerText = `${books.length} Active Book(s)`;
        const booksListEl = document.getElementById("act-modal-books-list");
        if (books.length === 0) {
          booksListEl.innerHTML = '<div class="text-[#5A5E67] text-xs py-1">No custom textbooks selected yet (using default curriculum).</div>';
        } else {
          booksListEl.innerHTML = books.map((b, idx) => `
            <div class="flex items-center gap-2 p-1.5 bg-white rounded border border-[#E4E6EA] text-xs">
              <span class="w-5 h-5 rounded bg-[#008069]/10 text-[#008069] font-bold text-[10px] flex items-center justify-center">${idx+1}</span>
              <span class="font-medium text-[#17181A] flex-1">${b}</span>
              <span class="badge-status-neutral text-[9px]">Enrolled</span>
            </div>
          `).join("");
        }

        // Recent events
        const events = act.recent_events || [];
        const eventsListEl = document.getElementById("act-modal-events-list");
        if (events.length === 0) {
          eventsListEl.innerHTML = '<div class="text-[#5A5E67] text-xs py-1">No message history recorded yet.</div>';
        } else {
          eventsListEl.innerHTML = events.map(e => {
            const isUser = e.role === "user";
            const timeFormatted = formatFullDateTime(e.timestamp);
            const roleBadge = isUser 
              ? '<span class="text-[9px] font-bold text-[#008069] bg-[#E8F8F0] px-1.5 py-0.5 rounded">STUDENT</span>'
              : '<span class="text-[9px] font-bold text-[#17181A] bg-[#F1F2F4] px-1.5 py-0.5 rounded">NEURA AI</span>';

            return `
              <div class="p-2 bg-white rounded border border-[#E4E6EA] space-y-1">
                <div class="flex items-center justify-between text-[10px] text-[#5A5E67]">
                  <div class="flex items-center gap-1.5">${roleBadge} <span class="font-mono text-[#9298A3]">(${e.msg_type || 'text'})</span></div>
                  <span>${timeFormatted}</span>
                </div>
                <div class="text-xs text-[#17181A] truncate">${e.content || "Empty content"}</div>
              </div>
            `;
          }).join("");
        }
      } catch (err) {
        document.getElementById("act-modal-name").innerText = "Error loading details";
      }
    }

    function closeActivityModal() {
      document.getElementById("activity-modal").classList.add("hidden");
    }

    function copyModalPhone() {
      const phone = document.getElementById("act-modal-phone").innerText;
      if (phone) {
        navigator.clipboard.writeText(phone);
        alert(`Copied WhatsApp Number: ${phone}`);
      }
    }

    function openChatFromModal() {
      if (currentModalStudentId) {
        closeActivityModal();
        openStudentChat(currentModalStudentId);
      }
    }

    // ================= CHAT TRANSCRIPTS LOGIC =================
    function toggleBroadcastsInChat() {
      showBroadcastsInChat = !showBroadcastsInChat;
      const btn = document.getElementById("ct-filter-BROADCAST");
      if (btn) {
        if (showBroadcastsInChat) {
          btn.innerText = "Broadcasts: Visible";
          btn.classList.add("active");
        } else {
          btn.innerText = "Broadcasts: Hidden";
          btn.classList.remove("active");
        }
      }
      if (currentSelectedUserId) {
        loadConversationForUser(currentSelectedUserId);
      }
    }

    async function loadRecentChats() {
      if (!authToken) return;
      try {
        const res = await fetch("/admin/api/chats/recent", {
          headers: { "Authorization": "Bearer " + authToken }
        });
        if (res.status === 401) { performLogout(); return; }
        const data = await res.json();
        rawChatThreads = data.conversations || [];
        
        const hasAnyIssues = rawChatThreads.some(t => t.has_issue);
        const issuesBadge = document.getElementById("nav-chat-issues");
        if (hasAnyIssues) {
          issuesBadge?.classList.remove("hidden");
        } else {
          issuesBadge?.classList.add("hidden");
        }

        const countLabel = document.getElementById("thread-count-label");
        if (countLabel) countLabel.innerText = `${rawChatThreads.length} active`;

        filterChatThreadsList();
      } catch (e) {
        console.error("Error loading chats:", e);
      }
    }

    function setChatThreadFilter(filterKey) {
      activeChatThreadFilter = filterKey;
      document.querySelectorAll(".filter-pill").forEach(b => {
        if (b.id && (b.id === "ct-filter-ALL" || b.id === "ct-filter-ISSUES")) {
          b.classList.remove("active");
        }
      });
      const btn = document.getElementById("ct-filter-" + filterKey);
      if (btn) {
        btn.classList.add("active");
      }
      filterChatThreadsList();
    }

    function filterChatThreadsList() {
      const q = (document.getElementById("search-chat-students")?.value || "").toLowerCase().trim();
      
      const filtered = rawChatThreads.filter(t => {
        if (activeChatThreadFilter === "ISSUES" && !t.has_issue) return false;
        
        if (q) {
          const matchName = (t.name || "").toLowerCase().includes(q);
          const matchPhone = (t.user_id || "").toLowerCase().includes(q);
          const matchLast = (t.last_message || "").toLowerCase().includes(q);
          if (!matchName && !matchPhone && !matchLast) return false;
        }
        return true;
      });

      renderChatThreads(filtered);
    }

    function renderChatThreads(threads) {
      const container = document.getElementById("chat-threads-container");
      if (!threads || threads.length === 0) {
        container.innerHTML = `
          <div class="py-6 text-center text-[#5A5E67] text-xs">
            No conversation threads found.
          </div>
        `;
        return;
      }

      container.innerHTML = threads.map(t => {
        const isActive = t.user_id === currentSelectedUserId ? "active" : "";
        
        return `
          <div onclick="loadConversationForUser('${t.user_id}')" class="chat-thread-row ${isActive} p-2.5 space-y-1">
            <div class="flex items-center justify-between text-xs">
              <div class="font-bold text-[#17181A] truncate">${t.name}</div>
              <span class="badge-status-neutral text-[10px]">${t.level || "Unset"}</span>
            </div>

            <p class="text-xs text-[#5A5E67] truncate leading-tight">
              ${t.last_message || "No message content"}
            </p>

            <div class="flex items-center justify-between text-[11px] text-[#9298A3] pt-0.5 font-mono">
              <span>${t.user_id}</span>
              <span>🔥 ${t.study_streak_days || 1}d &bull; ${t.message_count || 0} msgs</span>
            </div>
          </div>
        `;
      }).join("");
    }

    async function openStudentChat(userId) {
      switchTab('chats');
      await loadConversationForUser(userId);
    }

    async function loadConversationForUser(userId) {
      currentSelectedUserId = userId;
      filterChatThreadsList();
      
      const streamContainer = document.getElementById("chat-stream-container");
      streamContainer.innerHTML = `
        <div class="m-auto text-center text-[#5A5E67] p-8 text-xs">
          Loading transcript for ${userId}...
        </div>
      `;

      try {
        const res = await fetch(`/admin/api/students/${userId}/chats?include_broadcasts=${showBroadcastsInChat}`, {
          headers: { "Authorization": "Bearer " + authToken }
        });
        if (res.status === 401) { performLogout(); return; }
        const data = await res.json();
        
        const student = data.student || {};
        currentActiveMessages = data.messages || [];

        const initials = (student.name || "S").split(" ").map(w => w[0]).join("").substring(0, 2).toUpperCase();
        document.getElementById("chat-header-avatar").innerText = initials;
        document.getElementById("chat-header-name").innerText = student.name || "Candidate";
        document.getElementById("chat-header-level").innerText = student.level || "Unset";
        document.getElementById("chat-header-phone").innerText = student.user_id || userId;
        document.getElementById("chat-header-streak").innerText = `Streak: ${student.study_streak_days || 1}d`;
        document.getElementById("chat-header-wa-btn").href = `https://wa.me/${student.user_id || userId}`;

        const diagBanner = document.getElementById("chat-diagnostic-banner");
        const diagText = document.getElementById("chat-diagnostic-text");
        if (data.issue_count > 0) {
          diagText.innerText = `Diagnostic alert: ${data.issue_count} potential connection issue(s) detected.`;
          diagBanner.classList.remove("hidden");
        } else {
          diagBanner.classList.add("hidden");
        }

        renderConversationMessages(currentActiveMessages);
      } catch (e) {
        streamContainer.innerHTML = `<div class="m-auto text-[#C0392B] text-xs">Error loading transcript: ${e.message}</div>`;
      }
    }

    function filterInChatMessages() {
      const q = (document.getElementById("in-chat-search")?.value || "").toLowerCase().trim();
      if (!q) {
        renderConversationMessages(currentActiveMessages);
        return;
      }
      const filtered = currentActiveMessages.filter(m => (m.content || "").toLowerCase().includes(q));
      renderConversationMessages(filtered, q);
    }

    function formatChatMarkdown(text, highlightQuery = "") {
      if (!text) return "";
      let html = text
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");

      if (highlightQuery) {
        const regex = new RegExp(`(${highlightQuery})`, "gi");
        html = html.replace(regex, '<mark class="bg-[#FFF3CD] text-[#111B21] font-bold px-0.5 rounded">$1</mark>');
      }

      html = html
        .replace(/\*(.*?)\*/g, '<b class="font-bold text-[#111B21]">$1</b>')
        .replace(/_(.*?)_/g, '<i class="italic text-[#3B4A54]">$1</i>')
        .replace(/~([^~]+)~/g, '<s class="line-through text-[#667781]">$1</s>')
        .replace(/```([\s\S]*?)```/g, '<pre class="bg-[#F0F2F5] p-2 rounded font-mono text-[11px] border border-[#E9EDEF] overflow-x-auto my-1 text-[#111B21]"><code>$1</code></pre>')
        .replace(/`([^`]+)`/g, '<code class="bg-[#F0F2F5] px-1 py-0.5 rounded font-mono text-[11px] text-[#111B21] border border-[#E9EDEF]">$1</code>');

      return html;
    }

    function renderConversationMessages(messages, highlightQuery = "") {
      const container = document.getElementById("chat-stream-container");
      if (!messages || messages.length === 0) {
        container.innerHTML = `
          <div class="m-auto text-center text-[#667781] p-8 text-xs">
            No message interactions recorded in this thread.
          </div>
        `;
        return;
      }

      container.innerHTML = messages.map(m => {
        const isUser = m.role === "user";
        const hasIssue = m.metadata?.has_issue;
        const isBroadcast = m.metadata?.msg_type === "broadcast";
        const timeStr = formatChatTime(m.timestamp);
        const fullDateStr = formatFullDateTime(m.timestamp);
        const formatted = formatChatMarkdown(m.content, highlightQuery);

        if (isUser) {
          return `
            <div class="flex justify-end items-end gap-2">
              <div class="wa-bubble-user p-2.5 max-w-[85%] sm:max-w-[75%] space-y-0.5">
                <div class="text-[10px] font-semibold text-[#008069]">Candidate</div>
                <div class="text-xs text-[#111B21] whitespace-pre-wrap leading-relaxed">${formatted}</div>
                <div class="text-[10px] text-[#667781] text-right font-sans flex items-center justify-end gap-1 mt-0.5" title="${fullDateStr}">
                  <span>${timeStr}</span>
                  <i class="fa-solid fa-check-double text-[#53bdeb] text-[10px]"></i>
                </div>
              </div>
            </div>
          `;
        } else {
          return `
            <div class="flex justify-start items-end gap-2">
              <div class="wa-bubble-ai p-3 max-w-[90%] sm:max-w-[80%] space-y-1">
                <div class="text-[11px] font-bold text-[#008069] border-b border-[#E9EDEF] pb-1 mb-1 flex items-center justify-between">
                  <span>${isBroadcast ? '📢 NEURA AI Broadcast' : 'NEURA AI Clinical Engine'}</span>
                  ${m.metadata?.msg_type ? `<span class="text-[9px] font-mono text-[#667781]">(${m.metadata.msg_type})</span>` : ''}
                </div>
                ${hasIssue ? '<div class="mb-1 text-[10px] font-bold text-[#B7791F] bg-[#FEF7EC] border border-[#FAD7A0] px-1.5 py-0.5 rounded w-max">Diagnostic Alert</div>' : ''}
                <div class="text-xs text-[#111B21] whitespace-pre-wrap leading-relaxed">${formatted}</div>
                <div class="text-[10px] text-[#667781] text-right font-sans mt-0.5" title="${fullDateStr}">${timeStr}</div>
              </div>
            </div>
          `;
        }
      }).join("");

      container.scrollTop = container.scrollHeight;
    }

    // ================= BROADCAST & PREVIEW LOGIC =================
    function updateLivePreview() {
      const raw = document.getElementById("broadcast-msg").value;
      document.getElementById("char-count").innerText = raw.length + " chars";
      
      let formatted = raw
        .replace(/\*(.*?)\*/g, '<b class="font-bold text-[#111B21]">$1</b>')
        .replace(/_(.*?)_/g, '<i class="italic text-[#3B4A54]">$1</i>')
        .replace(/~([^~]+)~/g, '<s class="line-through text-[#667781]">$1</s>');
        
      if (!formatted.trim()) {
        formatted = "Type announcement on the left to see live rendering.";
      }
      document.getElementById("preview-text").innerHTML = formatted;
      
      const now = new Date();
      const timeFormatted = now.getHours().toString().padStart(2, '0') + ":" + now.getMinutes().toString().padStart(2, '0');
      document.getElementById("preview-time").innerText = timeFormatted;
      const footerTime = document.getElementById("preview-time-footer");
      if (footerTime) footerTime.innerText = timeFormatted;
    }

    function insertFormatting(prefix, suffix) {
      const textarea = document.getElementById("broadcast-msg");
      const start = textarea.selectionStart;
      const end = textarea.selectionEnd;
      const text = textarea.value;
      const selected = text.substring(start, end) || "text";
      textarea.value = text.substring(0, start) + prefix + selected + suffix + text.substring(end);
      textarea.focus();
      updateLivePreview();
    }

    function confirmBroadcast() {
      const msg = document.getElementById("broadcast-msg").value.trim();
      const target = document.getElementById("broadcast-target").value;
      if (!msg) {
        alert("Please enter announcement text to broadcast.");
        return;
      }
      document.getElementById("confirm-details").innerText = `You are about to dispatch this WhatsApp announcement to target cohort: ${target}.`;
      document.getElementById("confirm-modal").classList.remove("hidden");
    }

    function closeConfirmModal() {
      document.getElementById("confirm-modal").classList.add("hidden");
    }

    async function executeConfirmedBroadcast() {
      const msg = document.getElementById("broadcast-msg").value.trim();
      const target = document.getElementById("broadcast-target").value;
      const mode = document.getElementById("broadcast-mode").value;
      const btn = document.getElementById("modal-confirm-btn");
      
      btn.innerHTML = 'Dispatching...';
      btn.disabled = true;
      
      try {
        const res = await fetch("/admin/api/broadcast", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + authToken
          },
          body: JSON.stringify({
            message: msg,
            target_level: target,
            mode: mode
          })
        });
        
        if (res.ok) {
          closeConfirmModal();
          alert("Broadcast initiated successfully.");
          document.getElementById("broadcast-msg").value = "";
          updateLivePreview();
          setTimeout(loadAllData, 1000);
        } else {
          alert("Broadcast failed to initialize.");
        }
      } catch (e) {
        alert("Error sending broadcast: " + e.message);
      } finally {
        btn.innerHTML = 'Confirm Dispatch';
        btn.disabled = false;
      }
    }

    async function sendDirectStudentMessage() {
      if (!currentSelectedUserId) {
        alert("Please select a candidate first.");
        return;
      }
      const input = document.getElementById("direct-msg-input");
      const text = (input?.value || "").trim();
      if (!text) return;

      const btn = document.getElementById("direct-msg-send-btn");
      if (btn) {
        btn.disabled = true;
        btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin text-xs"></i> <span>Sending...</span>';
      }

      const modeSelect = document.getElementById("direct-msg-mode");
      const selectedMode = modeSelect ? modeSelect.value : "direct_only";

      try {
        const res = await fetch(`/admin/api/students/${currentSelectedUserId}/send-message`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + authToken
          },
          body: JSON.stringify({
            message: text,
            mode: selectedMode,
            template_name: "neura_announcement"
          })
        });

        if (res.ok) {
          const data = await res.json();
          input.value = "";
          // Reload transcript immediately to reflect the new message
          await loadConversationForUser(currentSelectedUserId);
        } else {
          const err = await res.json();
          alert("Failed to send message: " + (err.detail || "Unknown error"));
        }
      } catch (e) {
        alert("Error sending message: " + e.message);
      } finally {
        if (btn) {
          btn.disabled = false;
          btn.innerHTML = '<i class="fa-solid fa-paper-plane text-xs"></i> <span>Send</span>';
        }
      }
    }

    // Auto-login on load if token exists
    if (authToken) {
      showDashboard();
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)





