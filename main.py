import os
import re
import json
import traceback
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from starlette.background import BackgroundTask
from pydantic import BaseModel
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models
from motor.motor_asyncio import AsyncIOMotorClient

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
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Ensure payload index exists for book_title filtering
try:
    qdrant.create_payload_index(
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
SYSTEM_MEDICAL_PROMPT = """{user_context}You are NEURA AI, an elite medical study assistant designed for Nigerian medical students.
Your goal is to provide authoritative, textbook-grounded answers to medical queries, while being natural, conversational, and highly detailed.

RULES:
1. When answering NEW medical questions, use ONLY the provided Textbook Context. However, if the user asks a follow-up question, you MUST use the facts already established in the chat history.
2. Synthesize and simplify the textbook information into a clean, easy-to-digest structure (e.g., Overview, Mechanism, Clinical Uses) instead of quoting it word-for-word. Make it highly structured but simplified.
3. Keep the conversation natural and engaging. You remember previous messages in the chat history.
4. FORMATTING: You must NOT use asterisks (*) for bolding, or hashes (#) for headings. The user dislikes this formatting. Instead, use clean plain text, emojis (e.g., 📖, 💡, 🎯, 📌), UPPERCASE letters for headings, and simple bullet points (-) to structure your text beautifully. 
5. If separating by textbook perspective, use a format like "📖 PATHOLOGY PERSPECTIVE (ROBBINS):" instead of markdown headings. DO NOT mix the perspectives together into a single explanation.
6. Include 📖 IN-DEPTH EXPLANATION, 💡 KEY CLINICAL PEARLS, 📚 CITATION, and 🎯 STUDY HOOK to make it comprehensive yet readable, but again, without any asterisks or bolding.
7. If the retrieved context does not contain the answer, you MUST say "I'm sorry, but this information is not covered in your selected textbooks." DO NOT hallucinate.
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
        "model": "openai/gpt-4o-mini",
        "messages": messages,
        "temperature": 0.2
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(url, headers=headers, json=payload)
        if response.status_code != 200:
            print(f"OpenRouter Error Status {response.status_code}: {response.text}")
            raise HTTPException(status_code=500, detail=f"OpenRouter Error: {response.text}")
        data = response.json()
        return data["choices"][0]["message"]["content"]

async def send_whatsapp_cloud_msg(to_number: str, message_text: str):
    """Sends a text response directly to the student via Meta WhatsApp Cloud API"""
    url = f"https://graph.facebook.com/v19.0/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN.strip()}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message_text
        }
    }
    async with httpx.AsyncClient(timeout=20.0) as client:
        res = await client.post(url, headers=headers, json=payload)
        print(f"Meta Graph API Send Status {res.status_code}: {res.text}")

async def send_whatsapp_interactive_list(to_number: str, body_text: str, button_text: str, options: list):
    """Sends an Interactive List Message (max 10 options)"""
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
}

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
        elif "lippincott" in msg_lower and "lippincott" in b_lower:
            override_books.append(b)
        elif "robbins" in msg_lower and "robbins" in b_lower:
            override_books.append(b)
        elif "sembulingam" in msg_lower and "sembulingam" in b_lower:
            override_books.append(b)
            
    return override_books if override_books else preferred_books

def extract_medical_terms(user_msg: str) -> list:
    """Instantly extract medical keywords by stripping filler words and splitting on conjunctions.
    This is used ONLY for Qdrant search — the original message is still sent to the AI."""
    """Instantly extract medical keywords by stripping filler words and splitting on conjunctions. Preserves capitalization."""
    # We strip punctuation but keep case
    msg = re.sub(r'[^\w\s]', ' ', user_msg)
    words = msg.split()
    
    # Check lower case for stop words, but preserve original case
    meaningful_words = [w for w in words if w.lower() not in SEARCH_STOP_WORDS and len(w) > 2]
    
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
    
    # Add the raw message as a whole phrase to ensure context is kept
    if user_msg not in phrases:
        phrases.append(user_msg)
        
    print(f"🔍 Extracted search keywords: {phrases} (from: '{user_msg}')")
    return phrases

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

def search_qdrant(query_text: str, limit: int = 4, preferred_books: list = None) -> list:
    """Search Qdrant securely per textbook to guarantee every selected book gets equal representation."""
    try:
        query_vector = [e.tolist() for e in embedder.embed([query_text])][0]
        
        all_points = []
        
        # If no preferred books are selected, fall back to a generic global search
        if not preferred_books:
            return qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=limit
            ).points
            
        # Guarantee equal representation by querying Qdrant for EACH book
        for book in preferred_books:
            if not book or book.startswith("Skip"):
                continue
                
            try:
                hits = qdrant.query_points(
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
                ).points
                all_points.extend(hits)
            except Exception as e:
                print(f"⚠️ Failed to filter Qdrant for book '{book}': {e}")
                
        # Sort the combined hits from all books by score
        all_points.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        return all_points

    except Exception as outer_e:
        print(f"❌ Error in search_qdrant: {outer_e}")
        return []

def multi_search_qdrant(search_terms: list, preferred_books: list = None) -> list:
    """Run separate Qdrant searches for each extracted medical keyword, then deduplicate"""
    seen_texts = set()
    all_results = []
    
    # Search for each extracted medical term individually
    for term in search_terms:
        results = search_qdrant(term, limit=12, preferred_books=preferred_books)
        for point in results:
            text_key = point.payload.get("text", "")[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_results.append(point)
    
    # Cap at 24 results max to avoid overwhelming the LLM context
    all_results = all_results[:24]
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
        
    # Send menu for next_subject
    books = AVAILABLE_BOOKS.get(next_subject, [])
    if not books:
        books = ["Skip (None available yet)"]
        
    body_text = f"Please select your preferred textbook for *{next_subject}*:"
    await send_whatsapp_interactive_list(sender_phone, body_text, "Select Textbook", books)
    
    # Update state
    await users_col.update_one(
        {"user_id": sender_phone}, 
        {"$set": {"onboarding_step": f"ASK_BOOK_{next_subject}"}}
    )
    return True

async def complete_onboarding(sender_phone: str):
    user_doc = await users_col.find_one({"user_id": sender_phone})
    name = user_doc.get("name", "Student")
    level = user_doc.get("level", "")
    await users_col.update_one({"user_id": sender_phone}, {"$set": {"onboarding_step": "COMPLETED"}})
    final_msg = (f"Awesome, {name}! Your profile is all set up for {level}. You can now start asking me medical questions! 📚\n\n"
                 "⚙️ *Quick Commands:*\n"
                 "• Type */feedback* to share quick feedback\n"
                 "• Type */profile* to view your profile\n"
                 "• Type */update name* to change your name\n"
                 "• Type */update level* to change your level\n"
                 "• Type */update books* to change your textbooks\n\n"
                 "💬 _Help us improve! Share 2-min anonymous beta feedback anytime: https://forms.gle/dNr7SV5EUiqiFySx5_")
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
        all_book_options = [b for books in AVAILABLE_BOOKS.values() for b in books] + ["Skip (None available yet)"]
        is_menu_tap = user_msg in ["200L", "300L", "400L", "500L", "600L"] or user_msg in all_book_options or user_msg in ["START_ONBOARDING", "🚀 Start Setup"]
        
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
        
    # 4. Extract Books (Dynamic Subject Loop)
    if step.startswith("ASK_BOOK_"):
        current_subject = step.replace("ASK_BOOK_", "")
        
        valid_options = AVAILABLE_BOOKS.get(current_subject, [])
        is_skip = (user_msg == "Skip (None available yet)")
        
        if not is_skip and user_msg not in valid_options:
            await send_whatsapp_cloud_msg(sender_phone, "Please use the menu button to select a valid textbook, or tap Skip if none are available.")
            return True
            
        # Save book if not skipped
        if not is_skip:
            await users_col.update_one(
                {"user_id": sender_phone},
                {"$push": {"preferred_books_list": user_msg}}
            )
            
        # Move to next subject
        has_more = await send_next_subject_menu(sender_phone, level, current_subject)
        if not has_more:
            await complete_onboarding(sender_phone)
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
    q_match = re.search(r'Q(\d+)_([A-D])', selected_option.upper())
    active_quiz = user_doc.get("active_quiz") if user_doc else None

    # If student taps an MCQ option dropdown after the quiz is finished/cleared
    if not active_quiz:
        if q_match:
            await send_whatsapp_cloud_msg(
                sender_phone,
                "⚠️ This quiz session has already ended! To start a new practice quiz, tap '📝 Generate MCQs' under any medical answer!"
            )
            return True
        return False

    questions = active_quiz.get("questions", [])
    idx = active_quiz.get("current_idx", 0)
    score = active_quiz.get("score", 0)

    if idx >= len(questions):
        if q_match:
            await send_whatsapp_cloud_msg(
                sender_phone,
                "⚠️ This quiz session has already ended! To start a new practice quiz, tap '📝 Generate MCQs' under any medical answer!"
            )
            return True
        return False

    if q_match:
        tapped_q_num = int(q_match.group(1))
        choice = q_match.group(2)
        current_q_num = idx + 1
        
        if tapped_q_num != current_q_num:
            await send_whatsapp_cloud_msg(
                sender_phone,
                f"⚠️ You have already answered Question {tapped_q_num}! Please select your answer for Question {current_q_num} below."
            )
            return True
    else:
        match = re.search(r'\b([A-D])\b', selected_option.upper())
        if not match:
            return False
        choice = match.group(1)

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

async def process_whatsapp_message(sender_phone: str, user_msg: str):
    """Background task to run RAG & OpenRouter LLM and send WhatsApp reply"""
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

        # Check for profile commands first
        msg_lower = user_msg.strip().lower()
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

        # Step 1: Extract medical terms from the query_to_search
        medical_terms = extract_medical_terms(query_to_search)
        
        # Step 1.5: Check for explicit book overrides (e.g. if user says "Use pharmacology")
        active_books = get_explicit_book_override(query_to_search, preferred_books_list)
        
        # Step 2: Multi-search Qdrant with extracted terms + original query, filtered by active books
        if medical_terms:
            if query_to_search not in medical_terms:
                medical_terms.append(query_to_search)
            search_res = multi_search_qdrant(medical_terms, preferred_books=active_books)
        else:
            search_res = search_qdrant(query_to_search, limit=12, preferred_books=active_books)

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
        user_prompt = f"RETRIEVED TEXTBOOK CONTEXT:\n{formatted_context}\n\nSTUDENT QUESTION:\n{query_to_search}"
        
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

        ai_answer = await call_openrouter_llm(prompt_to_use, user_prompt, chat_history)

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

        # Send the detailed answer
        await send_whatsapp_cloud_msg(sender_phone, ai_answer)

        # Attach interactive follow-up button for quick MCQ generation if it was a medical question
        if intent != "QUIZ" and not user_msg.startswith("GENERATE_QUIZ"):
            try:
                topic_snippet = query_to_search[:50]
                await send_whatsapp_interactive_button(
                    sender_phone,
                    "💡 Want to test your MBBS exam knowledge on this topic?",
                    [
                        {"id": f"GENERATE_QUIZ:{topic_snippet}", "title": "📝 Generate MCQs"}
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
                        print(f"📩 Received msg ({msg_type}) from {sender_phone}: '{text_body}'")
                        # Process in background task to respond to Meta immediately (prevents timeout)
                        task = BackgroundTask(process_whatsapp_message, sender_phone, text_body)
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
