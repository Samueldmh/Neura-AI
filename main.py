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

app = FastAPI(title="NEURA AI Backend", version="2.0.0")

# Initialize FastEmbed & Qdrant Client
print("Initializing FastEmbed & Qdrant Client...")
print(f"QDRANT_URL: {QDRANT_URL}")
print(f"QDRANT_API_KEY Present: {bool(QDRANT_API_KEY)}")
print(f"OPENROUTER_API_KEY Present: {bool(OPENROUTER_API_KEY)}")
print(f"PHONE_NUMBER_ID: {PHONE_NUMBER_ID}")

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

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
1. When answering NEW medical questions, use ONLY the provided Textbook Context. However, if the user asks a follow-up question (like "explain it simpler" or "what are its side effects"), you MUST use the facts already established in the chat history.
2. If the user asks a very short keyword (like "antibiotics"), provide a detailed overview of the topic based on the context, covering key mechanisms, clinical uses, or classifications, and ask what specific aspect they want to focus on.
3. Keep the conversation natural and engaging. You remember previous messages in the chat history.
4. Structure your detailed medical answers using clear WhatsApp Markdown. Use sections like 📖 IN-DEPTH EXPLANATION, 💡 KEY CLINICAL PEARLS, 📚 CITATION, and 🎯 STUDY HOOK to make it comprehensive yet readable. (If just simplifying a previous answer, you can skip the rigid structure and just be conversational).
5. If they ask a NEW specific medical question that is completely absent from context, politely say you don't have that in your current textbooks and suggest they rephrase or try a related keyword. NEVER make up answers from "general medical knowledge". You ONLY know what is in the provided Textbook Context and chat history. DO NOT hallucinate under any circumstances.
6. If the user asks general questions about your capabilities (e.g., "what can you do", "who are you", "help"), gracefully introduce yourself. Explain that you can answer medical questions based on textbooks, generate practice MCQs, and simplify complex concepts. Ignore the retrieved textbook context for these meta-questions.
"""

SYSTEM_QUIZ_PROMPT = """{user_context}You are NEURA AI. Based ONLY on the retrieved medical textbook context, generate 3 high-yield MBBS exam-style Multiple Choice Questions (MCQs).
Format clearly for WhatsApp:
- Provide 4 options (A, B, C, D) for each question.
- Include a hidden/spoiler or separate Answer Key at the bottom with step-by-step rationale citing the textbook title and page number.
"""

def classify_intent(message: str) -> str:
    msg_lower = message.strip().lower()
    
    if msg_lower in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "who are you", "what is neura ai"]:
        return "GREETING"
    
    if any(k in msg_lower for k in ["mcq", "quiz", "practice question", "test me", "exam question", "questions on"]):
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
    "hey", "hi", "hello", "okay", "ok", "sure", "yes", "no", "thanks", "thank",
}

def extract_medical_terms(user_msg: str) -> list:
    """Instantly extract medical keywords by stripping filler words and splitting on conjunctions.
    This is used ONLY for Qdrant search — the original message is still sent to the AI."""
    # Split on conjunctions to handle multi-topic queries ("TRALI and TRACO")
    parts = re.split(r'\b(?:and|or|vs|versus|,)\b', user_msg, flags=re.IGNORECASE)
    
    terms = []
    for part in parts:
        # Remove punctuation except hyphens (for terms like "beta-blocker")
        cleaned = re.sub(r'[^\w\s-]', '', part)
        words = cleaned.strip().split()
        # Keep only non-stop words
        medical_words = [w for w in words if w.lower() not in SEARCH_STOP_WORDS]
        if medical_words:
            term = " ".join(medical_words)
            terms.append(term)
    
    print(f"🔍 Extracted search keywords: {terms} (from: '{user_msg}')")
    return terms

def search_qdrant(query_text: str, limit: int = 4) -> list:
    """Search Qdrant with a single query string, returns list of points"""
    query_vector = [e.tolist() for e in embedder.embed([query_text])][0]
    try:
        return qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit
        ).points
    except Exception:
        return qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit
        )

def multi_search_qdrant(search_terms: list) -> list:
    """Run separate Qdrant searches for each extracted medical keyword, then deduplicate"""
    seen_texts = set()
    all_results = []
    
    # Search for each extracted medical term individually
    for term in search_terms:
        results = search_qdrant(term, limit=4)
        for point in results:
            text_key = point.payload.get("text", "")[:100]
            if text_key not in seen_texts:
                seen_texts.add(text_key)
                all_results.append(point)
    
    # Cap at 8 results max to avoid overwhelming the LLM context
    all_results = all_results[:8]
    print(f"📚 Multi-search returned {len(all_results)} unique chunks from {len(search_terms)} keyword(s): {search_terms}")
    return all_results

async def extract_name_with_llm(user_msg: str) -> str:
    prompt = """Extract the person's first name from this message. 
If they just say a greeting, or "why do you need it", or there is clearly no name, return NONE.
Return ONLY the name, nothing else.
Examples:
- "I am Samuel" -> Samuel
- "Samuel" -> Samuel
- "Hi my name is John" -> John
- "Why do you want to know?" -> NONE
- "Hello" -> NONE"""
    try:
        res = await call_openrouter_llm(prompt, user_msg)
        return res.strip() if res.strip().upper() != "NONE" else None
    except:
        return None

def get_level_textbooks_message(level: str) -> str:
    if level in ["200L", "300L"]:
        return ("Great! Here are the subjects and textbooks currently available for 200L/300L:\n"
                "🦴 *Anatomy*: Keith Moore\n"
                "🫀 *Physiology*: (Uploading soon)\n"
                "🧬 *Biochemistry*: (Uploading soon)\n\n"
                "Reply with the names of the books you want to use, or type 'All' to use everything available.")
    elif level == "400L":
        return ("Great! Here are the subjects and textbooks currently available for 400L:\n"
                "🔬 *Histopathology*: Robbins and Cotran\n"
                "🩸 *Haematology*: Essentials of Haematology\n"
                "🧪 *Chemical Pathology*: (Uploading soon)\n"
                "🧫 *Microbiology*: (Uploading soon)\n\n"
                "Reply with the names of the books you want to use, or type 'All'.")
    elif level == "500L":
        return ("Great! Here are the subjects for 500L:\n"
                "👶 *Obstetrics & Gynaecology*: (Uploading soon)\n\n"
                "*Note: We will notify you when these are uploaded. For now, you can still ask general questions!*")
    elif level == "600L":
        return ("Great! Here are the subjects for 600L:\n"
                "🩺 *Medicine & Surgery*: (Uploading soon)\n\n"
                "*Note: We will notify you when these are uploaded. For now, you can still ask general questions!*")
    return None

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
        await send_whatsapp_cloud_msg(sender_phone, "Welcome to NEURA AI! 🧠 To give you the best study experience, what is your name?")
        return True
        
    step = user_doc.get("onboarding_step")
    name = user_doc.get("name", "Student")
    level = user_doc.get("level", "")
    
    if step == "COMPLETED":
        return False
        
    # 2. Extract Name
    if step == "ASK_NAME":
        extracted_name = await extract_name_with_llm(user_msg)
        if not extracted_name:
            await send_whatsapp_cloud_msg(sender_phone, "I didn't quite catch that! Please just type your first name so I know what to call you. 😊")
            return True
            
        await users_col.update_one({"user_id": sender_phone}, {"$set": {"name": extracted_name, "onboarding_step": "ASK_LEVEL"}})
        await send_whatsapp_cloud_msg(sender_phone, f"Nice to meet you, {extracted_name}! What is your current medical class/level? (e.g., 200L, 300L, 400L, 500L, 600L)")
        return True
        
    # 3. Extract Level
    if step == "ASK_LEVEL":
        import re
        match = re.search(r'(200|300|400|500|600)', user_msg)
        if not match:
            await send_whatsapp_cloud_msg(sender_phone, "I don't recognize that level. Please type 200L, 300L, 400L, 500L, or 600L.")
            return True
            
        new_level = f"{match.group(1)}L"
        book_msg = get_level_textbooks_message(new_level)
        
        await users_col.update_one({"user_id": sender_phone}, {"$set": {"level": new_level, "onboarding_step": "ASK_BOOKS"}})
        await send_whatsapp_cloud_msg(sender_phone, book_msg)
        return True
        
    # 4. Extract Books
    if step == "ASK_BOOKS":
        books = user_msg.strip()
        await users_col.update_one({"user_id": sender_phone}, {"$set": {"preferred_books": books, "onboarding_step": "COMPLETED"}})
        
        final_msg = (f"Awesome, {name}! Your profile is all set up for {level}. You can now start asking me medical questions! 📚\n\n"
                     "⚙️ *Profile Commands:*\n"
                     "• Type */profile* to view your profile\n"
                     "• Type */update name* to change your name\n"
                     "• Type */update level* to change your level\n"
                     "• Type */update books* to change your textbooks")
        await send_whatsapp_cloud_msg(sender_phone, final_msg)
        return True
        
    return False

async def process_whatsapp_message(sender_phone: str, user_msg: str):
    """Background task to run RAG & OpenRouter LLM and send WhatsApp reply"""
    try:
        # Check for profile commands first
        msg_lower = user_msg.strip().lower()
        if msg_lower.startswith("/") and users_col is not None:
            if msg_lower == "/profile":
                user_doc = await users_col.find_one({"user_id": sender_phone})
                if user_doc:
                    name = user_doc.get("name", "Student")
                    level = user_doc.get("level", "Unknown")
                    books = user_doc.get("preferred_books", "None")
                    await send_whatsapp_cloud_msg(sender_phone, f"👤 *Your Profile*\n• Name: {name}\n• Level: {level}\n• Books: {books}")
                return
            elif msg_lower == "/update name":
                await users_col.update_one({"user_id": sender_phone}, {"$set": {"onboarding_step": "ASK_NAME"}})
                await send_whatsapp_cloud_msg(sender_phone, "What would you like to change your name to?")
                return
            elif msg_lower == "/update level":
                await users_col.update_one({"user_id": sender_phone}, {"$set": {"onboarding_step": "ASK_LEVEL"}})
                await send_whatsapp_cloud_msg(sender_phone, "What is your new medical class/level? (e.g., 200L, 300L, 400L, 500L, 600L)")
                return
            elif msg_lower == "/update books":
                user_doc = await users_col.find_one({"user_id": sender_phone})
                level = user_doc.get("level", "400L") if user_doc else "400L"
                await users_col.update_one({"user_id": sender_phone}, {"$set": {"onboarding_step": "ASK_BOOKS"}})
                await send_whatsapp_cloud_msg(sender_phone, get_level_textbooks_message(level))
                return

        # Handle onboarding state machine
        is_onboarding = await handle_onboarding(sender_phone, user_msg)
        if is_onboarding:
            return

        intent = classify_intent(user_msg)
        
        if intent == "GREETING":
            greeting_msg = (
                "Hello! 👋 I'm *NEURA AI*, your medical study assistant.\n\n"
                "I can answer medical questions directly from your textbooks (*Lippincott Pharmacology*, *Hoffbrand's Haematology*, etc.) "
                "with exact citations, or generate practice MCQs for your MBBS exams!\n\n"
                "What concept are we studying today?"
            )
            await send_whatsapp_cloud_msg(sender_phone, greeting_msg)
            return

        # Step 1: Extract medical terms from the user's message
        medical_terms = extract_medical_terms(user_msg)
        
        # Step 2: Multi-search Qdrant with extracted terms + original query
        if medical_terms:
            search_res = multi_search_qdrant(medical_terms)
        else:
            # No medical terms found — search with raw query as fallback
            search_res = search_qdrant(user_msg, limit=4)

        if not search_res:
            await send_whatsapp_cloud_msg(sender_phone, "I couldn't find relevant textbook material for your question. Please try asking a specific medical topic!")
            return

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
            user_doc = await users_col.find_one({"user_id": sender_phone})
            if user_doc:
                name = user_doc.get("name", "Student")
                level = user_doc.get("level", "Unknown Level")
                books = user_doc.get("preferred_books", "Unknown")
                user_context_str = f"The student asking this question is {name}, a {level} medical student. Their preferred textbooks are: {books}. Tailor your explanation to their level.\n\n"

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
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": ai_answer}
            ]
            await chat_history_col.update_one(
                {"user_id": sender_phone},
                {"$push": {"messages": {"$each": new_msgs}}},
                upsert=True
            )

        await send_whatsapp_cloud_msg(sender_phone, ai_answer)

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
                    
                    if msg_type == "text":
                        text_body = msg.get("text", {}).get("body", "")
                        print(f"📩 Received text from {sender_phone}: '{text_body}'")
                        
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
