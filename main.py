import os
import json
import traceback
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from motor.motor_asyncio import AsyncIOMotorClient

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://76ce5d85-4701-4671-8c3f-02bcc741b078.us-west-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MONGO_URI = os.getenv("MONGO_URI", "")

COLLECTION_NAME = "neura_medical_knowledge"

app = FastAPI(title="NEURA AI Backend", version="1.0.0")

# Initialize FastEmbed & Qdrant Client
print(f"Initializing FastEmbed & Qdrant Client...")
print(f"QDRANT_URL: {QDRANT_URL}")
print(f"QDRANT_API_KEY Present: {bool(QDRANT_API_KEY)}")
print(f"OPENROUTER_API_KEY Present: {bool(OPENROUTER_API_KEY)}")

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

print(f"MONGO_URI Present: {bool(MONGO_URI)}")
mongo_client = AsyncIOMotorClient(MONGO_URI) if MONGO_URI else None
db = mongo_client.neura_db if mongo_client else None
chat_history_col = db.chat_history if db is not None else None

class QueryRequest(BaseModel):
    user_id: str
    message: str

# ==========================================
# 2. SYSTEM PROMPTS & INTENT ROUTER
# ==========================================
SYSTEM_MEDICAL_PROMPT = """You are NEURA AI, an elite medical study assistant designed for Nigerian medical students.
Your goal is to provide authoritative, textbook-grounded answers to medical queries, while being natural and conversational.

RULES:
1. When answering medical facts, use ONLY the provided Textbook Context.
2. If the user asks a very short keyword (like "antibiotics"), don't reject it! Give a broad summary of the keyword based on context, and ask them what specific aspect they want to know.
3. Keep the conversation natural. You remember previous messages in the chat history.
4. Structure detailed medical answers using WhatsApp Markdown (📌 SUMMARY, 💡 KEY CLINICAL PEARLS, 📚 CITATION, 🎯 STUDY HOOK).
5. If they ask a highly specific medical question that is completely absent from context, politely say you don't have that in your current textbooks and ask them to clarify. DO NOT hallucinate.
"""

SYSTEM_QUIZ_PROMPT = """You are NEURA AI. Based ONLY on the retrieved medical textbook context, generate 3 high-yield MBBS exam-style Multiple Choice Questions (MCQs).
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

# ==========================================
# 3. ENDPOINTS
# ==========================================
@app.get("/")
@app.head("/")
def root():
    return {
        "status": "online",
        "system": "NEURA AI Medical Backend v1.0",
        "qdrant_configured": bool(QDRANT_API_KEY),
        "openrouter_configured": bool(OPENROUTER_API_KEY)
    }

@app.post("/api/chat")
async def chat_endpoint(req: QueryRequest):
    try:
        user_msg = req.message.strip()
        intent = classify_intent(user_msg)
        
        if intent == "GREETING":
            return {
                "response": "Hello! 👋 I'm *NEURA AI*, your medical study assistant.\n\nI can answer medical questions directly from your textbooks (*Lippincott Pharmacology*, *Hoffbrand's Haematology*, etc.) with exact citations, or generate practice MCQs for your MBBS exams!\n\nWhat concept are we studying today?"
            }
        
        # Search Qdrant DB
        query_vector = [e.tolist() for e in embedder.embed([user_msg])][0]
        
        try:
            search_res = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=4
            ).points
        except Exception as q_err:
            print(f"Qdrant query_points failed, retrying search: {q_err}")
            search_res = qdrant.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=4
            )
        
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
        
        prompt_to_use = SYSTEM_QUIZ_PROMPT if intent == "QUIZ" else SYSTEM_MEDICAL_PROMPT
        
        # 1. Fetch Chat History from MongoDB
        chat_history = []
        if chat_history_col is not None:
            user_doc = await chat_history_col.find_one({"user_id": req.user_id})
            if user_doc and "messages" in user_doc:
                chat_history = user_doc["messages"][-6:] # Keep last 6 messages
        
        # 2. Call LLM
        ai_answer = await call_openrouter_llm(prompt_to_use, user_prompt, chat_history)
        
        # 3. Save to MongoDB
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
