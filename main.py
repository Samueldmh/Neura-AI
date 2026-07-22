import os
import json
import traceback
import httpx
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from fastembed import TextEmbedding
from qdrant_client import QdrantClient

# ==========================================
# 1. CONFIGURATION & ENVIRONMENT VARIABLES
# ==========================================
QDRANT_URL = os.getenv("QDRANT_URL", "https://76ce5d85-4701-4671-8c3f-02bcc741b078.us-west-1-0.aws.cloud.qdrant.io")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

COLLECTION_NAME = "neura_medical_knowledge"

app = FastAPI(title="NEURA AI Backend", version="1.0.0")

# Initialize FastEmbed & Qdrant Client
print(f"Initializing FastEmbed & Qdrant Client...")
print(f"QDRANT_URL: {QDRANT_URL}")
print(f"QDRANT_API_KEY Present: {bool(QDRANT_API_KEY)}")
print(f"OPENROUTER_API_KEY Present: {bool(OPENROUTER_API_KEY)}")

embedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

class QueryRequest(BaseModel):
    user_id: str
    message: str

# ==========================================
# 2. SYSTEM PROMPTS & INTENT ROUTER
# ==========================================
SYSTEM_MEDICAL_PROMPT = """You are NEURA AI, an elite medical study assistant designed for Nigerian medical students.
Your goal is to provide authoritative, textbook-grounded answers to medical queries.

STRICT RULES:
1. Answer the student's question accurately using ONLY the provided Textbook Context below.
2. Structure your answer using clear WhatsApp Markdown:
   - 📌 **SUMMARY / HIGH-YIELD DEFINITION**: Clear 2-sentence direct answer.
   - 💡 **KEY PATHOPHYSIOLOGY / CLINICAL PEARLS**: Clean bullet points with bold emphasis.
   - 📚 **TEXTBOOK CITATION**: State the exact Textbook Title and Page Number provided in the context.
   - 🎯 **STUDY HOOK**: Ask if they want 3 practice MCQs or clinical case questions on this topic.
3. If the provided context does NOT contain enough information to answer the question, state:
   "I could not find exact coverage of this topic in your indexed medical textbooks. Please check your spelling or ask another topic!"
4. DO NOT make up or hallucinate medical facts outside the retrieved context.
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

async def call_openrouter_llm(system_prompt: str, user_prompt: str) -> str:
    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set on Render!")
        
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://neura-ai.org",
        "X-Title": "NEURA AI Medical Assistant"
    }
    
    payload = {
        "model": "google/gemini-2.0-flash-001",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
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
        ai_answer = await call_openrouter_llm(prompt_to_use, user_prompt)
        
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
