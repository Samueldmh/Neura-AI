import asyncio
from unittest.mock import AsyncMock
import main

class MockUsersCol:
    def __init__(self):
        self.data = {}
        
    async def find_one(self, query):
        return self.data.get(query["user_id"])
        
    async def insert_one(self, doc):
        self.data[doc["user_id"]] = doc
        
    async def update_one(self, query, update_cmd):
        uid = query["user_id"]
        if uid not in self.data:
            self.data[uid] = {"user_id": uid}
            
        if "$set" in update_cmd:
            for k, v in update_cmd["$set"].items():
                self.data[uid][k] = v
        if "$push" in update_cmd:
            for k, v in update_cmd["$push"].items():
                if k not in self.data[uid]:
                    self.data[uid][k] = []
                self.data[uid][k].append(v)
                
    async def delete_one(self, query):
        uid = query["user_id"]
        if uid in self.data:
            del self.data[uid]

mock_col = MockUsersCol()
main.users_col = mock_col
main.chat_history_col = None

# Mock network calls
async def mock_send_cloud_msg(sender, msg):
    print(f"[WhatsApp TEXT] -> {msg}")

async def mock_send_interactive_list(sender, body, btn, options):
    print(f"[WhatsApp LIST] -> Body: '{body}' | Options: {options}")

async def mock_send_interactive_btn(sender, body, options):
    print(f"[WhatsApp BTN] -> Body: '{body}' | Options: {options}")

main.send_whatsapp_cloud_msg = mock_send_cloud_msg
main.send_whatsapp_interactive_list = mock_send_interactive_list
main.send_whatsapp_interactive_button = mock_send_interactive_btn

async def run_simulations():
    print("--- SIMULATING USER SENDING '/reset' ---")
    await main.process_whatsapp_message("test_user", "/reset")
    
    print("\n--- SIMULATING USER TRIGGERING ONBOARDING ---")
    await main.process_whatsapp_message("test_user", "START_ONBOARDING")
    
    print("\n--- SIMULATING USER SENDING NAME 'Samuel' ---")
    main.call_openrouter_llm = AsyncMock(return_value="Samuel")
    await main.process_whatsapp_message("test_user", "Samuel")
    
    print("\n--- SIMULATING USER SELECTING LEVEL '400L' ---")
    await main.process_whatsapp_message("test_user", "400L")
    
    print("\n--- SIMULATING USER SELECTING BOOK FOR HISTOPATHOLOGY ---")
    await main.process_whatsapp_message("test_user", "Robbins Basic Pathology 10th Edition 2017 (1)")

    print("\n--- SIMULATING USER SELECTING BOOK FOR CHEMICAL PATHOLOGY ---")
    await main.process_whatsapp_message("test_user", "Skip (None available yet)")
    
    print("\n--- CURRENT DB STATE ---")
    print(mock_col.data.get("test_user"))

if __name__ == "__main__":
    asyncio.run(run_simulations())
