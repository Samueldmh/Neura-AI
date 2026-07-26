import asyncio
import unittest
from unittest.mock import AsyncMock
import main

class MockUsersCol:
    def __init__(self):
        self.data = {}
        
    async def find_one(self, query):
        return self.data.get(query["user_id"])
        
    async def insert_one(self, doc):
        self.data[doc["user_id"]] = doc.copy()
        
    async def update_one(self, query, update_cmd):
        uid = query["user_id"]
        if uid not in self.data:
            self.data[uid] = {
                "user_id": uid,
                "wallet_balance_ngn": 0.0,
                "total_spent_ngn": 0.0,
                "transaction_history": []
            }
            
        doc = self.data[uid]
        
        if "$set" in update_cmd:
            for k, v in update_cmd["$set"].items():
                doc[k] = v
                
        if "$inc" in update_cmd:
            for k, v in update_cmd["$inc"].items():
                doc[k] = doc.get(k, 0.0) + v
                
        if "$push" in update_cmd:
            for k, v in update_cmd["$push"].items():
                if k not in doc or not isinstance(doc[k], list):
                    doc[k] = []
                doc[k].append(v)
                
        if "$unset" in update_cmd:
            for k in update_cmd["$unset"]:
                doc.pop(k, None)

    async def delete_one(self, query):
        uid = query["user_id"]
        if uid in self.data:
            del self.data[uid]

class TestWalletBillingEngine(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_users = MockUsersCol()
        main.users_col = self.mock_users
        main.chat_history_col = None
        
        self.sent_cloud_msgs = []
        self.sent_list_msgs = []
        self.sent_btn_msgs = []
        
        async def mock_send_cloud(to_number, msg):
            self.sent_cloud_msgs.append((to_number, msg))
            
        async def mock_send_list(to_number, body, button, options):
            self.sent_list_msgs.append((to_number, body, button, options))
            
        async def mock_send_btn(to_number, body, buttons):
            self.sent_btn_msgs.append((to_number, body, buttons))
            
        main.send_whatsapp_cloud_msg = mock_send_cloud
        main.send_whatsapp_interactive_list = mock_send_list
        main.send_whatsapp_interactive_button = mock_send_btn

    # 1. Schema Defaults & Safe Fallbacks
    async def test_schema_defaults_and_fallbacks(self):
        user_id = "2348000000001"
        await main.process_whatsapp_message(user_id, "Hello")
        doc = await self.mock_users.find_one({"user_id": user_id})
        
        self.assertIsNotNone(doc)
        self.assertEqual(doc.get("wallet_balance_ngn", 0.0), 0.0)
        self.assertEqual(doc.get("total_spent_ngn", 0.0), 0.0)
        self.assertEqual(doc.get("transaction_history", []), [])
        
        # Legacy fallback check
        self.mock_users.data["legacy_user"] = {"user_id": "legacy_user", "name": "Legacy"}
        fetched = await self.mock_users.find_one({"user_id": "legacy_user"})
        self.assertEqual(fetched.get("wallet_balance_ngn", 0.0), 0.0)
        self.assertEqual(fetched.get("total_spent_ngn", 0.0), 0.0)
        self.assertEqual(fetched.get("transaction_history", []), [])

    # 2. Token Cost Math & 8.0x Markup
    def test_token_cost_math_and_profit_multiplier(self):
        actual_cost_ngn, deduction_ngn = main.calculate_api_cost_ngn(1000, 500, usd_to_ngn=1500.0)
        self.assertAlmostEqual(actual_cost_ngn, 0.675, places=5)
        self.assertAlmostEqual(deduction_ngn, 5.40, places=5)

        actual_b, deduction_b = main.calculate_api_cost_ngn(1_000_000, 1_000_000, usd_to_ngn=1500.0)
        self.assertAlmostEqual(actual_b, 1125.0, places=5)
        self.assertAlmostEqual(deduction_b, 9000.0, places=5)

    # 3. Wallet Deduction and Transaction Logging
    async def test_wallet_deduction_and_transaction_log(self):
        user_id = "2348000000002"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 1000.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        tx = await main.deduct_user_wallet(user_id, prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        doc = await self.mock_users.find_one({"user_id": user_id})
        
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 994.60, places=4)
        self.assertAlmostEqual(doc["total_spent_ngn"], 5.40, places=4)
        self.assertEqual(len(doc["transaction_history"]), 1)
        
        tx_rec = doc["transaction_history"][0]
        self.assertTrue(tx_rec["tx_id"].startswith("tx_"))
        self.assertEqual(tx_rec["type"], "deduction")
        self.assertAlmostEqual(tx_rec["amount_ngn"], 5.40, places=4)
        self.assertEqual(tx_rec["tokens_used"], 1500)
        self.assertEqual(tx_rec["details"]["prompt_tokens"], 1000)
        self.assertEqual(tx_rec["details"]["completion_tokens"], 500)
        self.assertEqual(tx_rec["details"]["profit_multiplier"], 8.0)

    # 4. Low Balance (< ₦20.00) Interception
    async def test_low_balance_interception(self):
        user_id = "2348000000003"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "TestStudent",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 15.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        mock_llm = AsyncMock(return_value=("Answer", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}))
        main.call_openrouter_llm = mock_llm
        
        mock_qdrant_point = type("Point", (), {"payload": {"book_title": "Robbins", "text": "Pathology text"}})()
        main.multi_search_qdrant = lambda *args, **kwargs: [mock_qdrant_point]
        
        await main.process_whatsapp_message(user_id, "Explain inflammation in detail")
        
        mock_llm.assert_not_called()
        self.assertTrue(len(self.sent_btn_msgs) > 0)
        to, body, buttons = self.sent_btn_msgs[-1]
        self.assertEqual(to, user_id)
        self.assertIn("Insufficient Wallet Balance", body)
        self.assertIn("₦15.00", body)
        
        button_ids = [b["id"] for b in buttons]
        self.assertIn("TOPUP_5000", button_ids)
        self.assertIn("TOPUP_10000", button_ids)
        self.assertIn("TOPUP_20000", button_ids)

    # 5. System Command Bypass (With ₦0.00 Balance)
    async def test_system_command_bypass(self):
        user_id = "2348000000004"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "Alex",
            "level": "300L",
            "preferred_books_list": ["Textbook of Biochemistry For Medical Students 7th Edition"],
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 100.0,
            "transaction_history": []
        }
        
        # /wallet
        await main.process_whatsapp_message(user_id, "/wallet")
        to, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to, user_id)
        self.assertIn("Current Balance:* ₦0.00", msg)
        self.assertIn("Total Spent:* ₦100.00", msg)
        
        # /deposit
        await main.process_whatsapp_message(user_id, "/deposit")
        to, body, buttons = self.sent_btn_msgs[-1]
        self.assertEqual(to, user_id)
        self.assertIn("Select Top-Up Amount", body)
        
        # /profile
        await main.process_whatsapp_message(user_id, "/profile")
        to, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Alex", msg)
        self.assertIn("Wallet Balance: ₦0.00", msg)
        
        # /update name
        await main.process_whatsapp_message(user_id, "/update name")
        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertEqual(doc["onboarding_step"], "ASK_NAME")

    # 6. Sufficient Balance Query & Deduction
    async def test_sufficient_balance_flow(self):
        user_id = "2348000000005"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "Grace",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 500.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        mock_llm_resp = ("Pathology explanation of inflammation.", {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500})
        main.call_openrouter_llm = AsyncMock(return_value=mock_llm_resp)
        
        mock_qdrant_point = type("Point", (), {"payload": {"book_title": "Robbins", "page_number": 42, "text": "Inflammation response"}})()
        main.multi_search_qdrant = lambda *args, **kwargs: [mock_qdrant_point]
        
        await main.process_whatsapp_message(user_id, "What is acute inflammation?")
        
        to, msg = self.sent_cloud_msgs[0]
        self.assertEqual(to, user_id)
        self.assertIn("Pathology explanation of inflammation.", msg)
        
        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 494.60, places=4)
        self.assertAlmostEqual(doc["total_spent_ngn"], 5.40, places=4)
        self.assertEqual(len(doc["transaction_history"]), 1)

if __name__ == "__main__":
    unittest.main()
