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

    # 1. MongoDB User Model Extension, Schema Defaults & Safe Fallbacks
    async def test_schema_defaults_and_fallbacks(self):
        user_id = "2348000000001"
        await main.process_whatsapp_message(user_id, "Hello")
        doc = await self.mock_users.find_one({"user_id": user_id})
        
        self.assertIsNotNone(doc)
        self.assertEqual(doc.get("wallet_balance_ngn", 0.0), 0.0)
        self.assertEqual(doc.get("total_spent_ngn", 0.0), 0.0)
        self.assertEqual(doc.get("transaction_history", []), [])
        
        # Test helper function get_user_wallet_balance
        bal = await main.get_user_wallet_balance(user_id)
        self.assertEqual(bal, 0.0)

        # Legacy fallback check (user document lacking wallet fields)
        self.mock_users.data["legacy_user"] = {"user_id": "legacy_user", "name": "LegacyStudent"}
        legacy_bal = await main.get_user_wallet_balance("legacy_user")
        self.assertEqual(legacy_bal, 0.0)

    # 2. Token Cost Math & 8.0x Profit Multiplier Billing Engine
    def test_token_cost_math_and_profit_multiplier(self):
        # 1000 prompt tokens ($0.00015), 500 completion tokens ($0.00030)
        # Total USD = $0.00045 * 1500 NGN/USD = 0.675 NGN
        # Deduction = 0.675 * 8.0 = 5.40 NGN
        actual_cost_ngn, deduction_ngn = main.calculate_api_cost_ngn(1000, 500, usd_to_ngn=1500.0)
        self.assertAlmostEqual(actual_cost_ngn, 0.675, places=5)
        self.assertAlmostEqual(deduction_ngn, 5.40, places=5)

        # 1M prompt + 1M completion
        # Input: $0.15, Output: $0.60 => $0.75 * 1500 = 1125 NGN
        # Deduction = 1125 * 8.0 = 9000 NGN
        actual_b, deduction_b = main.calculate_api_cost_ngn(1_000_000, 1_000_000, usd_to_ngn=1500.0)
        self.assertAlmostEqual(actual_b, 1125.0, places=5)
        self.assertAlmostEqual(deduction_b, 9000.0, places=5)

    # 3. Wallet Balance Lookup, Crediting and Deducting Helpers
    async def test_wallet_crediting_and_deduction(self):
        user_id = "2348000000002"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Credit wallet 1,000 NGN
        credit_tx = await main.credit_user_wallet(user_id, 1000.0, "Test Deposit")
        bal_after_credit = await main.get_user_wallet_balance(user_id)
        self.assertEqual(bal_after_credit, 1000.0)
        self.assertEqual(credit_tx["type"], "credit")
        self.assertEqual(credit_tx["amount_ngn"], 1000.0)

        # Deduct usage (1000 prompt, 500 completion -> 5.40 NGN deduction)
        tx = await main.deduct_user_wallet(user_id, prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        doc = await self.mock_users.find_one({"user_id": user_id})
        
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 994.60, places=4)
        self.assertAlmostEqual(doc["total_spent_ngn"], 5.40, places=4)
        self.assertEqual(len(doc["transaction_history"]), 2)
        
        deduction_tx = doc["transaction_history"][1]
        self.assertTrue(deduction_tx["tx_id"].startswith("tx_"))
        self.assertEqual(deduction_tx["type"], "deduction")
        self.assertAlmostEqual(deduction_tx["amount_ngn"], 5.40, places=4)
        self.assertEqual(deduction_tx["tokens_used"], 1500)
        self.assertEqual(deduction_tx["details"]["prompt_tokens"], 1000)
        self.assertEqual(deduction_tx["details"]["completion_tokens"], 500)
        self.assertEqual(deduction_tx["details"]["profit_multiplier"], 8.0)

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
        
        # LLM should NOT be called due to low balance interception
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
        
        # /wallet command
        await main.process_whatsapp_message(user_id, "/wallet")
        to, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to, user_id)
        self.assertIn("Current Balance:* ₦0.00", msg)
        self.assertIn("Total Spent:* ₦100.00", msg)
        
        # /deposit command
        await main.process_whatsapp_message(user_id, "/deposit")
        to, body, buttons = self.sent_btn_msgs[-1]
        self.assertEqual(to, user_id)
        self.assertIn("Select Top-Up Amount", body)
        
        # /profile command
        await main.process_whatsapp_message(user_id, "/profile")
        to, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Alex", msg)
        self.assertIn("Wallet Balance: ₦0.00", msg)

    # 6. Interactive Top-Up Execution Flow
    async def test_interactive_topup_and_recovery(self):
        user_id = "2348000000005"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "Grace",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 5.0, # low balance
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Select top-up option (returns Paystack checkout link)
        await main.process_whatsapp_message(user_id, "TOPUP_5000")
        to, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Paystack", msg)
        self.assertIn("₦5,000.00", msg)

        # Simulate Paystack webhook payment credit
        await main.credit_user_wallet(user_id, 5000.0, "Paystack Deposit (ref_test)")
        bal = await main.get_user_wallet_balance(user_id)
        self.assertEqual(bal, 5005.0)

        # Now query should succeed and deduct
        mock_llm_resp = ("Pathology explanation of inflammation.", {"prompt_tokens": 1000, "completion_tokens": 500, "total_tokens": 1500})
        main.call_openrouter_llm = AsyncMock(return_value=mock_llm_resp)
        mock_qdrant_point = type("Point", (), {"payload": {"book_title": "Robbins", "page_number": 42, "text": "Inflammation response"}})()
        main.multi_search_qdrant = lambda *args, **kwargs: [mock_qdrant_point]

        await main.process_whatsapp_message(user_id, "What is acute inflammation?")
        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 4999.60, places=4)
        self.assertAlmostEqual(doc["total_spent_ngn"], 5.40, places=4)

if __name__ == "__main__":
    unittest.main()
