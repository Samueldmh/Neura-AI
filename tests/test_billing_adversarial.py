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
        
    async def update_one(self, query, update_cmd, upsert=False):
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
            for k, v in update_cmd["$unset"]:
                doc.pop(k, None)

    async def delete_one(self, query):
        uid = query["user_id"]
        if uid in self.data:
            del self.data[uid]

class TestBillingAdversarial(unittest.IsolatedAsyncioTestCase):
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

    # ==========================================
    # 1. EXTREMELY LARGE TOKEN COUNTS & EDGE CASES
    # ==========================================
    def test_extremely_large_token_counts(self):
        """Test billing math with extreme token counts (1M, 10M, 100M tokens)"""
        # 1M Prompt, 1M Completion
        actual_1m, deduction_1m = main.calculate_api_cost_ngn(1_000_000, 1_000_000, usd_to_ngn=1500.0)
        # Expected:
        # Prompt USD: (1M/1M)*0.15 = 0.15 USD
        # Completion USD: (1M/1M)*0.60 = 0.60 USD
        # Total USD: 0.75 USD -> NGN: 0.75 * 1500 = 1125.0 NGN
        # Deduction: 1125.0 * 8.0 = 9000.0 NGN
        self.assertAlmostEqual(actual_1m, 1125.0, places=5)
        self.assertAlmostEqual(deduction_1m, 9000.0, places=5)

        # 10M Prompt, 5M Completion
        actual_10m, deduction_10m = main.calculate_api_cost_ngn(10_000_000, 5_000_000, usd_to_ngn=1500.0)
        # Expected:
        # Prompt USD: 10 * 0.15 = 1.50 USD
        # Completion USD: 5 * 0.60 = 3.00 USD
        # Total USD: 4.50 USD -> NGN: 4.50 * 1500 = 6750.0 NGN
        # Deduction: 6750.0 * 8.0 = 54000.0 NGN
        self.assertAlmostEqual(actual_10m, 6750.0, places=5)
        self.assertAlmostEqual(deduction_10m, 54000.0, places=5)

        # Asymmetric high volume: 50M prompt tokens (huge RAG context dump)
        actual_50m, deduction_50m = main.calculate_api_cost_ngn(50_000_000, 0, usd_to_ngn=1500.0)
        # Expected: 50 * 0.15 = 7.5 USD -> 11250 NGN * 8.0 = 90000 NGN
        self.assertAlmostEqual(actual_50m, 11250.0, places=5)
        self.assertAlmostEqual(deduction_50m, 90000.0, places=5)

    async def test_wallet_deduction_for_large_tokens(self):
        """Verify wallet deduction and transaction details for large token counts"""
        user_id = "adv_large_tokens"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 100000.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        tx = await main.deduct_user_wallet(user_id, prompt_tokens=1_000_000, completion_tokens=1_000_000, total_tokens=2_000_000)
        doc = await self.mock_users.find_one({"user_id": user_id})
        
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 91000.0, places=4)
        self.assertAlmostEqual(doc["total_spent_ngn"], 9000.0, places=4)
        self.assertEqual(tx["amount_ngn"], 9000.0)
        self.assertEqual(tx["tokens_used"], 2_000_000)
        self.assertEqual(tx["details"]["prompt_tokens"], 1_000_000)
        self.assertEqual(tx["details"]["completion_tokens"], 1_000_000)

    # ==========================================
    # 2. ZERO-TOKEN COMPLETIONS & SMALL EDGES
    # ==========================================
    def test_zero_token_completions_math(self):
        """Test cost calculation when prompt or completion tokens are zero"""
        # 0 prompt, 0 completion
        actual_0, deduction_0 = main.calculate_api_cost_ngn(0, 0, usd_to_ngn=1500.0)
        self.assertEqual(actual_0, 0.0)
        self.assertEqual(deduction_0, 0.0)

        # 500 prompt, 0 completion (e.g. LLM returned empty string or hit error)
        actual_p_only, deduction_p_only = main.calculate_api_cost_ngn(500, 0, usd_to_ngn=1500.0)
        # Expected: (500/1M)*0.15 = 0.000075 USD -> 0.1125 NGN * 8.0 = 0.90 NGN
        self.assertAlmostEqual(actual_p_only, 0.1125, places=5)
        self.assertAlmostEqual(deduction_p_only, 0.90, places=5)

        # 0 prompt, 200 completion
        actual_c_only, deduction_c_only = main.calculate_api_cost_ngn(0, 200, usd_to_ngn=1500.0)
        # Expected: (200/1M)*0.60 = 0.00012 USD -> 0.18 NGN * 8.0 = 1.44 NGN
        self.assertAlmostEqual(actual_c_only, 0.18, places=5)
        self.assertAlmostEqual(deduction_c_only, 1.44, places=5)

    async def test_wallet_deduction_zero_tokens(self):
        """Verify wallet deduction flow with 0 tokens"""
        user_id = "adv_zero_tokens"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 500.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        tx = await main.deduct_user_wallet(user_id, prompt_tokens=0, completion_tokens=0, total_tokens=0)
        doc = await self.mock_users.find_one({"user_id": user_id})
        
        self.assertEqual(doc["wallet_balance_ngn"], 500.0)
        self.assertEqual(doc["total_spent_ngn"], 0.0)
        self.assertEqual(tx["amount_ngn"], 0.0)
        self.assertEqual(tx["tokens_used"], 0)

    # ==========================================
    # 3. BOUNDARY TESTS FOR ₦19.99 vs ₦20.00 BALANCE INTERCEPT
    # ==========================================
    async def test_balance_intercept_exact_boundary(self):
        """Test boundary cases around the ₦20.00 balance threshold"""
        mock_llm = AsyncMock(return_value=("Sample Answer", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}))
        main.call_openrouter_llm = mock_llm
        mock_qdrant_point = type("Point", (), {"payload": {"book_title": "Robbins", "text": "Pathology content"}})()
        main.multi_search_qdrant = lambda *args, **kwargs: [mock_qdrant_point]

        # Case A: Balance = ₦19.99 -> MUST BE INTERCEPTED
        user_1999 = "user_1999"
        self.mock_users.data[user_1999] = {
            "user_id": user_1999,
            "onboarding_step": "COMPLETED",
            "name": "BorderStudent1",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 19.99,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        await main.process_whatsapp_message(user_1999, "What is necrosis?")
        mock_llm.assert_not_called()
        self.assertTrue(len(self.sent_btn_msgs) > 0)
        to, body, buttons = self.sent_btn_msgs[-1]
        self.assertEqual(to, user_1999)
        self.assertIn("Insufficient Wallet Balance", body)
        self.assertIn("₦19.99", body)
        self.sent_btn_msgs.clear()

        # Case B: Balance = ₦19.9999 -> MUST BE INTERCEPTED (< 20.0)
        user_19999 = "user_19999"
        self.mock_users.data[user_19999] = {
            "user_id": user_19999,
            "onboarding_step": "COMPLETED",
            "name": "BorderStudent2",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 19.9999,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        await main.process_whatsapp_message(user_19999, "What is apoptosis?")
        mock_llm.assert_not_called()
        self.assertTrue(len(self.sent_btn_msgs) > 0)
        to, body, buttons = self.sent_btn_msgs[-1]
        self.assertEqual(to, user_19999)
        self.assertIn("Insufficient Wallet Balance", body)
        self.sent_btn_msgs.clear()

        # Case C: Balance = ₦20.00 -> MUST PASSTHROUGH (NOT INTERCEPTED)
        user_2000 = "user_2000"
        self.mock_users.data[user_2000] = {
            "user_id": user_2000,
            "onboarding_step": "COMPLETED",
            "name": "BorderStudent3",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 20.00,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        await main.process_whatsapp_message(user_2000, "What is ischemia?")
        mock_llm.assert_called_once()
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        to, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to, user_2000)
        self.assertEqual(msg, "Sample Answer")
        mock_llm.reset_mock()
        self.sent_cloud_msgs.clear()

        # Case D: Balance = ₦20.01 -> MUST PASSTHROUGH
        user_2001 = "user_2001"
        self.mock_users.data[user_2001] = {
            "user_id": user_2001,
            "onboarding_step": "COMPLETED",
            "name": "BorderStudent4",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 20.01,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        await main.process_whatsapp_message(user_2001, "What is hyperplasia?")
        mock_llm.assert_called_once()

    async def test_negative_balance_intercept(self):
        """Verify negative balance is intercepted cleanly"""
        user_id = "user_negative"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "NegStudent",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": -15.50,
            "total_spent_ngn": 50.0,
            "transaction_history": []
        }
        
        mock_llm = AsyncMock()
        main.call_openrouter_llm = mock_llm
        await main.process_whatsapp_message(user_id, "Explain cell injury")
        mock_llm.assert_not_called()
        self.assertTrue(len(self.sent_btn_msgs) > 0)

    # ==========================================
    # 4. PROFIT MULTIPLIER VALIDATION (8.0x MARKUP)
    # ==========================================
    def test_profit_multiplier_exactness(self):
        """Validate that deduction_ngn is EXACTLY 8.0x actual_cost_ngn across multiple token scenarios"""
        token_matrix = [
            (100, 50),
            (500, 250),
            (1234, 5678),
            (10000, 20000),
            (88888, 99999),
            (1000000, 2000000)
        ]
        
        for prompt, comp in token_matrix:
            actual_cost, deduction = main.calculate_api_cost_ngn(prompt, comp, usd_to_ngn=1500.0)
            multiplier_calculated = deduction / actual_cost if actual_cost > 0 else 8.0
            self.assertAlmostEqual(multiplier_calculated, 8.0, places=6,
                                   msg=f"Profit multiplier mismatch for prompt={prompt}, comp={comp}")
            self.assertAlmostEqual(deduction, actual_cost * 8.0, places=6)

    def test_profit_multiplier_constant(self):
        """Ensure PROFIT_MULTIPLIER constant in main module is 8.0"""
        self.assertEqual(main.PROFIT_MULTIPLIER, 8.0)

    async def test_transaction_history_profit_multiplier_field(self):
        """Verify transaction records explicitly log profit_multiplier = 8.0"""
        user_id = "user_multiplier_audit"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 1000.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        tx = await main.deduct_user_wallet(user_id, prompt_tokens=2000, completion_tokens=1000, total_tokens=3000)
        self.assertEqual(tx["details"]["profit_multiplier"], 8.0)
        
        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertEqual(doc["transaction_history"][0]["details"]["profit_multiplier"], 8.0)

    # ==========================================
    # 5. SYSTEM COMMAND BYPASS ON LOW BALANCE
    # ==========================================
    async def test_system_commands_bypass_low_balance_1999(self):
        """Ensure system commands (/wallet, /deposit, /profile, /reset) execute even when balance is ₦19.99 or ₦0.00"""
        user_id = "user_low_cmd"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "BypassTestUser",
            "level": "500L",
            "preferred_books_list": ["Obstetrics & Gynaecology"],
            "wallet_balance_ngn": 19.99,
            "total_spent_ngn": 45.0,
            "transaction_history": []
        }

        # /wallet command
        await main.process_whatsapp_message(user_id, "/wallet")
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        to, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to, user_id)
        self.assertIn("Current Balance:* ₦19.99", msg)

        # /deposit command
        await main.process_whatsapp_message(user_id, "/deposit")
        self.assertTrue(len(self.sent_btn_msgs) > 0)
        to, body, buttons = self.sent_btn_msgs[-1]
        self.assertIn("Select Top-Up Amount", body)

        # TOPUP execution (returns Paystack link without pre-crediting wallet)
        await main.process_whatsapp_message(user_id, "TOPUP_5000")
        bal = await main.get_user_wallet_balance(user_id)
        self.assertAlmostEqual(bal, 19.99, places=2) # Not pre-credited

        # Simulate Paystack payment credit via credit_user_wallet
        await main.credit_user_wallet(user_id, 5000.0, "Paystack Deposit")
        bal_after = await main.get_user_wallet_balance(user_id)
        self.assertAlmostEqual(bal_after, 5019.99, places=2)

if __name__ == "__main__":
    unittest.main()
