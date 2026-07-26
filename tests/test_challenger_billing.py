import asyncio
import unittest
import datetime
from unittest.mock import AsyncMock
import main

class MockUsersCol:
    def __init__(self):
        self.data = {}
        self._lock = asyncio.Lock()
        
    async def find_one(self, query):
        async with self._lock:
            doc = self.data.get(query["user_id"])
            return doc.copy() if doc else None
        
    async def insert_one(self, doc):
        async with self._lock:
            self.data[doc["user_id"]] = doc.copy()
        
    async def update_one(self, query, update_cmd, upsert=False):
        async with self._lock:
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
        async with self._lock:
            uid = query["user_id"]
            if uid in self.data:
                del self.data[uid]


class TestChallengerBillingEngine(unittest.IsolatedAsyncioTestCase):
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

    # =========================================================================
    # 1. High Volume Concurrent Wallet Deductions
    # =========================================================================
    async def test_high_volume_concurrent_wallet_deductions(self):
        """Stress test 100 concurrent wallet deductions on a single user and across multiple users."""
        user_id = "user_concurrent_stress"
        initial_balance = 100000.0
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": initial_balance,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # 100 concurrent wallet deduction requests
        # Each deduction: 1000 prompt tokens, 500 completion tokens -> 5.40 NGN deduction
        num_requests = 100
        expected_deduction_per_req = 5.40
        expected_total_deduction = num_requests * expected_deduction_per_req

        tasks = [
            main.deduct_user_wallet(user_id, prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
            for _ in range(num_requests)
        ]
        results = await asyncio.gather(*tasks)

        doc = await self.mock_users.find_one({"user_id": user_id})
        
        self.assertEqual(len(results), num_requests)
        self.assertEqual(len(doc["transaction_history"]), num_requests)
        self.assertAlmostEqual(doc["wallet_balance_ngn"], initial_balance - expected_total_deduction, places=2)
        self.assertAlmostEqual(doc["total_spent_ngn"], expected_total_deduction, places=2)

    async def test_multi_user_concurrent_deductions(self):
        """Stress test concurrent deductions across 50 distinct users simultaneously."""
        num_users = 50
        for i in range(num_users):
            uid = f"multi_user_{i}"
            self.mock_users.data[uid] = {
                "user_id": uid,
                "wallet_balance_ngn": 1000.0,
                "total_spent_ngn": 0.0,
                "transaction_history": []
            }

        # Concurrently deduct from all 50 users (2 deductions per user = 100 tasks)
        tasks = []
        for i in range(num_users):
            uid = f"multi_user_{i}"
            tasks.append(main.deduct_user_wallet(uid, prompt_tokens=1000, completion_tokens=500, total_tokens=1500))
            tasks.append(main.deduct_user_wallet(uid, prompt_tokens=2000, completion_tokens=1000, total_tokens=3000))

        await asyncio.gather(*tasks)

        for i in range(num_users):
            uid = f"multi_user_{i}"
            doc = await self.mock_users.find_one({"user_id": uid})
            # Tx 1: (1000, 500) -> 5.40 NGN
            # Tx 2: (2000, 1000) -> 10.80 NGN
            # Total spent: 16.20 NGN, Balance: 983.80 NGN
            self.assertEqual(len(doc["transaction_history"]), 2)
            self.assertAlmostEqual(doc["wallet_balance_ngn"], 983.80, places=2)
            self.assertAlmostEqual(doc["total_spent_ngn"], 16.20, places=2)

    # =========================================================================
    # 2. Micro-Deductions (1 Token Prompt, 1 Token Completion)
    # =========================================================================
    def test_micro_deductions_math(self):
        """Verify API cost calculation for 1 prompt token & 1 completion token."""
        actual_cost_ngn, deduction_ngn = main.calculate_api_cost_ngn(1, 1, usd_to_ngn=1500.0)
        # 1 prompt token = 1/1M * $0.15 = $0.00000015
        # 1 comp token   = 1/1M * $0.60 = $0.00000060
        # Total USD = $0.00000075 -> NGN: 0.00000075 * 1500 = 0.001125 NGN
        # Deduction (8x): 0.001125 * 8.0 = 0.009 NGN
        self.assertAlmostEqual(actual_cost_ngn, 0.001125, places=7)
        self.assertAlmostEqual(deduction_ngn, 0.009, places=7)

    async def test_micro_deductions_wallet_execution(self):
        """Verify wallet deduction execution and precision for micro-deductions."""
        user_id = "user_micro_deduction"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 100.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # 100 sequential micro-deductions of 0.009 NGN each
        for _ in range(100):
            await main.deduct_user_wallet(user_id, prompt_tokens=1, completion_tokens=1, total_tokens=2)

        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertEqual(len(doc["transaction_history"]), 100)
        # Total deduction: 100 * 0.009 NGN = 0.90 NGN
        self.assertAlmostEqual(doc["total_spent_ngn"], 0.90, places=3)
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 99.10, places=3)

    # =========================================================================
    # 3. Zero Token Queries
    # =========================================================================
    def test_zero_token_queries_math(self):
        """Verify math for 0 tokens prompt and 0 tokens completion."""
        actual_cost_ngn, deduction_ngn = main.calculate_api_cost_ngn(0, 0, usd_to_ngn=1500.0)
        self.assertEqual(actual_cost_ngn, 0.0)
        self.assertEqual(deduction_ngn, 0.0)

    async def test_zero_token_wallet_deduction(self):
        """Verify wallet state when zero token query is processed."""
        user_id = "user_zero_token"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 50.0,
            "total_spent_ngn": 10.0,
            "transaction_history": []
        }

        tx = await main.deduct_user_wallet(user_id, prompt_tokens=0, completion_tokens=0, total_tokens=0)
        doc = await self.mock_users.find_one({"user_id": user_id})

        self.assertEqual(doc["wallet_balance_ngn"], 50.0)
        self.assertEqual(doc["total_spent_ngn"], 10.0)
        self.assertEqual(tx["amount_ngn"], 0.0)
        self.assertEqual(tx["tokens_used"], 0)
        self.assertEqual(len(doc["transaction_history"]), 1)

    # =========================================================================
    # 4. Extremely Large Prompt/Completion Token Queries (100,000+ Tokens)
    # =========================================================================
    def test_extremely_large_token_queries_math(self):
        """Verify cost math for 100,000 prompt and 100,000 completion tokens."""
        prompt_tokens = 100_000
        completion_tokens = 100_000
        actual_cost_ngn, deduction_ngn = main.calculate_api_cost_ngn(prompt_tokens, completion_tokens, usd_to_ngn=1500.0)
        
        # Prompt: (100k/1M)*0.15 = 0.015 USD
        # Completion: (100k/1M)*0.60 = 0.060 USD
        # Total USD = 0.075 USD -> NGN: 0.075 * 1500 = 112.5 NGN
        # Deduction: 112.5 * 8.0 = 900.0 NGN
        self.assertAlmostEqual(actual_cost_ngn, 112.50, places=4)
        self.assertAlmostEqual(deduction_ngn, 900.00, places=4)

    async def test_extremely_large_token_wallet_execution(self):
        """Verify wallet deduction for 100k token query."""
        user_id = "user_large_100k"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 5000.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        tx = await main.deduct_user_wallet(user_id, prompt_tokens=100_000, completion_tokens=100_000, total_tokens=200_000)
        doc = await self.mock_users.find_one({"user_id": user_id})

        self.assertAlmostEqual(doc["wallet_balance_ngn"], 4100.0, places=2)
        self.assertAlmostEqual(doc["total_spent_ngn"], 900.0, places=2)
        self.assertEqual(tx["amount_ngn"], 900.0)
        self.assertEqual(tx["tokens_used"], 200_000)
        self.assertEqual(tx["details"]["prompt_tokens"], 100_000)
        self.assertEqual(tx["details"]["completion_tokens"], 100_000)

    # =========================================================================
    # 5. Exact Boundary Tests for Balance Interception (₦19.99 vs ₦20.00 vs ₦20.01)
    # =========================================================================
    async def test_exact_boundary_interception_1999(self):
        """Balance of ₦19.99 must be intercepted and block LLM processing."""
        user_id = "user_boundary_1999"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "BoundaryUser1999",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 19.99,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        mock_llm = AsyncMock(return_value=("Mock response", {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}))
        main.call_openrouter_llm = mock_llm
        mock_qdrant_point = type("Point", (), {"payload": {"book_title": "Robbins", "text": "Pathology text"}})()
        main.multi_search_qdrant = lambda *args, **kwargs: [mock_qdrant_point]

        await main.process_whatsapp_message(user_id, "What is acute inflammation?")

        # LLM must NOT be called
        mock_llm.assert_not_called()
        self.assertTrue(len(self.sent_btn_msgs) > 0)
        to, body, buttons = self.sent_btn_msgs[-1]
        self.assertEqual(to, user_id)
        self.assertIn("Insufficient Wallet Balance", body)
        self.assertIn("₦19.99", body)

    async def test_exact_boundary_interception_2000(self):
        """Balance of ₦20.00 must PASSTHROUGH (NOT be intercepted)."""
        user_id = "user_boundary_2000"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "BoundaryUser2000",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 20.00,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        mock_llm = AsyncMock(return_value=("Pathology answer for 20.00 NGN balance", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}))
        main.call_openrouter_llm = mock_llm
        mock_qdrant_point = type("Point", (), {"payload": {"book_title": "Robbins", "text": "Pathology text"}})()
        main.multi_search_qdrant = lambda *args, **kwargs: [mock_qdrant_point]

        await main.process_whatsapp_message(user_id, "What is necrosis?")

        # LLM MUST be called
        mock_llm.assert_called_once()
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        to, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to, user_id)
        self.assertIn("Pathology answer", msg)

    async def test_exact_boundary_interception_2001(self):
        """Balance of ₦20.01 must PASSTHROUGH (NOT be intercepted)."""
        user_id = "user_boundary_2001"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "BoundaryUser2001",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 20.01,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        mock_llm = AsyncMock(return_value=("Pathology answer for 20.01 NGN balance", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}))
        main.call_openrouter_llm = mock_llm
        mock_qdrant_point = type("Point", (), {"payload": {"book_title": "Robbins", "text": "Pathology text"}})()
        main.multi_search_qdrant = lambda *args, **kwargs: [mock_qdrant_point]

        await main.process_whatsapp_message(user_id, "What is apoptosis?")

        # LLM MUST be called
        mock_llm.assert_called_once()
        self.assertTrue(len(self.sent_cloud_msgs) > 0)

    # =========================================================================
    # 6. Verification of Transaction History Ordering, Metadata & Integrity
    # =========================================================================
    async def test_transaction_history_ordering_and_metadata(self):
        """Verify transaction history maintains chronological order, unique IDs, and complete metadata."""
        user_id = "user_tx_history_audit"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Sequence of transactions: Credit -> Deduct -> Credit -> Deduct
        tx1 = await main.credit_user_wallet(user_id, 5000.0, "Initial Deposit")
        await asyncio.sleep(0.01) # ensure timestamp separation
        tx2 = await main.deduct_user_wallet(user_id, prompt_tokens=2000, completion_tokens=1000, total_tokens=3000)
        await asyncio.sleep(0.01)
        tx3 = await main.credit_user_wallet(user_id, 2000.0, "Second Deposit")
        await asyncio.sleep(0.01)
        tx4 = await main.deduct_user_wallet(user_id, prompt_tokens=500, completion_tokens=250, total_tokens=750)

        doc = await self.mock_users.find_one({"user_id": user_id})
        history = doc["transaction_history"]

        self.assertEqual(len(history), 4)

        # 1. Ordering check: Timestamps must be monotonically non-decreasing
        timestamps = [datetime.datetime.fromisoformat(t["timestamp"]) for t in history]
        for i in range(len(timestamps) - 1):
            self.assertLessEqual(timestamps[i], timestamps[i + 1])

        # 2. Metadata completeness and type check
        types = [t["type"] for t in history]
        self.assertEqual(types, ["credit", "deduction", "credit", "deduction"])

        # 3. Tx IDs uniqueness
        tx_ids = [t["tx_id"] for t in history]
        self.assertEqual(len(set(tx_ids)), 4)
        for tid in tx_ids:
            self.assertTrue(tid.startswith("tx_"))

        # 4. Deduction metadata check
        deduction_record = history[1]
        self.assertEqual(deduction_record["type"], "deduction")
        self.assertEqual(deduction_record["tokens_used"], 3000)
        self.assertIn("details", deduction_record)
        details = deduction_record["details"]
        self.assertEqual(details["prompt_tokens"], 2000)
        self.assertEqual(details["completion_tokens"], 1000)
        self.assertEqual(details["profit_multiplier"], 8.0)
        self.assertIn("actual_api_cost_ngn", details)

if __name__ == "__main__":
    unittest.main()
