import asyncio
import unittest
from unittest.mock import AsyncMock, patch
import json
import os
import main

REAL_SEND_INTERACTIVE_LIST = main.send_whatsapp_interactive_list

class ConcurrentMockUsersCol:
    """
    Mock MongoDB Users Collection that simulates MongoDB's single-document atomic updates
    using asyncio.Lock per document.
    """
    def __init__(self):
        self.data = {}
        self.locks = {}

    def _get_lock(self, uid):
        if uid not in self.locks:
            self.locks[uid] = asyncio.Lock()
        return self.locks[uid]

    async def find_one(self, query):
        uid = query.get("user_id")
        async with self._get_lock(uid):
            doc = self.data.get(uid)
            return doc.copy() if doc else None

    async def insert_one(self, doc):
        uid = doc.get("user_id")
        async with self._get_lock(uid):
            if uid in self.data:
                raise Exception(f"Duplicate key error: {uid}")
            self.data[uid] = doc.copy()

    async def update_one(self, query, update_cmd, upsert=False):
        uid = query.get("user_id")
        async with self._get_lock(uid):
            if uid not in self.data:
                if not upsert:
                    # Without upsert=True, MongoDB does NOT update or create any document!
                    return type("UpdateResult", (), {"matched_count": 0, "modified_count": 0})()
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
                    doc[k] = round(doc.get(k, 0.0) + v, 6)

            if "$push" in update_cmd:
                for k, v in update_cmd["$push"].items():
                    if k not in doc or not isinstance(doc[k], list):
                        doc[k] = []
                    doc[k].append(v)

            if "$unset" in update_cmd:
                for k in update_cmd["$unset"]:
                    doc.pop(k, None)

            return type("UpdateResult", (), {"matched_count": 1, "modified_count": 1})()

    async def delete_one(self, query):
        uid = query.get("user_id")
        async with self._get_lock(uid):
            if uid in self.data:
                del self.data[uid]

class TestTopUpCardChallenge(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_users = ConcurrentMockUsersCol()
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

        self.orig_send_list = REAL_SEND_INTERACTIVE_LIST
        main.send_whatsapp_cloud_msg = mock_send_cloud
        main.send_whatsapp_interactive_list = mock_send_list
        main.send_whatsapp_interactive_button = mock_send_btn

    # ==========================================
    # 1. GIT BRANCH VERIFICATION
    # ==========================================
    def test_01_git_branch_isolation(self):
        head_path = os.path.join(os.path.dirname(__file__), ".git", "HEAD")
        self.assertTrue(os.path.exists(head_path), "Git HEAD file must exist")
        with open(head_path, "r") as f:
            content = f.read().strip()
        self.assertIn("refs/heads/feature/pay-as-you-go", content, "Git branch must be feature/pay-as-you-go")

    # ==========================================
    # 2. INTERACTIVE TOP-UP CARD FORMATTING & META API STRUCTURES
    # ==========================================
    async def test_02_topup_button_card_structure(self):
        """Test payload structures produced by topup command & low balance interceptor"""
        user_id = "2348000000001"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Trigger /deposit
        await main.process_whatsapp_message(user_id, "/deposit")
        self.assertTrue(len(self.sent_btn_msgs) > 0, "Should send interactive button card")
        to, body, buttons = self.sent_btn_msgs[-1]

        self.assertEqual(to, user_id)
        self.assertIn("Select Top-Up Amount", body)
        self.assertEqual(len(buttons), 3, "Meta WhatsApp Interactive Buttons limited to max 3")

        # Verify button IDs and Titles
        for btn in buttons:
            self.assertIn(btn["id"], ["TOPUP_5000", "TOPUP_10000", "TOPUP_20000"])
            self.assertLessEqual(len(btn["id"]), 256, f"Button ID '{btn['id']}' exceeds Meta 256 char limit")
            self.assertLessEqual(len(btn["title"]), 20, f"Button title '{btn['title']}' exceeds Meta 20 char limit")

    async def test_03_topup_list_card_structure(self):
        """Test formatting if topup card is rendered as a Meta WhatsApp Interactive List"""
        user_id = "2348000000002"
        body_text = "Select Wallet Top-Up Package"
        button_text = "Choose Amount"
        options = [
            {"id": "TOPUP_5000", "title": "NGN 5,000", "description": "5,000 NGN Wallet Credit"},
            {"id": "TOPUP_10000", "title": "NGN 10,000", "description": "10,000 NGN Wallet Credit"},
            {"id": "TOPUP_20000", "title": "NGN 20,000", "description": "20,000 NGN Wallet Credit"}
        ]

        captured_payloads = []
        async def mock_post(*args, **kwargs):
            payload = kwargs.get("json")
            if payload is None and len(args) > 2:
                payload = args[2]
            captured_payloads.append(payload)
            return type("Response", (), {"status_code": 200, "text": "OK"})()

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            # Call original function to test HTTP payload structure
            await self.orig_send_list(user_id, body_text, button_text, options)

        self.assertEqual(len(captured_payloads), 1)
        payload = captured_payloads[0]

        self.assertEqual(payload.get("messaging_product"), "whatsapp")
        self.assertEqual(payload.get("type"), "interactive")
        interactive = payload.get("interactive", {})
        self.assertEqual(interactive.get("type"), "list")
        self.assertEqual(interactive.get("body", {}).get("text"), body_text)

        action = interactive.get("action", {})
        self.assertLessEqual(len(action.get("button", "")), 20, "List button text must be <= 20 chars")

        sections = action.get("sections", [])
        self.assertEqual(len(sections), 1)
        rows = sections[0].get("rows", [])
        self.assertEqual(len(rows), 3)

        for row in rows:
            self.assertIn(row["id"], ["TOPUP_5000", "TOPUP_10000", "TOPUP_20000"])
            self.assertLessEqual(len(row["id"]), 200, "Row ID must be <= 200 chars")
            self.assertLessEqual(len(row["title"]), 24, "Row title must be <= 24 chars")
            self.assertLessEqual(len(row.get("description", "")), 72, "Row description must be <= 72 chars")

    async def test_04_topup_execution_and_case_sensitivity(self):
        """Test executing top-up option IDs via webhook parsing"""
        user_id = "2348000000003"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # 1. Standard UPPERCASE button reply: TOPUP_5000
        await main.process_whatsapp_message(user_id, "TOPUP_5000")
        bal = await main.get_user_wallet_balance(user_id)
        self.assertEqual(bal, 0.0) # Wallet not credited pre-payment
        await main.credit_user_wallet(user_id, 5000.0, "Paystack Deposit (TOPUP_5000)")
        self.assertEqual(await main.get_user_wallet_balance(user_id), 5000.0)

        # 2. Lowercase reply: topup_10000
        await main.process_whatsapp_message(user_id, "topup_10000")
        bal = await main.get_user_wallet_balance(user_id)
        self.assertEqual(bal, 5000.0) # Wallet not credited pre-payment
        await main.credit_user_wallet(user_id, 10000.0, "Paystack Deposit (TOPUP_10000)")
        self.assertEqual(await main.get_user_wallet_balance(user_id), 15000.0)

        # 3. Mixed case reply: TopUp_20000
        await main.process_whatsapp_message(user_id, "TopUp_20000")
        bal = await main.get_user_wallet_balance(user_id)
        self.assertEqual(bal, 15000.0) # Wallet not credited pre-payment
        await main.credit_user_wallet(user_id, 20000.0, "Paystack Deposit (TOPUP_20000)")
        self.assertEqual(await main.get_user_wallet_balance(user_id), 35000.0)

        # Check transaction history records count
        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertEqual(len(doc["transaction_history"]), 3)

    # ==========================================
    # 3. MONGODB TRANSACTION LEDGER CONSISTENCY
    # ==========================================
    async def test_05_transaction_ledger_integrity(self):
        """Verify ledger record schema, timestamps, IDs, and balance consistency"""
        user_id = "2348000000004"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Step A: Credit NGN 10,000
        credit_tx = await main.credit_user_wallet(user_id, 10000.0, "Deposit via Paystack")
        self.assertTrue(credit_tx["tx_id"].startswith("tx_"))
        self.assertEqual(credit_tx["type"], "credit")
        self.assertEqual(credit_tx["amount_ngn"], 10000.0)
        self.assertIn("timestamp", credit_tx)

        # Step B: Deduct usage (2000 prompt tokens, 1000 completion tokens)
        # Cost math: (2000/1M * 0.15 + 1000/1M * 0.60) * 1500 * 8.0 = 10.80 NGN
        deduct_tx = await main.deduct_user_wallet(user_id, prompt_tokens=2000, completion_tokens=1000, total_tokens=3000)
        self.assertTrue(deduct_tx["tx_id"].startswith("tx_"))
        self.assertEqual(deduct_tx["type"], "deduction")
        self.assertAlmostEqual(deduct_tx["amount_ngn"], 10.80, places=2)
        self.assertEqual(deduct_tx["tokens_used"], 3000)

        # Inspect DB state directly
        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 9989.20, places=2)
        self.assertAlmostEqual(doc["total_spent_ngn"], 10.80, places=2)
        self.assertEqual(len(doc["transaction_history"]), 2)

        # Verify transaction timestamps order
        t1 = doc["transaction_history"][0]["timestamp"]
        t2 = doc["transaction_history"][1]["timestamp"]
        self.assertLessEqual(t1, t2)

    async def test_06_unregistered_user_ledger_edge_case(self):
        """Test behavior when credit_user_wallet/deduct_user_wallet is called for a non-existent user doc"""
        unregistered_uid = "2348000000099"
        # Ensure user does not exist in collection
        doc_before = await self.mock_users.find_one({"user_id": unregistered_uid})
        self.assertIsNone(doc_before)

        # Call credit_user_wallet
        tx = await main.credit_user_wallet(unregistered_uid, 5000.0, "Test Credit")

        # Because update_one is called without upsert=True in main.py, matched_count is 0!
        doc_after = await self.mock_users.find_one({"user_id": unregistered_uid})
        self.assertIsNone(doc_after, "Vulnerability Check: credit_user_wallet without upsert=True does not create user doc in DB!")

    # ==========================================
    # 4. ASYNC CONCURRENCY OF CREDITS AND DEDUCTIONS
    # ==========================================
    async def test_07_concurrent_credits(self):
        """Test 50 concurrent wallet credit operations"""
        user_id = "2348000000005"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Run 50 concurrent credits of NGN 100.00 each
        tasks = [main.credit_user_wallet(user_id, 100.0, f"Concurrent Credit {i}") for i in range(50)]
        results = await asyncio.gather(*tasks)

        self.assertEqual(len(results), 50)
        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 5000.0, places=2)
        self.assertEqual(len(doc["transaction_history"]), 50)

    async def test_08_concurrent_deductions(self):
        """Test 50 concurrent wallet deduction operations"""
        user_id = "2348000000006"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 10000.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Run 50 concurrent deductions (1000 prompt, 500 completion -> 5.40 NGN each)
        tasks = [main.deduct_user_wallet(user_id, 1000, 500, 1500) for i in range(50)]
        results = await asyncio.gather(*tasks)

        self.assertEqual(len(results), 50)
        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 9730.0, places=2)
        self.assertAlmostEqual(doc["total_spent_ngn"], 270.0, places=2)
        self.assertEqual(len(doc["transaction_history"]), 50)

    async def test_09_concurrent_mixed_credits_and_deductions(self):
        """Test 50 concurrent mixed credit & deduction operations running simultaneously"""
        user_id = "2348000000007"
        initial_bal = 1000.0
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": initial_bal,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        credit_tasks = [main.credit_user_wallet(user_id, 200.0, f"Mixed Credit {i}") for i in range(25)]
        deduct_tasks = [main.deduct_user_wallet(user_id, 1000, 500, 1500) for i in range(25)]

        all_tasks = []
        for c, d in zip(credit_tasks, deduct_tasks):
            all_tasks.append(c)
            all_tasks.append(d)

        await asyncio.gather(*all_tasks)

        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 5865.0, places=2)
        self.assertAlmostEqual(doc["total_spent_ngn"], 135.0, places=2)
        self.assertEqual(len(doc["transaction_history"]), 50)

    async def test_10_concurrent_low_balance_race_condition(self):
        """Vulnerability Test: Concurrent queries when wallet balance is near threshold (25.00 NGN) with async I/O latency"""
        user_id = "2348000000008"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "RaceStudent",
            "level": "400L",
            "preferred_books_list": ["Robbins Basic Pathology 10th Edition 2017 (1)"],
            "wallet_balance_ngn": 25.0, # Balance > 20.00 threshold
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Simulate real network latency (50ms) during LLM call to allow concurrent messages to read stale balance
        async def mock_async_llm(*args, **kwargs):
            await asyncio.sleep(0.05)
            return ("Detailed Medical Response", {"prompt_tokens": 2777, "completion_tokens": 1388, "total_tokens": 4165})

        main.call_openrouter_llm = mock_async_llm

        mock_qdrant_point = type("Point", (), {"payload": {"book_title": "Robbins", "text": "Pathology detail"}})()
        main.multi_search_qdrant = lambda *args, **kwargs: [mock_qdrant_point]

        # Launch 3 concurrent medical query requests simultaneously
        tasks = [
            main.process_whatsapp_message(user_id, "Explain pathology topic 1"),
            main.process_whatsapp_message(user_id, "Explain pathology topic 2"),
            main.process_whatsapp_message(user_id, "Explain pathology topic 3")
        ]
        await asyncio.gather(*tasks)

        doc = await self.mock_users.find_one({"user_id": user_id})
        final_balance = doc["wallet_balance_ngn"]
        print(f"\n[Race Condition Test] Final Balance after 3 concurrent requests on 25.00 NGN: NGN {final_balance:.2f}")
        # With async I/O delay during LLM call, all 3 requests pass low balance check simultaneously, driving balance negative (-19.98 NGN)
        self.assertLess(final_balance, 0.0, "Empirical proof: concurrent requests bypass low balance check causing negative balance!")

if __name__ == "__main__":
    unittest.main()
