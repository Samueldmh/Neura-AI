import asyncio
import json
import hmac
import hashlib
import unittest
from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
import main

class MockUsersCol:
    def __init__(self):
        self.data = {}
        
    async def find_one(self, query):
        if "user_id" in query:
            return self.data.get(query["user_id"])
        if "$or" in query:
            for user in self.data.values():
                tx_list = user.get("transaction_history", [])
                for tx in tx_list:
                    for cond in query["$or"]:
                        for key, ref_val in cond.items():
                            if key == "transaction_history.reference" and tx.get("reference") == ref_val:
                                return user
                            if key == "transaction_history.details.reference" and tx.get("details", {}).get("reference") == ref_val:
                                return user
        return None
        
    async def insert_one(self, doc):
        self.data[doc["user_id"]] = doc.copy()
        
    async def update_one(self, query, update_cmd, upsert=False):
        uid = query.get("user_id")
        if not uid and "$or" in query:
            doc = await self.find_one(query)
            if doc:
                uid = doc["user_id"]
        
        if not uid:
            return
            
        if uid not in self.data:
            if upsert:
                self.data[uid] = {
                    "user_id": uid,
                    "wallet_balance_ngn": 0.0,
                    "total_spent_ngn": 0.0,
                    "transaction_history": []
                }
            else:
                return

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

    async def delete_one(self, query):
        uid = query.get("user_id")
        if uid in self.data:
            del self.data[uid]

class TestAdversarialM5_2(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.mock_users = MockUsersCol()
        main.users_col = self.mock_users
        main.chat_history_col = None
        main.PAYSTACK_SECRET_KEY = "sk_test_neura_ai_secret_key_2026"
        
        self.sent_cloud_msgs = []
        self.sent_btn_msgs = []
        
        async def mock_send_cloud(to_number, msg):
            self.sent_cloud_msgs.append((to_number, msg))
            
        async def mock_send_btn(to_number, body, buttons):
            self.sent_btn_msgs.append((to_number, body, buttons))
            
        main.send_whatsapp_cloud_msg = mock_send_cloud
        main.send_whatsapp_interactive_button = mock_send_btn
        self.client = TestClient(main.app)

    def generate_signature(self, body_bytes: bytes, secret_key: str = None) -> str:
        key = secret_key if secret_key is not None else main.PAYSTACK_SECRET_KEY
        return hmac.new(key.encode("utf-8"), body_bytes, hashlib.sha512).hexdigest()

    # ==========================================
    # 1. PAYSTACK WEBHOOK ADVERSARIAL TESTS
    # ==========================================
    def test_webhook_missing_signature_returns_401(self):
        response = self.client.post("/webhook/paystack", json={"event": "charge.success"})
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing x-paystack-signature", response.json()["detail"])

    def test_webhook_invalid_signature_returns_401(self):
        body = json.dumps({"event": "charge.success"}).encode("utf-8")
        headers = {"x-paystack-signature": "invalid_sig_123"}
        response = self.client.post("/webhook/paystack", content=body, headers=headers)
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid Paystack signature", response.json()["detail"])

    def test_webhook_non_charge_success_event_ignored(self):
        payload = {"event": "charge.failed", "data": {"reference": "ref_failed_123"}}
        body = json.dumps(payload).encode("utf-8")
        sig = self.generate_signature(body)
        response = self.client.post("/webhook/paystack", content=body, headers={"x-paystack-signature": sig})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ignored")
        self.assertEqual(response.json()["event"], "charge.failed")

    async def test_webhook_replay_attack_idempotency(self):
        user_id = "2348011112222"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 100.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "ref_replay_test_001",
                "amount": 500000, # ₦5,000.00
                "metadata": {"phone_number": user_id}
            }
        }
        body = json.dumps(payload).encode("utf-8")
        sig = self.generate_signature(body)
        
        # First call -> Credited
        res1 = self.client.post("/webhook/paystack", content=body, headers={"x-paystack-signature": sig})
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["status"], "success")
        self.assertEqual(res1.json()["new_balance"], 5100.0)

        # Second call (Replay Attack) -> Ignored as Duplicate
        res2 = self.client.post("/webhook/paystack", content=body, headers={"x-paystack-signature": sig})
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["status"], "success")
        self.assertEqual(res2.json()["message"], "Duplicate event ignored")
        
        # Verify wallet balance remained 5100.0 and was not double-credited to 10100.0
        bal = await main.get_user_wallet_balance(user_id)
        self.assertEqual(bal, 5100.0)

    async def test_webhook_fallback_phone_from_email(self):
        user_id = "2348099998888"
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "ref_email_fallback_001",
                "amount": 1000000, # ₦10,000.00
                "customer": {"email": "2348099998888@neura-ai.org"}
            }
        }
        body = json.dumps(payload).encode("utf-8")
        sig = self.generate_signature(body)
        
        res = self.client.post("/webhook/paystack", content=body, headers={"x-paystack-signature": sig})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "success")
        bal = await main.get_user_wallet_balance(user_id)
        self.assertEqual(bal, 10000.0)

    def test_webhook_missing_phone_number_returns_ignored(self):
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "ref_no_phone_001",
                "amount": 500000,
                "customer": {"email": "anonymous@gmail.com"}
            }
        }
        body = json.dumps(payload).encode("utf-8")
        sig = self.generate_signature(body)
        
        res = self.client.post("/webhook/paystack", content=body, headers={"x-paystack-signature": sig})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ignored")
        self.assertEqual(res.json()["reason"], "Missing phone_number")

    # ==========================================
    # 2. TOPUP COMMAND HANDLERS ADVERSARIAL TESTS
    # ==========================================
    async def test_deposit_command_invalid_amount_parsing(self):
        user_id = "2348022223333"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        # /deposit with non-numeric text -> handled as invalid float -> minimum deposit warning
        await main.process_whatsapp_message(user_id, "/deposit abc")
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        _, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Minimum Deposit Amount is ₦5,000", msg)

    async def test_deposit_command_boundary_4999_99_rejected(self):
        user_id = "2348022223334"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        await main.process_whatsapp_message(user_id, "/deposit 4999.99")
        _, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Minimum Deposit Amount is ₦5,000", msg)

    async def test_deposit_command_boundary_5000_accepted(self):
        user_id = "2348022223335"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        await main.process_whatsapp_message(user_id, "/deposit 5000")
        _, msg = self.sent_cloud_msgs[-1]
        self.assertIn("To complete your deposit of *₦5,000.00*", msg)
        self.assertIn("paystack", msg.lower())

    async def test_topup_button_callbacks_case_insensitive(self):
        user_id = "2348022223336"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        await main.process_whatsapp_message(user_id, "topup_10000")
        _, msg = self.sent_cloud_msgs[-1]
        self.assertIn("To complete your deposit of *₦10,000.00*", msg)

    # ==========================================
    # 3. BALANCE INTERCEPTOR (< ₦20) ADVERSARIAL TESTS
    # ==========================================
    async def test_balance_interceptor_boundary_19_99_blocked(self):
        user_id = "2348033334444"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "BoundUser",
            "level": "500L",
            "wallet_balance_ngn": 19.99,
            "total_spent_ngn": 500.0,
            "transaction_history": []
        }
        
        mock_llm = AsyncMock()
        main.call_openrouter_llm = mock_llm
        
        await main.process_whatsapp_message(user_id, "What is pre-eclampsia?")
        
        mock_llm.assert_not_called()
        self.assertTrue(len(self.sent_btn_msgs) > 0)
        _, body, buttons = self.sent_btn_msgs[-1]
        self.assertIn("Insufficient Wallet Balance", body)
        self.assertIn("₦19.99", body)

    async def test_balance_interceptor_boundary_20_00_allowed(self):
        user_id = "2348033334445"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "PassUser",
            "level": "500L",
            "preferred_books_list": ["Obstetrics & Gynaecology"],
            "wallet_balance_ngn": 20.00,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        mock_llm = AsyncMock(return_value=("Pre-eclampsia is a multi-system disorder...", {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}))
        main.call_openrouter_llm = mock_llm
        
        mock_point = type("Point", (), {"payload": {"book_title": "Obstetrics", "text": "Pre-eclampsia text"}})()
        main.multi_search_qdrant = lambda *args, **kwargs: [mock_point]
        
        await main.process_whatsapp_message(user_id, "What is pre-eclampsia?")
        
        mock_llm.assert_called_once()
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        _, msg = self.sent_cloud_msgs[0]
        self.assertIn("Pre-eclampsia is a multi-system disorder", msg)

    async def test_balance_interceptor_system_commands_bypass_when_zero_balance(self):
        user_id = "2348033334446"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "ZeroUser",
            "level": "200L",
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        # Test /wallet command
        await main.process_whatsapp_message(user_id, "/wallet")
        _, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Current Balance:* ₦0.00", msg)
        
        # Test /deposit command
        await main.process_whatsapp_message(user_id, "/deposit")
        _, body, _ = self.sent_btn_msgs[-1]
        self.assertIn("Select Top-Up Amount", body)
        
        # Test /profile command
        await main.process_whatsapp_message(user_id, "/profile")
        _, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Your Profile", msg)

    # ==========================================
    # 4. BILLING DEDUCTION FORMULA (cost * 8.0) TESTS
    # ==========================================
    def test_billing_deduction_formula_exact_math(self):
        # prompt: 1000 ($0.00015), completion: 500 ($0.00030)
        # total USD: $0.00045 * 1500 NGN/USD = 0.675 NGN
        # deduction = 0.675 * 8.0 = 5.40 NGN
        actual_ngn, deduction_ngn = main.calculate_api_cost_ngn(1000, 500, usd_to_ngn=1500.0)
        self.assertAlmostEqual(actual_ngn, 0.675, places=5)
        self.assertAlmostEqual(deduction_ngn, 5.40, places=5)
        self.assertEqual(deduction_ngn, actual_ngn * 8.0)

    def test_billing_deduction_formula_zero_tokens(self):
        actual_ngn, deduction_ngn = main.calculate_api_cost_ngn(0, 0, usd_to_ngn=1500.0)
        self.assertEqual(actual_ngn, 0.0)
        self.assertEqual(deduction_ngn, 0.0)

    async def test_deduct_user_wallet_state_updates(self):
        user_id = "2348044445555"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 100.0,
            "total_spent_ngn": 10.0,
            "transaction_history": []
        }
        
        tx = await main.deduct_user_wallet(user_id, prompt_tokens=2000, completion_tokens=1000, total_tokens=3000)
        # prompt 2000 => $0.00030, completion 1000 => $0.00060 -> $0.00090 * 1500 = 1.35 NGN
        # deduction = 1.35 * 8.0 = 10.80 NGN
        self.assertAlmostEqual(tx["amount_ngn"], 10.80, places=4)
        
        doc = await self.mock_users.find_one({"user_id": user_id})
        self.assertAlmostEqual(doc["wallet_balance_ngn"], 89.20, places=4)
        self.assertAlmostEqual(doc["total_spent_ngn"], 20.80, places=4)
        self.assertEqual(len(doc["transaction_history"]), 1)
        self.assertEqual(doc["transaction_history"][0]["type"], "deduction")
        self.assertEqual(doc["transaction_history"][0]["details"]["profit_multiplier"], 8.0)

if __name__ == "__main__":
    unittest.main()
