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
        uid = query.get("user_id")
        if uid in self.data:
            del self.data[uid]

class TestPaystackIntegrationAndWebhook(unittest.IsolatedAsyncioTestCase):
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

    # 1. Test initialize_paystack_transaction minimum deposit & payment URL
    async def test_initialize_paystack_transaction_validation(self):
        # Reject deposits < ₦5,000
        with self.assertRaises(ValueError) as ctx:
            await main.initialize_paystack_transaction("2348123456789", 4999.0)
        self.assertIn("Minimum deposit amount is ₦5,000", str(ctx.exception))
        
        with self.assertRaises(ValueError):
            await main.initialize_paystack_transaction("2348123456789", 1000.0)

        # Accept deposits >= ₦5,000 and return payment URL
        auth_url = await main.initialize_paystack_transaction("2348123456789", 5000.0)
        self.assertTrue(isinstance(auth_url, str))
        self.assertIn("paystack", auth_url.lower())

    # 2. Test Webhook HMAC SHA512 Signature Validation (Valid & Invalid)
    def test_webhook_hmac_signature_validation(self):
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "TEST_REF_SIG_123",
                "amount": 500000,
                "metadata": {"phone_number": "2348123456789"}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        
        # Missing signature header -> 401
        res_missing = self.client.post("/webhook/paystack", data=raw_body, headers={"Content-Type": "application/json"})
        self.assertEqual(res_missing.status_code, 401)
        self.assertIn("Missing", res_missing.json()["detail"])

        # Invalid signature -> 401
        res_invalid = self.client.post(
            "/webhook/paystack",
            data=raw_body,
            headers={
                "Content-Type": "application/json",
                "x-paystack-signature": "invalid_signature_hash_12345"
            }
        )
        self.assertEqual(res_invalid.status_code, 401)
        self.assertIn("Invalid", res_invalid.json()["detail"])

        # Valid signature -> 200 OK
        valid_sig = hmac.new(
            main.PAYSTACK_SECRET_KEY.encode("utf-8"),
            raw_body,
            hashlib.sha512
        ).hexdigest()

        res_valid = self.client.post(
            "/webhook/paystack",
            data=raw_body,
            headers={
                "Content-Type": "application/json",
                "x-paystack-signature": valid_sig
            }
        )
        self.assertEqual(res_valid.status_code, 200)
        self.assertEqual(res_valid.json()["status"], "success")

    # 3. Test Webhook charge.success Processing & Wallet Crediting & Receipt Notification
    def test_charge_success_processing_and_receipt(self):
        user_id = "2348123456789"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 1000.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }
        
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "PAYSTACK_TX_999",
                "amount": 1000000, # 10,000 NGN (1,000,000 kobo)
                "metadata": {"phone_number": user_id}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        valid_sig = hmac.new(
            main.PAYSTACK_SECRET_KEY.encode("utf-8"),
            raw_body,
            hashlib.sha512
        ).hexdigest()

        res = self.client.post(
            "/webhook/paystack",
            data=raw_body,
            headers={
                "Content-Type": "application/json",
                "x-paystack-signature": valid_sig
            }
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "success")
        self.assertEqual(res.json()["reference"], "PAYSTACK_TX_999")

        # Verify wallet credited (1000 + 10000 = 11000 NGN)
        user_doc = self.mock_users.data[user_id]
        self.assertEqual(user_doc["wallet_balance_ngn"], 11000.0)
        self.assertEqual(len(user_doc["transaction_history"]), 1)
        tx = user_doc["transaction_history"][0]
        self.assertEqual(tx["amount_ngn"], 10000.0)
        self.assertEqual(tx["reference"], "PAYSTACK_TX_999")

        # Verify automated WhatsApp receipt notification sent
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        to_num, receipt_body = self.sent_cloud_msgs[-1]
        self.assertEqual(to_num, user_id)
        self.assertIn("Payment Received & Wallet Credited!", receipt_body)
        self.assertIn("₦10,000.00", receipt_body)
        self.assertIn("PAYSTACK_TX_999", receipt_body)
        self.assertIn("₦11,000.00", receipt_body)

    # 4. Test Idempotency Check Against Transaction History
    def test_webhook_idempotency_prevents_double_crediting(self):
        user_id = "2348123456789"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 500.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        payload = {
            "event": "charge.success",
            "data": {
                "reference": "REF_IDEMPOTENT_001",
                "amount": 500000, # 5,000 NGN
                "metadata": {"phone_number": user_id}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        valid_sig = hmac.new(
            main.PAYSTACK_SECRET_KEY.encode("utf-8"),
            raw_body,
            hashlib.sha512
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "x-paystack-signature": valid_sig
        }

        # First webhook call -> Success
        res1 = self.client.post("/webhook/paystack", data=raw_body, headers=headers)
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["status"], "success")
        self.assertEqual(self.mock_users.data[user_id]["wallet_balance_ngn"], 5500.0)
        self.assertEqual(len(self.mock_users.data[user_id]["transaction_history"]), 1)

        # Duplicate webhook call (same payload/reference) -> Ignored for idempotency
        res2 = self.client.post("/webhook/paystack", data=raw_body, headers=headers)
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["status"], "success")
        self.assertIn("Duplicate", res2.json()["message"])
        
        # Balance must remain 5500.0 NGN, transaction count must remain 1
        self.assertEqual(self.mock_users.data[user_id]["wallet_balance_ngn"], 5500.0)
        self.assertEqual(len(self.mock_users.data[user_id]["transaction_history"]), 1)

    # 5. Test WhatsApp Commands (/wallet, /balance, /deposit, /topup)
    async def test_whatsapp_wallet_and_deposit_commands(self):
        user_id = "2348123456789"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "Sarah",
            "level": "500L",
            "preferred_books_list": ["Medicine & Surgery"],
            "wallet_balance_ngn": 5000.0,
            "total_spent_ngn": 100.0,
            "transaction_history": []
        }

        # Test /wallet command (balance: 5000, spent: 100, est_queries: ~250)
        await main.process_whatsapp_message(user_id, "/wallet")
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to_num, user_id)
        self.assertIn("Current Balance:* ₦5,000.00", msg)
        self.assertIn("Total Spent:* ₦100.00", msg)
        self.assertIn("Est. Queries Remaining:* ~250", msg)

        # Test /balance alias command
        await main.process_whatsapp_message(user_id, "/balance")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Current Balance:* ₦5,000.00", msg)

        # Test /deposit command (interactive buttons)
        await main.process_whatsapp_message(user_id, "/deposit")
        self.assertTrue(len(self.sent_btn_msgs) > 0)
        to_num, body, buttons = self.sent_btn_msgs[-1]
        self.assertEqual(to_num, user_id)
        self.assertIn("Select Top-Up Amount", body)
        btn_ids = [b["id"] for b in buttons]
        self.assertIn("TOPUP_5000", btn_ids)
        self.assertIn("TOPUP_10000", btn_ids)
        self.assertIn("TOPUP_20000", btn_ids)

        # Test /deposit with custom amount below 5000 (rejection)
        await main.process_whatsapp_message(user_id, "/deposit 2000")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Minimum Deposit Amount is ₦5,000", msg)

        # Test /deposit with valid custom amount (>= 5000) -> returns Paystack payment link
        await main.process_whatsapp_message(user_id, "/deposit 7500")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertIn("To complete your deposit of *₦7,500.00*", msg)
        self.assertIn("paystack", msg.lower())

        # Test interactive button selection TOPUP_5000 -> returns Paystack payment link
        await main.process_whatsapp_message(user_id, "TOPUP_5000")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertIn("To complete your deposit of *₦5,000.00*", msg)
        self.assertIn("paystack", msg.lower())

if __name__ == "__main__":
    unittest.main()
