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

class TestPaystackAdversarial(unittest.IsolatedAsyncioTestCase):
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

    # -------------------------------------------------------------
    # Category 1: Paystack Webhook HMAC-SHA512 Signature Security
    # -------------------------------------------------------------

    def test_adv_hmac_missing_signature_header(self):
        """Test webhook request missing x-paystack-signature header is rejected (401)"""
        payload = {"event": "charge.success", "data": {"reference": "ADV_MISSING_SIG_1"}}
        raw_body = json.dumps(payload).encode("utf-8")
        
        res = self.client.post("/webhook/paystack", data=raw_body, headers={"Content-Type": "application/json"})
        self.assertEqual(res.status_code, 401)
        self.assertIn("Missing", res.json()["detail"])

    def test_adv_hmac_invalid_signature_header(self):
        """Test webhook request with random invalid signature header is rejected (401)"""
        payload = {"event": "charge.success", "data": {"reference": "ADV_INVALID_SIG_1"}}
        raw_body = json.dumps(payload).encode("utf-8")
        
        res = self.client.post(
            "/webhook/paystack",
            data=raw_body,
            headers={
                "Content-Type": "application/json",
                "x-paystack-signature": "deadbeef1234567890abcdef"
            }
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid", res.json()["detail"])

    def test_adv_hmac_forged_payload_body(self):
        """Test webhook payload tampered after generating signature is rejected (401)"""
        original_payload = {
            "event": "charge.success",
            "data": {
                "reference": "ADV_FORGED_1",
                "amount": 500000, # 5,000 NGN
                "metadata": {"phone_number": "2348011112222"}
            }
        }
        original_raw = json.dumps(original_payload).encode("utf-8")
        valid_sig = hmac.new(
            main.PAYSTACK_SECRET_KEY.encode("utf-8"),
            original_raw,
            hashlib.sha512
        ).hexdigest()

        # Forged payload: attacker changes amount to 50,000 NGN or changes reference
        forged_payload = {
            "event": "charge.success",
            "data": {
                "reference": "ADV_FORGED_1",
                "amount": 5000000, # Tampered amount!
                "metadata": {"phone_number": "2348011112222"}
            }
        }
        forged_raw = json.dumps(forged_payload).encode("utf-8")

        # Send forged payload with valid signature computed for original payload
        res = self.client.post(
            "/webhook/paystack",
            data=forged_raw,
            headers={
                "Content-Type": "application/json",
                "x-paystack-signature": valid_sig
            }
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid", res.json()["detail"])

    def test_adv_hmac_wrong_secret_key(self):
        """Test webhook signed with attacker's secret key is rejected (401)"""
        payload = {"event": "charge.success", "data": {"reference": "ADV_WRONG_SECRET_1"}}
        raw_body = json.dumps(payload).encode("utf-8")
        
        attacker_sig = hmac.new(
            b"sk_test_attacker_fake_secret_key",
            raw_body,
            hashlib.sha512
        ).hexdigest()

        res = self.client.post(
            "/webhook/paystack",
            data=raw_body,
            headers={
                "Content-Type": "application/json",
                "x-paystack-signature": attacker_sig
            }
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("Invalid", res.json()["detail"])

    def test_adv_hmac_valid_signature_accepted(self):
        """Test webhook with legitimate HMAC SHA-512 signature is accepted (200 OK)"""
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "ADV_VALID_SIG_1",
                "amount": 500000,
                "metadata": {"phone_number": "2348011112222"}
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

    # -------------------------------------------------------------
    # Category 2: Replay Attack & Idempotency Tests
    # -------------------------------------------------------------

    def test_adv_replay_attack_identical_event_twice(self):
        """Test replay attack: sending identical charge.success event twice does NOT double credit"""
        user_id = "2348033334444"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 1000.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        payload = {
            "event": "charge.success",
            "data": {
                "reference": "REF_REPLAY_TEST_100",
                "amount": 500000, # ₦5,000 NGN
                "metadata": {"phone_number": user_id}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        valid_sig = hmac.new(
            main.PAYSTACK_SECRET_KEY.encode("utf-8"),
            raw_body,
            hashlib.sha512
        ).hexdigest()
        headers = {"Content-Type": "application/json", "x-paystack-signature": valid_sig}

        # 1st request -> success, balance 1000 + 5000 = 6000
        res1 = self.client.post("/webhook/paystack", data=raw_body, headers=headers)
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["status"], "success")
        self.assertEqual(self.mock_users.data[user_id]["wallet_balance_ngn"], 6000.0)
        self.assertEqual(len(self.mock_users.data[user_id]["transaction_history"]), 1)

        # 2nd request (Replay Attack) -> ignored gracefully, balance stays 6000
        res2 = self.client.post("/webhook/paystack", data=raw_body, headers=headers)
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["status"], "success")
        self.assertIn("Duplicate", res2.json()["message"])
        self.assertEqual(self.mock_users.data[user_id]["wallet_balance_ngn"], 6000.0)
        self.assertEqual(len(self.mock_users.data[user_id]["transaction_history"]), 1)

        # 3rd request (Triplicate Attempt) -> still ignored, balance stays 6000
        res3 = self.client.post("/webhook/paystack", data=raw_body, headers=headers)
        self.assertEqual(res3.status_code, 200)
        self.assertIn("Duplicate", res3.json()["message"])
        self.assertEqual(self.mock_users.data[user_id]["wallet_balance_ngn"], 6000.0)
        self.assertEqual(len(self.mock_users.data[user_id]["transaction_history"]), 1)

    # -------------------------------------------------------------
    # Category 3: Minimum Deposit Limit Enforcement Tests
    # -------------------------------------------------------------

    async def test_adv_minimum_deposit_1000_rejected(self):
        """Test deposit of ₦1,000 is rejected with ValueError"""
        with self.assertRaises(ValueError) as ctx:
            await main.initialize_paystack_transaction("2348055556666", 1000.0)
        self.assertIn("Minimum deposit amount is ₦5,000", str(ctx.exception))

    async def test_adv_minimum_deposit_4999_rejected(self):
        """Test boundary deposit of ₦4,999 is rejected with ValueError"""
        with self.assertRaises(ValueError) as ctx:
            await main.initialize_paystack_transaction("2348055556666", 4999.0)
        self.assertIn("Minimum deposit amount is ₦5,000", str(ctx.exception))

    async def test_adv_minimum_deposit_5000_accepted(self):
        """Test deposit of exactly ₦5,000 is accepted and returns payment URL"""
        auth_url = await main.initialize_paystack_transaction("2348055556666", 5000.0)
        self.assertIsInstance(auth_url, str)
        self.assertIn("paystack", auth_url.lower())

    async def test_adv_whatsapp_deposit_command_limits(self):
        """Test WhatsApp /deposit command rejects ₦1,000 and accepts ₦5,000"""
        user_id = "2348077778888"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # /deposit 1000 -> Rejection response
        await main.process_whatsapp_message(user_id, "/deposit 1000")
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to_num, user_id)
        self.assertIn("Minimum Deposit Amount is ₦5,000", msg)

        # /deposit 5000 -> Successful payment link
        await main.process_whatsapp_message(user_id, "/deposit 5000")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertIn("To complete your deposit of *₦5,000.00*", msg)
        self.assertIn("paystack", msg.lower())

if __name__ == "__main__":
    unittest.main()
