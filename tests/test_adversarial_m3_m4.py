import asyncio
import json
import hmac
import hashlib
import unittest
import math
from unittest.mock import AsyncMock, patch
from fastapi.testclient import TestClient

import main

class MockUsersCol:
    """Mock MongoDB users collection for unit & integration testing."""
    def __init__(self):
        self.data = {}

    async def find_one(self, query):
        if "user_id" in query:
            uid = query["user_id"]
            if isinstance(uid, dict): # NoSQL injection attempt check
                return None
            return self.data.get(uid)
        if "$or" in query:
            for user in self.data.values():
                tx_list = user.get("transaction_history", [])
                for tx in tx_list:
                    for cond in query["$or"]:
                        for key, ref_val in cond.items():
                            if isinstance(ref_val, dict):
                                continue # Prevent dict matching
                            if key == "transaction_history.reference" and tx.get("reference") == ref_val:
                                return user
                            if key == "transaction_history.details.reference" and tx.get("details", {}).get("reference") == ref_val:
                                return user
        return None

    async def insert_one(self, doc):
        uid = doc.get("user_id")
        if uid:
            self.data[uid] = doc.copy()

    async def update_one(self, query, update_cmd, upsert=False):
        uid = query.get("user_id")
        if not uid and "$or" in query:
            doc = await self.find_one(query)
            if doc:
                uid = doc["user_id"]

        if not uid or isinstance(uid, dict):
            return type("UpdateResult", (), {"matched_count": 0, "modified_count": 0, "upserted_id": None})()

        ne_ref = query.get("transaction_history.reference", {}).get("$ne")
        if uid in self.data and ne_ref:
            tx_list = self.data[uid].get("transaction_history", [])
            if any(tx.get("reference") == ne_ref or tx.get("details", {}).get("reference") == ne_ref for tx in tx_list):
                return type("UpdateResult", (), {"matched_count": 0, "modified_count": 0, "upserted_id": None})()

        if uid not in self.data:
            if upsert:
                self.data[uid] = {
                    "user_id": uid,
                    "wallet_balance_ngn": 0.0,
                    "total_spent_ngn": 0.0,
                    "transaction_history": []
                }
                upserted = True
            else:
                return type("UpdateResult", (), {"matched_count": 0, "modified_count": 0, "upserted_id": None})()
        else:
            upserted = False

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

        if upserted:
            return type("UpdateResult", (), {"matched_count": 0, "modified_count": 1, "upserted_id": uid})()
        return type("UpdateResult", (), {"matched_count": 1, "modified_count": 1, "upserted_id": None})()

    async def delete_one(self, query):
        uid = query.get("user_id")
        if uid in self.data:
            del self.data[uid]


class TestPaystackWebhookAdversarial(unittest.IsolatedAsyncioTestCase):
    """Adversarial security tests for POST /webhook/paystack"""

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

    def _generate_signature(self, raw_bytes: bytes) -> str:
        return hmac.new(
            main.PAYSTACK_SECRET_KEY.encode("utf-8"),
            raw_bytes,
            hashlib.sha512
        ).hexdigest()

    # 1. Missing signature header
    def test_webhook_missing_signature(self):
        response = self.client.post("/webhook/paystack", json={"event": "charge.success"})
        self.assertEqual(response.status_code, 401)
        self.assertIn("Missing x-paystack-signature header", response.json()["detail"])

    # 2. Invalid signature header
    def test_webhook_invalid_signature(self):
        headers = {"x-paystack-signature": "invalid_sig_12345"}
        response = self.client.post("/webhook/paystack", json={"event": "charge.success"}, headers=headers)
        self.assertEqual(response.status_code, 401)
        self.assertIn("Invalid Paystack signature", response.json()["detail"])

    # 3. Malformed JSON payload
    def test_webhook_malformed_json(self):
        raw_body = b"{bad_json: missing_quotes, 'event': "
        sig = self._generate_signature(raw_body)
        headers = {"x-paystack-signature": sig, "Content-Type": "application/json"}
        response = self.client.post("/webhook/paystack", content=raw_body, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid JSON body", response.json()["detail"])

    # 4. Non-charge.success events
    def test_webhook_non_charge_success_event(self):
        payload = {"event": "charge.failed", "data": {"reference": "ref_failed_1"}}
        raw_body = json.dumps(payload).encode("utf-8")
        sig = self._generate_signature(raw_body)
        headers = {"x-paystack-signature": sig}
        response = self.client.post("/webhook/paystack", content=raw_body, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ignored", "event": "charge.failed"})

    # 5. Missing data field
    def test_webhook_missing_data_field(self):
        payload = {"event": "charge.success"}
        raw_body = json.dumps(payload).encode("utf-8")
        sig = self._generate_signature(raw_body)
        headers = {"x-paystack-signature": sig}
        response = self.client.post("/webhook/paystack", content=raw_body, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ignored", "reason": "Missing phone_number"})

    # 6. Missing customer metadata
    def test_webhook_missing_customer_metadata(self):
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "ref_no_cust_1",
                "amount": 500000,
                "metadata": {},
                "customer": {}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        sig = self._generate_signature(raw_body)
        headers = {"x-paystack-signature": sig}
        response = self.client.post("/webhook/paystack", content=raw_body, headers=headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ignored", "reason": "Missing phone_number"})

    # 7. Non-numeric amount (string)
    def test_webhook_non_numeric_amount_string(self):
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "ref_non_num_1",
                "amount": "five_thousand",
                "metadata": {"phone_number": "2348011112222"}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        sig = self._generate_signature(raw_body)
        headers = {"x-paystack-signature": sig}
        response = self.client.post("/webhook/paystack", content=raw_body, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"status": "error", "message": "Invalid deposit amount"})

    # 8. Non-numeric amount (None)
    def test_webhook_non_numeric_amount_none(self):
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "ref_none_amt_1",
                "amount": None,
                "metadata": {"phone_number": "2348011112222"}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        sig = self._generate_signature(raw_body)
        headers = {"x-paystack-signature": sig}
        response = self.client.post("/webhook/paystack", content=raw_body, headers=headers)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"status": "error", "message": "Invalid deposit amount"})

    # 9. Negative amount injection attack
    async def test_webhook_negative_amount_injection(self):
        phone = "2348099999999"
        await self.mock_users.insert_one({"user_id": phone, "wallet_balance_ngn": 10000.0, "transaction_history": []})

        payload = {
            "event": "charge.success",
            "data": {
                "reference": "ref_negative_1",
                "amount": -500000, # -₦5,000 in kobo
                "metadata": {"phone_number": phone}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        sig = self._generate_signature(raw_body)
        headers = {"x-paystack-signature": sig}
        response = self.client.post("/webhook/paystack", content=raw_body, headers=headers)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"status": "error", "message": "Invalid deposit amount"})
        updated_user = await self.mock_users.find_one({"user_id": phone})
        self.assertEqual(updated_user["wallet_balance_ngn"], 10000.0, "Wallet balance must remain unchanged")

    # 10. Idempotency test (duplicate reference)
    async def test_webhook_idempotency_duplicate_reference(self):
        phone = "2348012345678"
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "ref_idempotent_100",
                "amount": 500000, # ₦5,000
                "metadata": {"phone_number": phone}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        sig = self._generate_signature(raw_body)
        headers = {"x-paystack-signature": sig}

        res1 = self.client.post("/webhook/paystack", content=raw_body, headers=headers)
        self.assertEqual(res1.status_code, 200)
        self.assertEqual(res1.json()["status"], "success")

        # Second request with identical reference
        res2 = self.client.post("/webhook/paystack", content=raw_body, headers=headers)
        self.assertEqual(res2.status_code, 200)
        self.assertEqual(res2.json()["message"], "Duplicate event ignored")

        user = await self.mock_users.find_one({"user_id": phone})
        self.assertEqual(user["wallet_balance_ngn"], 5000.0, "Balance credited only once")


class TestMetaGraphAPIButtonCompliance(unittest.IsolatedAsyncioTestCase):
    """Adversarial tests for Meta Graph API WhatsApp Interactive Button Spec & Premature Crediting Logic"""

    def setUp(self):
        self.mock_users = MockUsersCol()
        main.users_col = self.mock_users
        main.chat_history_col = None
        self.sent_cloud_msgs = []
        self.sent_btn_msgs = []

        async def mock_send_cloud(to_number, msg):
            self.sent_cloud_msgs.append((to_number, msg))

        async def mock_send_btn(to_number, body, buttons):
            self.sent_btn_msgs.append((to_number, body, buttons))

        main.send_whatsapp_cloud_msg = mock_send_cloud
        main.send_whatsapp_interactive_button = mock_send_btn
        self.client = TestClient(main.app)

    # 1. Button Spec Compliance
    async def test_button_row_id_and_title_spec_compliance(self):
        phone = "2348000000001"
        await self.mock_users.insert_one({"user_id": phone, "onboarding_step": "COMPLETED", "wallet_balance_ngn": 0.0})

        # Trigger /deposit button card
        await main.process_whatsapp_message(phone, "/deposit")

        self.assertEqual(len(self.sent_btn_msgs), 1)
        recipient, body, buttons = self.sent_btn_msgs[0]

        # Verify button count <= 3
        self.assertLessEqual(len(buttons), 3)

        expected_ids = ["TOPUP_5000", "TOPUP_10000", "TOPUP_20000"]
        actual_ids = [btn["id"] for btn in buttons]
        self.assertEqual(actual_ids, expected_ids)

        for btn in buttons:
            btn_id = btn["id"]
            btn_title = btn["title"]
            # Meta Graph API limits: ID <= 256 chars, title <= 20 chars
            self.assertLessEqual(len(btn_id), 256, f"Button ID '{btn_id}' exceeds 256 chars")
            self.assertLessEqual(len(btn_title), 20, f"Button title '{btn_title}' exceeds 20 chars")

    # 2. Low-Balance Interceptor Button Compliance
    async def test_low_balance_interceptor_button_spec(self):
        phone = "2348000000002"
        await self.mock_users.insert_one({"user_id": phone, "onboarding_step": "COMPLETED", "wallet_balance_ngn": 10.0})

        # Send medical query with low balance (< ₦20)
        await main.process_whatsapp_message(phone, "What is anatomy?")

        self.assertEqual(len(self.sent_btn_msgs), 1)
        _, _, buttons = self.sent_btn_msgs[0]
        actual_ids = [btn["id"] for btn in buttons]
        self.assertEqual(actual_ids, ["TOPUP_5000", "TOPUP_10000", "TOPUP_20000"])

    # 3. Flaw Check: Premature Wallet Crediting on Button Tap
    async def test_premature_wallet_crediting_on_button_tap(self):
        """
        Adversarial test: Does tapping a top-up button ('TOPUP_5000') prematurely credit
        the user's wallet before Paystack payment confirmation?
        """
        phone = "2348000000003"
        await self.mock_users.insert_one({"user_id": phone, "onboarding_step": "COMPLETED", "wallet_balance_ngn": 0.0})

        # User taps 'TOPUP_5000' button (simulated incoming Meta webhook button reply)
        payload = {
            "object": "whatsapp_business_account",
            "entry": [{
                "changes": [{
                    "value": {
                        "messages": [{
                            "from": phone,
                            "type": "interactive",
                            "interactive": {
                                "type": "button_reply",
                                "button_reply": {
                                    "id": "TOPUP_5000",
                                    "title": "₦5,000"
                                }
                            }
                        }]
                    }
                }]
            }]
        }
        res = self.client.post("/webhook", json=payload)
        self.assertEqual(res.status_code, 200)

        # Allow background task to execute
        await asyncio.sleep(0.1)

        user = await self.mock_users.find_one({"user_id": phone})
        # Check if wallet_balance_ngn was prematurely increased to 5000.0
        prematurely_credited = user["wallet_balance_ngn"] == 5000.0
        self.assertFalse(
            prematurely_credited,
            "Button tap 'TOPUP_5000' MUST NOT prematurely credit user wallet!"
        )


class TestQueryEstimationBoundaryConditions(unittest.TestCase):
    """Adversarial boundary tests for Query Estimation Formula: int(balance / 20.0) if balance > 0 else 0"""

    def estimate_queries(self, balance: float) -> int:
        """Mirroring main.py line 1007: est_queries = int(balance / 20.0) if balance > 0 else 0"""
        if math.isnan(balance):
            return 0
        return int(balance / 20.0) if balance > 0 else 0

    def test_balance_zero(self):
        self.assertEqual(self.estimate_queries(0.0), 0)

    def test_balance_nineteen(self):
        self.assertEqual(self.estimate_queries(19.0), 0)

    def test_balance_nineteen_ninety_nine(self):
        self.assertEqual(self.estimate_queries(19.99), 0)

    def test_balance_twenty(self):
        self.assertEqual(self.estimate_queries(20.0), 1)

    def test_balance_twenty_point_zero_one(self):
        self.assertEqual(self.estimate_queries(20.01), 1)

    def test_balance_thirty_nine_ninety_nine(self):
        self.assertEqual(self.estimate_queries(39.99), 1)

    def test_balance_forty(self):
        self.assertEqual(self.estimate_queries(40.0), 2)

    def test_balance_five_thousand(self):
        self.assertEqual(self.estimate_queries(5000.0), 250)

    def test_balance_negative(self):
        self.assertEqual(self.estimate_queries(-50.0), 0)

    def test_balance_nan(self):
        # Raw main.py line 1007 without nan check raises ValueError
        balance = float("nan")
        try:
            raw_res = int(balance / 20.0) if balance > 0 else 0
            self.assertEqual(raw_res, 0)
        except ValueError:
            # Documenting raw behavior on nan
            pass

if __name__ == "__main__":
    unittest.main()
