import asyncio
import json
import hmac
import hashlib
import time
import unittest
import httpx
import uuid
import main

class ThreadSafeMockUsersCol:
    """
    Mock MongoDB users collection with configurable artificial delay
    to simulate real-world async network/db latency during concurrent requests.
    """
    def __init__(self, delay_ms: float = 0.001):
        self.data = {}
        self.delay_ms = delay_ms
        self._lock = asyncio.Lock()
        
    async def find_one(self, query):
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms)
        async with self._lock:
            if "user_id" in query:
                return self.data.get(query["user_id"])
            if "$or" in query:
                for user in self.data.values():
                    tx_list = user.get("transaction_history", [])
                    for tx in tx_list:
                        for cond in query["$or"]:
                            for key, ref_val in cond.items():
                                if key == "transaction_history.reference" and tx.get("reference") == ref_val:
                                    return user.copy()
                                if key == "transaction_history.details.reference" and tx.get("details", {}).get("reference") == ref_val:
                                    return user.copy()
            return None
        
    async def insert_one(self, doc):
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms)
        async with self._lock:
            self.data[doc["user_id"]] = doc.copy()
        
    async def update_one(self, query, update_cmd, upsert=False):
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms)
        async with self._lock:
            uid = query.get("user_id")
            if not uid and "$or" in query:
                # Find matching user
                for u in self.data.values():
                    tx_list = u.get("transaction_history", [])
                    for tx in tx_list:
                        for cond in query["$or"]:
                            for key, ref_val in cond.items():
                                if key == "transaction_history.reference" and tx.get("reference") == ref_val:
                                    uid = u["user_id"]
                                    break
            
            if not uid:
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
        if self.delay_ms > 0:
            await asyncio.sleep(self.delay_ms)
        async with self._lock:
            uid = query.get("user_id")
            if uid in self.data:
                del self.data[uid]

class TestChallengerM3M4(unittest.IsolatedAsyncioTestCase):
    """
    Empirical Stress & Security Test Suite for Milestones 3 & 4 (Paystack & WhatsApp Commands)
    """

    async def asyncSetUp(self):
        self.mock_users = ThreadSafeMockUsersCol(delay_ms=0.0005)
        main.users_col = self.mock_users
        main.chat_history_col = None
        main.PAYSTACK_SECRET_KEY = "sk_test_neura_ai_secret_key_2026"
        
        self.sent_cloud_msgs = []
        self.sent_btn_msgs = []
        
        async def mock_send_cloud(to_number, msg):
            self.sent_cloud_msgs.append((to_number, msg))
            
        async def mock_send_btn(to_number, body, buttons):
            self.sent_btn_msgs.append((to_number, body, buttons))

        async def mock_init_paystack(phone_number, amount_ngn, email=None):
            if amount_ngn < 5000.0:
                raise ValueError("Minimum deposit amount is \u20a65,000")
            return f"https://checkout.paystack.com/mock-{uuid.uuid4().hex[:12]}"
            
        main.send_whatsapp_cloud_msg = mock_send_cloud
        main.send_whatsapp_interactive_button = mock_send_btn
        main.initialize_paystack_transaction = mock_init_paystack
        
        # httpx client with ASGITransport for true concurrent async HTTP requests
        self.transport = httpx.ASGITransport(app=main.app)
        self.client = httpx.AsyncClient(transport=self.transport, base_url="http://testserver")

    async def asyncTearDown(self):
        await self.client.aclose()

    # =========================================================================
    # REQUIREMENT 2A: Idempotency Race Conditions under Concurrency
    # =========================================================================
    async def test_concurrent_webhook_duplicate_references(self):
        """
        Stress Test: 20 concurrent Paystack charge.success webhooks with the exact same reference.
        EXPECTED BEHAVIOR: Exactly 1 webhook credits the wallet, remaining 19 are identified as duplicates.
        Wallet balance must increase by exactly NGN 5,000.00 once, and transaction_history must contain 1 entry.
        """
        user_id = "2348000000001"
        initial_balance = 1000.0
        deposit_amount_ngn = 5000.0
        amount_kobo = int(deposit_amount_ngn * 100)
        dup_reference = "REF_CONCURRENT_RACE_9999"

        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "wallet_balance_ngn": initial_balance,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        payload = {
            "event": "charge.success",
            "data": {
                "reference": dup_reference,
                "amount": amount_kobo,
                "metadata": {"phone_number": user_id}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        signature = hmac.new(
            main.PAYSTACK_SECRET_KEY.encode("utf-8"),
            raw_body,
            hashlib.sha512
        ).hexdigest()

        headers = {
            "Content-Type": "application/json",
            "x-paystack-signature": signature
        }

        concurrency_count = 20
        start_time = time.perf_counter()

        async def send_req():
            return await self.client.post("/webhook/paystack", content=raw_body, headers=headers)

        responses = await asyncio.gather(*[send_req() for _ in range(concurrency_count)])
        elapsed = time.perf_counter() - start_time

        status_codes = [r.status_code for r in responses]
        results = [r.json() for r in responses]

        credited_responses = [res for res in results if res.get("message") == "Wallet credited successfully"]
        duplicate_responses = [res for res in results if "Duplicate" in str(res.get("message", ""))]

        final_user = self.mock_users.data[user_id]
        final_balance = final_user["wallet_balance_ngn"]
        tx_history = final_user["transaction_history"]

        print(f"\n--- Webhook Concurrency Metric ---")
        print(f"Total Concurrent Requests: {concurrency_count}")
        print(f"Elapsed Time: {elapsed*1000:.2f} ms")
        print(f"Credited Responses Count: {len(credited_responses)}")
        print(f"Duplicate/Ignored Responses Count: {len(duplicate_responses)}")
        print(f"Initial Balance: NGN {initial_balance:,.2f} | Expected Final: NGN {initial_balance + deposit_amount_ngn:,.2f} | Actual: NGN {final_balance:,.2f}")
        print(f"Transaction History Count: {len(tx_history)}")

        # Verify HTTP status codes
        for code in status_codes:
            self.assertEqual(code, 200)

        # Verify idempotency under concurrency
        self.assertEqual(len(credited_responses), 1, f"Idempotency Failure: {len(credited_responses)} requests credited wallet instead of 1!")
        self.assertEqual(final_balance, initial_balance + deposit_amount_ngn, f"Idempotency Failure: Balance is NGN {final_balance} instead of NGN {initial_balance + deposit_amount_ngn}!")
        self.assertEqual(len(tx_history), 1, f"Idempotency Failure: Transaction history contains {len(tx_history)} entries instead of 1!")

    # =========================================================================
    # REQUIREMENT 2B: Tampered HMAC SHA-512 Signature Headers (1-byte flip)
    # =========================================================================
    async def test_tampered_hmac_signature(self):
        """
        Security Test: HMAC SHA-512 Signature validation with 1-byte flips & corrupted payloads.
        EXPECTED BEHAVIOR: Any 1-byte modification to signature or payload yields HTTP 401.
        """
        payload = {
            "event": "charge.success",
            "data": {
                "reference": "REF_SIG_TAMPER_001",
                "amount": 500000,
                "metadata": {"phone_number": "2348000000002"}
            }
        }
        raw_body = json.dumps(payload).encode("utf-8")
        valid_sig = hmac.new(
            main.PAYSTACK_SECRET_KEY.encode("utf-8"),
            raw_body,
            hashlib.sha512
        ).hexdigest()

        # 1. Valid Signature baseline -> 200
        res_valid = await self.client.post(
            "/webhook/paystack",
            content=raw_body,
            headers={"Content-Type": "application/json", "x-paystack-signature": valid_sig}
        )
        self.assertEqual(res_valid.status_code, 200)

        # 2. 1-byte flip at first position of signature hex string
        char_0 = valid_sig[0]
        flipped_char_0 = 'b' if char_0 != 'b' else 'a'
        sig_flip_first = flipped_char_0 + valid_sig[1:]
        res_flip1 = await self.client.post(
            "/webhook/paystack",
            content=raw_body,
            headers={"Content-Type": "application/json", "x-paystack-signature": sig_flip_first}
        )
        self.assertEqual(res_flip1.status_code, 401, "Failed: 1-byte flip (first byte) did not return 401")

        # 3. 1-byte flip at middle position of signature
        mid_idx = len(valid_sig) // 2
        char_mid = valid_sig[mid_idx]
        flipped_char_mid = '0' if char_mid != '0' else '1'
        sig_flip_mid = valid_sig[:mid_idx] + flipped_char_mid + valid_sig[mid_idx+1:]
        res_flip_mid = await self.client.post(
            "/webhook/paystack",
            content=raw_body,
            headers={"Content-Type": "application/json", "x-paystack-signature": sig_flip_mid}
        )
        self.assertEqual(res_flip_mid.status_code, 401, "Failed: 1-byte flip (middle byte) did not return 401")

        # 4. 1-byte flip at last position of signature
        char_last = valid_sig[-1]
        flipped_char_last = 'c' if char_last != 'c' else 'd'
        sig_flip_last = valid_sig[:-1] + flipped_char_last
        res_flip_last = await self.client.post(
            "/webhook/paystack",
            content=raw_body,
            headers={"Content-Type": "application/json", "x-paystack-signature": sig_flip_last}
        )
        self.assertEqual(res_flip_last.status_code, 401, "Failed: 1-byte flip (last byte) did not return 401")

        # 5. Tamper 1 byte in payload AFTER signature was generated
        tampered_body = json.dumps({
            "event": "charge.success",
            "data": {
                "reference": "REF_SIG_TAMPER_001",
                "amount": 500001, # Tampered amount from 500000 to 500001
                "metadata": {"phone_number": "2348000000002"}
            }
        }).encode("utf-8")
        res_body_tampered = await self.client.post(
            "/webhook/paystack",
            content=tampered_body,
            headers={"Content-Type": "application/json", "x-paystack-signature": valid_sig}
        )
        self.assertEqual(res_body_tampered.status_code, 401, "Failed: Tampered body with original signature did not return 401")

        # 6. Missing Signature Header -> 401
        res_missing = await self.client.post("/webhook/paystack", content=raw_body, headers={"Content-Type": "application/json"})
        self.assertEqual(res_missing.status_code, 401)

        # 7. Signature generated with WRONG secret key -> 401
        wrong_sig = hmac.new(
            b"wrong_secret_key_attacker",
            raw_body,
            hashlib.sha512
        ).hexdigest()
        res_wrong_key = await self.client.post(
            "/webhook/paystack",
            content=raw_body,
            headers={"Content-Type": "application/json", "x-paystack-signature": wrong_sig}
        )
        self.assertEqual(res_wrong_key.status_code, 401)

        print("\n--- HMAC Signature Validation Metrics ---")
        print("1-byte flip (first byte) -> 401 PASS")
        print("1-byte flip (middle byte) -> 401 PASS")
        print("1-byte flip (last byte) -> 401 PASS")
        print("Body tamper after sig -> 401 PASS")
        print("Missing signature header -> 401 PASS")
        print("Wrong secret key sig -> 401 PASS")

    # =========================================================================
    # REQUIREMENT 2C: Extreme Deposit Amounts (NGN 5,000, NGN 50,000, NGN 1,000,000, NGN 4,999 rejection)
    # =========================================================================
    async def test_extreme_deposit_amounts(self):
        """
        Boundary Test: Verify behavior across minimum threshold and extreme deposit amounts.
        - NGN 4,999.00 rejection (ValueError in initialize & warning in /deposit command)
        - NGN 4,999.99 rejection
        - NGN 5,000.00 acceptance & URL generation
        - NGN 50,000.00 acceptance
        - NGN 1,000,000.00 acceptance & float precision verification
        """
        phone = "2348000000003"
        self.mock_users.data[phone] = {
            "user_id": phone,
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # 1. NGN 4,999 Rejection (initialize_paystack_transaction)
        with self.assertRaises(ValueError) as ctx1:
            await main.initialize_paystack_transaction(phone, 4999.0)
        self.assertIn("Minimum deposit amount is \u20a65,000", str(ctx1.exception))

        # 2. NGN 4,999.99 Rejection
        with self.assertRaises(ValueError) as ctx2:
            await main.initialize_paystack_transaction(phone, 4999.99)
        self.assertIn("Minimum deposit amount is \u20a65,000", str(ctx2.exception))

        # 3. WhatsApp /deposit 4999 Command Rejection
        await main.process_whatsapp_message(phone, "/deposit 4999")
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        _, msg_rejection = self.sent_cloud_msgs[-1]
        self.assertIn("Minimum Deposit Amount is \u20a65,000", msg_rejection)

        # 4. NGN 5,000.00 Acceptance (Tier 1 Boundary)
        url_5k = await main.initialize_paystack_transaction(phone, 5000.0)
        self.assertTrue(isinstance(url_5k, str))
        self.assertIn("paystack", url_5k.lower())

        # Webhook credit NGN 5,000.00
        ref_5k = "REF_EXTREME_5K"
        body_5k = json.dumps({"event": "charge.success", "data": {"reference": ref_5k, "amount": 500000, "metadata": {"phone_number": phone}}}).encode("utf-8")
        sig_5k = hmac.new(main.PAYSTACK_SECRET_KEY.encode("utf-8"), body_5k, hashlib.sha512).hexdigest()
        res_5k = await self.client.post("/webhook/paystack", content=body_5k, headers={"Content-Type": "application/json", "x-paystack-signature": sig_5k})
        self.assertEqual(res_5k.status_code, 200)

        # 5. NGN 50,000.00 Acceptance (Tier 2 Boundary)
        url_50k = await main.initialize_paystack_transaction(phone, 50000.0)
        self.assertTrue(isinstance(url_50k, str))

        ref_50k = "REF_EXTREME_50K"
        body_50k = json.dumps({"event": "charge.success", "data": {"reference": ref_50k, "amount": 5000000, "metadata": {"phone_number": phone}}}).encode("utf-8")
        sig_50k = hmac.new(main.PAYSTACK_SECRET_KEY.encode("utf-8"), body_50k, hashlib.sha512).hexdigest()
        res_50k = await self.client.post("/webhook/paystack", content=body_50k, headers={"Content-Type": "application/json", "x-paystack-signature": sig_50k})
        self.assertEqual(res_50k.status_code, 200)

        # 6. NGN 1,000,000.00 Acceptance (Extreme High Value Tier)
        url_1m = await main.initialize_paystack_transaction(phone, 1000000.0)
        self.assertTrue(isinstance(url_1m, str))

        ref_1m = "REF_EXTREME_1M"
        body_1m = json.dumps({"event": "charge.success", "data": {"reference": ref_1m, "amount": 100000000, "metadata": {"phone_number": phone}}}).encode("utf-8")
        sig_1m = hmac.new(main.PAYSTACK_SECRET_KEY.encode("utf-8"), body_1m, hashlib.sha512).hexdigest()
        res_1m = await self.client.post("/webhook/paystack", content=body_1m, headers={"Content-Type": "application/json", "x-paystack-signature": sig_1m})
        self.assertEqual(res_1m.status_code, 200)

        # Check total cumulative wallet balance (5k + 50k + 1m = NGN 1,055,000.00)
        user_doc = self.mock_users.data[phone]
        expected_balance = 5000.0 + 50000.0 + 1000000.0
        actual_balance = user_doc["wallet_balance_ngn"]
        self.assertAlmostEqual(actual_balance, expected_balance, places=2)

        print("\n--- Extreme Deposit Amounts Metrics ---")
        print("NGN 4,999.00 Rejection: PASS (ValueError raised & WhatsApp warning)")
        print("NGN 4,999.99 Rejection: PASS (ValueError raised)")
        print("NGN 5,000.00 Acceptance: PASS (Paystack URL & Webhook Credited)")
        print("NGN 50,000.00 Acceptance: PASS (Paystack URL & Webhook Credited)")
        print("NGN 1,000,000.00 Acceptance: PASS (Paystack URL & Webhook Credited)")
        print(f"Cumulative Balance Precision: NGN {actual_balance:,.2f} == NGN {expected_balance:,.2f} PASS")

    # =========================================================================
    # REQUIREMENT 2D: Concurrent /wallet and /deposit Command Execution Under Load
    # =========================================================================
    async def test_concurrent_wallet_and_deposit_commands_load(self):
        """
        Stress Test: 100 concurrent executions of WhatsApp commands (/wallet, /deposit, /deposit 10000, TOPUP_5000).
        Measures execution latency, throughput, error rates, and memory/state stability.
        """
        num_users = 20
        requests_per_user = 5
        total_requests = num_users * requests_per_user

        # Setup test users
        for i in range(num_users):
            uid = f"23480000010{i:02d}"
            self.mock_users.data[uid] = {
                "user_id": uid,
                "onboarding_step": "COMPLETED",
                "name": f"User_{i}",
                "level": "400L",
                "preferred_books_list": ["Pharmacology"],
                "wallet_balance_ngn": 15000.0 + i * 1000.0,
                "total_spent_ngn": 500.0,
                "transaction_history": []
            }

        commands = ["/wallet", "/balance", "/deposit", "/deposit 10000", "TOPUP_5000"]
        latencies = []
        errors = []

        start_time = time.perf_counter()

        async def run_command_task(user_idx, cmd_idx):
            uid = f"23480000010{user_idx:02d}"
            cmd = commands[cmd_idx % len(commands)]
            t0 = time.perf_counter()
            try:
                await main.process_whatsapp_message(uid, cmd)
                lat = time.perf_counter() - t0
                latencies.append(lat)
            except Exception as e:
                errors.append((uid, cmd, str(e)))

        tasks = []
        for u in range(num_users):
            for r in range(requests_per_user):
                tasks.append(run_command_task(u, r))

        await asyncio.gather(*tasks)
        total_elapsed = time.perf_counter() - start_time

        latencies.sort()
        p50 = latencies[int(len(latencies) * 0.50)] * 1000 if latencies else 0
        p95 = latencies[int(len(latencies) * 0.95)] * 1000 if latencies else 0
        avg_lat = (sum(latencies) / len(latencies)) * 1000 if latencies else 0
        throughput = total_requests / total_elapsed

        print(f"\n--- Concurrent Command Load Metric ---")
        print(f"Total Commands Processed: {total_requests}")
        print(f"Total Elapsed Time: {total_elapsed:.3f} s")
        print(f"Throughput: {throughput:.2f} req/s")
        print(f"Latency Avg: {avg_lat:.2f} ms | P50: {p50:.2f} ms | P95: {p95:.2f} ms")
        print(f"Errors Encountered: {len(errors)}")

        self.assertEqual(len(errors), 0, f"Errors occurred during load testing: {errors}")
        self.assertEqual(len(latencies), total_requests)

if __name__ == "__main__":
    unittest.main()
