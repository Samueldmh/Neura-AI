import asyncio
import unittest
from unittest.mock import AsyncMock, patch, MagicMock
import json
import math
import sys
import httpx

# Mock Heavy ML / External Client Initializations BEFORE importing main.py
sys.modules['fastembed'] = MagicMock()
sys.modules['qdrant_client'] = MagicMock()
sys.modules['qdrant_client.models'] = MagicMock()

import main

ORIGINAL_SEND_INTERACTIVE_LIST = main.send_whatsapp_interactive_list

class MockUsersCol:
    def __init__(self):
        self.data = {}
        
    async def find_one(self, query):
        uid = query.get("user_id")
        doc = self.data.get(uid)
        return doc.copy() if doc else None
        
    async def insert_one(self, doc):
        uid = doc.get("user_id")
        self.data[uid] = doc.copy()
        
    async def update_one(self, query, update_cmd, upsert=False):
        uid = query.get("user_id")
        if uid not in self.data:
            if not upsert and "$set" not in update_cmd and "$inc" not in update_cmd:
                return type("UpdateResult", (), {"matched_count": 0, "modified_count": 0})()
            self.data[uid] = {
                "user_id": uid,
                "onboarding_step": "COMPLETED",
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

        return type("UpdateResult", (), {"matched_count": 1, "modified_count": 1})()

    async def delete_one(self, query):
        uid = query.get("user_id")
        if uid in self.data:
            del self.data[uid]

class TestAdversarialM2(unittest.IsolatedAsyncioTestCase):
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
    # REQUIREMENT 1: Floating Point Accuracy of 8.0x Profit Multiplier Formula
    # =========================================================================
    def test_floating_point_multiplier_accuracy(self):
        """
        Tests 8.0x Profit Multiplier formula over a vast spectrum of synthetic token usage values.
        Ensures exact mathematical integrity (deduction_ngn == 8.0 * actual_cost_ngn),
        no floating point drift or off-by-one errors across prime, fractional, zero, and huge values.
        """
        test_cases = [
            (0, 0),
            (1, 0),
            (0, 1),
            (1, 1),
            (7, 13),
            (123, 456),
            (999, 999),
            (1001, 2003),
            (15555, 27777),
            (100_000, 50_000),
            (1_000_000, 1_000_000),
            (12_345_678, 98_765_432),
            (100_000_000, 500_000_000),
        ]
        
        rate = 1500.0
        multiplier = 8.0
        
        for prompt_tok, comp_tok in test_cases:
            actual_ngn, deduction_ngn = main.calculate_api_cost_ngn(prompt_tok, comp_tok, usd_to_ngn=rate)
            
            # Theoretical calculation using exact floating point
            expected_input_usd = (prompt_tok / 1_000_000.0) * 0.15
            expected_output_usd = (comp_tok / 1_000_000.0) * 0.60
            expected_actual_ngn = (expected_input_usd + expected_output_usd) * rate
            expected_deduction_ngn = expected_actual_ngn * multiplier
            
            # Assert equality to 10 decimal places to catch precision loss
            self.assertAlmostEqual(actual_ngn, expected_actual_ngn, places=10, 
                                   msg=f"Failed actual_ngn accuracy for ({prompt_tok}, {comp_tok})")
            self.assertAlmostEqual(deduction_ngn, expected_deduction_ngn, places=10, 
                                   msg=f"Failed deduction_ngn accuracy for ({prompt_tok}, {comp_tok})")
            
            # Assert multiplier relationship holds exactly
            self.assertAlmostEqual(deduction_ngn, actual_ngn * 8.0, places=10,
                                   msg=f"Multiplier ratio mismatch for ({prompt_tok}, {comp_tok})")

            # Check transaction record rounding precision logic in deduct_user_wallet
            tx = asyncio.run(main.deduct_user_wallet("test_user_fp", prompt_tok, comp_tok, prompt_tok + comp_tok))
            self.assertEqual(tx["amount_ngn"], round(deduction_ngn, 4))
            self.assertEqual(tx["details"]["actual_api_cost_ngn"], round(actual_ngn, 6))

    # =========================================================================
    # REQUIREMENT 2: Command Interception Bypass Analysis
    # =========================================================================
    async def test_command_interception_bypass(self):
        """
        Tests command interception for all command variations:
        /wallet, /deposit, /profile, /reset, /update *, /feedback, topup_*
        with various casing, whitespace, leading/trailing junk, and parameters.
        Also tests if command bypass works when wallet balance is low (0.0 NGN).
        """
        user_phone = "2348000000099"
        
        # User with 0.0 balance (Low balance trigger condition) and completed onboarding
        self.mock_users.data[user_phone] = {
            "user_id": user_phone,
            "name": "Alex",
            "level": "400L",
            "onboarding_step": "COMPLETED",
            "preferred_books_list": ["Robbins Basic Pathology"],
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        commands_to_test = [
            ("/wallet", "cloud", "💳 *NEURA AI Wallet*"),
            ("/WALLET", "cloud", "💳 *NEURA AI Wallet*"),
            (" /wallet ", "cloud", "💳 *NEURA AI Wallet*"),
            ("/balance", "cloud", "💳 *NEURA AI Wallet*"),
            ("/deposit", "btn", "💳 *Select Top-Up Amount*"),
            ("/DEPOSIT", "btn", "💳 *Select Top-Up Amount*"),
            (" /deposit ", "btn", "💳 *Select Top-Up Amount*"),
            ("/topup", "btn", "💳 *Select Top-Up Amount*"),
            ("/profile", "cloud", "👤 *Your Profile*"),
            ("/PROFILE", "cloud", "👤 *Your Profile*"),
            (" /profile ", "cloud", "👤 *Your Profile*"),
            ("/reset", "btn", "✅ Your profile and chat history have been completely reset!"),
            ("/feedback", "cloud", "📝 *NEURA AI Beta Feedback Survey*"),
            ("/FEEDBACK", "cloud", "📝 *NEURA AI Beta Feedback Survey*"),
            ("/update name", "cloud", "What would you like to change your name to?"),
            ("/UPDATE NAME", "cloud", "What would you like to change your name to?"),
            ("/update level", "list", "What is your new medical class/level?"),
            ("/update books", "list", "Please select your preferred textbook"),
            ("/update", "cloud", "⚙️ *Available Update Commands:*"),
            ("/update invalid_param", "cloud", "⚙️ *Available Update Commands:*"),
            ("topup_5000", "cloud", "Top-Up"),
            ("TOPUP_10000", "cloud", "Top-Up"),
            (" topup_20000 ", "cloud", "Top-Up"),
        ]

        for cmd, msg_type, expected_text_substring in commands_to_test:
            self.sent_cloud_msgs.clear()
            self.sent_btn_msgs.clear()
            self.sent_list_msgs.clear()
            
            # Re-ensure user doc exists if reset deleted it
            if user_phone not in self.mock_users.data:
                self.mock_users.data[user_phone] = {
                    "user_id": user_phone,
                    "name": "Alex",
                    "level": "400L",
                    "onboarding_step": "COMPLETED",
                    "preferred_books_list": ["Robbins Basic Pathology"],
                    "wallet_balance_ngn": 0.0,
                    "total_spent_ngn": 0.0,
                    "transaction_history": []
                }

            await main.process_whatsapp_message(user_phone, cmd)

            # Verification that command was intercepted and low balance card was NOT shown!
            if msg_type == "cloud":
                self.assertTrue(len(self.sent_cloud_msgs) > 0, f"Expected cloud message for '{cmd}'")
                self.assertIn(expected_text_substring, self.sent_cloud_msgs[-1][1], f"Content mismatch for '{cmd}'")
            elif msg_type == "btn":
                self.assertTrue(len(self.sent_btn_msgs) > 0, f"Expected button message for '{cmd}'")
                self.assertIn(expected_text_substring, self.sent_btn_msgs[-1][1], f"Content mismatch for '{cmd}'")
            elif msg_type == "list":
                self.assertTrue(len(self.sent_list_msgs) > 0, f"Expected list message for '{cmd}'")
                self.assertIn(expected_text_substring, self.sent_list_msgs[-1][1], f"Content mismatch for '{cmd}'")

    async def test_command_bypass_adversarial_variations(self):
        """
        Adversarial test for command injection / bypass attempts that DO NOT match exact command rules.
        Tests if user queries containing command words (e.g. 'tell me about /wallet' or 'how do I update my profile?')
        are NOT intercepted as system commands and pass through to medical search/low-balance check.
        """
        user_phone = "2348000000088"
        self.mock_users.data[user_phone] = {
            "user_id": user_phone,
            "onboarding_step": "COMPLETED",
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0
        }

        non_command_inputs = [
            "what is a /wallet?",
            "how to /deposit money",
            "show me /profile info",
            "please /reset my account",
            "can I /update my books?",
            "give me /feedback on this question",
            "topup_5000_fake",
            "my topup_10000 failed"
        ]

        for text in non_command_inputs:
            self.sent_btn_msgs.clear()
            self.sent_cloud_msgs.clear()

            await main.process_whatsapp_message(user_phone, text)

            # Because balance is 0.0, non-commands MUST trigger the Low-Balance Interceptor button card!
            self.assertEqual(len(self.sent_btn_msgs), 1, f"Input '{text}' should NOT be intercepted as command, must trigger low balance card!")
            self.assertIn("Insufficient Wallet Balance", self.sent_btn_msgs[0][1])

    async def test_missing_onboarding_step_unhandled_none(self):
        """
        Adversarial test for missing onboarding_step field in user MongoDB doc.
        Detects if handle_onboarding crashes when onboarding_step is None.
        """
        user_phone = "2348000000066"
        self.mock_users.data[user_phone] = {
            "user_id": user_phone,
            # onboarding_step intentionally omitted (None)
            "wallet_balance_ngn": 100.0
        }

        try:
            await main.process_whatsapp_message(user_phone, "/profile")
            # If no exception, check if command worked
            self.assertTrue(len(self.sent_cloud_msgs) > 0)
        except AttributeError as ae:
            self.fail(f"AttributeError encountered when onboarding_step is None: {ae}")

    # =========================================================================
    # REQUIREMENT 3: Low-Balance Card Payload Meta Graph API Compliance
    # =========================================================================
    async def test_low_balance_card_payload_meta_compliance(self):
        """
        Verifies that the low-balance interception card payload complies with Meta Graph API rules:
        - Interactive type: 'button'
        - Header / Body text constraints
        - Button count <= 3
        - Button title length <= 20 chars
        - Button ID length <= 256 chars
        - No duplicate button IDs
        """
        user_phone = "2348000000077"
        self.mock_users.data[user_phone] = {
            "user_id": user_phone,
            "onboarding_step": "COMPLETED",
            "wallet_balance_ngn": 15.50 # < 20.0 NGN
        }

        self.sent_btn_msgs.clear()
        await main.process_whatsapp_message(user_phone, "What is the mechanism of action of aspirin?")

        # Assert low balance card was sent
        self.assertEqual(len(self.sent_btn_msgs), 1)
        to_num, body_text, buttons = self.sent_btn_msgs[0]

        # Test button card payload rules
        self.assertEqual(to_num, user_phone)
        self.assertIn("Insufficient Wallet Balance", body_text)
        self.assertIn("₦15.50", body_text)
        
        # Meta Graph API Rule 1: Interactive buttons must be between 1 and 3 buttons
        self.assertGreaterEqual(len(buttons), 1)
        self.assertLessEqual(len(buttons), 3)

        button_ids = set()
        for btn in buttons:
            btn_id = btn.get("id", "")
            btn_title = btn.get("title", "")

            # Meta Graph API Rule 2: Title length <= 20 characters
            self.assertTrue(len(btn_title) <= 20, f"Button title '{btn_title}' exceeds Meta 20 char limit ({len(btn_title)})")
            
            # Meta Graph API Rule 3: ID length <= 256 characters
            self.assertTrue(len(btn_id) <= 256, f"Button ID '{btn_id}' exceeds Meta limit")
            
            # Meta Graph API Rule 4: Non-empty strings
            self.assertTrue(len(btn_title.strip()) > 0, "Button title cannot be empty")
            self.assertTrue(len(btn_id.strip()) > 0, "Button ID cannot be empty")
            
            # Meta Graph API Rule 5: Unique button IDs
            self.assertNotIn(btn_id, button_ids, f"Duplicate button ID '{btn_id}' found in interactive card payload")
            button_ids.add(btn_id)

    async def test_meta_graph_api_list_payload_rules(self):
        """
        Directly tests send_whatsapp_interactive_list to ensure compliance with Meta Graph API list rules:
        - Max 10 list rows per section
        - List button text <= 20 characters
        - Row title <= 24 characters
        - Row description <= 72 characters
        - Row ID <= 200 characters
        """
        captured_payloads = []

        async def mock_post(*args, **kwargs):
            payload = kwargs.get("json")
            if payload is None and len(args) > 2:
                payload = args[2]
            captured_payloads.append(payload)
            return MagicMock(status_code=200, text='{"status":"ok"}')

        with patch.object(httpx.AsyncClient, "post", side_effect=mock_post):
            options = [
                {"id": f"OPT_{i}", "title": f"Option Title {i} Extra Long Text That Exceeds Limit", "description": "D" * 100}
                for i in range(15) # Pass 15 options to test truncation/capping
            ]

            await ORIGINAL_SEND_INTERACTIVE_LIST(
                "2348000000000",
                "Header body text",
                "Choose Option Action Button Exceeding 20 Chars",
                options
            )

            self.assertEqual(len(captured_payloads), 1)
            payload = captured_payloads[0]

            self.assertEqual(payload["messaging_product"], "whatsapp")
            self.assertEqual(payload["type"], "interactive")
            interactive = payload["interactive"]
            self.assertEqual(interactive["type"], "list")
            
            action = interactive["action"]
            # Check list action button <= 20 chars
            self.assertLessEqual(len(action["button"]), 20)
            
            rows = action["sections"][0]["rows"]
            # Check max 10 rows cap
            self.assertLessEqual(len(rows), 10)

            for row in rows:
                self.assertLessEqual(len(row["title"]), 24, f"Row title '{row['title']}' exceeds 24 char limit")
                if "description" in row:
                    self.assertLessEqual(len(row["description"]), 72, f"Row description exceeds 72 char limit")
                self.assertLessEqual(len(row["id"]), 200)

if __name__ == "__main__":
    unittest.main()
