import asyncio
import unittest
from unittest.mock import AsyncMock
import json
import re
import main

class MockUsersCol:
    def __init__(self):
        self.data = {}
        
    async def find_one(self, query):
        uid = query.get("user_id")
        if uid:
            return self.data.get(uid)
        return None
        
    async def insert_one(self, doc):
        self.data[doc["user_id"]] = doc.copy()
        
    async def update_one(self, query, update_cmd, upsert=False):
        uid = query.get("user_id")
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
                doc[k] = round(doc.get(k, 0.0) + v, 4)
                
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

class TestWhatsAppWalletCommands(unittest.IsolatedAsyncioTestCase):
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

    # 1. Test /wallet and /balance command output formatting
    async def test_wallet_and_balance_command_execution(self):
        user_id = "2348111222333"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "name": "Chidi",
            "level": "400L",
            "wallet_balance_ngn": 15000.0,
            "total_spent_ngn": 2500.0,
            "transaction_history": []
        }

        # Run /wallet command
        await main.process_whatsapp_message(user_id, "/wallet")
        self.assertTrue(len(self.sent_cloud_msgs) > 0)
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to_num, user_id)
        self.assertIn("💳 *NEURA AI Wallet*", msg)
        self.assertIn("Current Balance:* ₦15,000.00", msg)
        self.assertIn("Total Spent:* ₦2,500.00", msg)
        # est_queries = 15000 / 20 = 750
        self.assertIn("Est. Queries Remaining:* ~750", msg)

        # Run /balance alias command
        await main.process_whatsapp_message(user_id, "/balance")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to_num, user_id)
        self.assertIn("Current Balance:* ₦15,000.00", msg)

    # 2. Test /deposit and /topup menu card generation
    async def test_deposit_and_topup_card_generation(self):
        user_id = "2348111222334"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "wallet_balance_ngn": 0.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Trigger /deposit without amount
        await main.process_whatsapp_message(user_id, "/deposit")
        self.assertTrue(len(self.sent_btn_msgs) > 0)
        to_num, body, buttons = self.sent_btn_msgs[-1]
        self.assertEqual(to_num, user_id)
        self.assertIn("Select Top-Up Amount", body)
        self.assertEqual(len(buttons), 3)
        btn_ids = [b["id"] for b in buttons]
        self.assertIn("TOPUP_5000", btn_ids)
        self.assertIn("TOPUP_10000", btn_ids)
        self.assertIn("TOPUP_20000", btn_ids)

        # Trigger /topup alias without amount
        await main.process_whatsapp_message(user_id, "/topup")
        to_num, body, buttons = self.sent_btn_msgs[-1]
        self.assertIn("Select Top-Up Amount", body)

    # 3. Test /deposit with custom amounts (minimum ₦5,000 validation)
    async def test_deposit_custom_amount_validation(self):
        user_id = "2348111222335"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "wallet_balance_ngn": 500.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Deposit below 5000 -> Rejection
        await main.process_whatsapp_message(user_id, "/deposit 3000")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertIn("Minimum Deposit Amount is ₦5,000", msg)

        # Deposit >= 5000 -> Link generation
        await main.process_whatsapp_message(user_id, "/deposit 8000")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertIn("To complete your deposit of *₦8,000.00*", msg)
        self.assertIn("paystack", msg.lower())

    # 4. Test Topup Button Callbacks (TOPUP_5000, TOPUP_10000, TOPUP_20000)
    async def test_topup_button_callbacks(self):
        user_id = "2348111222336"
        self.mock_users.data[user_id] = {
            "user_id": user_id,
            "onboarding_step": "COMPLETED",
            "wallet_balance_ngn": 100.0,
            "total_spent_ngn": 0.0,
            "transaction_history": []
        }

        # Click TOPUP_5000
        await main.process_whatsapp_message(user_id, "TOPUP_5000")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertEqual(to_num, user_id)
        self.assertIn("To complete your deposit of *₦5,000.00*", msg)
        self.assertIn("paystack", msg.lower())

        # Click TOPUP_10000
        await main.process_whatsapp_message(user_id, "TOPUP_10000")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertIn("To complete your deposit of *₦10,000.00*", msg)

        # Click TOPUP_20000
        await main.process_whatsapp_message(user_id, "TOPUP_20000")
        to_num, msg = self.sent_cloud_msgs[-1]
        self.assertIn("To complete your deposit of *₦20,000.00*", msg)

if __name__ == "__main__":
    unittest.main()
