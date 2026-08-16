use crate::config::AppConfig;
use crate::models::{ListOptionItem, PaystackInitResponse, UserDoc, WalletTransaction};
use crate::whatsapp::{
    send_whatsapp_cloud_msg, send_whatsapp_cta_url_button, send_whatsapp_interactive_list,
};
use chrono::Utc;
use hmac::{Hmac, Mac};
use mongodb::bson::doc;
use mongodb::Collection;
use regex::Regex;
use serde_json::json;
use sha2::Sha512;
use tracing::{error, info};
use uuid::Uuid;

type HmacSha512 = Hmac<Sha512>;

pub const PROFIT_MULTIPLIER: f64 = 8.0;
pub const MIN_DEPOSIT_NGN: u64 = 5000;
pub const LOW_BALANCE_THRESHOLD_NGN: f64 = 20.0;

/// Initializes a Flutterwave payment and returns the hosted checkout URL.
pub async fn initialize_flutterwave_transaction(
    http: &reqwest::Client,
    secret_key: &str,
    amount_ngn: u64,
    email: &str,
    phone: &str,
    callback_url: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let url = "https://api.flutterwave.com/v3/payments";
    let reference = format!("NEURA_{}_{}", phone, Uuid::new_v4().simple());

    let payload = json!({
        "tx_ref": reference,
        "amount": amount_ngn.to_string(),
        "currency": "NGN",
        "redirect_url": callback_url,
        "customer": {
            "email": email,
            "phonenumber": phone,
            "name": "Student"
        },
        "customizations": {
            "title": "NEURA AI Wallet Top-Up",
            "description": format!("NEURA AI MBBS Study Assistant (₦{})", amount_ngn),
            "logo": "https://raw.githubusercontent.com/Samueldmh/Neura-AI/main/assets/logo.png"
        }
    });

    let res = http
        .post(url)
        .bearer_auth(secret_key.trim())
        .json(&payload)
        .send()
        .await?;

    let v: serde_json::Value = res.json().await?;
    if v.get("status").and_then(|s| s.as_str()) == Some("success") {
        if let Some(link) = v.pointer("/data/link").and_then(|l| l.as_str()) {
            return Ok(link.to_string());
        }
    }

    Err(format!("Flutterwave init failed: {:?}", v).into())
}

/// Initializes a Paystack transaction and returns the checkout URL (Fallback).
pub async fn initialize_paystack_transaction(
    http: &reqwest::Client,
    secret_key: &str,
    amount_ngn: u64,
    email: &str,
    phone: &str,
    callback_url: &str,
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let url = "https://api.paystack.co/transaction/initialize";
    let amount_kobo = amount_ngn * 100;
    let reference = format!("NEURA_{}_{}", phone, Uuid::new_v4().simple());

    let payload = json!({
        "amount": amount_kobo,
        "email": email,
        "reference": reference,
        "callback_url": callback_url,
        "metadata": {
            "phone_number": phone,
            "custom_fields": [
                {
                    "display_name": "Phone Number",
                    "variable_name": "phone_number",
                    "value": phone
                },
                {
                    "display_name": "Product",
                    "variable_name": "product",
                    "value": "NEURA AI Wallet Credit"
                }
            ]
        }
    });

    let res = http
        .post(url)
        .bearer_auth(secret_key.trim())
        .json(&payload)
        .send()
        .await?;

    let init_res: PaystackInitResponse = res.json().await?;
    if init_res.status {
        if let Some(data) = init_res.data {
            return Ok(data.authorization_url);
        }
    }

    Err(format!("Paystack init failed: {}", init_res.message).into())
}

/// Verifies Paystack HMAC-SHA512 webhook signature.
pub fn verify_paystack_hmac(secret_key: &str, payload_bytes: &[u8], signature_header: &str) -> bool {
    let mut mac = match HmacSha512::new_from_slice(secret_key.trim().as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(payload_bytes);
    let expected = hex::encode(mac.finalize().into_bytes());
    expected.eq_ignore_ascii_case(signature_header.trim())
}

/// Calculates deduction in NGN based on token count with an 8.0x Profit Multiplier (~85% margin).
pub fn calculate_query_cost_ngn(prompt_tokens: usize, completion_tokens: usize) -> f64 {
    // DeepSeek V4 Flash: $0.14 / 1M prompt, $0.28 / 1M completion
    // NGN/USD ~ 1550
    let prompt_cost_usd = (prompt_tokens as f64) * 0.00000014;
    let completion_cost_usd = (completion_tokens as f64) * 0.00000028;
    let raw_usd = prompt_cost_usd + completion_cost_usd;
    let raw_ngn = raw_usd * 1550.0;

    let marked_up = raw_ngn * PROFIT_MULTIPLIER;
    marked_up.max(1.50) // Minimum ₦1.50 per query
}

/// Sends the hybrid deposit menu with quick presets and custom amount prompt.
pub async fn send_deposit_menu(
    http: &reqwest::Client,
    config: &AppConfig,
    sender_phone: &str,
    current_balance: f64,
) {
    let body_text = format!(
        "💳 *NEURA AI Wallet Top-Up*\n\n• Current Balance: *₦{:.2}*\n• Minimum Deposit: *₦5,000*\n\nTap a quick tier below, or simply reply with any custom amount (e.g. *7500* or *15000*):",
        current_balance
    );

    let options = vec![
        ListOptionItem {
            id: "DEPOSIT_5000".to_string(),
            title: "₦5,000 Deposit".to_string(),
            description: Some("~250 In-Depth Medical Explanations".to_string()),
        },
        ListOptionItem {
            id: "DEPOSIT_10000".to_string(),
            title: "₦10,000 Deposit".to_string(),
            description: Some("~550 In-Depth Medical Explanations".to_string()),
        },
        ListOptionItem {
            id: "DEPOSIT_20000".to_string(),
            title: "₦20,000 Deposit".to_string(),
            description: Some("~1,200 In-Depth Medical Explanations".to_string()),
        },
    ];

    send_whatsapp_interactive_list(
        http,
        config,
        sender_phone,
        &body_text,
        "Select Deposit",
        &options,
    )
    .await;
}

/// Parses deposit selection or custom typed amount and sends an In-App Paystack CTA button.
pub async fn handle_deposit_request(
    http: &reqwest::Client,
    config: &AppConfig,
    users_col: &Collection<UserDoc>,
    sender_phone: &str,
    user_msg: &str,
) -> bool {
    let msg_trim = user_msg.trim();
    let upper = msg_trim.to_uppercase();

    let mut amount_ngn: Option<u64> = None;

    if upper == "DEPOSIT_5000" {
        amount_ngn = Some(5000);
    } else if upper == "DEPOSIT_10000" {
        amount_ngn = Some(10000);
    } else if upper == "DEPOSIT_20000" {
        amount_ngn = Some(20000);
    } else {
        // Check for custom amount: "/deposit 7500" or "7500" or "₦7,500"
        let re_num = Regex::new(r"^(?:/deposit\s+)?(?:₦|NGN\s*)?(\d{1,3}(?:,\d{3})*|\d+)(?:\.00)?$").unwrap();
        if let Some(caps) = re_num.captures(msg_trim) {
            let num_str = caps[1].replace(',', "");
            if let Ok(parsed) = num_str.parse::<u64>() {
                amount_ngn = Some(parsed);
            }
        }
    }

    let amt = match amount_ngn {
        Some(a) => a,
        None => return false,
    };

    if amt < MIN_DEPOSIT_NGN {
        let warning = format!(
            "⚠️ Minimum deposit amount is *₦5,000*.\n\nPlease choose ₦5,000 or more (e.g. *5000*, *7500*, *10000*)."
        );
        send_whatsapp_cloud_msg(http, config, sender_phone, &warning).await;
        return true;
    }

    let email = format!("user_{}@neura.ai", sender_phone.replace('+', ""));
    let callback_url = format!("{}/api/payment-complete", config.base_url);

    let auth_res = if !config.flutterwave_secret_key.is_empty() {
        initialize_flutterwave_transaction(http, &config.flutterwave_secret_key, amt, &email, sender_phone, &callback_url).await
    } else {
        initialize_paystack_transaction(http, &config.paystack_secret_key, amt, &email, sender_phone, &callback_url).await
    };

    match auth_res {
        Ok(auth_url) => {
            let card_body = format!(
                "💳 *NEURA AI In-App Checkout*\n\n• Amount: *₦{}*\n• Gateway: *Flutterwave*\n• Status: *Ready*\n\nTap the button below to complete your deposit directly inside WhatsApp (Supports all Nigerian Cards, Bank Transfer & USSD):",
                amt
            );
            let btn_title = format!("Pay ₦{} Now", amt);
            send_whatsapp_cta_url_button(
                http,
                config,
                sender_phone,
                &card_body,
                &btn_title,
                &auth_url,
            )
            .await;

            let _ = users_col
                .update_one(
                    doc! { "user_id": sender_phone },
                    doc! { "$set": { "awaiting_custom_deposit": false } },
                )
                .await;
        }
        Err(e) => {
            error!("Failed to initialize checkout: {}", e);
            send_whatsapp_cloud_msg(
                http,
                config,
                sender_phone,
                "Sorry, we couldn't generate the payment link right now. Please verify your payment gateway setup or try again in a moment!",
            )
            .await;
        }
    }

    true
}

/// Credits user wallet upon confirmed Paystack charge.success webhook.
pub async fn credit_user_wallet(
    http: &reqwest::Client,
    config: &AppConfig,
    users_col: &Collection<UserDoc>,
    sender_phone: &str,
    amount_kobo: u64,
    reference: &str,
) {
    let amount_ngn = amount_kobo as f64 / 100.0;
    let now_str = Utc::now().to_rfc3339();

    let tx = WalletTransaction {
        tx_id: Uuid::new_v4().to_string(),
        tx_type: "deposit".to_string(),
        amount_ngn,
        reference: Some(reference.to_string()),
        description: format!("Paystack Wallet Deposit (₦{:.2})", amount_ngn),
        timestamp: now_str,
    };

    let tx_bson = mongodb::bson::to_bson(&tx).unwrap_or(mongodb::bson::Bson::Null);

    // Idempotent credit: only update if reference not already present in transactions
    let filter = doc! {
        "user_id": sender_phone,
        "transactions.reference": { "$ne": reference }
    };

    let update = doc! {
        "$inc": { "wallet_balance_ngn": amount_ngn },
        "$push": { "transactions": tx_bson }
    };

    let res = users_col.update_one(filter, update).await;

    match res {
        Ok(update_result) if update_result.modified_count > 0 => {
            info!("✅ Successfully credited ₦{:.2} to {}", amount_ngn, sender_phone);

            // Fetch new balance
            let balance = users_col
                .find_one(doc! { "user_id": sender_phone })
                .await
                .ok()
                .flatten()
                .map(|u| u.wallet_balance_ngn)
                .unwrap_or(amount_ngn);

            let receipt = format!(
                "🎉 *PAYMENT RECEIVED!*\n\n• Amount Credited: *₦{:.2}*\n• New Wallet Balance: *₦{:.2}*\n• Ref: _{}_\n\nYou can now continue asking medical questions with full textbook grounding! 🧠⚡",
                amount_ngn, balance, reference
            );
            send_whatsapp_cloud_msg(http, config, sender_phone, &receipt).await;
        }
        Ok(_) => {
            info!("ℹ️ Transaction {} already credited (idempotent ignore)", reference);
        }
        Err(e) => {
            error!("Database error crediting wallet: {}", e);
        }
    }
}

/// Deducts query cost from user's wallet in MongoDB.
pub async fn deduct_user_wallet(
    users_col: &Collection<UserDoc>,
    sender_phone: &str,
    amount_ngn: f64,
    description: &str,
) {
    let now_str = Utc::now().to_rfc3339();
    let tx = WalletTransaction {
        tx_id: Uuid::new_v4().to_string(),
        tx_type: "query_deduction".to_string(),
        amount_ngn,
        reference: None,
        description: description.to_string(),
        timestamp: now_str,
    };

    let tx_bson = mongodb::bson::to_bson(&tx).unwrap_or(mongodb::bson::Bson::Null);

    let _ = users_col
        .update_one(
            doc! { "user_id": sender_phone },
            doc! {
                "$inc": {
                    "wallet_balance_ngn": -amount_ngn,
                    "total_spent_ngn": amount_ngn
                },
                "$push": { "transactions": tx_bson }
            },
        )
        .await;
}
