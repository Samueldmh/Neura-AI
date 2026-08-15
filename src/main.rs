mod billing;
mod config;
mod llm;
mod models;
mod onboarding;
mod queue;
mod quiz;
mod rag;
mod whatsapp;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{Html, IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use config::{AppConfig, COLLECTION_NAME, SYSTEM_MEDICAL_PROMPT, SYSTEM_QUIZ_PROMPT};
use llm::{call_openrouter_llm, rewrite_query_with_context, stream_openrouter_llm_to_whatsapp};
use models::{
    ButtonOptionItem, ChatHistoryDoc, ChatMessage, QueryRequest, UserDoc, WebhookPayload,
    WebhookVerificationParams,
};
use mongodb::bson::doc;
use mongodb::{Client as MongoClient, Collection, Database};
use onboarding::{complete_onboarding, handle_onboarding, send_next_subject_menu};
use qdrant_client::client::QdrantClient;
use qdrant_client::config::QdrantConfig;
use queue::UserQueueManager;
use quiz::{handle_quiz_answer, start_interactive_quiz};
use rag::{extract_medical_terms, get_explicit_book_override, is_followup_question, RagEngine};
use serde_json::json;
use std::env;
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use tracing::{error, info};
use whatsapp::{
    format_whatsapp_text, send_whatsapp_cloud_msg, send_whatsapp_interactive_button,
    send_whatsapp_interactive_list,
};

#[derive(Clone)]
pub struct AppState {
    pub config: AppConfig,
    pub http: reqwest::Client,
    pub db: Option<Database>,
    pub users_col: Option<Collection<UserDoc>>,
    pub chat_history_col: Option<Collection<ChatHistoryDoc>>,
    pub rag: Option<RagEngine>,
    pub queue: UserQueueManager,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "neura_ai=info,tower_http=info".into()),
        )
        .init();

    let config = AppConfig::from_env();
    info!("🚀 Initializing NEURA AI Rust Backend v2.0.0");
    info!("QDRANT_URL: {}", config.qdrant_url);
    info!("QDRANT_API_KEY Present: {}", !config.qdrant_api_key.is_empty());
    info!("OPENROUTER_API_KEY Present: {}", !config.openrouter_api_key.is_empty());
    info!("PAYSTACK_SECRET_KEY Present: {}", !config.paystack_secret_key.is_empty());
    info!("PHONE_NUMBER_ID: {}", config.phone_number_id);

    let http = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(30))
        .build()
        .unwrap();

    // MongoDB connection
    let (db, users_col, chat_history_col) = if !config.mongo_uri.is_empty() {
        match MongoClient::with_uri_str(&config.mongo_uri).await {
            Ok(client) => {
                info!("✅ Connected to MongoDB successfully!");
                let database = client.database("neura_db");
                let u_col = database.collection::<UserDoc>("users");
                let c_col = database.collection::<ChatHistoryDoc>("chat_history");
                (Some(database), Some(u_col), Some(c_col))
            }
            Err(e) => {
                error!("⚠️ Failed to connect to MongoDB: {}", e);
                (None, None, None)
            }
        }
    } else {
        info!("⚠️ MONGO_URI not provided; running without persistent user storage");
        (None, None, None)
    };

    // Qdrant & FastEmbed connection
    let rag = if !config.qdrant_api_key.is_empty() {
        let mut q_cfg = QdrantConfig::from_url(&config.qdrant_url);
        q_cfg.set_api_key(&config.qdrant_api_key);
        match QdrantClient::new(Some(q_cfg)) {
            Ok(q_client) => {
                let arc_client = Arc::new(q_client);
                match RagEngine::new(arc_client) {
                    Ok(engine) => {
                        info!("✅ FastEmbed (BAAI/bge-small-en-v1.5) and Qdrant initialized!");
                        Some(engine)
                    }
                    Err(e) => {
                        error!("⚠️ FastEmbed model initialization error: {}", e);
                        None
                    }
                }
            }
            Err(e) => {
                error!("⚠️ Failed to initialize Qdrant Client: {}", e);
                None
            }
        }
    } else {
        None
    };

    let queue = UserQueueManager::new();

    let state = AppState {
        config,
        http,
        db,
        users_col,
        chat_history_col,
        rag,
        queue,
    };

    let app = Router::new()
        .route("/", get(root_handler))
        .route("/webhook", get(webhook_verify_handler).post(webhook_message_handler))
        .route("/webhook/paystack", post(paystack_webhook_handler))
        .route("/api/payment-complete", get(payment_complete_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/books", get(books_handler))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let port = env::var("PORT").unwrap_or_else(|_| "8000".to_string());
    let addr: SocketAddr = format!("0.0.0.0:{}", port).parse().unwrap();
    info!("⚡ Server running on http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn root_handler() -> Json<serde_json::Value> {
    Json(json!({
        "status": "online",
        "service": "NEURA AI Medical Backend (Rust)",
        "version": "2.0.0",
        "runtime": "Rust / Axum / Tokio",
        "billing": "Paystack In-App WebView + Dynamic Token Multiplier"
    }))
}

async fn payment_complete_handler() -> Html<&'static str> {
    Html(r#"
    <!DOCTYPE html>
    <html>
    <head>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Payment Successful - NEURA AI</title>
        <style>
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background-color: #f8fafc; color: #0f172a; text-align: center; }
            .card { background: white; padding: 32px; border-radius: 16px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1); max-width: 400px; margin: 20px; }
            .icon { font-size: 48px; margin-bottom: 16px; }
            h1 { font-size: 24px; margin: 0 0 8px; color: #16a34a; }
            p { color: #64748b; margin: 0 0 24px; font-size: 16px; line-height: 1.5; }
            .btn { display: inline-block; background: #2563eb; color: white; padding: 12px 24px; border-radius: 8px; text-decoration: none; font-weight: 600; }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="icon">✅</div>
            <h1>Payment Confirmed!</h1>
            <p>Your NEURA AI wallet has been successfully credited. You can close this window and return to WhatsApp.</p>
        </div>
    </body>
    </html>
    "#)
}

async fn paystack_webhook_handler(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let signature = headers
        .get("x-paystack-signature")
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default();

    if !billing::verify_paystack_hmac(&state.config.paystack_secret_key, &body, signature) {
        error!("❌ Paystack Webhook signature verification failed!");
        return (StatusCode::UNAUTHORIZED, "Invalid signature").into_response();
    }

    if let Ok(payload) = serde_json::from_slice::<models::PaystackWebhookPayload>(&body) {
        if payload.event.as_deref() == Some("charge.success") {
            if let Some(data) = payload.data {
                let amount = data.amount.unwrap_or(0);
                let reference = data.reference.unwrap_or_default();
                let phone = data
                    .metadata
                    .and_then(|m| {
                        m.get("phone_number")
                            .and_then(|p| p.as_str().map(|s| s.to_string()))
                    })
                    .or_else(|| data.customer.and_then(|c| c.phone))
                    .unwrap_or_default();

                if !phone.is_empty() && amount > 0 {
                    info!(
                        "💳 Confirmed payment of {} kobo from {} (Ref: {})",
                        amount, phone, reference
                    );
                    if let Some(users_col) = &state.users_col {
                        billing::credit_user_wallet(
                            &state.http,
                            &state.config,
                            users_col,
                            &phone,
                            amount,
                            &reference,
                        )
                        .await;
                    }
                }
            }
        }
    }

    (StatusCode::OK, "Webhook processed").into_response()
}

async fn webhook_verify_handler(
    State(state): State<AppState>,
    Query(params): Query<WebhookVerificationParams>,
) -> Response {
    if params.mode.as_deref() == Some("subscribe")
        && params.verify_token.as_deref() == Some(state.config.verify_token.as_str())
    {
        info!("✅ Meta Webhook Verification Successful!");
        params.challenge.unwrap_or_default().into_response()
    } else {
        error!("❌ Meta Webhook Verification Failed: Invalid Token");
        (StatusCode::FORBIDDEN, "Verification token mismatch").into_response()
    }
}

async fn webhook_message_handler(
    State(state): State<AppState>,
    Json(payload): Json<WebhookPayload>,
) -> Response {
    if let Some(entries) = payload.entry {
        for entry in entries {
            if let Some(changes) = entry.changes {
                for change in changes {
                    if let Some(val) = change.value {
                        if let Some(messages) = val.messages {
                            for msg in messages {
                                let sender_phone = msg.from;
                                let msg_type = msg.msg_type;
                                let is_tagged_reply = msg.context.and_then(|c| c.id).is_some();

                                let mut text_body = String::new();
                                if msg_type == "text" {
                                    if let Some(t) = msg.text {
                                        text_body = t.body;
                                    }
                                } else if msg_type == "interactive" {
                                    if let Some(inter) = msg.interactive {
                                        if inter.interactive_type == "list_reply" {
                                            if let Some(lr) = inter.list_reply {
                                                text_body = if !lr.id.is_empty() { lr.id } else { lr.title };
                                            }
                                        } else if inter.interactive_type == "button_reply" {
                                            if let Some(br) = inter.button_reply {
                                                text_body = if !br.id.is_empty() { br.id } else { br.title };
                                            }
                                        }
                                    }
                                }

                                if !text_body.is_empty() {
                                    info!(
                                        "📩 Received msg ({}) from {} (Tagged: {}): '{}'",
                                        msg_type, sender_phone, is_tagged_reply, text_body
                                    );
                                    let state_clone = state.clone();
                                    tokio::spawn(async move {
                                        process_whatsapp_message(state_clone, sender_phone, text_body, is_tagged_reply).await;
                                    });
                                    return Json(json!({ "status": "processing" })).into_response();
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Json(json!({ "status": "ignored" })).into_response()
}

async fn process_whatsapp_message(
    state: AppState,
    sender_phone: String,
    user_msg: String,
    _is_tagged_reply: bool,
) {
    // 1. Deduplication check (drops rapid duplicate taps within 3s)
    if state.queue.is_duplicate_action(&sender_phone, &user_msg).await {
        info!("🔇 Dropping duplicate action within 3s from {}: '{}'", sender_phone, user_msg);
        return;
    }

    // 2. Rate limiter check (30 messages per minute)
    if !state.queue.check_rate_limit(&sender_phone).await {
        send_whatsapp_cloud_msg(
            &state.http,
            &state.config,
            &sender_phone,
            "⏳ You're studying fast! Please wait a moment before sending another message.",
        ).await;
        return;
    }

    // 3. Acquire user-specific async lock (ensures sequential processing per user while other users run in parallel)
    let user_lock = state.queue.get_user_lock(&sender_phone).await;
    let _lock_guard = user_lock.lock().await;

    let mut user_doc = None;
    let mut preferred_books_list = Vec::new();
    let mut name = "Student".to_string();
    let mut level = "Unknown Level".to_string();
    let mut wallet_balance = 0.0;
    let mut total_spent = 0.0;

    if let Some(users_col) = &state.users_col {
        if let Ok(Some(u)) = users_col.find_one(doc! { "user_id": &sender_phone }).await {
            name = u.name.clone().unwrap_or_else(|| "Student".to_string());
            level = u.level.clone().unwrap_or_else(|| "Unknown Level".to_string());
            preferred_books_list = u.preferred_books_list.clone();
            wallet_balance = u.wallet_balance_ngn;
            total_spent = u.total_spent_ngn;
            user_doc = Some(u);
        }
    }

    // Profile & Wallet commands
    let msg_lower = user_msg.trim().to_lowercase();
    if msg_lower.starts_with('/')
        || msg_lower == "topup_wallet"
        || msg_lower == "start_deposit"
        || msg_lower == "clearwallet"
        || msg_lower == "deposit"
        || msg_lower == "wallet"
    {
        if let Some(users_col) = &state.users_col {
            if msg_lower == "/clearwallet"
                || msg_lower == "/clear_wallet"
                || msg_lower == "/clear wallet"
                || msg_lower == "/resetwallet"
                || msg_lower == "/reset_wallet"
                || msg_lower == "/reset wallet"
                || msg_lower == "/emptywallet"
                || msg_lower == "/empty_wallet"
                || msg_lower == "/empty wallet"
                || msg_lower == "clearwallet"
            {
                let _ = users_col
                    .update_one(
                        doc! { "user_id": &sender_phone },
                        doc! { "$set": { "wallet_balance_ngn": 0.0 } },
                    )
                    .upsert(true)
                    .await;
                send_whatsapp_cloud_msg(
                    &state.http,
                    &state.config,
                    &sender_phone,
                    "🗑️ *Wallet Cleared!*\n\nYour wallet balance has been reset to *₦0.00* for testing.\n\nType */deposit* to test depositing funds again, or ask a question to test the low-balance prompt!",
                )
                .await;
                return;
            }

            if msg_lower == "/wallet" || msg_lower == "/balance" || msg_lower == "wallet" || msg_lower == "balance" {
                let est_queries = (wallet_balance / 20.0).floor() as u64;
                let wallet_msg = format!(
                    "💳 *NEURA AI Wallet*\n\n• Available Balance: *₦{:.2}*\n• Total Spent: *₦{:.2}*\n• Estimated Queries Remaining: *~{}*\n\nType */deposit* to top up your wallet with any custom amount (min ₦5,000)!",
                    wallet_balance, total_spent, est_queries
                );
                send_whatsapp_interactive_button(
                    &state.http,
                    &state.config,
                    &sender_phone,
                    &wallet_msg,
                    &[ButtonOptionItem {
                        id: "TOPUP_WALLET".to_string(),
                        title: "💳 Deposit ₦5,000+".to_string(),
                    }],
                )
                .await;
                return;
            }

            if msg_lower == "/deposit" || msg_lower == "/topup" || msg_lower == "topup_wallet" || msg_lower == "start_deposit" || msg_lower == "deposit" || msg_lower == "topup" {
                billing::send_deposit_menu(&state.http, &state.config, &sender_phone, wallet_balance).await;
                return;
            }

            if msg_lower.starts_with("/deposit ") || msg_lower.starts_with("deposit ") {
                if billing::handle_deposit_request(&state.http, &state.config, users_col, &sender_phone, &user_msg).await {
                    return;
                }
            }

            match msg_lower.as_str() {
                "/reset" => {
                    let _ = users_col.delete_one(doc! { "user_id": &sender_phone }).await;
                    if let Some(ch_col) = &state.chat_history_col {
                        let _ = ch_col.delete_one(doc! { "user_id": &sender_phone }).await;
                    }
                    send_whatsapp_interactive_button(
                        &state.http,
                        &state.config,
                        &sender_phone,
                        "✅ Your profile and chat history have been completely reset!\n\nTap the button below to set up your profile:",
                        &[ButtonOptionItem {
                            id: "START_ONBOARDING".to_string(),
                            title: "🚀 Start Setup".to_string(),
                        }],
                    )
                    .await;
                    return;
                }
                "/profile" => {
                    let books_str = if !preferred_books_list.is_empty() {
                        preferred_books_list
                            .iter()
                            .map(|b| format!("  - {}", b))
                            .collect::<Vec<_>>()
                            .join("\n")
                    } else {
                        "  - None".to_string()
                    };
                    let profile_msg = format!(
                        "👤 *Your Profile*\n• Name: {}\n• Level: {}\n• Wallet Balance: ₦{:.2}\n• Books:\n{}\n\n📝 *Feedback Survey:* https://forms.gle/dNr7SV5EUiqiFySx5",
                        name, level, wallet_balance, books_str
                    );
                    send_whatsapp_cloud_msg(&state.http, &state.config, &sender_phone, &profile_msg).await;
                    return;
                }
                "/feedback" => {
                    let feedback = "📝 *NEURA AI Beta Feedback Survey*\n\nYour feedback helps us make NEURA AI 10x better for medical students!\n\nThis survey is 100% anonymous (takes under 2 minutes):\n👉 https://forms.gle/dNr7SV5EUiqiFySx5\n\nThank you for beta testing NEURA AI! 🧠⚡";
                    send_whatsapp_cloud_msg(&state.http, &state.config, &sender_phone, feedback).await;
                    return;
                }
                "/update name" => {
                    let _ = users_col
                        .update_one(
                            doc! { "user_id": &sender_phone },
                            doc! { "$set": { "onboarding_step": "ASK_NAME" } },
                        )
                        .await;
                    send_whatsapp_cloud_msg(&state.http, &state.config, &sender_phone, "What would you like to change your name to?").await;
                    return;
                }
                "/update level" => {
                    let _ = users_col
                        .update_one(
                            doc! { "user_id": &sender_phone },
                            doc! { "$set": { "onboarding_step": "ASK_LEVEL" } },
                        )
                        .await;
                    let options = ["200L", "300L", "400L", "500L", "600L"]
                        .iter()
                        .map(|l| models::ListOptionItem {
                            id: l.to_string(),
                            title: l.to_string(),
                            description: None,
                        })
                        .collect::<Vec<_>>();
                    send_whatsapp_interactive_list(
                        &state.http,
                        &state.config,
                        &sender_phone,
                        "What is your new medical class/level?",
                        "Select Level",
                        &options,
                    )
                    .await;
                    return;
                }
                "/update books" => {
                    let _ = users_col
                        .update_one(
                            doc! { "user_id": &sender_phone },
                            doc! { "$set": { "preferred_books_list": [] } },
                        )
                        .await;
                    let has_subjects = send_next_subject_menu(&state.http, &state.config, users_col, &sender_phone, &level, None).await;
                    if !has_subjects {
                        complete_onboarding(&state.http, &state.config, users_col, &sender_phone).await;
                    }
                    return;
                }
                _ => {}
            }
        }
    }

    // Handle In-App Paystack Deposit selection or custom amount typing
    if let Some(users_col) = &state.users_col {
        if billing::handle_deposit_request(&state.http, &state.config, users_col, &sender_phone, &user_msg).await {
            return;
        }
    }

    // Handle Quiz Answer
    if let (Some(users_col), Some(u)) = (&state.users_col, &user_doc) {
        if u.active_quiz.is_some() {
            if handle_quiz_answer(&state.http, &state.config, users_col, &sender_phone, &user_msg, u).await {
                return;
            }
        }
    }

    // Handle Onboarding State Machine
    if let Some(users_col) = &state.users_col {
        if handle_onboarding(&state.http, &state.config, users_col, &sender_phone, &user_msg).await {
            return;
        }
    }

    // Low Balance Interception
    if wallet_balance < billing::LOW_BALANCE_THRESHOLD_NGN {
        let low_bal_card = format!(
            "⚠️ *Insufficient Wallet Balance (₦{:.2})*\n\nTo continue asking clinical questions and practicing MBBS MCQs, please top up your wallet (minimum deposit is ₦5,000).\n\nTap below to deposit:",
            wallet_balance
        );
        send_whatsapp_interactive_button(
            &state.http,
            &state.config,
            &sender_phone,
            &low_bal_card,
            &[ButtonOptionItem {
                id: "TOPUP_WALLET".to_string(),
                title: "💳 Deposit ₦5,000+".to_string(),
            }],
        )
        .await;
        return;
    }

    // Fetch chat history early
    let mut history_messages = Vec::new();
    if let Some(ch_col) = &state.chat_history_col {
        if let Ok(Some(ch)) = ch_col.find_one(doc! { "user_id": &sender_phone }).await {
            history_messages = ch.messages.iter().rev().take(6).cloned().collect::<Vec<_>>();
            history_messages.reverse();
        }
    }

    // RAG and AI Medical Response Pipeline
    let mut query_to_search = user_msg.clone();
    let mut intent = "EXPLANATION";

    if user_msg.starts_with("GENERATE_QUIZ:") {
        query_to_search = user_msg.replace("GENERATE_QUIZ:", "").trim().to_string();
        intent = "QUIZ";
    }

    // Contextual Query Resolution for follow-up questions
    if !history_messages.is_empty() && is_followup_question(&query_to_search) && intent == "EXPLANATION" {
        let rewritten = rewrite_query_with_context(
            &state.http,
            &state.config.openrouter_api_key,
            &query_to_search,
            &history_messages,
        )
        .await;
        info!("🔄 Contextual Query Rewritten: '{}' -> '{}'", query_to_search, rewritten);
        query_to_search = rewritten;
    }

    let explicit_books = get_explicit_book_override(&query_to_search, &preferred_books_list);

    let search_res = if let Some(rag) = &state.rag {
        let medical_terms = extract_medical_terms(&query_to_search);
        rag.multi_search_qdrant(&medical_terms, &explicit_books).await
    } else {
        Vec::new()
    };

    if user_msg.starts_with("GENERATE_QUIZ:") {
        if let Some(users_col) = &state.users_col {
            start_interactive_quiz(
                &state.http,
                &state.config,
                users_col,
                &sender_phone,
                &query_to_search,
                &search_res,
            )
            .await;
            return;
        }
    }

    if search_res.is_empty() {
        let fallback = if !preferred_books_list.is_empty() {
            "I'm sorry, but this specific topic is not found in your currently selected textbooks.\n\nType */update books* to add more textbooks or verify the spelling!"
        } else {
            "I couldn't find relevant textbook material for your question. Please try asking a specific medical topic!"
        };
        send_whatsapp_cloud_msg(&state.http, &state.config, &sender_phone, fallback).await;
        return;
    }

    let mut context_blocks = Vec::new();
    for (idx, point) in search_res.iter().take(10).enumerate() {
        let book_str = point
            .payload
            .get("book_title")
            .and_then(|v| v.as_str())
            .unwrap_or("Textbook");
        let text_str = point
            .payload
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        context_blocks.push(format!("[Chunk {} | Book: {}]\n{}", idx + 1, book_str, text_str));
    }
    let formatted_context = context_blocks.join("\n\n");

    let base_prompt = if intent == "QUIZ" {
        SYSTEM_QUIZ_PROMPT
    } else {
        SYSTEM_MEDICAL_PROMPT
    };
    let system_prompt = base_prompt.replace("{user_context}", &format!("User Name: {}, Class: {}\n", name, level));

    let user_prompt = format!(
        "RETRIEVED TEXTBOOK CONTEXT:\n{}\n\nSTUDENT QUESTION:\n{}",
        formatted_context, user_msg
    );

    let final_response = stream_openrouter_llm_to_whatsapp(
        &state.http,
        &state.config,
        &system_prompt,
        &user_prompt,
        &sender_phone,
        &history_messages,
    )
    .await;

    // Dynamic Token Billing Deduction (8.0x Profit Multiplier)
    if let Some(users_col) = &state.users_col {
        let est_prompt_tokens = 1500 + (user_prompt.len() / 4);
        let est_completion_tokens = final_response.len() / 4;
        let query_charge_ngn = billing::calculate_query_cost_ngn(est_prompt_tokens, est_completion_tokens);

        billing::deduct_user_wallet(
            users_col,
            &sender_phone,
            query_charge_ngn,
            "Medical Clinical Query / RAG Explanation",
        )
        .await;

        info!(
            "💰 Deducted ₦{:.2} from {} (Prompt: {} tok, Compl: {} tok, ~85% net margin)",
            query_charge_ngn, sender_phone, est_prompt_tokens, est_completion_tokens
        );
    }

    // Send follow-up interactive buttons
    let short_topic: String = query_to_search.chars().take(20).collect();
    let buttons = vec![
        ButtonOptionItem {
            id: format!("GENERATE_QUIZ:{}", short_topic),
            title: "📝 Generate MCQs".to_string(),
        },
        ButtonOptionItem {
            id: format!("CLINICAL_SOCRATIC:{}", short_topic),
            title: "💡 Socratic Question".to_string(),
        },
    ];

    send_whatsapp_interactive_button(
        &state.http,
        &state.config,
        &sender_phone,
        "Would you like to practice exam questions on this topic?",
        &buttons,
    )
    .await;

    // Update chat history in MongoDB
    if let Some(ch_col) = &state.chat_history_col {
        let user_entry = ChatMessage {
            role: "user".to_string(),
            content: user_msg,
        };
        let bot_entry = ChatMessage {
            role: "assistant".to_string(),
            content: final_response,
        };

        let _ = ch_col
            .update_one(
                doc! { "user_id": &sender_phone },
                doc! {
                    "$push": {
                        "messages": {
                            "$each": [
                                mongodb::bson::to_bson(&user_entry).unwrap(),
                                mongodb::bson::to_bson(&bot_entry).unwrap(),
                            ],
                            "$slice": -10
                        }
                    }
                },
            )
            .upsert(true)
            .await;
    }
}

async fn chat_handler(
    State(state): State<AppState>,
    Json(req): Json<QueryRequest>,
) -> Json<serde_json::Value> {
    let search_res = if let Some(rag) = &state.rag {
        let terms = extract_medical_terms(&req.message);
        rag.multi_search_qdrant(&terms, &[]).await
    } else {
        Vec::new()
    };

    if search_res.is_empty() {
        return Json(json!({
            "response": "I couldn't find relevant textbook material for your question. Please try asking a specific medical topic!"
        }));
    }

    let mut context_blocks = Vec::new();
    for (idx, point) in search_res.iter().take(10).enumerate() {
        let book_str = point
            .payload
            .get("book_title")
            .and_then(|v| v.as_str())
            .unwrap_or("Textbook");
        let text_str = point
            .payload
            .get("text")
            .and_then(|v| v.as_str())
            .unwrap_or_default();
        context_blocks.push(format!("[Context {} | Book: {}]\n{}", idx + 1, book_str, text_str));
    }

    let system_prompt = SYSTEM_MEDICAL_PROMPT.replace("{user_context}", "");
    let user_prompt = format!(
        "RETRIEVED TEXTBOOK CONTEXT:\n{}\n\nSTUDENT QUESTION:\n{}",
        context_blocks.join("\n\n"),
        req.message
    );

    let res = call_openrouter_llm(&state.http, &state.config.openrouter_api_key, &system_prompt, &user_prompt, &[])
        .await
        .unwrap_or_else(|e| format!("Error calling AI: {}", e));

    Json(json!({ "response": format_whatsapp_text(&res) }))
}

async fn books_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let books = config::get_available_books();
    Json(json!({
        "collection": COLLECTION_NAME,
        "available_books": books
    }))
}
