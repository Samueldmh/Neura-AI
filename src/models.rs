use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct QueryRequest {
    pub user_id: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WebhookVerificationParams {
    #[serde(rename = "hub.mode")]
    pub mode: Option<String>,
    #[serde(rename = "hub.verify_token")]
    pub verify_token: Option<String>,
    #[serde(rename = "hub.challenge")]
    pub challenge: Option<String>,
}

// ==========================================
// WhatsApp Incoming Webhook Payload Models
// ==========================================

#[derive(Debug, Serialize, Deserialize)]
pub struct WebhookPayload {
    pub object: Option<String>,
    pub entry: Option<Vec<WebhookEntry>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WebhookEntry {
    pub id: Option<String>,
    pub changes: Option<Vec<WebhookChange>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WebhookChange {
    pub value: Option<WebhookValue>,
    pub field: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct WebhookValue {
    pub messaging_product: Option<String>,
    pub metadata: Option<serde_json::Value>,
    pub messages: Option<Vec<IncomingMessage>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct IncomingMessage {
    pub from: String,
    pub id: String,
    pub timestamp: Option<String>,
    #[serde(rename = "type")]
    pub msg_type: String,
    pub text: Option<TextBody>,
    pub interactive: Option<InteractiveBody>,
    pub context: Option<MessageContext>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TextBody {
    pub body: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MessageContext {
    pub id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InteractiveBody {
    #[serde(rename = "type")]
    pub interactive_type: String,
    pub list_reply: Option<ListReply>,
    pub button_reply: Option<ButtonReply>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ListReply {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ButtonReply {
    pub id: String,
    pub title: String,
}

// ==========================================
// MongoDB Documents & Wallet System
// ==========================================

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
pub struct UserDoc {
    pub user_id: String,
    pub name: Option<String>,
    pub level: Option<String>,
    pub onboarding_step: Option<String>,
    #[serde(default)]
    pub preferred_books_list: Vec<String>,
    pub active_quiz: Option<ActiveQuiz>,
    #[serde(default)]
    pub wallet_balance_ngn: f64,
    #[serde(default)]
    pub total_spent_ngn: f64,
    #[serde(default)]
    pub transactions: Vec<WalletTransaction>,
    pub awaiting_custom_deposit: Option<bool>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct WalletTransaction {
    pub tx_id: String,
    pub tx_type: String, // "deposit" | "query_deduction"
    pub amount_ngn: f64,
    pub reference: Option<String>,
    pub description: String,
    pub timestamp: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ActiveQuiz {
    pub topic: String,
    pub questions: Vec<QuizQuestion>,
    pub current_idx: usize,
    pub score: usize,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct QuizQuestion {
    pub q_num: usize,
    pub vignette: String,
    pub option_a: String,
    pub option_b: String,
    pub option_c: String,
    pub option_d: String,
    pub correct_option: String,
    pub explanation: String,
    pub book_source: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ChatHistoryDoc {
    pub user_id: String,
    pub messages: Vec<ChatMessage>,
}

// ==========================================
// Paystack Models
// ==========================================

#[derive(Debug, Serialize, Deserialize)]
pub struct PaystackInitResponse {
    pub status: bool,
    pub message: String,
    pub data: Option<PaystackInitData>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PaystackInitData {
    pub authorization_url: String,
    pub access_code: String,
    pub reference: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PaystackWebhookPayload {
    pub event: Option<String>,
    pub data: Option<PaystackWebhookData>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PaystackWebhookData {
    pub id: Option<u64>,
    pub amount: Option<u64>, // in Kobo
    pub reference: Option<String>,
    pub status: Option<String>,
    pub channel: Option<String>,
    pub customer: Option<PaystackCustomer>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PaystackCustomer {
    pub email: Option<String>,
    pub phone: Option<String>,
    pub customer_code: Option<String>,
}

// ==========================================
// WhatsApp Interactive Outgoing Types
// ==========================================

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ListOptionItem {
    pub id: String,
    pub title: String,
    pub description: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct ButtonOptionItem {
    pub id: String,
    pub title: String,
}
