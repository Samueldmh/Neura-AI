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
// MongoDB Documents
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
