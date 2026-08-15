use crate::config::AppConfig;
use crate::models::{ButtonOptionItem, ListOptionItem};
use regex::Regex;
use serde_json::json;
use tracing::{error, info};

pub fn format_whatsapp_text(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut result = text.to_string();

    // 1. Remove markdown hashes
    let re_hash = Regex::new(r"(?m)^\s*#{1,6}\s*").unwrap();
    result = re_hash.replace_all(&result, "").to_string();
    result = result.replace("###", "").replace("##", "");

    // 2. Fix double asterisks **text** -> *text*
    let re_double_ast = Regex::new(r"\*\*(.*?)\*\*").unwrap();
    result = re_double_ast.replace_all(&result, "*$1*").to_string();

    // Fix spaces inside asterisks: "* text *" -> "*text*"
    let re_space_ast1 = Regex::new(r"\*\s+([^\*\n]+?)\s+\*").unwrap();
    result = re_space_ast1.replace_all(&result, "*$1*").to_string();
    let re_space_ast2 = Regex::new(r"\*\s+([^\*\n]+?)\*").unwrap();
    result = re_space_ast2.replace_all(&result, "*$1*").to_string();
    let re_space_ast3 = Regex::new(r"\*([^\*\n]+?)\s+\*").unwrap();
    result = re_space_ast3.replace_all(&result, "*$1*").to_string();

    // 3. Add space around bold text if missing to prevent word merging
    let re_bold_merge1 = Regex::new(r"([a-zA-Z0-9])(\*[^\*\n]+\*)").unwrap();
    result = re_bold_merge1.replace_all(&result, "$1 $2").to_string();
    let re_bold_merge2 = Regex::new(r"(\*[^\*\n]+\*)([a-zA-Z0-9])").unwrap();
    result = re_bold_merge2.replace_all(&result, "$1 $2").to_string();

    // 4. Fix missing space after colons/commas
    let re_colon_space = Regex::new(r"(\*[^\*\n]+\*:)([^\s\n])").unwrap();
    result = re_colon_space.replace_all(&result, "$1 $2").to_string();

    // 5. Ensure space after bullet hyphens
    let re_bullet = Regex::new(r"(?m)^-([a-zA-Z0-9\*])").unwrap();
    result = re_bullet.replace_all(&result, "- $1").to_string();

    // 6. Ensure double-line spacing before list items
    let re_list_space = Regex::new(r"([^\n])\n(-|\*|[0-9]+\.)\s+").unwrap();
    result = re_list_space.replace_all(&result, "$1\n\n$2 ").to_string();

    // 7. Add newline before emojis
    let re_emoji = Regex::new(r"([^\n])([📖💡📚🎯])").unwrap();
    result = re_emoji.replace_all(&result, "$1\n\n$2").to_string();

    // 8. Clean up orphan asterisks safely per line
    let re_valid_bold = Regex::new(r"\*([^\s\*][^\*]*[^\s\*]|[^\s\*])\*").unwrap();
    let re_word_ast = Regex::new(r"([a-zA-Z0-9])\*([a-zA-Z0-9])").unwrap();

    let mut cleaned_lines = Vec::new();
    for line in result.split('\n') {
        let mut placeholders = Vec::new();
        let masked = re_valid_bold.replace_all(line, |caps: &regex::Captures| {
            placeholders.push(caps[0].to_string());
            format!("__VALID_BOLD_{}__", placeholders.len() - 1)
        });

        let mut line_str = re_word_ast.replace_all(&masked, "$1 $2").to_string();
        line_str = line_str.replace('*', "");

        for (idx, ph) in placeholders.iter().enumerate() {
            line_str = line_str.replace(&format!("__VALID_BOLD_{}__", idx), ph);
        }
        cleaned_lines.push(line_str);
    }
    result = cleaned_lines.join("\n");

    // 9. Normalize blank lines and clean lingering spaces
    let re_blank = Regex::new(r"\n{3,}").unwrap();
    result = re_blank.replace_all(&result, "\n\n").to_string();
    let re_multi_space = Regex::new(r" +").unwrap();
    result = re_multi_space.replace_all(&result, " ").to_string();
    let re_period_space = Regex::new(r" \.").unwrap();
    result = re_period_space.replace_all(&result, ".").to_string();
    let re_comma_space = Regex::new(r" ,").unwrap();
    result = re_comma_space.replace_all(&result, ",").to_string();

    result.trim().to_string()
}

pub async fn send_whatsapp_cloud_msg(
    http: &reqwest::Client,
    config: &AppConfig,
    to_number: &str,
    message_text: &str,
) {
    let sanitized = format_whatsapp_text(message_text);
    if sanitized.is_empty() {
        return;
    }

    let url = format!(
        "https://graph.facebook.com/v19.0/{}/messages",
        config.phone_number_id
    );

    // Meta WhatsApp limit is ~4096 chars. Split into ~3500 char chunks by paragraph.
    let chunks: Vec<String> = if sanitized.len() > 3500 {
        let paragraphs: Vec<&str> = sanitized.split("\n\n").collect();
        let mut result_chunks = Vec::new();
        let mut current_chunk = String::new();

        for p in paragraphs {
            if current_chunk.len() + p.len() + 2 <= 3500 {
                if !current_chunk.is_empty() {
                    current_chunk.push_str("\n\n");
                }
                current_chunk.push_str(p);
            } else {
                if !current_chunk.is_empty() {
                    result_chunks.push(current_chunk);
                }
                current_chunk = p.to_string();
            }
        }
        if !current_chunk.is_empty() {
            result_chunks.push(current_chunk);
        }
        result_chunks
    } else {
        vec![sanitized]
    };

    for chunk in chunks {
        let payload = json!({
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": to_number,
            "type": "text",
            "text": {
                "preview_url": false,
                "body": chunk
            }
        });

        match http
            .post(&url)
            .bearer_auth(config.whatsapp_token.trim())
            .json(&payload)
            .send()
            .await
        {
            Ok(res) => {
                let status = res.status();
                let text = res.text().await.unwrap_or_default();
                info!("Meta Graph API Send Status {}: {}", status, text);
            }
            Err(e) => {
                error!("Failed to send WhatsApp message: {}", e);
            }
        }
    }
}

pub async fn send_whatsapp_interactive_list(
    http: &reqwest::Client,
    config: &AppConfig,
    to_number: &str,
    body_text: &str,
    button_text: &str,
    options: &[ListOptionItem],
) {
    let sanitized_body = format_whatsapp_text(body_text);
    let url = format!(
        "https://graph.facebook.com/v19.0/{}/messages",
        config.phone_number_id
    );

    let rows: Vec<serde_json::Value> = options
        .iter()
        .take(10)
        .map(|opt| {
            let mut row = json!({
                "id": opt.id.chars().take(200).collect::<String>(),
                "title": opt.title.chars().take(24).collect::<String>().trim(),
            });
            if let Some(desc) = &opt.description {
                let d = desc.chars().take(72).collect::<String>().trim().to_string();
                if !d.is_empty() {
                    row["description"] = json!(d);
                }
            }
            row
        })
        .collect();

    let button_label = button_text.chars().take(20).collect::<String>();

    let payload = json!({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "list",
            "body": {
                "text": sanitized_body
            },
            "action": {
                "button": button_label,
                "sections": [
                    {
                        "title": "Available Options",
                        "rows": rows
                    }
                ]
            }
        }
    });

    if let Err(e) = http
        .post(&url)
        .bearer_auth(config.whatsapp_token.trim())
        .json(&payload)
        .send()
        .await
    {
        error!("Failed to send interactive list: {}", e);
    }
}

pub async fn send_whatsapp_interactive_button(
    http: &reqwest::Client,
    config: &AppConfig,
    to_number: &str,
    body_text: &str,
    buttons: &[ButtonOptionItem],
) {
    let sanitized_body = format_whatsapp_text(body_text);
    let url = format!(
        "https://graph.facebook.com/v19.0/{}/messages",
        config.phone_number_id
    );

    let button_objs: Vec<serde_json::Value> = buttons
        .iter()
        .take(3)
        .map(|b| {
            json!({
                "type": "reply",
                "reply": {
                    "id": b.id.chars().take(256).collect::<String>(),
                    "title": b.title.chars().take(20).collect::<String>().trim()
                }
            })
        })
        .collect();

    let payload = json!({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "button",
            "body": {
                "text": sanitized_body
            },
            "action": {
                "buttons": button_objs
            }
        }
    });

    if let Err(e) = http
        .post(&url)
        .bearer_auth(config.whatsapp_token.trim())
        .json(&payload)
        .send()
        .await
    {
        error!("Failed to send interactive button: {}", e);
    }
}
