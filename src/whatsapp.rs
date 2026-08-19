use crate::config::AppConfig;
use crate::models::{ButtonOptionItem, ListOptionItem};
use regex::Regex;
use serde_json::json;
use tracing::{error, info};

pub fn convert_markdown_tables_to_whatsapp_cards(text: &str) -> String {
    let lines: Vec<&str> = text.split('\n').collect();
    let mut output_lines: Vec<String> = Vec::new();
    let mut i = 0;

    let re_delimiter = Regex::new(r"^[\-\s:]+$").unwrap();
    let re_hr = Regex::new(r"^\s*[\-_]{3,}\s*$").unwrap();

    while i < lines.len() {
        let line = lines[i].trim();
        if line.starts_with('|') && line.ends_with('|') && line.chars().filter(|&c| c == '|').count() >= 2 {
            let mut table_rows: Vec<Vec<String>> = Vec::new();
            while i < lines.len() {
                let row_raw = lines[i].trim();
                if row_raw.starts_with('|') && row_raw.ends_with('|') {
                    let parts: Vec<&str> = row_raw.split('|').collect();
                    if parts.len() >= 3 {
                        let cells: Vec<String> = parts[1..parts.len() - 1]
                            .iter()
                            .map(|c| c.trim().to_string())
                            .collect();
                        let is_delimiter = cells.iter().all(|c| c.is_empty() || re_delimiter.is_match(c));
                        if !is_delimiter {
                            table_rows.push(cells);
                        }
                    }
                    i += 1;
                } else {
                    break;
                }
            }

            if !table_rows.is_empty() {
                let headers = table_rows[0].clone();
                let data_rows = if table_rows.len() > 1 {
                    &table_rows[1..]
                } else {
                    &[]
                };
                output_lines.push(String::new());
                for r in data_rows {
                    if r.is_empty() || r.iter().all(|c| c.is_empty()) {
                        continue;
                    }
                    let primary_title = &r[0];
                    output_lines.push(format!("- *{}*", primary_title));
                    for c_idx in 1..std::cmp::min(headers.len(), r.len()) {
                        let default_col = format!("Detail {}", c_idx);
                        let col_name = if c_idx < headers.len() && !headers[c_idx].is_empty() {
                            &headers[c_idx]
                        } else {
                            &default_col
                        };
                        let val = &r[c_idx];
                        if !val.is_empty() {
                            output_lines.push(format!("  • *{}:* {}", col_name, val));
                        }
                    }
                    output_lines.push(String::new());
                }
            }
            continue;
        } else {
            if re_hr.is_match(line) {
                output_lines.push(String::new());
            } else {
                output_lines.push(lines[i].to_string());
            }
            i += 1;
        }
    }
    output_lines.join("\n")
}

pub fn strip_conversational_preambles(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let preamble_patterns = [
        r"(?i)^\s*(?:[\*#_]+\s*)?(?:certainly|absolutely|sure|sure thing|of course|hello|hi|hey|greetings)[\s\w,!.*#_]*?(?:here is|here are|i(?:'ve| have)\s+attached|let(?:'s| us)\s+(?:dive into|examine|look at|explore)|below is|as requested)[^\n]*(?:\n+|\:\s*|\.\s*)",
        r"(?i)^\s*(?:[\*#_]+\s*)?(?:certainly|absolutely|sure|sure thing|of course|hello|hi|hey|greetings)[,\s!\*#_.]+(?:\w+[,\s!\*#_.]+){0,2}(?:\n+|$)",
        r"(?i)^\s*(?:[\*#_]+\s*)?(?:here is|here are|below is|i have attached|i've attached|i am attaching|i'm attaching)\s+[^\n]*(?:\n+|\:\s*|\.\s*)",
        r"(?i)^\s*(?:[\*#_]+\s*)?(?:based on (?:the )?(?:retrieved )?(?:textbook )?(?:context|material|information)[^\n,.]*|according to (?:the )?(?:retrieved )?(?:textbook )?(?:context|material|information)[^\n,.]*)[\*#_\s:,.]*(?:\n+|$)?",
    ];

    let regexes: Vec<Regex> = preamble_patterns
        .iter()
        .map(|pat| Regex::new(pat).unwrap())
        .collect();

    let mut current = text.to_string();
    let mut modified = true;

    while modified {
        modified = false;
        for re in &regexes {
            if let Some(mat) = re.find(&current) {
                if mat.start() == 0 {
                    let new_text = current[mat.end()..].trim_start().to_string();
                    if new_text != current {
                        current = new_text;
                        modified = true;
                        break;
                    }
                }
            }
        }
    }

    current
}

pub fn strip_figure_citations(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    let mut result = text.to_string();

    // 1. Trailing figure/page tags in citations section (e.g. "- Robbins Pathology, Figure 12.8" -> "- Robbins Pathology")
    let re_cit = Regex::new(r"(?mi)^(-\s+[^\n,]+?)(?:,\s*(?:figure|fig\.?|table|plate|chart|p\.|page)[^\r\n]*)$").unwrap();
    result = re_cit.replace_all(&result, "$1").to_string();

    // 2. Parenthetical citations
    let re_paren = Regex::new(r"(?i)\(\s*(?:see\s+|refer to\s+|as shown in\s+|as seen in\s+|shown in\s+)?(?:figure|fig\.?|table|plate|chart)\s+\d+[-.:]\d+[a-z]?(?:\s+(?:from|in|of)\s+[A-Za-z\s]+)?(?:\s+(?:below|above))?(?:\s+for\s+[^)\n]*)?\s*\)").unwrap();
    result = re_paren.replace_all(&result, "").to_string();

    // 3. Introductory / transitional clauses
    let re_intro = Regex::new(r"(?i)\b(?:as\s+(?:shown|illustrated|seen|depicted|noted)\s+(?:in|on)\s+)?(?:figure|fig\.?|table|plate|chart)\s+\d+[-.:]\d+[a-z]?(?:\s+(?:from|in|of)\s+[A-Za-z\s]+)?(?:\s+(?:below|above))?(?:,\s*|\s+demonstrates\s+|\s+shows\s+|\s+illustrates\s+)").unwrap();
    result = re_intro.replace_all(&result, " ").to_string();

    // 4. Directive phrases / sentences
    let re_directive = Regex::new(r"(?i)\b(?:refer to|see)\s+(?:figure|fig\.?|table|plate|chart)\s+\d+[-.:]\d+[a-z]?(?:\s+(?:from|in|of)\s+[A-Za-z\s]+)?(?:\s+(?:below|above))?(?:\s+for\s+[^\n.]*)?").unwrap();
    result = re_directive.replace_all(&result, "").to_string();

    // 5. Standalone figure mentions
    let re_standalone = Regex::new(r"(?i)\b(?:figure|fig\.?|table|plate|chart)\s+\d+[-.:]\d+[a-z]?(?:\s+(?:from|in|of)\s+[A-Za-z]+)?").unwrap();
    result = re_standalone.replace_all(&result, "").to_string();

    // Cleanup whitespace & punctuation artifacts
    let re_multi_space = Regex::new(r" +").unwrap();
    result = re_multi_space.replace_all(&result, " ").to_string();
    let re_period_space = Regex::new(r" \.").unwrap();
    result = re_period_space.replace_all(&result, ".").to_string();
    let re_comma_space = Regex::new(r" ,").unwrap();
    result = re_comma_space.replace_all(&result, ",").to_string();
    let re_empty_paren = Regex::new(r"\(\s*\)").unwrap();
    result = re_empty_paren.replace_all(&result, "").to_string();

    let re_empty_line = Regex::new(r"^\s*[.,:;!?\- ]*\s*$").unwrap();
    let cleaned_lines: Vec<&str> = result
        .split('\n')
        .filter(|line| {
            if line.trim().starts_with('-') {
                true
            } else {
                !re_empty_line.is_match(line)
            }
        })
        .collect();

    cleaned_lines.join("\n")
}

pub fn format_whatsapp_text(text: &str) -> String {
    if text.is_empty() {
        return String::new();
    }

    // 0. Convert raw markdown tables to readable bullet cards & remove --- lines
    let mut result = convert_markdown_tables_to_whatsapp_cards(text);

    // 1. Deterministically strip opening preambles & conversational greetings
    result = strip_conversational_preambles(&result);

    // 2. Deterministically strip fabricated figure/table citations
    result = strip_figure_citations(&result);

    // 3. Remove markdown hashes
    let re_hash = Regex::new(r"(?m)^\s*#{1,6}\s*").unwrap();
    result = re_hash.replace_all(&result, "").to_string();
    result = result.replace("###", "").replace("##", "");

    // 4. Fix double asterisks **text** -> *text*
    let re_double_ast = Regex::new(r"\*\*(.*?)\*\*").unwrap();
    result = re_double_ast.replace_all(&result, "*$1*").to_string();

    // Fix spaces inside asterisks: "* text *" -> "*text*"
    let re_space_ast1 = Regex::new(r"\*\s+([^\*\n]+?)\s+\*").unwrap();
    result = re_space_ast1.replace_all(&result, "*$1*").to_string();
    let re_space_ast2 = Regex::new(r"\*\s+([^\*\n]+?)\*").unwrap();
    result = re_space_ast2.replace_all(&result, "*$1*").to_string();
    let re_space_ast3 = Regex::new(r"\*([^\*\n]+?)\s+\*").unwrap();
    result = re_space_ast3.replace_all(&result, "*$1*").to_string();

    // 5. Add space around bold text if missing to prevent word merging
    let re_bold_merge1 = Regex::new(r"([a-zA-Z0-9])(\*[^\*\n]+\*)").unwrap();
    result = re_bold_merge1.replace_all(&result, "$1 $2").to_string();
    let re_bold_merge2 = Regex::new(r"(\*[^\*\n]+\*)([a-zA-Z0-9])").unwrap();
    result = re_bold_merge2.replace_all(&result, "$1 $2").to_string();

    // 6. Fix missing space after colons/commas
    let re_colon_space = Regex::new(r"(\*[^\*\n]+\*:)([^\s\n])").unwrap();
    result = re_colon_space.replace_all(&result, "$1 $2").to_string();

    // 7. Ensure space after bullet hyphens
    let re_bullet = Regex::new(r"(?m)^-([a-zA-Z0-9\*])").unwrap();
    result = re_bullet.replace_all(&result, "- $1").to_string();

    // 8. Ensure double-line spacing before list items
    let re_list_space = Regex::new(r"([^\n])\n(-|\*|[0-9]+\.)\s+").unwrap();
    result = re_list_space.replace_all(&result, "$1\n\n$2 ").to_string();

    // 9. Add newline before emojis
    let re_emoji = Regex::new(r"([^\n])([📖💡📚🎯🔑])").unwrap();
    result = re_emoji.replace_all(&result, "$1\n\n$2").to_string();

    // 10. Clean up orphan asterisks safely per line
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

    // 11. Normalize blank lines and clean lingering spaces
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

pub async fn send_whatsapp_cta_url_button(
    http: &reqwest::Client,
    config: &AppConfig,
    to_number: &str,
    body_text: &str,
    button_label: &str,
    url_target: &str,
) {
    let sanitized_body = format_whatsapp_text(body_text);
    let url = format!(
        "https://graph.facebook.com/v19.0/{}/messages",
        config.phone_number_id
    );

    let display_title = button_label.chars().take(20).collect::<String>();

    let payload = json!({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "interactive",
        "interactive": {
            "type": "cta_url",
            "body": {
                "text": sanitized_body
            },
            "action": {
                "name": "cta_url",
                "parameters": {
                    "display_text": display_title,
                    "url": url_target
                }
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
        error!("Failed to send CTA URL button: {}", e);
    }
}

pub async fn send_whatsapp_image_url(
    http: &reqwest::Client,
    config: &AppConfig,
    to_number: &str,
    image_url: &str,
    caption: &str,
) {
    let url = format!(
        "https://graph.facebook.com/v19.0/{}/messages",
        config.phone_number_id
    );

    let payload = json!({
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "image",
        "image": {
            "link": image_url,
            "caption": caption
        }
    });

    if let Err(e) = http
        .post(&url)
        .bearer_auth(config.whatsapp_token.trim())
        .json(&payload)
        .send()
        .await
    {
        error!("Failed to send image: {}", e);
    }
}
