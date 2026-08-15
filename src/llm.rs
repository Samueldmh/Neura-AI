use crate::config::AppConfig;
use crate::models::ChatMessage;
use crate::whatsapp::send_whatsapp_cloud_msg;
use eventsource_stream::Eventsource;
use futures_util::StreamExt;
use serde_json::json;
use tracing::{error, info};

pub async fn call_openrouter_llm(
    http: &reqwest::Client,
    api_key: &str,
    system_prompt: &str,
    user_prompt: &str,
    chat_history: &[ChatMessage],
) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let url = "https://openrouter.ai/api/v1/chat/completions";

    let mut messages = vec![json!({
        "role": "system",
        "content": system_prompt
    })];

    for msg in chat_history {
        messages.push(json!({
            "role": msg.role,
            "content": msg.content
        }));
    }

    messages.push(json!({
        "role": "user",
        "content": user_prompt
    }));

    let payload = json!({
        "model": "deepseek/deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2500
    });

    let res = http
        .post(url)
        .bearer_auth(api_key.trim())
        .header("HTTP-Referer", "https://neura-ai.org")
        .header("X-Title", "NEURA AI Medical Assistant")
        .json(&payload)
        .send()
        .await?;

    let json_val: serde_json::Value = res.json().await?;
    let content = json_val["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or_default()
        .to_string();

    Ok(content)
}

pub async fn rewrite_query_with_context(
    http: &reqwest::Client,
    api_key: &str,
    user_msg: &str,
    chat_history: &[ChatMessage],
) -> String {
    let system_prompt = r#"You are an expert medical search query rewriter.
Given the chat history and the user's latest follow-up question, rewrite it into a single, standalone search query that includes the exact drug, disease, or medical concept being discussed.
If the query is already complete and standalone, return it unchanged.
Return ONLY the rewritten query text, without explanations, quotes, or conversational filler.
Examples:
- History: "Tell me about Chloroquine", Query: "What is its antidote?" -> "What is the antidote for Chloroquine?"
- History: "Explain Tetralogy of Fallot", Query: "Can it be repaired surgically?" -> "Surgical repair of Tetralogy of Fallot"
- History: "Describe Phenobarbital", Query: "Side effects?" -> "Side effects of Phenobarbital""#;

    let url = "https://openrouter.ai/api/v1/chat/completions";

    let mut messages = vec![json!({
        "role": "system",
        "content": system_prompt
    })];

    for msg in chat_history.iter().rev().take(4).cloned().collect::<Vec<_>>().into_iter().rev() {
        let truncated: String = msg.content.chars().take(300).collect();
        messages.push(json!({
            "role": msg.role,
            "content": truncated
        }));
    }

    messages.push(json!({
        "role": "user",
        "content": format!("Latest question to rewrite: \"{}\"", user_msg)
    }));

    let payload = json!({
        "model": "deepseek/deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 80
    });

    match http
        .post(url)
        .bearer_auth(api_key.trim())
        .header("HTTP-Referer", "https://neura-ai.org")
        .header("X-Title", "NEURA AI Query Rewriter")
        .json(&payload)
        .send()
        .await
    {
        Ok(res) => {
            if let Ok(json_val) = res.json::<serde_json::Value>().await {
                if let Some(content) = json_val["choices"][0]["message"]["content"].as_str() {
                    let cleaned = content.trim().trim_matches('"').to_string();
                    if !cleaned.is_empty() {
                        return cleaned;
                    }
                }
            }
        }
        Err(e) => {
            error!("Error in query rewriting LLM call: {}", e);
        }
    }

    user_msg.to_string()
}

pub async fn stream_openrouter_llm_to_whatsapp(
    http: &reqwest::Client,
    config: &AppConfig,
    system_prompt: &str,
    user_prompt: &str,
    sender_phone: &str,
    chat_history: &[ChatMessage],
) -> String {
    let url = "https://openrouter.ai/api/v1/chat/completions";

    let mut messages = vec![json!({
        "role": "system",
        "content": system_prompt
    })];

    for msg in chat_history {
        messages.push(json!({
            "role": msg.role,
            "content": msg.content
        }));
    }

    messages.push(json!({
        "role": "user",
        "content": user_prompt
    }));

    let payload = json!({
        "model": "deepseek/deepseek-v4-flash",
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": 2500,
        "stream": true
    });

    let res = match http
        .post(url)
        .bearer_auth(config.openrouter_api_key.trim())
        .header("HTTP-Referer", "https://neura-ai.org")
        .header("X-Title", "NEURA AI Medical Assistant")
        .json(&payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            error!("Error initiating LLM stream: {}", e);
            let fallback = call_openrouter_llm(
                http,
                &config.openrouter_api_key,
                system_prompt,
                user_prompt,
                chat_history,
            )
            .await
            .unwrap_or_else(|_| "Sorry, I had trouble generating a response.".to_string());
            send_whatsapp_cloud_msg(http, config, sender_phone, &fallback).await;
            return fallback;
        }
    };

    let mut event_stream = res.bytes_stream().eventsource();
    let mut full_text = String::new();
    let mut current_chunk = String::new();

    while let Some(event_res) = event_stream.next().await {
        match event_res {
            Ok(event) => {
                if event.data == "[DONE]" {
                    break;
                }

                if let Ok(data_json) = serde_json::from_str::<serde_json::Value>(&event.data) {
                    if let Some(choices) = data_json.get("choices").and_then(|c| c.as_array()) {
                        if let Some(first_choice) = choices.first() {
                            if let Some(content) = first_choice["delta"]["content"].as_str() {
                                full_text.push_str(content);
                                current_chunk.push_str(content);

                                // If paragraph break and threshold exceeded, send chunk to WhatsApp
                                if current_chunk.contains("\n\n") && current_chunk.len() > 1500 {
                                    if let Some(idx) = current_chunk.rfind("\n\n") {
                                        let p1 = current_chunk[..idx].trim().to_string();
                                        let last_char = p1.chars().last().unwrap_or(' ');
                                        if (['.', '!', '?', '"', '>'].contains(&last_char) && p1.len() > 100)
                                            || current_chunk.len() > 3500
                                        {
                                            send_whatsapp_cloud_msg(http, config, sender_phone, &p1).await;
                                            current_chunk = current_chunk[idx + 2..].to_string();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            Err(e) => {
                error!("Error parsing SSE stream event: {}", e);
            }
        }
    }

    if !current_chunk.trim().is_empty() {
        send_whatsapp_cloud_msg(http, config, sender_phone, current_chunk.trim()).await;
    }

    full_text
}
