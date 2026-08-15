use crate::config::{AppConfig, SYSTEM_INTERACTIVE_QUIZ_PROMPT};
use crate::llm::call_openrouter_llm;
use crate::models::{ActiveQuiz, ListOptionItem, QuizQuestion, UserDoc};
use crate::whatsapp::{send_whatsapp_cloud_msg, send_whatsapp_interactive_list};
use mongodb::bson::doc;
use mongodb::Collection;
use qdrant_client::qdrant::ScoredPoint;
use regex::Regex;
use tracing::error;

pub async fn start_interactive_quiz(
    http: &reqwest::Client,
    config: &AppConfig,
    users_col: &Collection<UserDoc>,
    sender_phone: &str,
    topic: &str,
    search_res: &[ScoredPoint],
) {
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
    let user_prompt = format!(
        "RETRIEVED TEXTBOOK CONTEXT:\n{}\n\nTOPIC TO TEST: {}",
        formatted_context, topic
    );

    match call_openrouter_llm(
        http,
        &config.openrouter_api_key,
        SYSTEM_INTERACTIVE_QUIZ_PROMPT,
        &user_prompt,
        &[],
    )
    .await
    {
        Ok(json_raw) => {
            let re_fence = Regex::new(r"```(?:json)?\s*").unwrap();
            let mut cleaned = re_fence.replace_all(&json_raw, "").to_string();
            if let Some(idx) = cleaned.rfind("```") {
                cleaned.truncate(idx);
            }
            let cleaned = cleaned.trim();

            match serde_json::from_str::<Vec<QuizQuestion>>(cleaned) {
                Ok(questions) if !questions.is_empty() => {
                    let quiz_state = ActiveQuiz {
                        topic: topic.to_string(),
                        questions,
                        current_idx: 0,
                        score: 0,
                    };

                    let quiz_bson = match mongodb::bson::to_bson(&quiz_state) {
                        Ok(b) => b,
                        Err(e) => {
                            error!("Error converting quiz state to BSON: {}", e);
                            return;
                        }
                    };

                    let _ = users_col
                        .update_one(
                            doc! { "user_id": sender_phone },
                            doc! { "$set": { "active_quiz": quiz_bson } },
                        )
                        .await;

                    send_quiz_question(http, config, sender_phone, &quiz_state).await;
                }
                Ok(_) | Err(_) => {
                    send_whatsapp_cloud_msg(
                        http,
                        config,
                        sender_phone,
                        "Sorry, I had trouble parsing the quiz questions. Please tap '📝 Generate MCQs' to try again!",
                    )
                    .await;
                }
            }
        }
        Err(e) => {
            error!("Error generating quiz from LLM: {}", e);
            send_whatsapp_cloud_msg(
                http,
                config,
                sender_phone,
                "Sorry, I had trouble generating practice MCQs right now. Please try again in a moment!",
            )
            .await;
        }
    }
}

pub async fn send_quiz_question(
    http: &reqwest::Client,
    config: &AppConfig,
    sender_phone: &str,
    quiz_state: &ActiveQuiz,
) {
    let idx = quiz_state.current_idx;
    let total = quiz_state.questions.len();

    if idx >= total {
        let pct = (quiz_state.score as f32 / total as f32 * 100.0).round() as usize;
        let trophy = if pct >= 80 {
            "🏆 Outstanding!"
        } else if pct >= 60 {
            "👏 Good job!"
        } else {
            "💪 Keep practicing!"
        };

        let summary = format!(
            "🎯 *QUIZ COMPLETED!*\n\n{}\n• Topic: *{}*\n• Score: *{}/{}* ({}%)\n\nAsk me any medical concept to review or tap '📝 Generate MCQs' to practice another topic!",
            trophy, quiz_state.topic, quiz_state.score, total, pct
        );

        send_whatsapp_cloud_msg(http, config, sender_phone, &summary).await;
        return;
    }

    let q = &quiz_state.questions[idx];
    let q_num = idx + 1;

    let question_text = format!(
        "📝 *Question {} of {}* (Topic: {})\n\n{}\n\n*A)* {}\n*B)* {}\n*C)* {}\n*D)* {}",
        q_num, total, quiz_state.topic, q.vignette, q.option_a, q.option_b, q.option_c, q.option_d
    );

    let options_list = vec![
        ListOptionItem {
            id: format!("Q{}_A", q_num),
            title: "Option A".to_string(),
            description: Some(q.option_a.chars().take(72).collect()),
        },
        ListOptionItem {
            id: format!("Q{}_B", q_num),
            title: "Option B".to_string(),
            description: Some(q.option_b.chars().take(72).collect()),
        },
        ListOptionItem {
            id: format!("Q{}_C", q_num),
            title: "Option C".to_string(),
            description: Some(q.option_c.chars().take(72).collect()),
        },
        ListOptionItem {
            id: format!("Q{}_D", q_num),
            title: "Option D".to_string(),
            description: Some(q.option_d.chars().take(72).collect()),
        },
    ];

    send_whatsapp_interactive_list(http, config, sender_phone, &question_text, "Select Option", &options_list).await;
}

pub async fn handle_quiz_answer(
    http: &reqwest::Client,
    config: &AppConfig,
    users_col: &Collection<UserDoc>,
    sender_phone: &str,
    selected_option: &str,
    user_doc: &UserDoc,
) -> bool {
    let re_q = Regex::new(r"Q(\d+)_([A-D])").unwrap();
    let upper = selected_option.to_uppercase();

    let active_quiz = match &user_doc.active_quiz {
        Some(q) => q,
        None => {
            if re_q.is_match(&upper) {
                send_whatsapp_cloud_msg(
                    http,
                    config,
                    sender_phone,
                    "⚠️ This quiz session has already ended! To start a new practice quiz, tap '📝 Generate MCQs' under any medical answer!",
                )
                .await;
                return true;
            }
            return false;
        }
    };

    let idx = active_quiz.current_idx;
    if idx >= active_quiz.questions.len() {
        if re_q.is_match(&upper) {
            send_whatsapp_cloud_msg(
                http,
                config,
                sender_phone,
                "⚠️ This quiz session has already ended! To start a new practice quiz, tap '📝 Generate MCQs' under any medical answer!",
            )
            .await;
            return true;
        }
        return false;
    }

    let choice = if let Some(caps) = re_q.captures(&upper) {
        let tapped_q_num: usize = caps[1].parse().unwrap_or(0);
        let current_q_num = idx + 1;
        if tapped_q_num != current_q_num {
            send_whatsapp_cloud_msg(
                http,
                config,
                sender_phone,
                &format!(
                    "⚠️ You have already answered Question {}! Please select your answer for Question {} below.",
                    tapped_q_num, current_q_num
                ),
            )
            .await;
            return true;
        }
        caps[2].to_string()
    } else {
        let re_single = Regex::new(r"\b([A-D])\b").unwrap();
        match re_single.captures(&upper) {
            Some(caps) => caps[1].to_string(),
            None => return false,
        }
    };

    let q = &active_quiz.questions[idx];
    let correct = q.correct_option.trim().to_uppercase();
    let is_correct = choice == correct;

    let mut new_score = active_quiz.score;
    let feedback_header = if is_correct {
        new_score += 1;
        format!("✅ *CORRECT!* (Option {})", correct)
    } else {
        format!(
            "❌ *INCORRECT!* (Your Choice: {} | Correct Answer: Option {})",
            choice, correct
        )
    };

    let feedback_msg = format!(
        "{}\n\n📖 *Textbook Rationale ({}):*\n{}",
        feedback_header, q.book_source, q.explanation
    );
    send_whatsapp_cloud_msg(http, config, sender_phone, &feedback_msg).await;

    let mut updated_quiz = active_quiz.clone();
    updated_quiz.current_idx = idx + 1;
    updated_quiz.score = new_score;

    let quiz_bson = mongodb::bson::to_bson(&updated_quiz).unwrap_or(mongodb::bson::Bson::Null);
    let _ = users_col
        .update_one(
            doc! { "user_id": sender_phone },
            doc! { "$set": { "active_quiz": quiz_bson } },
        )
        .await;

    send_quiz_question(http, config, sender_phone, &updated_quiz).await;
    true
}
