use crate::config::{get_available_books, get_curriculum, AppConfig};
use crate::llm::call_openrouter_llm;
use crate::models::{ListOptionItem, UserDoc};
use crate::whatsapp::{send_whatsapp_cloud_msg, send_whatsapp_interactive_list};
use mongodb::bson::doc;
use mongodb::Collection;
use tracing::error;

pub async fn extract_name_with_llm(
    http: &reqwest::Client,
    api_key: &str,
    user_msg: &str,
) -> Option<String> {
    let prompt = r#"Extract the person's first name from this message. 
If they just say a greeting, or "why do you need it", or if it is random gibberish (e.g., "asdfgh"), or if there is clearly no name, return NONE.
Return ONLY the name, nothing else.
Examples:
- "I am Samuel" -> Samuel
- "Samuel" -> Samuel
- "Hi my name is John" -> John
- "Why do you want to know?" -> NONE
- "Hello" -> NONE
- "dhjdsf" -> NONE"#;

    match call_openrouter_llm(http, api_key, prompt, user_msg, &[]).await {
        Ok(res) => {
            let name = res.trim().to_string();
            if name.to_uppercase() != "NONE" && !name.is_empty() {
                Some(name)
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

pub async fn send_subject_book_menu(
    http: &reqwest::Client,
    config: &AppConfig,
    users_col: &Collection<UserDoc>,
    sender_phone: &str,
    level: &str,
    subject: &str,
) -> bool {
    let user_doc = users_col
        .find_one(doc! { "user_id": sender_phone })
        .await
        .ok()
        .flatten()
        .unwrap_or_default();

    let preferred_books = user_doc.preferred_books_list;
    let available_books_map = get_available_books();
    let all_books = available_books_map
        .get(subject)
        .cloned()
        .unwrap_or_default();

    if all_books.is_empty() {
        let body_text = format!("No textbooks currently indexed for *{}*.", subject);
        let options = vec![ListOptionItem {
            id: "SKIP_SUBJECT".to_string(),
            title: "⏭️ Skip this subject".to_string(),
            description: Some("Continue to next subject".to_string()),
        }];
        send_whatsapp_interactive_list(http, config, sender_phone, &body_text, "Select Option", &options).await;
    } else if all_books.len() == 1 {
        let single_book = all_books[0];
        let body_text = format!("Please select your preferred textbook for *{}*:", subject);
        let options = vec![ListOptionItem {
            id: single_book.to_string(),
            title: single_book.chars().take(24).collect(),
            description: Some(single_book.chars().take(72).collect()),
        }];
        send_whatsapp_interactive_list(http, config, sender_phone, &body_text, "Select Textbook", &options).await;
    } else {
        let selected_for_subject: Vec<&str> = all_books
            .iter()
            .cloned()
            .filter(|b| preferred_books.contains(&b.to_string()))
            .collect();

        let mut checklist_lines = Vec::new();
        for b in &all_books {
            let b_display = b.split(':').next().unwrap_or(b).chars().take(40).collect::<String>();
            if selected_for_subject.contains(b) {
                checklist_lines.push(format!("• [✓] *{}*", b_display));
            } else {
                checklist_lines.push(format!("• [  ] {}", b_display));
            }
        }
        let checklist_str = checklist_lines.join("\n");

        let body_text = if !selected_for_subject.is_empty() {
            format!(
                "📚 *{} Textbooks* ({}/{} Selected):\n{}\n\nTap a book below to add/remove, or tap Finish when you're done!",
                subject,
                selected_for_subject.len(),
                all_books.len(),
                checklist_str
            )
        } else {
            format!(
                "📚 *{} Textbooks* (0/{} Selected):\n{}\n\nTap a textbook below to select it:",
                subject,
                all_books.len(),
                checklist_str
            )
        };

        let mut options = Vec::new();
        if !selected_for_subject.is_empty() {
            options.push(ListOptionItem {
                id: format!("FINISH_SUBJECT_{}", subject),
                title: "✅ Finish & Next Subject".to_string(),
                description: Some(format!("Proceed with {} selected book(s)", selected_for_subject.len())),
            });
        }

        for b in &all_books {
            if selected_for_subject.contains(b) {
                options.push(ListOptionItem {
                    id: format!("TOGGLE_{}", b),
                    title: format!("❌ Remove: {}", b).chars().take(24).collect(),
                    description: Some(format!("Remove {}", b).chars().take(72).collect()),
                });
            } else {
                options.push(ListOptionItem {
                    id: format!("TOGGLE_{}", b),
                    title: format!("➕ Add: {}", b).chars().take(24).collect(),
                    description: Some(format!("Add {}", b).chars().take(72).collect()),
                });
            }
        }

        if selected_for_subject.is_empty() {
            options.push(ListOptionItem {
                id: "SKIP_SUBJECT".to_string(),
                title: "⏭️ Skip this subject".to_string(),
                description: Some("Do not select any textbook for this subject".to_string()),
            });
        }

        send_whatsapp_interactive_list(http, config, sender_phone, &body_text, "Select / Toggle", &options).await;
    }

    let _ = users_col
        .update_one(
            doc! { "user_id": sender_phone },
            doc! { "$set": { "onboarding_step": format!("ASK_BOOK_{}", subject) } },
        )
        .await;

    true
}

pub async fn send_next_subject_menu(
    http: &reqwest::Client,
    config: &AppConfig,
    users_col: &Collection<UserDoc>,
    sender_phone: &str,
    level: &str,
    current_subject: Option<&str>,
) -> bool {
    let curriculum = get_curriculum();
    let subjects = match curriculum.get(level) {
        Some(s) => s,
        None => return false,
    };

    let next_subject = match current_subject {
        None => subjects.first().cloned(),
        Some(curr) => {
            if let Some(pos) = subjects.iter().position(|&s| s == curr) {
                subjects.get(pos + 1).cloned()
            } else {
                subjects.first().cloned()
            }
        }
    };

    match next_subject {
        Some(subj) => send_subject_book_menu(http, config, users_col, sender_phone, level, subj).await,
        None => false,
    }
}

pub async fn complete_onboarding(
    http: &reqwest::Client,
    config: &AppConfig,
    users_col: &Collection<UserDoc>,
    sender_phone: &str,
) {
    let user_doc = users_col
        .find_one(doc! { "user_id": sender_phone })
        .await
        .ok()
        .flatten()
        .unwrap_or_default();

    let name = user_doc.name.as_deref().unwrap_or("Student");
    let level = user_doc.level.as_deref().unwrap_or("");
    let preferred_books = user_doc.preferred_books_list;

    let _ = users_col
        .update_one(
            doc! { "user_id": sender_phone },
            doc! { "$set": { "onboarding_step": "COMPLETED" } },
        )
        .await;

    let books_summary = if !preferred_books.is_empty() {
        preferred_books
            .iter()
            .map(|b| format!("• {}", b))
            .collect::<Vec<String>>()
            .join("\n")
    } else {
        "• None selected (searching general medical knowledge)".to_string()
    };

    let final_msg = format!(
        "🎉 Awesome, {}! Your profile is all set up for *{}*.\n\n📚 *Your Selected Textbooks:*\n{}\n\nYou can now start asking me medical questions directly from these textbooks! 🧠⚡\n\n⚙️ *Quick Commands:*\n• Type */feedback* to share quick feedback\n• Type */profile* to view your profile\n• Type */update name* to change your name\n• Type */update level* to change your level\n• Type */update books* to change your textbooks\n\n💬 _Help us improve! Share 2-min anonymous beta feedback anytime: https://forms.gle/dNr7SV5EUiqiFySx5_",
        name, level, books_summary
    );

    send_whatsapp_cloud_msg(http, config, sender_phone, &final_msg).await;
}

pub async fn handle_onboarding(
    http: &reqwest::Client,
    config: &AppConfig,
    users_col: &Collection<UserDoc>,
    sender_phone: &str,
    user_msg: &str,
) -> bool {
    let user_doc = match users_col.find_one(doc! { "user_id": sender_phone }).await {
        Ok(Some(u)) => u,
        Ok(None) => {
            let new_user = UserDoc {
                user_id: sender_phone.to_string(),
                onboarding_step: Some("ASK_NAME".to_string()),
                ..Default::default()
            };
            let _ = users_col.insert_one(new_user).await;
            let welcome = "Hello! 👋 I'm *NEURA AI*, your elite medical study assistant.\n\nI can answer medical questions directly from your textbooks with exact citations, or generate practice MCQs for your MBBS exams!\n\nTo give you the best personalized study experience, what is your first name?";
            send_whatsapp_cloud_msg(http, config, sender_phone, welcome).await;
            return true;
        }
        Err(e) => {
            error!("Error fetching user: {}", e);
            return false;
        }
    };

    let step = user_doc.onboarding_step.as_deref().unwrap_or("");
    let level = user_doc.level.as_deref().unwrap_or("");

    if step == "COMPLETED" {
        let is_menu_tap = ["200L", "300L", "400L", "500L", "600L"].contains(&user_msg)
            || user_msg == "START_ONBOARDING"
            || user_msg == "🚀 Start Setup"
            || user_msg.starts_with("TOGGLE_")
            || user_msg.starts_with("FINISH_SUBJECT_")
            || user_msg.starts_with("DONE_SUBJECT_");

        if is_menu_tap {
            let warning = "⚠️ Your profile is already completed!\n\nTo change your level or textbooks, please use the profile commands:\n• Type */update level* to change your level\n• Type */update books* to change your textbooks\n• Type */reset* to start over";
            send_whatsapp_cloud_msg(http, config, sender_phone, warning).await;
            return true;
        }
        return false;
    }

    if step == "ASK_NAME" {
        match extract_name_with_llm(http, &config.openrouter_api_key, user_msg).await {
            Some(extracted) => {
                let _ = users_col
                    .update_one(
                        doc! { "user_id": sender_phone },
                        doc! { "$set": { "name": &extracted, "onboarding_step": "ASK_LEVEL" } },
                    )
                    .await;

                let prompt = format!("Nice to meet you, {}! What is your current medical class/level?", extracted);
                let options = ["200L", "300L", "400L", "500L", "600L"]
                    .iter()
                    .map(|l| ListOptionItem {
                        id: l.to_string(),
                        title: l.to_string(),
                        description: None,
                    })
                    .collect::<Vec<_>>();
                send_whatsapp_interactive_list(http, config, sender_phone, &prompt, "Select Level", &options).await;
                return true;
            }
            None => {
                send_whatsapp_cloud_msg(
                    http,
                    config,
                    sender_phone,
                    "I didn't quite catch that, or it didn't look like a real name! Please type your real first name so I know what to call you. 😊",
                )
                .await;
                return true;
            }
        }
    }

    if step == "ASK_LEVEL" {
        if !["200L", "300L", "400L", "500L", "600L"].contains(&user_msg) {
            send_whatsapp_cloud_msg(http, config, sender_phone, "Please use the menu button to select your level.").await;
            return true;
        }

        let _ = users_col
            .update_one(
                doc! { "user_id": sender_phone },
                doc! { "$set": { "level": user_msg, "preferred_books_list": [] } },
            )
            .await;

        let has_subjects = send_next_subject_menu(http, config, users_col, sender_phone, user_msg, None).await;
        if !has_subjects {
            complete_onboarding(http, config, users_col, sender_phone).await;
        }
        return true;
    }

    if step.starts_with("ASK_BOOK_") {
        let current_subject = step.replace("ASK_BOOK_", "");
        let available_books_map = get_available_books();
        let all_subject_books = available_books_map.get(current_subject.as_str()).cloned().unwrap_or_default();
        let preferred_books = user_doc.preferred_books_list.clone();

        // A. Finish / Skip
        let is_finish = [
            "SKIP_SUBJECT",
            "Skip (None available yet)",
            "⏭️ Skip this subject",
            "✅ Finish & Next Subject",
            "➡️ Next Subject",
        ]
        .contains(&user_msg)
            || user_msg.starts_with("FINISH_SUBJECT_")
            || user_msg.starts_with("DONE_SUBJECT_");

        if is_finish {
            let has_more = send_next_subject_menu(
                http,
                config,
                users_col,
                sender_phone,
                level,
                Some(&current_subject),
            )
            .await;
            if !has_more {
                complete_onboarding(http, config, users_col, sender_phone).await;
            }
            return true;
        }

        // B. Handle Toggle / Book Selection
        let mut raw_book = user_msg;
        if raw_book.starts_with("TOGGLE_") {
            raw_book = &raw_book["TOGGLE_".len()..];
        }

        let matched_book = all_subject_books.iter().find(|&&b| {
            raw_book == b || b.starts_with(raw_book) || raw_book.starts_with(&b[..b.len().min(24)])
        });

        let matched = match matched_book {
            Some(&b) => b,
            None => {
                send_whatsapp_cloud_msg(
                    http,
                    config,
                    sender_phone,
                    "Please use the menu button to select/toggle your textbook, or tap Finish.",
                )
                .await;
                return true;
            }
        };

        if all_subject_books.len() <= 1 {
            if !preferred_books.contains(&matched.to_string()) {
                let _ = users_col
                    .update_one(
                        doc! { "user_id": sender_phone },
                        doc! { "$push": { "preferred_books_list": matched } },
                    )
                    .await;
            }
            let has_more = send_next_subject_menu(
                http,
                config,
                users_col,
                sender_phone,
                level,
                Some(&current_subject),
            )
            .await;
            if !has_more {
                complete_onboarding(http, config, users_col, sender_phone).await;
            }
            return true;
        }

        // Multi-book toggle
        if preferred_books.contains(&matched.to_string()) {
            let _ = users_col
                .update_one(
                    doc! { "user_id": sender_phone },
                    doc! { "$pull": { "preferred_books_list": matched } },
                )
                .await;
        } else {
            let _ = users_col
                .update_one(
                    doc! { "user_id": sender_phone },
                    doc! { "$push": { "preferred_books_list": matched } },
                )
                .await;
        }

        send_subject_book_menu(http, config, users_col, sender_phone, level, &current_subject).await;
        return true;
    }

    false
}
