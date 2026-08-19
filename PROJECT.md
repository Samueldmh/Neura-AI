# Project: Neura-AI Medical Illustration & Diagram Engine Overhaul

## Architecture
- **Runtime**: Python 3.14 (FastAPI, httpx, motor, qdrant-client, fastembed) & Rust (Actix-web / Tokio high-performance service).
- **Prompt & Sanitization Subsystem**: `SYSTEM_MEDICAL_PROMPT`, `SYSTEM_QUIZ_PROMPT`, `format_whatsapp_text()` regex pipeline (Python & Rust).
- **Visual Retrieval Subsystem**:
  - `detect_visual_intent_modality()`: Classifies visual intent into `FLOWCHART_SCHEMATIC`, `HISTOLOGY_MICROSCOPY`, and `ANATOMICAL_MAP`.
  - `VERIFIED_MEDICAL_ATLAS`: Curated 120+ high-resolution SVG/PNG pathway flowcharts and textbook cycle schematics across 11 medical disciplines (200L–600L).
  - `retrieve_real_medical_diagram()`: Atlas-first resolution with micrograph demotion and decorated Wikimedia Commons search fallback.
  - `_reject_micrograph_candidate()`: Deterministic regex rejection filter preventing unannotated blood smears, single-cell crops, and histology stains for flowchart queries.
- **Verification Subsystem**:
  - `test_multidomain_verification_suite.py`: 50-topic multi-domain benchmark covering 200L to 600L medical subjects with 100% flowchart accuracy, zero rate limits, and natural WhatsApp formatting verification.
- **Git Branch Synchronization**:
  - Validated across `main` and `feature/pay-as-you-go`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | F1: Ban Fabricated Citations in Prompts | Update `SYSTEM_MEDICAL_PROMPT` and `SYSTEM_QUIZ_PROMPT` to strictly forbid "Figure X-Y" and robotic preambles | M1 | R1 Survey |
| 2 | F2: WhatsApp Response Sanitizer Pipeline | Implement deterministic regex-based figure citation removal and preamble stripping in `format_whatsapp_text` | M1 | R1 Survey |
| 3 | F3: Intelligent Visual Intent Router | Modality-aware intent classification (`FLOWCHART_SCHEMATIC` vs `HISTOLOGY_MICROSCOPY` vs `ANATOMICAL_MAP`) | M2 | R2/R3 Survey |
| 4 | F4: Universal Verified Medical Atlas | Expand `VERIFIED_MEDICAL_ATLAS` across 11 medical disciplines (Parasitology, Microbiology, Biochemistry, Physiology, Pharmacology, Immunology, Hematology, Surgery, O&G, Pediatrics, Internal Medicine) | M2 | R2/R3 Survey |
| 5 | F5: Micrograph Demotion & Fallback Filter | Negative regex filter rejecting micrographs/smears for flowchart queries + decorated Wikimedia fallback | M2 | R3 Survey |
| 6 | F6: 50-Topic Multi-Domain Verification Suite | Automated 50-topic benchmark suite spanning 200L–600L medical disciplines | M3 | R4 Survey |
| 7 | F7: Multi-Branch Verification & Git Sync | Commit and verify changes across `main` and `feature/pay-as-you-go` branches | M3 | R4 Survey |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Natural Conversational Delivery & Prompt Sanitization | Update prompts and `format_whatsapp_text` in Python and Rust; eliminate fabricated figure citations and robotic preambles | None | DONE |
| M2 | Universal Flowchart Engine & Micrograph Demotion Filter | Implement modality routing, expand `VERIFIED_MEDICAL_ATLAS` to 120+ entries across all 11 disciplines, implement micrograph rejection filter | M1 | DONE |
| M3 | 50-Topic Verification Suite & Multi-Branch Git Sync | Implement and run 50-topic verification suite, verify all acceptance criteria, and sync/commit across `main` and `feature/pay-as-you-go` | M2 | DONE |

## Code Layout
- `main.py`: Core FastAPI application, prompts, sanitizers, visual intent router, `VERIFIED_MEDICAL_ATLAS`, and diagram retriever.
- `src/config.rs`: Rust system prompts and configurations.
- `src/whatsapp.rs`: Rust WhatsApp message formatting and sanitization.
- `src/main.rs`: Rust RAG pipeline and message handlers.
- `test_multidomain_verification_suite.py`: Automated 50-topic verification suite.

## Interface Contracts
### Visual Intent Router ↔ Diagram Retriever
- `detect_visual_intent_modality(user_msg: str, ai_answer: str) -> str`: Returns `"FLOWCHART_SCHEMATIC"`, `"HISTOLOGY_MICROSCOPY"`, `"ANATOMICAL_MAP"`, or `"NONE"`.
- `retrieve_real_medical_diagram(topic_or_candidates, modality: str = "FLOWCHART_SCHEMATIC") -> tuple[str, str] | None`: Returns `(title, image_url)` or `None`.
- `_reject_micrograph_candidate(title: str, url: str, modality: str) -> bool`: Returns `True` if candidate should be rejected (e.g. contains `smear`, `biopsy`, `stain`, `histolog`, `micrograph` when modality is `FLOWCHART_SCHEMATIC`).

### Sanitizer Contract
- `format_whatsapp_text(text: str) -> str`:
  - Input: raw model output string.
  - Guarantees: strips `Figure X-Y` / `Fig. X.Y`, strips opening conversational/robotic preambles, converts markdown tables to structured cards, formats bold words cleanly.
