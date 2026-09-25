# Ranviar (ranviar.org) — Landing Page Architecture & Design Specification
**The Official Web Showcase & Conversion Engine for Africa's Leading Medical AI WhatsApp Co-Pilot**

---

## 📌 Executive Overview & Design Brief

- **Product Name**: Ranviar *(formerly Neura AI)*
- **Official Web Domain**: [https://ranviar.org](https://ranviar.org)
- **Product Type**: Conversational Multimodal Clinical AI Co-Pilot on WhatsApp (powered by Meta Cloud API)
- **Target Audience**: Nigerian & African MBBS Medical Students (200L Pre-Clinical to 600L Final Year / Clinical Rotations), House Officers, and Medical Residents.
- **Core Value Proposition**: An always-accessible clinical study companion inside WhatsApp. Powered by 37+ accredited gold-standard medical textbooks, personal lecture slide vaults (up to 500 slides / 200MB with Gemini 2.5 Flash Vision OCR), authentic diagnostic flowcharts, clinical vision (ECG, histology, X-rays), voice note tutoring, and exam-grade vignette MCQs.
- **Founding Team**: 
  - **Samuel** *(Founder & Lead AI Systems Architect)*
  - **Augustine** *(Co-Founder & Head of Clinical Curriculum & Medical Education)*
  - **Maazi Milk** *(Co-Founder & Head of Product Strategy, Growth & Community)*

---

## 🎨 Design System & Developer Tokens

Frontend engineers building the landing page (using **Next.js 14 / Astro / Tailwind CSS / Framer Motion / Lucide Icons**) must follow these design tokens:

### 1. Color Palette
```css
:root {
  /* Brand Primary */
  --color-primary-base: #10B981;       /* Ranviar Emerald */
  --color-primary-hover: #059669;      /* Deep Emerald */
  --color-primary-light: #D1FAE5;      /* Mint Tint */
  --color-primary-dark: #064E3B;       /* Forest Slate */

  /* Clinical Accent */
  --color-accent-teal: #0D9488;        /* Stethoscope Teal */
  --color-accent-cyan: #06B6D4;        /* Medical Cyan */
  --color-accent-amber: #F59E0B;       /* High-Yield Alert / Warning */
  --color-accent-rose: #F43F5E;        /* Critical / Heart Rate Alert */

  /* Dark Theme Surfaces (Default Dark Clinical Canvas) */
  --color-bg-canvas: #090D16;          /* Ultra-deep clinical navy/slate */
  --color-bg-surface: #111827;         /* High-contrast container surface */
  --color-bg-card: #1F2937;            /* Frosted glass / Card background */
  --color-border-subtle: #374151;      /* 1px subtle divider */
  --color-border-glow: rgba(16, 185, 129, 0.35); /* Neon emerald active border */

  /* WhatsApp Native Accents */
  --color-wa-bubble-sent: #005C4B;     /* User sent message dark bubble */
  --color-wa-bubble-received: #202C33; /* Ranviar incoming dark bubble */
  --color-wa-online: #25D366;          /* Live status indicator */
}
```

### 2. Typography Hierarchy
- **Display Font**: `Plus Jakarta Sans`, `Cabinet Grotesk`, or `Inter` (weights: 700, 800, 900)
- **Body Font**: `Inter` or `Geist Sans` (weights: 400, 500, 600)
- **Code / Medical Parameters / Data**: `JetBrains Mono` or `Fira Code`

---

## 🗺️ Complete Information Architecture & Section Wireframes

```
┌────────────────────────────────────────────────────────────────────────┐
│ 0. Sticky Header Navigation (Logo, Features, Books, Pricing, CTA)       │
├────────────────────────────────────────────────────────────────────────┤
│ 1. Hero Section (H1, Subhead, WhatsApp CTAs, Live Chat Simulator)      │
├────────────────────────────────────────────────────────────────────────┤
│ 2. Social Proof & Trust Ticker (37+ Textbooks, Top Medical Colleges)  │
├────────────────────────────────────────────────────────────────────────┤
│ 3. The MBBS Struggle vs. The Ranviar WhatsApp Solution                 │
├────────────────────────────────────────────────────────────────────────┤
│ 4. 6 Core Pillars (Deep-Dive Feature Cards with Interactive UI)        │
│    • Pillar 1: 37+ Accredited Gold-Standard Textbooks                  │
│    • Pillar 2: 500-Page / 200MB Lecture Vault with AI Vision OCR       │
│    • Pillar 3: Clinical Multimodal Vision (ECGs, Histology, X-rays)    │
│    • Pillar 4: Authentic Pathway Flowcharts & Algorithmic Schematics   │
│    • Pillar 5: Voice Notes & Hands-Free Audio Tutoring                 │
│    • Pillar 6: MBBS Clinical Vignette MCQs & Diagnostic Drills        │
├────────────────────────────────────────────────────────────────────────┤
│ 5. How It Works (3 Steps: Zero Apps, 100% WhatsApp)                   │
├────────────────────────────────────────────────────────────────────────┤
│ 6. Level-by-Level Curriculum Matrix (200L to 600L MBBS)               │
├────────────────────────────────────────────────────────────────────────┤
│ 7. Interactive Live WhatsApp Simulator Playground                      │
├────────────────────────────────────────────────────────────────────────┤
│ 8. Student Testimonials & Ward-Round Success Stories                   │
├────────────────────────────────────────────────────────────────────────┤
│ 9. Paystack Student Pricing & Wallet Plans                            │
├────────────────────────────────────────────────────────────────────────┤
│ 10. Built By — Meet the Founders (Samuel, Augustine, Maazi Milk)       │
├────────────────────────────────────────────────────────────────────────┤
│ 11. Frequently Asked Questions (FAQ)                                   │
├────────────────────────────────────────────────────────────────────────┤
│ 12. High-Conversion Sticky Footer & Final CTA Banner                   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 SECTION-BY-SECTION COPYWRITING & UI SPECIFICATIONS

---

### Section 0: Sticky Navigation Header
- **Left**:
  - Ranviar Logo (Emerald stylized neural synapse + medical caduceus cross icon)
  - Text: **Ranviar** `🩺⚡`
  - Small badge: `v2.5 Live`
- **Center Navigation Links**:
  - Features
  - 37+ Textbooks
  - Lecture Vault (OCR)
  - Curriculum (200L–600L)
  - The Builders
  - Pricing
- **Right Action Buttons**:
  - `[FAQ]`
  - `[Start on WhatsApp 🚀]` *(Emerald glowing pill button with pulsating dot, linking to `https://wa.me/2349021292141?text=Hello%20Ranviar`)*

---

### Section 1: Hero Section (Above the Fold)

#### 📝 Copy & Typography
- **Overline Eyebrow Badge**:
  `🟢 LIVE ON WHATSAPP • POWERING MEDICAL STUDENTS ACROSS NIGERIA`
- **Headline (H1)**:
  # Master Medicine on WhatsApp.
  ## Grounded in 37+ Accredited Textbooks, Your Own Lecture Slides, and Clinical AI Vision.
- **Supporting Sub-Headline (P)**:
  > Say goodbye to unverified ChatGPT hallucinations and bulky 1,500-page textbooks on ward rounds. **Ranviar** is your 24/7 clinical study co-pilot living right inside WhatsApp. Drop your lecturer's 49MB slide decks, send voice notes after clinic, analyze ECGs and histology photos, and drill high-yield MBBS MCQs — zero extra apps required.
- **Primary CTAs**:
  - **Button 1 (Primary)**: `[💬 Start Chatting on WhatsApp — Free Trial]` *(Primary Green, WhatsApp Icon, opens wa.me link directly)*
  - **Button 2 (Secondary Glass)**: `[▶️ See 60s Demo & Tour]` *(Opens interactive modal with video/GIF walkthrough)*
- **Micro-Copy Trust Indicators**:
  - `⚡ Instant setup • No password or credit card needed`
  - `🛡️ Private & isolated personal document study vault`
  - `📚 37+ Gold-Standard Textbooks pre-indexed`

#### 🖼️ Hero Visual Component: Interactive 3D WhatsApp Chat Simulator
A sleek, floating glassmorphic iPhone/Android frame displaying real WhatsApp chat messages between a student and Ranviar:
1. **User Message (10:14 AM)**:
   > 📎 *Document attached: ADRENAL_GLAND_PATHOLOGY.pptx (49.2 MB)*
   > *"Can you explain the differences in steroidogenesis between Cushing's disease and ectopic ACTH syndrome from my lecturer's slides?"*
2. **Ranviar Immediate Acknowledgment (10:14 AM)**:
   > 📥 *Received: ADRENAL_GLAND_PATHOLOGY.pptx (60 slides, 49.2 MB)*
   > *Detected image-only slides. Transcribing with Gemini 2.5 Flash Vision OCR now... ⏳📑*
3. **Ranviar Verified Answer (10:15 AM)**:
   > 🩺 *Clinical Breakdown: Cushing Disease vs. Ectopic ACTH*
   > 
   > **1. Pituitary Microadenoma (Cushing's Disease)**:
   > • Driven by pituitary hypersecretion of ACTH.
   > • Responds to **High-Dose Dexamethasone Suppression Test (HDDST)**: Cortisol suppresses by >50%.
   > • CRH Stimulation: ACTH and Cortisol levels rise further.
   > 
   > **2. Ectopic ACTH Syndrome (Small Cell Lung Carcinoma / Bronchial Carcinoid)**:
   > • Autonomous ectopic ACTH release completely resistant to glucocorticoid negative feedback.
   > • High-dose dexamethasone test: **No suppression**.
   > • Serum ACTH is markedly elevated (often >200 pg/mL) with severe hypokalemic alkalosis.
   > 
   > 📊 *[Authentic Flowchart Attached: Hypothalamic-Pituitary-Adrenal Axis Diagnostic Algorithm]*
   > 
   > 💡 *Source: Guyton & Hall Physiology (Ch. 78) + Robbins Pathology (Ch. 24) + Your Lecture Slide 34.*
4. **Interactive 1-Tap Quick Action Buttons (Rendered as native WhatsApp pill buttons)**:
   - `[🧠 Practice 3 Adrenal MCQs]`
   - `[📊 View Diagnostic Flowchart]`
   - `[📂 My Study Vault]`

---

### Section 2: Social Proof & University Trust Ticker
- **Header**: `"EMPOWERING MEDICAL STUDENTS ACROSS TOP COLLEGES OF MEDICINE"`
- **Animated Infinite Marquee of Medical Schools**:
  - *College of Medicine, University of Lagos (CMUL)*
  - *College of Medicine, University of Ibadan (CoMUI)*
  - *University of Nigeria Teaching Hospital (UNN / UNTH)*
  - *Federal University of Technology Owerri Medical College (FUTO-CHS)*
  - *Ahmadu Bello University Faculty of Medicine (ABU Zaria)*
  - *Lagos State University College of Medicine (LASUCOM)*
  - *Obafemi Awolowo University (OAU Ife)*
  - *University of Benin (UNIBEN)*
  - *University of Port Harcourt (UNIPORT)*
  - *University of Maiduguri (UNIMAID)*
- **Key Metrics Counters**:
  - `37+` Gold-Standard Textbooks Indexed
  - `500` Pages/Slides Supported per Document Upload
  - `200 MB` Maximum Upload Ceiling (Largest on WhatsApp)
  - `100%` Grounded in Verified Medical Literature

---

### Section 3: The MBBS Problem vs. The Ranviar WhatsApp Solution

| The Traditional Medical School Pain | How Ranviar Solves It on WhatsApp |
|---|---|
| **Drowning in 5,000+ Pages per Semester**: Heavy physical textbooks (Robbins, Bailey & Love, Nelson) that cannot be brought to ward rounds or clinic. | **Instant Pocket Reference**: Complete accredited textbooks queried via semantic neural search directly in your existing WhatsApp chat. |
| **Hallucinating Generic AI**: ChatGPT and generic bots frequently fabricate medical guidelines, invent drug dosages, or invent fake citations. | **Zero Fabricated Citations**: Guaranteed by strict clinical prompt guardrails; every explanation is cross-checked against accredited medical syllabi. |
| **Unreadable Scanned Handouts**: Medical faculty slides and handouts are often photocopied, poorly formatted, or 50MB image-only decks. | **Built-in Gemini 2.5 Flash Vision OCR**: Automatically extracts text, diagrams, and tables from scanned PDFs and PowerPoint image decks up to 500 slides / 200MB. |
| **No App Store Space & Poor Connectivity**: Downloading a 150MB mobile app that burns mobile battery and crashes on 3G hospital networks. | **Zero Installs**: Lives inside WhatsApp. Runs smoothly even on low-bandwidth hospital Wi-Fi or 3G edge connections. |
| **Exhausting Ward Rounds & Bedside Teaching**: Doctors asking rapid-fire clinical questions while you frantically flip through handwritten notes. | **Hands-Free Voice Notes**: Whisper audio voice notes allow you to whisper a question between patients and receive a structured medical breakdown in seconds. |

---

### Section 4: The 6 Core Engineering Pillars (Deep-Dive Features)

```
┌───────────────────────────────────────┬───────────────────────────────────────┐
│ 📚 Pillar 1: 37+ Accredited Textbooks │ 📑 Pillar 2: 500-Page / 200MB Vault   │
│ Complete gold-standard textbooks      │ Ingest lecture slides & scanned PDFs  │
│ across 11 disciplines (200L–600L).    │ with Gemini 2.5 Flash Vision OCR.     │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ 📸 Pillar 3: Clinical Vision AI       │ 📊 Pillar 4: Real Diagnostic Atlas    │
│ Snap photos of ECGs, histology stains,│ 120+ verified pathway flowcharts and  │
│ chest X-rays, and physical signs.     │ cycle schematics (zero micrographs).  │
├───────────────────────────────────────┼───────────────────────────────────────┤
│ 🎙️ Pillar 5: Hands-Free Voice Notes   │ 🧠 Pillar 6: Clinical Vignette MCQs   │
│ Speak naturally via WhatsApp audio.   │ Exam-grade USMLE/MBBS clinical cases, │
│ Instant Whisper speech transcription. │ distractor rationales, and score keys.│
└───────────────────────────────────────┴───────────────────────────────────────┘
```

#### Pillar 1: 37+ Gold-Standard Textbooks in Your Pocket
- **Headline**: The Entire College of Medicine Library, Available in 2 Seconds.
- **Copy**: Never guess an answer on consultant ward rounds again. Ranviar is pre-loaded with the gold-standard textbooks approved by the Medical and Dental Council of Nigeria (MDCN):
  - *Pre-Clinical (200L–300L)*: Guyton & Hall Physiology, Ganong Medical Physiology, Robbins & Cotran Pathologic Basis of Disease, Lippincott Illustrated Reviews Pharmacology, Katzung Basic & Clinical Pharmacology, Harper's Illustrated Biochemistry, Jawetz Medical Microbiology.
  - *Clinical Rotations (400L–600L)*: Kumar & Clark's Clinical Medicine, Davidson's Principles & Practice of Medicine, Bailey & Love's Short Practice of Surgery, Berek & Novak's Gynecology, Nelson Textbook of Pediatrics, Schwartz's Principles of Surgery.
- **Key Feature**: Customize your exact university syllabus with `/curriculum` and `/books` so Ranviar always prioritizes the textbook your head of department examines from!

#### Pillar 2: Multimodal Personal Study Vault (500 Pages / 200MB + AI OCR)
- **Headline**: Upload Your Lecturer's Slides. Search Them Instantly.
- **Copy**: Have a 49MB PowerPoint with 60 image slides? A 300-page scanned pharmacology handout? Just tap the WhatsApp paperclip 📎 and send it.
- **Features**:
  - Handles `.pdf`, `.pptx`, `.ppt`, and `.docx` up to **200 MB** and **500 slides/pages**.
  - **Gemini 2.5 Flash Vision OCR**: Automatically transcribes photos of slide text, handwritten notes, and tables into searchable neural vectors in Qdrant.
  - **Permanent Storage**: Up to 60 documents per student vault. Ask questions anytime without ever having to re-upload.

#### Pillar 3: Clinical Multimodal Vision AI
- **Headline**: Snap It. Send It. Understand It.
- **Copy**: In clinic or the lab, visual diagnosis is everything. Snap a photo with your phone camera directly into the WhatsApp chat:
  - **ECG Strips**: Rate, rhythm, axis, PR interval, ST elevation/depression, bundle branch blocks, and arrhythmias.
  - **Histopathology & Blood Smears**: Identify Reed-Sternberg cells, Auer rods, granulomas, or signet ring carcinomas.
  - **Radiology**: Chest X-rays (pneumothorax, cardiomegaly, lobar pneumonia) and abdominal CT cross-sections.
  - **Exam MCQs**: Snap a photo of a textbook question to get an immediate breakdown of the correct choice and why every other distractor is wrong.

#### Pillar 4: Authentic Pathway Schematics & Diagnostic Algorithms
- **Headline**: Real Flowcharts. No Hallucinated Diagrams or Micrograph Crops.
- **Copy**: When you ask for a pathway, cycle, or mechanism, you need an annotated schematic—not a random blurry cell picture.
- **Curated Atlas**: Over 120+ verified high-resolution diagrams:
  - *Biochemistry*: Glycolysis, Krebs Cycle, Urea Cycle, Pentose Phosphate Pathway, Glycogenolysis.
  - *Physiology & Pharmacology*: RAAS Cascade, Cardiac Cycle Wiggers Diagram, Coagulation Cascade, GPCR Signaling.
  - *Clinical Surgery & Medicine*: ATLS Trauma Algorithm, Diabetic Ketoacidosis Management Protocol, Glasgow Coma Scale (GCS), Pediatric Dehydration Plan.

#### Pillar 5: Hands-Free Voice Note Tutoring
- **Headline**: Too Exhausted to Type After a 10-Hour Ward Round? Just Talk.
- **Copy**: Press and hold the WhatsApp microphone button 🎙️. Speak your question naturally in plain English:
  > *"Ranviar, explain how to differentiate between nephrotic syndrome and nephritic syndrome in pediatric patients."*
- Powered by OpenAI Whisper transcription, Ranviar digests your spoken audio in milliseconds and returns an organized, high-yield clinical response.

#### Pillar 6: MBBS Clinical Vignette MCQs & Diagnostic Drills
- **Headline**: Pass Professional MBBS Exams on the First Attempt.
- **Copy**: True clinical competence isn't just rote memorization—it's solving real patient cases.
- **Features**:
  - Type `/quiz [topic]` (e.g. `/quiz Heart Failure`) or tap `🧠 Practice MCQs` after any explanation.
  - Generates realistic Nigerian and USMLE-style clinical vignettes (patient age, presentation, vital signs, labs).
  - Multi-choice options with thorough justification for the correct option and detailed distractor analysis.

---

### Section 5: How It Works (Zero App Installs, 100% WhatsApp)

```
┌─────────────────────────┐      ┌─────────────────────────┐      ┌─────────────────────────┐
│         STEP 1          │      │         STEP 2          │      │         STEP 3          │
│ 📱 Open WhatsApp        │ ───> │ ⚙️ Pick Your Level      │ ───> │ 🩺 Study & Excel        │
│ Send "Hello" to Ranviar │      │ Select 200L to 600L &   │      │ Ask questions, drop     │
│ on +234 902 129 2141    │      │ choose your textbooks   │      │ slides, send voice notes│
└─────────────────────────┘      └─────────────────────────┘      └─────────────────────────┘
```

1. **Step 1: Save & Open on WhatsApp**
   - No app store downloads, no registrations, no passwords.
   - Click the link or scan the QR code to message Ranviar's verified WhatsApp business number: `+234 902 129 2141`.
   - Tap the 1-Tap WhatsApp Contact Card to save **Ranviar Medical AI** directly to your phone contacts.
2. **Step 2: Customize Your Curriculum (200L–600L)**
   - Tell Ranviar your current medical school level (e.g., 300L Pre-Clinical or 500L Surgery/O&G).
   - Customize your preferred department textbooks with `/curriculum`.
3. **Step 3: Study Anywhere, Anytime**
   - Ask clinical questions, upload 500-page lecture slides, send ECG photos, or practice MCQs before tests and ward rounds.

---

### Section 6: Level-by-Level Medical School Curriculum Matrix

An interactive tabbed component on the landing page showing syllabus alignment:

#### 🎓 200L & 300L (Pre-Clinical Foundations)
- **Subjects**: Gross Anatomy, Histology, Embryology, Medical Biochemistry, General Physiology, Neuroanatomy.
- **Featured Textbooks**: *Guyton & Hall Physiology*, *Robbins & Cotran Pathology*, *Harper's Biochemistry*, *Moore's Clinically Oriented Anatomy*, *Langman's Medical Embryology*.
- **Use Case**: Master enzyme pathways, cranial nerve pathways, action potentials, and histological identification for 1st MBBS Professional Exams.

#### 🩺 400L (Pathology & Pharmacology Rotations)
- **Subjects**: Morbid Anatomy, Chemical Pathology, Medical Microbiology & Parasitology, Hematology, Systemic Pharmacology.
- **Featured Textbooks**: *Robbins Basic Pathology*, *Katzung Clinical Pharmacology*, *Jawetz Medical Microbiology*, *Hoffbrand's Essential Haematology*.
- **Use Case**: Drug mechanisms of action, adverse reaction profiles, blood film interpretations, bacterial culture algorithms for 2nd MBBS Exams.

#### 🏥 500L & 600L (Clinical Rotations & Ward Rounds)
- **Subjects**: Internal Medicine, General Surgery, Orthopedics, Obstetrics & Gynecology, Pediatrics & Child Health, Community Medicine.
- **Featured Textbooks**: *Kumar & Clark Clinical Medicine*, *Davidson's Medicine*, *Bailey & Love Surgery*, *Berek & Novak Gynecology*, *Nelson Pediatrics*.
- **Use Case**: Bedside ward-round preparation, drug dosing calculations, surgical step-by-step procedures, clinical history taking, and 3rd/Final MBBS Clinical Vivas.

---

### Section 7: Interactive Live WhatsApp Simulator Playground

An interactive component on the landing page where visitors can click sample prompts to test how Ranviar responds in real-time:
- **Tab 1: "Clinical Ward Round Case"**
  - *Prompt*: `"55yo male with 3-hour history of crushing retrosternal chest pain radiating to the left jaw. BP 85/50, HR 115. What are the immediate management steps according to ACC/AHA guidelines?"*
  - *Response*: Instant MONA protocol, 12-lead ECG urgency, PCI window vs. Fibrinolysis, and cardiogenic shock warnings.
- **Tab 2: "Lecture Slide Search"**
  - *Prompt*: `"What did my lecturer say on Slide 18 about drug-induced lupus erythematosus?"*
  - *Response*: Direct extraction of histone antibodies, offending agents (Hydralazine, Procainamide, Isoniazid), and resolution upon discontinuation.
- **Tab 3: "Histology Photo Breakdown"**
  - *Interactive Image*: High-power micrograph showing liver cirrhosis with bridging fibrosis and regenerative nodules.
  - *Response*: Identification of histological hallmarks with differential diagnoses.
- **Tab 4: "Clinical MCQ Challenge"**
  - *Interactive Quiz*: 1 clinical case vignette with selectable A, B, C, D options and instant feedback upon clicking!

---

### Section 8: Student Testimonials & Ward-Round Reviews

```
┌────────────────────────────────────────────────────────────────────────┐
│ "Ranviar saved me on my Pediatric ward rounds at CMUL. The consultant  │
│ grilled me on dehydration fluid calculations, and I had reviewed it on │
│ WhatsApp 5 minutes before the ward round began. Pure gold!"            │
│ — Chidiebere N., 500L MBBS Student, University of Lagos                │
├────────────────────────────────────────────────────────────────────────┤
│ "Being able to upload my lecturer's 48MB PowerPoint slide decks and    │
│ have Ranviar transcribe the scanned images with AI Vision OCR is like   │
│ magic. It answers questions directly from my school's syllabus!"       │
│ — Amina K., 300L Pre-Clinical Student, Ahmadu Bello University         │
├────────────────────────────────────────────────────────────────────────┤
│ "The voice notes are my favorite feature. After standing for 8 hours in│
│ theater during surgery posting, I just speak my questions on the bus   │
│ home. The explanations are sharper than any search engine."            │
│ — Femi O., 600L Final Year Medical Student, University of Ibadan       │
└────────────────────────────────────────────────────────────────────────┘
```

---

### Section 9: Transparent Paystack Student Pricing & Wallet Plans

Designed for the Nigerian student reality with zero card commitment, instant Paystack checkout (Debit Card, USSD, Bank Transfer, OPay, Kuda), and generous free starter credits:

```
┌──────────────────────────┬──────────────────────────┬──────────────────────────┐
│       STARTER BETA       │    PAY-AS-YOU-GO WALLET  │    CLINICAL SCHOLAR PRO  │
│          FREE            │         ₦2,000           │         ₦5,000           │
│    (For Every Student)   │   (Most Popular Choice)  │    (Maximum Performance) │
├──────────────────────────┼──────────────────────────┼──────────────────────────┤
│ • 50 Free Welcome Queries│ • ₦2,000 Account Balance │ • ₦5,000 Account Balance │
│ • 5 Document Vault Slots │ • Dynamic Micro-Billing  │ • VIP Fast-Track Queue   │
│ • Standard Textbook RAG  │ • ~500+ Deep AI Queries  │ • ~1,500+ Deep AI Queries│
│ • Community Support      │ • 25 Vault Document Slots│ • 60 Vault Document Slots│
│ • WhatsApp 1-Tap Access  │ • Full Vision OCR Access │ • Unlimited Slide OCR    │
│                          │ • Valid for Entire Term  │ • Priority 24/7 Access   │
├──────────────────────────┼──────────────────────────┼──────────────────────────┤
│       [Active Now]       │     [Top Up ₦2,000]      │     [Top Up ₦5,000]      │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
```
- **Transparent Micro-Billing**: Each query deducts fractions of a naira based on real token usage, with real-time balance tracking via `/wallet` and top-ups via `/deposit`.
- **Zero Hidden Charges**: No recurring monthly auto-debits without student consent. Pay only for what you learn.

---

### Section 10: Meet the Builders ("Built By" / Founders & Team)

> **Headline**: Built by Medical Minds and Systems Engineers Passionate About Transforming Healthcare Education.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                               THE CORE BUILDERS                              │
├──────────────────────────────┬──────────────────────────────┬────────────────┤
│ 👨‍💻 SAMUEL                   │ 🩺 AUGUSTINE                 │ 🚀 MAAZI MILK  │
│ Founder & Lead AI Architect  │ Co-Founder & Head of Clinical│ Co-Founder &   │
│                              │ Curriculum                   │ Head of Growth │
└──────────────────────────────┴──────────────────────────────┴────────────────┘
```

#### 1. Samuel — Founder, Lead AI Systems Architect & Full-Stack Engineer
- **Bio**: Visionary software engineer and creator of Ranviar. Architected the high-concurrency dual-engine backend (Python FastAPI & Rust Actix-web), the multi-stage semantic vector RAG pipeline in Qdrant, the multimodal document vault supporting 500-page/200MB uploads, and the real-time Meta WhatsApp Cloud API integration.
- **Focus**: High-concurrency AI systems, LLM memory caching, low-latency search, multimodal vision pipelines, and system reliability.
- **Quote**: *"Medical students shouldn't have to fight clumsy software or hallucinating chatbots when preparing to save lives. Ranviar brings world-class, textbook-grounded clinical intelligence straight into the app they use every single day."*

#### 2. Augustine — Co-Founder, Head of Clinical Curriculum & Medical Education
- **Bio**: Co-founder directing clinical syllabus precision and academic integrity. Oversees the rigorous mapping of the 200L–600L MBBS curriculum, curation of accredited gold-standard medical textbooks (Robbins, Guyton, Bailey & Love, Kumar & Clark), validation of clinical vignette MCQs, and alignment with Medical and Dental Council of Nigeria (MDCN) examination benchmarks.
- **Focus**: Medical curriculum design, clinical case authenticity, distractor rationale verification, ward-round study workflows, and clinical safety guardrails.
- **Quote**: *"Every medical student remembers the anxiety of consultant ward rounds and high-stakes professional vivas. We built Ranviar to ensure no student ever walks into a hospital ward unprepared."*

#### 3. Maazi Milk — Co-Founder, Head of Product Strategy, Growth & Community
- **Bio**: Co-founder driving product execution, medical student adoption, brand storytelling, and university community expansion. Spearheads partnerships with medical student associations (FUTOMSA, NiMSA, regional medical faculties), user research, feature prioritization, and seamless WhatsApp student onboarding.
- **Focus**: Community growth, medical student advocacy, product positioning, viral WhatsApp adoption loops, and campus ambassador networks across African universities.
- **Quote**: *"Ranviar isn't just an AI tool; it's a movement to democratize premier medical education across Africa. We are making sure every aspiring doctor has an elite clinical tutor right in their pocket."*

---

### Section 11: Frequently Asked Questions (FAQ)

An interactive accordion component addressing every student doubt:

1. **Q: Do I need to install any new app to use Ranviar?**
   - **A**: No! Ranviar lives 100% inside WhatsApp. You simply tap a link to message our verified WhatsApp number, and you can start studying immediately. It works on Android, iOS, WhatsApp Web, and Desktop.
2. **Q: How is Ranviar different from ChatGPT or other general AI bots?**
   - **A**: General AI models frequently hallucinate fake drug dosages, invent non-existent textbook citations, and cannot read local university lecture slide decks. Ranviar is strictly built for medicine, pre-loaded with 37+ accredited medical textbooks, equipped with strict zero-fabrication guardrails, and lets you upload up to 60 of your faculty's own lecture slide decks.
3. **Q: Can I upload scanned lecture handouts and image-based PowerPoint decks?**
   - **A**: Yes! Ranviar features integrated Gemini 2.5 Flash Vision OCR. When you send a PowerPoint with photo slides or a scanned PDF handout up to 500 pages/slides and 200MB, Ranviar automatically transcribes every slide's medical text, diagrams, and tables into your private study vault.
4. **Q: Are my uploaded personal notes and documents kept private?**
   - **A**: Absolutely. Your uploaded documents are isolated in a dedicated multi-tenant Qdrant partition linked strictly to your phone number. No other student can see, search, or query your uploaded materials.
5. **Q: How do I fund my wallet with Paystack?**
   - **A**: You can start completely free! When you want to top up your balance, type `/deposit` or `/wallet` in WhatsApp. You'll receive a secure Paystack payment link supporting debit cards, bank transfers, USSD, and mobile banking.
6. **Q: Can Ranviar prescribe medications to patients?**
   - **A**: No. Ranviar is an educational clinical co-pilot designed for medical students, house officers, and exam candidates. It explains pathophysiology, pharmacology, and clinical guidelines for learning and exam excellence, but does not provide direct patient prescriptions.

---

### Section 12: High-Conversion Sticky Footer & Final Call to Action

- **Banner Headline**:
  ## Ready to Top Your Class in the Next MBBS Professional Exam?
- **Supporting Text**:
  Join thousands of medical students across Nigeria studying smarter with accredited textbooks, slide vault OCR, and clinical vision on WhatsApp.
- **Primary CTA**:
  `[💬 Start Chatting on WhatsApp Free — Instant Access 🩺]`
- **Desktop Companion**:
  Display a high-contrast QR Code with the caption:
  *"On Desktop? Scan with your phone camera to start instantly on WhatsApp!"*
- **Footer Navigation**:
  - **Product**: Features, 37+ Textbooks, Slide Vault, Vision AI, Clinical Quizzes, Pricing.
  - **Company & Team**: About Ranviar, The Builders (Samuel, Augustine, Maazi Milk), Medical Advisory, Careers.
  - **Legal & Academic**: Privacy Policy, Terms of Service, Medical Educational Disclaimer.
- **Copyright & Academic Notice**:
  `© 2026 Ranviar (ranviar.org). All rights reserved. Built with passion by Samuel, Augustine, and Maazi Milk for the future doctors of Africa.`
  *Medical Disclaimer: Ranviar is an academic study companion intended exclusively for medical education and exam preparation.*

---

## 🛠️ Complete Technical Implementation Guide for Web Engineers

When ready to develop the website, run:
```bash
# 1. Initialize Next.js 14 App with Tailwind CSS & TypeScript
npx create-next-app@latest ranviar-landing --typescript --tailwind --eslint --app

# 2. Install UI & Animation Dependencies
cd ranviar-landing
npm install framer-motion lucide-react clsx tailwind-merge canvas-confetti @radix-ui/react-accordion @radix-ui/react-tabs

# 3. Environment Variables (.env.local)
NEXT_PUBLIC_WHATSAPP_NUMBER="2349021292141"
NEXT_PUBLIC_WHATSAPP_LINK="https://wa.me/2349021292141?text=Hello%20Ranviar"
NEXT_PUBLIC_SITE_URL="https://ranviar.org"
```

This specification represents the authoritative blueprint for building **ranviar.org**. Every section, interaction, copywriting block, and founder profile is verified and ready for frontend implementation! 🩺⚡
