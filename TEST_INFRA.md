# E2E Test Infra: Neura-AI Medical Illustration & Diagram Engine

## Test Philosophy
- Opaque-box, requirement-driven. Derives test cases strictly from `ORIGINAL_REQUEST.md`.
- Methodology: Category-Partition across 11 disciplines + Boundary Value Analysis + Natural Language Response Inspection + Multi-branch validation.

## Feature Inventory & Test Mapping
| # | Feature | Source | Verification Mechanism |
|---|---------|--------|------------------------|
| 1 | F1: Ban Fabricated Citations | ORIGINAL_REQUEST §R1 | Regex assertion `re.search(r'\b(figure|fig\.?)\s+\d+[-.:]\d+', text)` == None |
| 2 | F2: WhatsApp Response Sanitization | ORIGINAL_REQUEST §R1 | Assert no robotic preambles, proper card formatting, clean bolding |
| 3 | F3: Intelligent Visual Intent Routing | ORIGINAL_REQUEST §R3 | Assert modality classification for 50+ diverse queries |
| 4 | F4: Universal Flowchart Atlas | ORIGINAL_REQUEST §R2 | 50 benchmark queries across 200L–600L resolve to authentic schematics with 0ms latency |
| 5 | F5: Micrograph Demotion Filter | ORIGINAL_REQUEST §R3 | Assert 0% micrograph/smear return on pathway/cycle/mechanism queries |
| 6 | F6: 50-Topic Verification Suite | ORIGINAL_REQUEST §R4 | Comprehensive runner `test_multidomain_verification_suite.py` passes 100% |
| 7 | F7: Multi-Branch Sync & Readiness | ORIGINAL_REQUEST §AC3 | Git status and test passes on `main` and `feature/pay-as-you-go` |

## 50-Topic Multi-Domain Coverage (200L - 600L)
Spans all 11 disciplines:
- Parasitology (5 topics: Plasmodium, Schistosoma, Leishmania, Entamoeba, Trypanosoma)
- Microbiology (5 topics: Bacterial Endospore, Viral Replication, HIV, Gram Cell Wall, Bacteriophage)
- Biochemistry (5 topics: Glycolysis, Krebs, Urea Cycle, Pentose Phosphate Pathway, Gluconeogenesis)
- Physiology (5 topics: Action Potential, Cardiac Wiggers, RAAS, Neuromuscular Junction, Nephron Countercurrent)
- Pharmacology (5 topics: GPCR Gs/Gq/Gi, Tyrosine Kinase, Adrenergic/Cholinergic, Beta-blockers MOA, Warfarin/Heparin)
- Immunology (5 topics: Complement Cascade, B/T Cell Differentiation, Hypersensitivity I-IV, TCR MHC, Clonal Selection)
- Hematology (5 topics: Hematopoiesis, Coagulation Cascade, Hemoglobin Dissociation, Iron Metabolism, Platelet Plug)
- Surgery (5 topics: ATLS Trauma Algorithm, Burns Rule of Nines, Glasgow Coma Scale, Acute Abdomen, Surgical Sepsis)
- O&G (5 topics: Menstrual Cycle Hormones, Cardinal Movements of Labor, APGAR Scoring, Preeclampsia/Eclampsia, Bishop Score)
- Pediatrics (5 topics: IMCI Triage Algorithm, Tetralogy of Fallot, Developmental Milestones, Neonatal Resuscitation, Dehydration Plan)
- Internal Medicine & Pathology (5 topics: Meningitis CSF Algorithm, JVP Waveforms, ACLS Tachy/Bradycardia, Diabetic Ketoacidosis, Granuloma Formation)
