"""
Adversarial Stress Test Suite for Milestone 1 (Challenger 2).
Focus areas:
1. 35+ Complex Figure, Plate, Table, and Chart citation patterns across all syntax variants.
2. Complex Markdown Table Edge Cases (empty cells, 5+ columns, markdown in cells, no trailing pipes, immediate lists, etc.).
3. Medical Entity Preservation (entities with numbers, hyphens, colons, Greek letters, 'Figure-of-eight', 'CD4:CD8', etc.).
4. Conversational Preamble and Greeting variations under adversarial phrasing.
"""

import os
import sys
import re
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    format_whatsapp_text,
    strip_conversational_preambles,
    strip_figure_citations,
    convert_markdown_tables_to_whatsapp_cards,
)


class TestAdversarialFigureCitations(unittest.TestCase):
    """Stress tests stripping of 35+ distinct figure/plate/table citation patterns."""

    def test_01_parenthetical_hyphenated_figure(self):
        text = "Hepatic schizogony occurs first (Figure 12-4). Then merozoites enter RBCs."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 12-4", out)
        self.assertNotIn("()", out)
        self.assertIn("Hepatic schizogony occurs first", out)
        self.assertIn("Then merozoites enter RBCs.", out)

    def test_02_parenthetical_fig_with_subletter_and_source(self):
        text = "Councilman bodies are apoptotic hepatocytes (Fig. 1.2b from Robbins)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Fig. 1.2b", out)
        self.assertIn("Councilman bodies are apoptotic hepatocytes", out)

    def test_03_bracketed_or_parenthetical_plate(self):
        text = "Amyloid deposition with Congo red birefringence (Plate 43.1) in the glomerulus."
        out = format_whatsapp_text(text)
        self.assertNotIn("Plate 43.1", out)
        self.assertIn("Amyloid deposition with Congo red birefringence", out)
        self.assertIn("in the glomerulus", out)

    def test_04_refer_to_table_below(self):
        text = "For differential diagnosis, refer to Table 3-1 below for enzyme markers."
        out = format_whatsapp_text(text)
        self.assertNotIn("Table 3-1", out)
        self.assertNotIn("refer to Table", out)
        self.assertIn("For differential diagnosis", out)

    def test_05_as_shown_in_figure_hyphen(self):
        text = "As shown in Figure 46-9, the life cycle involves female Anopheles mosquitoes."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 46-9", out)
        self.assertIn("the life cycle involves female Anopheles mosquitoes", out)

    def test_06_figure_with_colon_title(self):
        text = "Figure 20.33: Pathogenesis of Atherosclerosis.\nEndothelial injury initiates lipid accumulation."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 20.33", out)
        self.assertIn("Endothelial injury initiates lipid accumulation", out)

    def test_07_see_figure_for_details(self):
        text = "The citric acid cycle generates reducing equivalents (see Figure 10-2 for details)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 10-2", out)
        self.assertIn("The citric acid cycle generates reducing equivalents", out)

    def test_08_as_illustrated_in_fig_with_subletter(self):
        text = "As illustrated in Fig. 5.4a, the action potential shows a plateau phase."
        out = format_whatsapp_text(text)
        self.assertNotIn("Fig. 5.4a", out)
        self.assertIn("the action potential shows a plateau phase", out)

    def test_09_depicted_in_plate_with_source(self):
        text = "As depicted in Plate 12.3 of Wheater, the brush border contains microvilli."
        out = format_whatsapp_text(text)
        self.assertNotIn("Plate 12.3", out)
        self.assertIn("the brush border contains microvilli", out)

    def test_10_as_noted_in_table(self):
        text = "As noted in Table 2-4, LDL targets vary by cardiovascular risk category."
        out = format_whatsapp_text(text)
        self.assertNotIn("Table 2-4", out)
        self.assertIn("LDL targets vary by cardiovascular risk category", out)

    def test_11_shown_in_chart(self):
        text = "As shown in Chart 8.1, childhood vaccination coverage has reduced incidence."
        out = format_whatsapp_text(text)
        self.assertNotIn("Chart 8.1", out)
        self.assertIn("childhood vaccination coverage has reduced incidence", out)

    def test_12_refer_to_figure_below_for_comparison(self):
        text = "To distinguish Crohn's from UC, (refer to Figure 9-3 below for comparison)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 9-3", out)
        self.assertIn("To distinguish Crohn's from UC", out)

    def test_13_see_fig_hyphen(self):
        text = "Gram-positive peptidoglycan crosslinking occurs extracellularly (see Fig 14-2)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Fig 14-2", out)
        self.assertIn("Gram-positive peptidoglycan crosslinking occurs extracellularly", out)

    def test_14_simple_parenthetical_plate(self):
        text = "Gaucher cells appear as crumpled tissue paper (Plate 8-2)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Plate 8-2", out)
        self.assertIn("Gaucher cells appear as crumpled tissue paper", out)

    def test_15_simple_parenthetical_table(self):
        text = "Multiple endocrine neoplasia types differ in organ involvement (Table 1.1)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Table 1.1", out)
        self.assertIn("Multiple endocrine neoplasia types differ in organ involvement", out)

    def test_16_simple_parenthetical_chart(self):
        text = "Growth velocity curves show peak height velocity during puberty (Chart 4.2)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Chart 4.2", out)
        self.assertIn("Growth velocity curves show peak height velocity during puberty", out)

    def test_17_standalone_figure_illustrates_verb(self):
        text = "Figure 11-1 illustrates the coagulation cascade leading to thrombin generation."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 11-1", out)
        self.assertIn("coagulation cascade leading to thrombin generation", out)

    def test_18_standalone_table_shows_verb(self):
        text = "Table 7-2 shows the Michaelis-Menten kinetic parameters of hexokinase and glucokinase."
        out = format_whatsapp_text(text)
        self.assertNotIn("Table 7-2", out)
        self.assertIn("Michaelis-Menten kinetic parameters of hexokinase and glucokinase", out)

    def test_19_standalone_plate_demonstrates_verb(self):
        text = "Plate 4.5 demonstrates the characteristic Congo red birefringence of amyloid plaques."
        out = format_whatsapp_text(text)
        self.assertNotIn("Plate 4.5", out)
        self.assertIn("characteristic Congo red birefringence of amyloid plaques", out)

    def test_20_standalone_fig_shows_verb(self):
        text = "Fig. 3.2 shows the Phase 0 rapid depolarization mediated by sodium influx."
        out = format_whatsapp_text(text)
        self.assertNotIn("Fig. 3.2", out)
        self.assertIn("Phase 0 rapid depolarization mediated by sodium influx", out)

    def test_21_citation_section_jawetz_figure(self):
        text = "📖 *IN-DEPTH EXPLANATION*\n\nText\n\n📚 *CITATIONS*\n- Jawetz Medical Microbiology, Figure 46-9\n- Medical Microbiology 28th Ed"
        out = format_whatsapp_text(text)
        self.assertIn("- Jawetz Medical Microbiology", out)
        self.assertNotIn("Figure 46-9", out)

    def test_22_citation_section_robbins_page_and_fig(self):
        text = "📚 *CITATIONS*\n- Robbins & Cotran Pathologic Basis of Disease, p. 787, Fig 20.33"
        out = format_whatsapp_text(text)
        self.assertIn("- Robbins & Cotran Pathologic Basis of Disease", out)
        self.assertNotIn("Fig 20.33", out)
        self.assertNotIn("p. 787", out)

    def test_23_citation_section_ganong_table(self):
        text = "📚 *CITATIONS*\n- Ganong's Review of Medical Physiology, Table 12-1"
        out = format_whatsapp_text(text)
        self.assertIn("- Ganong's Review of Medical Physiology", out)
        self.assertNotIn("Table 12-1", out)

    def test_24_citation_section_wheater_plate(self):
        text = "📚 *CITATIONS*\n- Wheater's Functional Histology, Plate 3.2"
        out = format_whatsapp_text(text)
        self.assertIn("- Wheater's Functional Histology", out)
        self.assertNotIn("Plate 3.2", out)

    def test_25_parenthetical_fig_source_guyton(self):
        text = "Renal countercurrent multiplication (see Figure 4-1A in Guyton) concentrates urine."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 4-1A", out)
        self.assertIn("Renal countercurrent multiplication", out)
        self.assertIn("concentrates urine", out)

    def test_26_refer_to_table_reference_values(self):
        text = "For normal CSF opening pressure, refer to Table 6-4 for reference values."
        out = format_whatsapp_text(text)
        self.assertNotIn("Table 6-4", out)
        self.assertIn("For normal CSF opening pressure", out)

    def test_27_as_seen_in_figure_below_cascade(self):
        text = "As seen in Figure 18-2 below, the complement cascade activates C3 convertase."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 18-2", out)
        self.assertIn("the complement cascade activates C3 convertase", out)

    def test_28_parenthetical_fig_subletter(self):
        text = "The electron transport chain complexes (Fig. 7-10b) create a proton gradient."
        out = format_whatsapp_text(text)
        self.assertNotIn("Fig. 7-10b", out)
        self.assertIn("The electron transport chain complexes", out)
        self.assertIn("create a proton gradient", out)

    def test_29_parenthetical_figure_colon_name(self):
        text = "Isovolumetric ventricular contraction occurs during systole (Figure 3.1: Cardiac Cycle)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 3.1", out)
        self.assertIn("Isovolumetric ventricular contraction occurs during systole", out)

    def test_30_figure_overview_header(self):
        text = "Figure 1-1: Overview of Metabolism\nGlycolysis converts glucose into pyruvate."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 1-1", out)
        self.assertIn("Glycolysis converts glucose into pyruvate", out)

    def test_31_parenthetical_plate_source_junqueira(self):
        text = "Goblet cells secrete protective mucin granules (Plate 22.1 from Junqueira)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Plate 22.1", out)
        self.assertIn("Goblet cells secrete protective mucin granules", out)

    def test_32_parenthetical_table_source_katzung(self):
        text = "Beta-blocker receptor selectivity varies significantly (Table 10.5 from Katzung)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Table 10.5", out)
        self.assertIn("Beta-blocker receptor selectivity varies significantly", out)

    def test_33_parenthetical_see_fig_period(self):
        text = "The pentose phosphate pathway produces NADPH (see Fig. 15-3)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Fig. 15-3", out)
        self.assertIn("The pentose phosphate pathway produces NADPH", out)

    def test_34_parenthetical_refer_to_plate(self):
        text = "Reed-Sternberg cells display owl-eyed nuclei (refer to Plate 9.4)."
        out = format_whatsapp_text(text)
        self.assertNotIn("Plate 9.4", out)
        self.assertIn("Reed-Sternberg cells display owl-eyed nuclei", out)

    def test_35_figure_from_first_aid(self):
        text = "Figure 8.12 from First Aid summarizes the antiarrhythmic classes."
        out = format_whatsapp_text(text)
        self.assertNotIn("Figure 8.12", out)
        self.assertIn("summarizes the antiarrhythmic classes", out)


class TestAdversarialMarkdownTables(unittest.TestCase):
    """Stress tests conversion of markdown tables into WhatsApp card format under messy or edge conditions."""

    def test_01_table_with_5_plus_columns(self):
        raw = """| Drug | Class | MOA | Clinical Use | Adverse Effects | Elimination |
| --- | --- | --- | --- | --- | --- |
| Metformin | Biguanide | AMPK activation | T2DM | Lactic acidosis, GI upset | Renal |
| Empagliflozin | SGLT2 inhibitor | Blocks glucose reabsorption | T2DM, Heart Failure | UTI, Mycotic infections | Renal/Hepatic |"""
        out = format_whatsapp_text(raw)
        self.assertNotIn("|", out)
        self.assertIn("- *Metformin*", out)
        self.assertIn("• *Class:* Biguanide", out)
        self.assertIn("• *MOA:* AMPK activation", out)
        self.assertIn("• *Clinical Use:* T2DM", out)
        self.assertIn("• *Adverse Effects:* Lactic acidosis, GI upset", out)
        self.assertIn("• *Elimination:* Renal", out)
        self.assertIn("- *Empagliflozin*", out)
        self.assertIn("• *Class:* SGLT2 inhibitor", out)

    def test_02_table_with_multiple_empty_cells(self):
        raw = """| Condition | Etiology | Key Lab | Imaging | First-Line Rx |
| --- | --- | --- | --- | --- |
| Appendicitis | Fecalith obstruction | Leukocytosis | Ultrasound / CT | Appendectomy |
| Cholecystitis | Gallstone impaction |  | Ultrasound |  |
| Diverticulitis |  | Leukocytosis |  | Ciprofloxacin + Metronidazole |"""
        out = format_whatsapp_text(raw)
        self.assertNotIn("|", out)
        self.assertIn("- *Appendicitis*", out)
        self.assertIn("- *Cholecystitis*", out)
        self.assertIn("• *Imaging:* Ultrasound", out)
        self.assertIn("- *Diverticulitis*", out)
        self.assertIn("• *Key Lab:* Leukocytosis", out)
        self.assertIn("• *First-Line Rx:* Ciprofloxacin + Metronidazole", out)

    def test_03_table_with_markdown_inside_cells(self):
        raw = """| Step | Enzyme | Substrate -> Product | Regulation |
| --- | --- | --- | --- |
| 1 | *Hexokinase* / *Glucokinase* | Glucose -> G6P | Inhibited by *G6P* (HK only) |
| 3 | **PFK-1** (rate-limiting) | F6P -> F-1,6-BP | Activated by **AMP**, **F-2,6-BP** |"""
        out = format_whatsapp_text(raw)
        self.assertNotIn("|", out)
        self.assertNotIn("**", out)
        self.assertIn("- *1*", out)
        self.assertIn("• *Enzyme:* *Hexokinase* / *Glucokinase*", out)
        self.assertIn("• *Regulation:* Inhibited by *G6P* (HK only)", out)
        self.assertIn("• *Substrate -> Product:* Glucose -> G6P", out)
        self.assertIn("- *3*", out)
        self.assertIn("*PFK-1* (rate-limiting)", out)

    def test_04_table_followed_immediately_by_bullet_list(self):
        raw = """| Parameter | Transudate | Exudate |
| --- | --- | --- |
| Pleural/Serum Protein | < 0.5 | > 0.5 |
| Pleural/Serum LDH | < 0.6 | > 0.6 |
- Transudative effusions are typically due to Heart Failure or Cirrhosis.
- Exudative effusions are caused by Malignancy, Infection, or PE."""
        out = format_whatsapp_text(raw)
        self.assertNotIn("|", out)
        self.assertIn("- *Pleural/Serum Protein*", out)
        self.assertIn("• *Transudate:* < 0.5", out)
        self.assertIn("• *Exudate:* > 0.5", out)
        self.assertIn("- *Pleural/Serum LDH*", out)
        self.assertIn("• *Transudate:* < 0.6", out)
        self.assertIn("• *Exudate:* > 0.6", out)
        self.assertIn("Heart Failure or Cirrhosis", out)
        self.assertIn("Malignancy, Infection, or PE", out)

    def test_05_table_with_complex_delimiter_alignments(self):
        raw = """| Factor | Intrinsic Pathway | Extrinsic Pathway |
| :--- | :---: | ---: |
| Trigger | Collagen / Surface | Tissue Factor (III) |
| Initial Factor | Factor XII | Factor VII |"""
        out = format_whatsapp_text(raw)
        self.assertNotIn("|", out)
        self.assertIn("- *Trigger*", out)
        self.assertIn("• *Intrinsic Pathway:* Collagen / Surface", out)
        self.assertIn("• *Extrinsic Pathway:* Tissue Factor (III)", out)
        self.assertIn("- *Initial Factor*", out)
        self.assertIn("• *Intrinsic Pathway:* Factor XII", out)

    def test_06_single_data_row_table(self):
        raw = """| Test | Target | Sensitivity | Specificity |
| --- | --- | --- | --- |
| Monospot | Heterophile antibodies | 85% | 98% |"""
        out = format_whatsapp_text(raw)
        self.assertNotIn("|", out)
        self.assertIn("- *Monospot*", out)
        self.assertIn("• *Target:* Heterophile antibodies", out)
        self.assertIn("• *Sensitivity:* 85%", out)
        self.assertIn("• *Specificity:* 98%", out)

    def test_07_two_consecutive_tables_separated_by_heading(self):
        raw = """| Microbe | Morphology |
| --- | --- |
| S. pneumoniae | Lancets |

💡 *GRAM NEGATIVE*

| Microbe | Morphology |
| --- | --- |
| N.誕生 | Kidney bean diplococci |"""
        out = format_whatsapp_text(raw)
        self.assertNotIn("|", out)
        self.assertIn("- *S. pneumoniae*", out)
        self.assertIn("• *Morphology:* Lancets", out)
        self.assertIn("💡 *GRAM NEGATIVE*", out)

    def test_08_table_without_trailing_pipe_handling(self):
        # Even if a malformed table with irregular row formats is passed, check that format_whatsapp_text does not crash
        raw = """| Gene | Chromosome | Associated Condition
| BRCA1 | 17q21 | Hereditary Breast/Ovarian Cancer
| TP53 | 17p13.1 | Li-Fraumeni Syndrome"""
        out = format_whatsapp_text(raw)
        # Should gracefully retain or format without exception
        self.assertIn("BRCA1", out)
        self.assertIn("TP53", out)


class TestMedicalEntityPreservation(unittest.TestCase):
    """Verifies that medical terms, ratios, acronyms, and formulas containing numbers/hyphens are NOT deleted."""

    def test_01_cd4_cd8_ratio(self):
        raw = "A reduced CD4:CD8 ratio (< 1.0) is indicative of inverted immunophenotype in HIV infection."
        out = format_whatsapp_text(raw)
        self.assertIn("CD4:CD8 ratio", out)
        self.assertIn("inverted immunophenotype in HIV infection", out)

    def test_02_12_lead_ecg_and_st_elevation(self):
        raw = "The 12-lead ECG demonstrates hyperacute T-waves and ST-segment elevation in leads V1-V4."
        out = format_whatsapp_text(raw)
        self.assertIn("12-lead ECG", out)
        self.assertIn("leads V1-V4", out)

    def test_03_hla_b27_and_ankylosing_spondylitis(self):
        raw = "HLA-B27 seropositivity is strongly associated with Ankylosing Spondylitis and reactive arthritis."
        out = format_whatsapp_text(raw)
        self.assertIn("HLA-B27", out)
        self.assertIn("Ankylosing Spondylitis", out)

    def test_04_diabetes_types(self):
        raw = "Type 1 Diabetes involves autoimmune beta-cell destruction, whereas Type 2 Diabetes is characterized by peripheral insulin resistance."
        out = format_whatsapp_text(raw)
        self.assertIn("Type 1 Diabetes", out)
        self.assertIn("Type 2 Diabetes", out)

    def test_05_figure_of_eight_bandage(self):
        raw = "Clavicle fractures in children are conservatively managed with a Figure-of-eight bandage or arm sling."
        out = format_whatsapp_text(raw)
        self.assertIn("Figure-of-eight bandage", out)
        self.assertIn("Clavicle fractures", out)

    def test_06_clotting_factors_roman(self):
        raw = "Hemophilia A is an X-linked recessive deficiency of Factor VIII, while Hemophilia B is a deficiency of Factor IX."
        out = format_whatsapp_text(raw)
        self.assertIn("Factor VIII", out)
        self.assertIn("Factor IX", out)

    def test_07_cytokines_and_interleukins(self):
        raw = "Activated Th1 cells release IL-2, TNF-alpha, and IFN-gamma to stimulate macrophage activation."
        out = format_whatsapp_text(raw)
        self.assertIn("IL-2", out)
        self.assertIn("TNF-alpha", out)
        self.assertIn("IFN-gamma", out)

    def test_08_ratios_vq_and_pao2_fio2(self):
        raw = "In ARDS, the PaO2/FiO2 ratio falls below 300 mmHg due to severe V/Q mismatch and intrapulmonary shunting."
        out = format_whatsapp_text(raw)
        self.assertIn("PaO2/FiO2 ratio", out)
        self.assertIn("V/Q mismatch", out)

    def test_09_ion_pumps_and_atpases(self):
        raw = "Digitalis inhibits the myocardial Na+/K+-ATPase pump, increasing intracellular Ca2+ via the Na+/Ca2+ exchanger."
        out = format_whatsapp_text(raw)
        self.assertIn("Na+/K+-ATPase", out)
        self.assertIn("Ca2+", out)
        self.assertIn("Na+/Ca2+ exchanger", out)

    def test_10_hba1c_and_glucose(self):
        raw = "An HbA1c level of >= 6.5% confirms the diagnosis of diabetes mellitus."
        out = format_whatsapp_text(raw)
        self.assertIn("HbA1c", out)
        self.assertIn(">= 6.5%", out)

    def test_11_receptors_5ht_and_adrenergic(self):
        raw = "Ondansetron is a selective 5-HT3 receptor antagonist. Prazosin acts on alpha-1A adrenergic receptors, whereas salbutamol is a beta-2 agonist."
        out = format_whatsapp_text(raw)
        self.assertIn("5-HT3 receptor", out)
        self.assertIn("alpha-1A", out)
        self.assertIn("beta-2 agonist", out)

    def test_12_g6pd_deficiency(self):
        raw = "G6PD deficiency leads to episodic hemolytic anemia when exposed to oxidative drugs (e.g. Primaquine, Dapsone)."
        out = format_whatsapp_text(raw)
        self.assertIn("G6PD deficiency", out)
        self.assertIn("Primaquine", out)

    def test_13_men_syndromes(self):
        raw = "MEN 1 involves pituitary, parathyroid, and pancreas tumors. MEN 2A adds medullary thyroid cancer and pheochromocytoma; MEN 2B also features mucosal neuromas."
        out = format_whatsapp_text(raw)
        self.assertIn("MEN 1", out)
        self.assertIn("MEN 2A", out)
        self.assertIn("MEN 2B", out)

    def test_14_vitamin_d_metabolites(self):
        raw = "Liver 25-hydroxylase produces 25-hydroxyvitamin D, and renal 1-alpha-hydroxylase produces active 1,25-dihydroxycholecalciferol."
        out = format_whatsapp_text(raw)
        self.assertIn("25-hydroxyvitamin D", out)
        self.assertIn("1,25-dihydroxycholecalciferol", out)

    def test_15_cyp450_enzymes(self):
        raw = "CYP3A4, CYP2D6, and CYP2C19 metabolize over 70% of clinical pharmaceuticals."
        out = format_whatsapp_text(raw)
        self.assertIn("CYP3A4", out)
        self.assertIn("CYP2D6", out)
        self.assertIn("CYP2C19", out)

    def test_16_chromosomal_translocations(self):
        raw = "CML is defined by the t(9;22) BCR-ABL1 translocation, whereas Burkitt lymphoma involves t(8;14) c-myc translocation."
        out = format_whatsapp_text(raw)
        self.assertIn("t(9;22)", out)
        self.assertIn("t(8;14)", out)

    def test_17_cancer_tnm_staging(self):
        raw = "A patient staged as T2N1M0 is classified as Stage IIIA based on regional lymph node involvement."
        out = format_whatsapp_text(raw)
        self.assertIn("T2N1M0", out)
        self.assertIn("Stage IIIA", out)


class TestAdversarialConversationalPreambles(unittest.TestCase):
    """Stress tests variations in conversational preambles and greeting patterns."""

    def test_01_hi_there_samuel_opener(self):
        raw = "Hi there Samuel, let's explore the pathophysiology of nephrotic syndrome.\n\n📖 *IN-DEPTH EXPLANATION*\n\nPodocyte effacement leads to massive proteinuria."
        out = format_whatsapp_text(raw)
        self.assertNotIn("Hi there Samuel", out)
        self.assertTrue(out.startswith("📖 *IN-DEPTH EXPLANATION*"))

    def test_02_greetings_student_opener(self):
        raw = "Greetings, as requested here is the pathway of gluconeogenesis.\n\n📖 *IN-DEPTH EXPLANATION*\n\nPyruvate carboxylase converts pyruvate to oxaloacetate."
        out = format_whatsapp_text(raw)
        self.assertNotIn("Greetings", out)
        self.assertTrue(out.startswith("📖 *IN-DEPTH EXPLANATION*"))

    def test_03_of_course_opener(self):
        raw = "Of course! Below is the detailed explanation for cardiac action potentials.\n\n📖 *IN-DEPTH EXPLANATION*\n\nPhase 0 involves rapid Na+ influx."
        out = format_whatsapp_text(raw)
        self.assertNotIn("Of course", out)
        self.assertTrue(out.startswith("📖 *IN-DEPTH EXPLANATION*"))

    def test_04_according_to_textbook_opener(self):
        raw = "According to the textbook material in Guyton Physiology,\n\nRenal autoregulation maintains GFR between 80-180 mmHg."
        out = format_whatsapp_text(raw)
        self.assertNotIn("According to the textbook", out)
        self.assertIn("Renal autoregulation maintains GFR", out)


if __name__ == "__main__":
    unittest.main()
