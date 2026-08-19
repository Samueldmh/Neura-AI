"""
Empirical Adversarial Test Suite for Milestone 1
Target: format_whatsapp_text() and regex sanitization in main.py
Tester: Challenger 1 (critic/specialist)

Tests 42 discrete adversarial scenarios:
- Varied opening preambles, greetings, conversational filler, figure announcements
- Case variations, multiline stacking, punctuation variants
- False positive guards for medical phrases & terminology (Factor VIII, Stage IV, Figure of 8, etc.)
- In-text and parenthetical fabricated figure citations
- Markdown tables to WhatsApp card transformations
- Markdown hash and bold formatting integrity
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
    SYSTEM_MEDICAL_PROMPT,
    SYSTEM_QUIZ_PROMPT,
)


class TestMilestone1Adversarial(unittest.TestCase):
    """Adversarial stress testing for M1 conversational delivery and response sanitization."""

    # =========================================================================
    # Group 1: Adversarial Opening Preambles & Conversational Filler (18 tests)
    # =========================================================================

    def test_adv_01_greeting_with_title_and_exclamation(self):
        raw = "Hello Dr. Samuel! Here is the authentic schematic:\n\n📖 *IN-DEPTH EXPLANATION*\n\nCardiac action potential Phase 0 involves rapid Na+ influx."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Hello Dr. Samuel", res)
        self.assertNotIn("Here is the authentic schematic", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Cardiac action potential Phase 0", res)

    def test_adv_02_stacked_multiline_preambles(self):
        raw = "Hi there!\nCertainly!\nI have attached the authentic textbook figure below:\n\n📖 *IN-DEPTH EXPLANATION*\n\nCitric acid cycle generates 3 NADH, 1 FADH2, and 1 GTP."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Hi there", res)
        self.assertNotIn("Certainly", res)
        self.assertNotIn("attached the authentic", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Citric acid cycle generates", res)

    def test_adv_03_bold_markdown_in_preamble(self):
        raw = "**Certainly, Samuel!** Below is the requested illustration for you:\n\n📖 *IN-DEPTH EXPLANATION*\n\n*Prazosin* is an alpha-1 selective adrenergic blocker."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Certainly, Samuel", res)
        self.assertNotIn("Below is the requested illustration", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("*Prazosin* is an alpha-1 selective", res)

    def test_adv_04_mixed_casing_preamble(self):
        raw = "cErTaiNLy! As requested, here is the diagram.\n\n📖 *IN-DEPTH EXPLANATION*\n\nGlycolysis converts glucose into pyruvate."
        res = format_whatsapp_text(raw)
        self.assertNotIn("cErTaiNLy", res)
        self.assertNotIn("As requested", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Glycolysis converts glucose", res)

    def test_adv_05_of_course_dive_into(self):
        raw = "Of course! Let's dive into the biochemical pathway of heme synthesis.\n\n📖 *IN-DEPTH EXPLANATION*\n\nALA synthase is the rate-limiting enzyme in heme synthesis."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Of course", res)
        self.assertNotIn("Let's dive into", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("ALA synthase is the rate-limiting enzyme", res)

    def test_adv_06_sure_thing_let_us_examine(self):
        raw = "Sure thing! Let us examine the stages of lobar pneumonia:\n\n📖 *IN-DEPTH EXPLANATION*\n\nLobar pneumonia progresses through congestion, red hepatization, grey hepatization, and resolution."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Sure thing", res)
        self.assertNotIn("Let us examine", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Lobar pneumonia progresses", res)

    def test_adv_07_greetings_here_are_the_steps(self):
        raw = "Greetings! Here are the steps of gluconeogenesis as requested:\n\n📖 *IN-DEPTH EXPLANATION*\n\nPyruvate carboxylase converts pyruvate to oxaloacetate."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Greetings", res)
        self.assertNotIn("Here are the steps", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Pyruvate carboxylase converts pyruvate", res)

    def test_adv_08_i_am_attaching_diagram(self):
        raw = "I am attaching the requested flowchart below.\n\n📖 *IN-DEPTH EXPLANATION*\n\nRenin-angiotensin-aldosterone system regulates arterial pressure."
        res = format_whatsapp_text(raw)
        self.assertNotIn("I am attaching", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Renin-angiotensin-aldosterone system", res)

    def test_adv_09_im_attaching_diagram(self):
        raw = "I'm attaching the requested diagram below:\n\n📖 *IN-DEPTH EXPLANATION*\n\nCoagulation cascade involves intrinsic and extrinsic pathways."
        res = format_whatsapp_text(raw)
        self.assertNotIn("I'm attaching", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Coagulation cascade involves", res)

    def test_adv_10_according_to_retrieved_context(self):
        raw = "According to the retrieved context from Robbins Pathology, granulomas contain multinucleated giant cells.\n\n📚 *CITATIONS*\n- Robbins Pathology"
        res = format_whatsapp_text(raw)
        self.assertNotIn("According to the retrieved context", res)
        self.assertIn("granulomas contain multinucleated giant cells", res)

    def test_adv_11_based_on_retrieved_textbook_material(self):
        raw = "Based on retrieved textbook material:\n\n📖 *IN-DEPTH EXPLANATION*\n\nMyocardial infarction leads to coagulative necrosis."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Based on retrieved textbook material", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Myocardial infarction leads to", res)

    def test_adv_12_standalone_certainly_exclamation(self):
        raw = "Certainly!\n\n📖 *IN-DEPTH EXPLANATION*\n\nTetralogy of Fallot comprises VSD, overriding aorta, pulmonary stenosis, and RV hypertrophy."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Certainly", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Tetralogy of Fallot comprises", res)

    def test_adv_13_standalone_hey_there(self):
        raw = "Hey there,\n\n📖 *IN-DEPTH EXPLANATION*\n\nCushing syndrome results from chronic glucocorticoid excess."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Hey there", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Cushing syndrome results from", res)

    def test_adv_14_below_is_diagram_announcement(self):
        raw = "Below is the diagram of the brachial plexus:\n\n📖 *IN-DEPTH EXPLANATION*\n\nRoots originate from C5-T1 anterior rami."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Below is the diagram", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Roots originate from C5-T1", res)

    def test_adv_15_here_are_key_diagnostic_algorithms(self):
        raw = "Here are the key diagnostic algorithms for meningitis:\n\n📖 *IN-DEPTH EXPLANATION*\n\nCSF findings in bacterial meningitis show elevated neutrophils and low glucose."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Here are the key diagnostic algorithms", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("CSF findings in bacterial meningitis", res)

    def test_adv_16_stacked_five_layer_preamble(self):
        raw = "Greetings!\nHello!\nCertainly Samuel!\nSure thing!\nHere is the requested diagram below:\n\n📖 *IN-DEPTH EXPLANATION*\n\nUrea cycle occurs partly in mitochondria and partly in cytoplasm."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Greetings", res)
        self.assertNotIn("Hello", res)
        self.assertNotIn("Certainly Samuel", res)
        self.assertNotIn("Sure thing", res)
        self.assertNotIn("Here is the requested diagram", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Urea cycle occurs partly in mitochondria", res)

    def test_adv_17_preamble_with_leading_whitespace_and_tabs(self):
        raw = "   \t  Certainly! Here is the algorithm below:\n\n📖 *IN-DEPTH EXPLANATION*\n\nGlasgow Coma Scale evaluates Eye, Verbal, and Motor responses."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Certainly", res)
        self.assertNotIn("Here is the algorithm", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Glasgow Coma Scale evaluates", res)

    def test_adv_18_absolutely_look_at(self):
        raw = "Absolutely! Let's look at the mechanism of action of Heparin:\n\n📖 *IN-DEPTH EXPLANATION*\n\nHeparin potentiates Antithrombin III activity."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Absolutely", res)
        self.assertNotIn("Let's look at", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Heparin potentiates Antithrombin III", res)

    # =========================================================================
    # Group 2: Medical Terminology & Legitimate Medical Phrases (14 tests)
    # =========================================================================

    def test_adv_19_sure_signs_of_appendicitis_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nSure signs of acute appendicitis include localized tenderness at McBurney's point and Rovsing's sign."
        res = format_whatsapp_text(raw)
        self.assertIn("Sure signs of acute appendicitis include", res)

    def test_adv_20_hello_like_antigen_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nHello-like antigen configurations do not occur in human leukocytes."
        res = format_whatsapp_text(raw)
        self.assertIn("Hello-like antigen configurations", res)

    def test_adv_21_according_to_starling_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nAccording to Starling's law of the heart, ventricular stroke volume increases with end-diastolic volume."
        res = format_whatsapp_text(raw)
        self.assertIn("According to Starling's law of the heart", res)

    def test_adv_22_based_on_clinical_presentation_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nBased on clinical presentation alone, differentiating bacterial and viral pneumonia is challenging."
        res = format_whatsapp_text(raw)
        self.assertIn("Based on clinical presentation alone", res)

    def test_adv_23_factor_viii_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nFactor VIII acts as a cofactor for Factor IXa in the activation of Factor X."
        res = format_whatsapp_text(raw)
        self.assertIn("Factor VIII", res)
        self.assertIn("Factor IXa", res)
        self.assertIn("Factor X", res)

    def test_adv_24_stage_iv_and_figo_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nStage IV breast cancer and FIGO Stage III cervical carcinoma require systemic therapy."
        res = format_whatsapp_text(raw)
        self.assertIn("Stage IV", res)
        self.assertIn("FIGO Stage III", res)

    def test_adv_25_type_1_diabetes_and_hla_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nType 1 diabetes mellitus is associated with HLA-DR3 and HLA-DR4 alleles."
        res = format_whatsapp_text(raw)
        self.assertIn("Type 1 diabetes mellitus", res)
        self.assertIn("HLA-DR3", res)
        self.assertIn("HLA-DR4", res)

    def test_adv_26_figure_of_eight_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nA Figure of 8 suture or bandage is employed in orthopedic and surgical interventions."
        res = format_whatsapp_text(raw)
        self.assertIn("Figure of 8 suture", res)

    def test_adv_27_table_1_receptors_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nTable 1 receptor families (T1R) mediate umami and sweet taste transduction."
        res = format_whatsapp_text(raw)
        self.assertIn("Table 1 receptor families", res)

    def test_adv_28_plate_boundary_and_platelet_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nPlatelet count below 150,000/mcL defines thrombocytopenia."
        res = format_whatsapp_text(raw)
        self.assertIn("Platelet count below", res)

    def test_adv_29_12_lead_ecg_and_st_elevation_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\n12-lead ECG demonstrates hyperacute T waves followed by ST elevation in leads V1-V4."
        res = format_whatsapp_text(raw)
        self.assertIn("12-lead ECG", res)
        self.assertIn("ST elevation", res)

    def test_adv_30_cd4_cd8_ratio_and_interleukins_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nCD4+ Th1 cells secrete IL-2, IFN-gamma, and TNF-beta to stimulate macrophage killing."
        res = format_whatsapp_text(raw)
        self.assertIn("CD4+ Th1 cells", res)
        self.assertIn("IL-2", res)
        self.assertIn("IFN-gamma", res)
        self.assertIn("TNF-beta", res)

    def test_adv_31_certainly_in_mid_sentence_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nBiopsy will certainly establish the histological diagnosis of amyloidosis via Congo red staining."
        res = format_whatsapp_text(raw)
        self.assertIn("will certainly establish the histological diagnosis", res)

    def test_adv_32_sure_in_mid_sentence_preserved(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nClinicians must make sure that electrolyte imbalances are corrected prior to administering digoxin."
        res = format_whatsapp_text(raw)
        self.assertIn("make sure that electrolyte imbalances", res)

    # =========================================================================
    # Group 3: Fabricated Figure Citations & Inline References (6 tests)
    # =========================================================================

    def test_adv_33_parenthetical_citations_complex(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nDuring schizogony (see Figure 46-9 from Jawetz Medical Microbiology), merozoites rupture RBCs (refer to Fig. 12.8 for details)."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Figure 46-9", res)
        self.assertNotIn("Fig. 12.8", res)
        self.assertIn("merozoites rupture RBCs", res)

    def test_adv_34_as_shown_in_table_and_plate(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nAs shown in Table 14-2 below, the enzyme kinetics follow Michaelis-Menten principles. As depicted in Plate 3.1, spherocytes lack central pallor."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Table 14-2", res)
        self.assertNotIn("Plate 3.1", res)
        self.assertIn("Michaelis-Menten principles", res)
        self.assertIn("spherocytes lack central pallor", res)

    def test_adv_35_figure_demonstrates_clause(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nFigure 43.5 from Lippincott demonstrates the stepwise activation of glycogen phosphorylase."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Figure 43.5", res)
        self.assertIn("stepwise activation of glycogen phosphorylase", res)

    def test_adv_36_citations_trailing_strip_multiple(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nPathophysiology summary.\n\n📚 *CITATIONS*\n- Jawetz Medical Microbiology, Figure 46-9\n- Robbins & Cotran Pathologic Basis of Disease, Fig 12.8\n- Katzung Basic & Clinical Pharmacology, Table 3.2\n- Wheater's Functional Histology, Plate 4.1"
        res = format_whatsapp_text(raw)
        self.assertIn("- Jawetz Medical Microbiology", res)
        self.assertNotIn("Figure 46-9", res)
        self.assertIn("- Robbins & Cotran Pathologic Basis of Disease", res)
        self.assertNotIn("Fig 12.8", res)
        self.assertIn("- Katzung Basic & Clinical Pharmacology", res)
        self.assertNotIn("Table 3.2", res)
        self.assertIn("- Wheater's Functional Histology", res)
        self.assertNotIn("Plate 4.1", res)

    def test_adv_37_standalone_see_fig_clause(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nOsteoclasts resorb bone matrix. See Fig 22-1 for bone remodeling cycle. Osteoblasts synthesize osteoid."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Fig 22-1", res)
        self.assertIn("Osteoclasts resorb bone matrix.", res)
        self.assertIn("Osteoblasts synthesize osteoid.", res)

    def test_adv_38_bracketed_figure_reference(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nComplement pathway activation occurs via three pathways (Figure 2.4). C3 convertase cleaves C3."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Figure 2.4", res)
        self.assertIn("C3 convertase cleaves C3.", res)

    # =========================================================================
    # Group 4: Markdown Tables, Bold Spacing & System Prompts (4 tests)
    # =========================================================================

    def test_adv_39_table_card_conversion_with_multilines(self):
        raw = """| Condition | Etiology | Key Finding |
| :--- | :--- | :--- |
| Hemophilia A | Factor VIII deficiency | Prolonged aPTT, normal PT |
| Von Willebrand Disease | vWF deficiency | Prolonged bleeding time & aPTT |"""
        res = format_whatsapp_text(raw)
        self.assertNotIn("|", res)
        self.assertIn("- *Hemophilia A*", res)
        self.assertIn("• *Etiology:* Factor VIII deficiency", res)
        self.assertIn("• *Key Finding:* Prolonged aPTT, normal PT", res)
        self.assertIn("- *Von Willebrand Disease*", res)
        self.assertIn("• *Etiology:* vWF deficiency", res)

    def test_adv_40_bold_spacing_and_hash_stripping(self):
        raw = "### 📖 IN-DEPTH EXPLANATION\n\nTreatment with**ACE inhibitors**improves*survival*in heart failure."
        res = format_whatsapp_text(raw)
        self.assertNotIn("###", res)
        self.assertNotIn("**", res)
        self.assertIn("*ACE inhibitors* improves *survival*", res)

    def test_adv_41_medical_prompt_zero_preamble_and_zero_fabricated_figures(self):
        self.assertIn("NO ROBOT TALK & NO PREAMBLES", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("ZERO FABRICATED FIGURE CITATIONS", SYSTEM_MEDICAL_PROMPT)
        self.assertNotIn("refer the student to the attached figure", SYSTEM_MEDICAL_PROMPT)

    def test_adv_42_quiz_prompt_zero_preambles_and_figures(self):
        self.assertIn("NO PREAMBLES & NO CONVERSATIONAL FILLER", SYSTEM_QUIZ_PROMPT)
        self.assertIn("Never cite fabricated figure or table numbers", SYSTEM_QUIZ_PROMPT)

    # =========================================================================
    # Group 5: Hostile Markdown-Wrapped Preambles & Multi-layer Stacks (8 tests)
    # =========================================================================

    def test_adv_43_h3_bold_certainly_exclamation(self):
        raw = "### **Certainly!**\n\n📖 *IN-DEPTH EXPLANATION*\n\nKetone bodies (acetoacetate and beta-hydroxybutyrate) are synthesized during fasting."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Certainly", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Ketone bodies (acetoacetate and beta-hydroxybutyrate)", res)

    def test_adv_44_underscore_certainly_samuel(self):
        raw = "__Certainly Samuel!__\n\n📖 *IN-DEPTH EXPLANATION*\n\nAldosterone increases Na+ reabsorption and K+ secretion in the collecting duct."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Certainly Samuel", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Aldosterone increases Na+ reabsorption", res)

    def test_adv_45_mixed_markdown_underscore_and_bold_announcement(self):
        raw = "_Certainly!_\n**Here is the authentic textbook figure below:**\n\n📖 *IN-DEPTH EXPLANATION*\n\nCitric acid cycle generates 3 NADH, 1 FADH2, and 1 GTP."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Certainly", res)
        self.assertNotIn("Here is the authentic", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Citric acid cycle generates 3 NADH", res)

    def test_adv_46_h3_bold_greetings_as_requested(self):
        raw = "### **Greetings!** As requested, here is the diagram:\n\n📖 *IN-DEPTH EXPLANATION*\n\nPyruvate dehydrogenase complex requires TPP, FAD, NAD+, CoA, and Lipoic acid."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Greetings", res)
        self.assertNotIn("As requested", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Pyruvate dehydrogenase complex requires", res)

    def test_adv_47_italic_sure_thing_lets_dive_into(self):
        raw = "*Sure thing, let's dive into the mechanism of action!*\n\n📖 *IN-DEPTH EXPLANATION*\n\nAtropine is a competitive muscarinic receptor antagonist."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Sure thing", res)
        self.assertNotIn("let's dive into", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Atropine is a competitive muscarinic", res)

    def test_adv_48_bold_based_on_retrieved_textbook_context(self):
        raw = "**Based on retrieved textbook context from Robbins:**\n\n📖 *IN-DEPTH EXPLANATION*\n\nCaseous necrosis is characteristic of tuberculosis granulomas."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Based on retrieved textbook context", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Caseous necrosis is characteristic", res)

    def test_adv_49_h3_italic_according_to_material(self):
        raw = "### *According to the textbook material:*\n\n📖 *IN-DEPTH EXPLANATION*\n\nWarfarin inhibits vitamin K epoxide reductase (VKORC1)."
        res = format_whatsapp_text(raw)
        self.assertNotIn("According to the textbook material", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Warfarin inhibits vitamin K epoxide reductase", res)

    def test_adv_50_multi_layer_hostile_markdown_preamble_stack(self):
        raw = "### **Hello Samuel!**\n*_Certainly!_*\n**_Here is the requested pathway below:_**\n\n📖 *IN-DEPTH EXPLANATION*\n\nFatty acid beta-oxidation occurs in mitochondria."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Hello Samuel", res)
        self.assertNotIn("Certainly", res)
        self.assertNotIn("Here is the requested pathway", res)
        self.assertTrue(res.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Fatty acid beta-oxidation occurs in mitochondria.", res)


if __name__ == "__main__":
    unittest.main()

