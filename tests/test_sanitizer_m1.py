"""
Unit test suite for Milestone 1: Natural Conversational Delivery & Response Sanitizer.
Tests 20 comprehensive edge cases covering:
- System prompt verification (anti-preamble and anti-figure citation directives)
- Opening conversational preambles (greetings, polite openers, context phrases)
- Fabricated figure, table, and plate citations (parenthetical, inline, citation suffixes)
- Markdown table conversion to WhatsApp bullet cards
- Bold styling and asterisks preservation
- Combined complex scenarios
- Medical acronym preservation
"""

import os
import sys
import re
import unittest

# Add project root to sys.path to enable importing main
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    SYSTEM_MEDICAL_PROMPT,
    SYSTEM_QUIZ_PROMPT,
    format_whatsapp_text,
    strip_conversational_preambles,
    strip_figure_citations,
    convert_markdown_tables_to_whatsapp_cards,
)


class TestPromptRulesM1(unittest.TestCase):
    """Verifies that system prompts strictly ban fabricated figure citations and preambles."""

    def test_medical_prompt_bans_preambles(self):
        self.assertIn("NO ROBOT TALK & NO PREAMBLES", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("Zero conversational filler", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("📖 *IN-DEPTH EXPLANATION*", SYSTEM_MEDICAL_PROMPT)

    def test_medical_prompt_bans_fabricated_figures(self):
        self.assertIn("ZERO FABRICATED FIGURE CITATIONS", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("Figure 46-9", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("Fig 12.8", SYSTEM_MEDICAL_PROMPT)

    def test_medical_prompt_silent_visual_integration(self):
        self.assertIn("WHEN ASKED FOR DIAGRAMS/ILLUSTRATIONS", SYSTEM_MEDICAL_PROMPT)
        self.assertNotIn("refer the student to the attached figure", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("Do NOT announce or refer to figure numbers", SYSTEM_MEDICAL_PROMPT)

    def test_quiz_prompt_anti_preamble_and_figures(self):
        self.assertIn("NO PREAMBLES & NO CONVERSATIONAL FILLER", SYSTEM_QUIZ_PROMPT)
        self.assertIn("Never cite fabricated figure or table numbers", SYSTEM_QUIZ_PROMPT)


class TestSanitizerM1(unittest.TestCase):
    """Verifies deterministic post-processing sanitization on response text."""

    def test_01_preamble_greeting_with_name(self):
        raw = "Certainly Samuel, here is the life cycle of Plasmodium falciparum.\n\n📖 *IN-DEPTH EXPLANATION*\n\nPlasmodium falciparum undergoes hepatic and erythrocytic schizogony."
        sanitized = format_whatsapp_text(raw)
        self.assertIsNone(re.search(r'(?i)certainly\s+samuel', sanitized))
        self.assertTrue(sanitized.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("hepatic and erythrocytic schizogony", sanitized)

    def test_02_preamble_attached_figure_announcement(self):
        raw = "I've attached the authentic textbook figure below showing the Krebs cycle.\n\n📖 *IN-DEPTH EXPLANATION*\n\nThe citric acid cycle occurs in the mitochondrial matrix."
        sanitized = format_whatsapp_text(raw)
        self.assertIsNone(re.search(r'(?i)i\x27?ve attached', sanitized))
        self.assertTrue(sanitized.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("mitochondrial matrix", sanitized)

    def test_03_preamble_sure_thing_opener(self):
        raw = "Sure thing! Let's examine the mechanism of action of Prazosin.\n\n📖 *IN-DEPTH EXPLANATION*\n\n*Prazosin* is a selective alpha-1 adrenergic antagonist."
        sanitized = format_whatsapp_text(raw)
        self.assertIsNone(re.search(r'(?i)sure thing', sanitized))
        self.assertTrue(sanitized.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("*Prazosin* is a selective", sanitized)

    def test_04_preamble_retrieved_context_opener(self):
        raw = "Based on the retrieved context from Lippincott Pharmacology, Prazosin lowers peripheral vascular resistance.\n\n📚 *CITATIONS*\n- Lippincott Pharmacology"
        sanitized = format_whatsapp_text(raw)
        self.assertIsNone(re.search(r'(?i)based on (?:the )?retrieved context', sanitized))
        self.assertIn("Prazosin lowers peripheral vascular resistance", sanitized)

    def test_05_parenthetical_figure_citation(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nDuring hepatic schizogony (see Figure 46-9 from Jawetz), sporozoites invade hepatocytes."
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("Figure 46-9", sanitized)
        self.assertIn("sporozoites invade hepatocytes", sanitized)

    def test_06_inline_refer_to_fig_citation(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nViral hepatitis leads to Councilman bodies (refer to Fig. 12.8)."
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("Fig. 12.8", sanitized)
        self.assertIn("Councilman bodies", sanitized)

    def test_07_direct_figure_demonstration_clause(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nFigure 43.5 from Lippincott demonstrates this enzymatic step in the pathway."
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("Figure 43.5", sanitized)
        self.assertIn("enzymatic step in the pathway", sanitized)

    def test_08_citations_section_trailing_figure_removal(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nExplanation text here.\n\n📚 *CITATIONS*\n- Jawetz Medical Microbiology, Figure 46-9\n- Robbins Basic Pathology, p. 787, Fig 20.33"
        sanitized = format_whatsapp_text(raw)
        self.assertIn("- Jawetz Medical Microbiology", sanitized)
        self.assertNotIn("Figure 46-9", sanitized)
        self.assertIn("- Robbins Basic Pathology", sanitized)
        self.assertNotIn("Fig 20.33", sanitized)

    def test_09_table_and_plate_citations(self):
        raw = "As outlined in Table 14.2 and Plate 3-1, sickle cell anemia results from a point mutation."
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("Table 14.2", sanitized)
        self.assertNotIn("Plate 3-1", sanitized)
        self.assertIn("sickle cell anemia results from a point mutation", sanitized)

    def test_10_markdown_table_to_whatsapp_cards(self):
        raw = """| Drug | Receptor | Indication |
| --- | --- | --- |
| Prazosin | Alpha-1 | Hypertension |
| Propranolol | Beta-1/2 | Angina |"""
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("|", sanitized)
        self.assertIn("- *Prazosin*", sanitized)
        self.assertIn("• *Receptor:* Alpha-1", sanitized)
        self.assertIn("• *Indication:* Hypertension", sanitized)
        self.assertIn("- *Propranolol*", sanitized)
        self.assertIn("• *Receptor:* Beta-1/2", sanitized)

    def test_11_markdown_table_with_empty_cells(self):
        raw = """| Microorganism | Gram Stain | Key Feature |
| :--- | :--- | :--- |
| Staphylococcus aureus | Gram positive | Catalase positive |
| Escherichia coli | Gram negative |  |"""
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("|", sanitized)
        self.assertIn("- *Staphylococcus aureus*", sanitized)
        self.assertIn("• *Gram Stain:* Gram positive", sanitized)
        self.assertIn("• *Key Feature:* Catalase positive", sanitized)
        self.assertIn("- *Escherichia coli*", sanitized)
        self.assertIn("• *Gram Stain:* Gram negative", sanitized)

    def test_12_bold_and_spacing_fixes(self):
        raw = "The drug*Prazosin*is an**alpha-1 blocker**acting on arterioles."
        sanitized = format_whatsapp_text(raw)
        self.assertIn("*Prazosin* is", sanitized)
        self.assertIn("*alpha-1 blocker*", sanitized)
        self.assertNotIn("**", sanitized)

    def test_13_header_hashes_and_emojis(self):
        raw = "### IN-DEPTH EXPLANATION\n\nHeart failure pathophysiology.\n### KEY CLINICAL PEARLS\n- Treat with ACE inhibitors."
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("#", sanitized)
        self.assertIn("IN-DEPTH EXPLANATION", sanitized)
        self.assertIn("KEY CLINICAL PEARLS", sanitized)

    def test_14_combined_complex_response(self):
        raw = """Certainly Samuel! Here is the authentic textbook figure you requested.

### 📖 IN-DEPTH EXPLANATION

Plasmodium falciparum life cycle involves two hosts (see Figure 46-9 from Jawetz).

| Stage | Location | Diagnostic Form |
| --- | --- | --- |
| Exo-erythrocytic | Liver | Schizont |
| Erythrocytic | RBC | Ring form (trophozoite) |

Refer to Fig. 12.8 for details on parasitemia.

### 📚 CITATIONS
- Jawetz Medical Microbiology, Figure 46-9
- Robbins Basic Pathology, p. 787"""

        sanitized = format_whatsapp_text(raw)
        self.assertIsNone(re.search(r'(?i)certainly\s+samuel', sanitized))
        self.assertIsNone(re.search(r'(?i)here is the authentic', sanitized))
        self.assertNotIn("Figure 46-9", sanitized)
        self.assertNotIn("Fig. 12.8", sanitized)
        self.assertNotIn("|", sanitized)
        self.assertIn("- *Exo-erythrocytic*", sanitized)
        self.assertIn("• *Location:* Liver", sanitized)
        self.assertIn("- *Erythrocytic*", sanitized)
        self.assertIn("• *Location:* RBC", sanitized)
        self.assertIn("- Jawetz Medical Microbiology", sanitized)
        self.assertIn("- Robbins Basic Pathology", sanitized)

    def test_15_clean_response_unmodified(self):
        raw = """📖 *IN-DEPTH EXPLANATION*

The renal clearance of inulin equals GFR because it is freely filtered and neither reabsorbed nor secreted.

💡 *KEY CLINICAL PEARLS*

- Creatinine slightly overestimates GFR due to mild tubular secretion.

📚 *CITATIONS*

- K Sembulingam Essentials of Medical Physiology"""

        sanitized = format_whatsapp_text(raw)
        self.assertTrue(sanitized.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("inulin equals GFR", sanitized)
        self.assertIn("💡 *KEY CLINICAL PEARLS*", sanitized)
        self.assertIn("📚 *CITATIONS*", sanitized)
        self.assertIn("- K Sembulingam Essentials of Medical Physiology", sanitized)

    def test_16_multiline_stacked_preambles(self):
        raw = """Hello Samuel!
Certainly!
Here is the requested diagram and breakdown below:

📖 *IN-DEPTH EXPLANATION*

Glycolysis produces a net yield of 2 ATP and 2 NADH per glucose molecule."""

        sanitized = format_whatsapp_text(raw)
        self.assertIsNone(re.search(r'(?i)hello\s+samuel', sanitized))
        self.assertIsNone(re.search(r'(?i)certainly', sanitized))
        self.assertIsNone(re.search(r'(?i)here is the requested', sanitized))
        self.assertTrue(sanitized.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Glycolysis produces a net yield", sanitized)

    def test_17_standalone_directive_sentence_removal(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nSporozoites enter the liver.\n\nRefer to Figure 12.8 for details on parasitemia.\n\nSchizonts rupture hepatocytes."
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("Figure 12.8", sanitized)
        self.assertNotIn("details on parasitemia", sanitized)
        self.assertIn("Sporozoites enter the liver.", sanitized)
        self.assertIn("Schizonts rupture hepatocytes.", sanitized)

    def test_18_medical_acronyms_unharmed(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nFactor VIII deficiency causes Hemophilia A. CD4+ T-helper cells produce IL-2 and IFN-gamma. HLA-B27 is associated with Ankylosing Spondylitis. 12-lead ECG reveals ST-elevation."
        sanitized = format_whatsapp_text(raw)
        self.assertIn("Factor VIII", sanitized)
        self.assertIn("CD4+", sanitized)
        self.assertIn("IL-2", sanitized)
        self.assertIn("HLA-B27", sanitized)
        self.assertIn("12-lead ECG", sanitized)

    def test_19_markdown_bold_preamble_stripping(self):
        raw = "**Certainly, Samuel!** Below is the requested illustration for you:\n\n📖 *IN-DEPTH EXPLANATION*\n\n*Prazosin* is an alpha-1 selective adrenergic blocker."
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("Certainly, Samuel", sanitized)
        self.assertNotIn("Below is the requested illustration", sanitized)
        self.assertTrue(sanitized.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("*Prazosin* is an alpha-1 selective", sanitized)

    def test_20_markdown_header_and_italic_preamble_stripping(self):
        raw = "### *Certainly!* As requested, below is the breakdown:\n\n📖 *IN-DEPTH EXPLANATION*\n\nGlycolysis converts glucose into pyruvate."
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("Certainly", sanitized)
        self.assertNotIn("below is the breakdown", sanitized)
        self.assertTrue(sanitized.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Glycolysis converts glucose", sanitized)

    def test_21_markdown_underscore_and_announcement_stripping(self):
        raw = "_Certainly!_\n**Here is the authentic textbook figure below:**\n\n📖 *IN-DEPTH EXPLANATION*\n\nCitric acid cycle generates NADH."
        sanitized = format_whatsapp_text(raw)
        self.assertNotIn("Certainly", sanitized)
        self.assertNotIn("Here is the authentic", sanitized)
        self.assertTrue(sanitized.startswith("📖 *IN-DEPTH EXPLANATION*"))
        self.assertIn("Citric acid cycle generates", sanitized)


if __name__ == "__main__":
    unittest.main()

