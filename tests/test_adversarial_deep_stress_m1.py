"""
Deep Stress Test Suite for Milestone 1: Medical Classifications & Complex Edge Cases
Target: format_whatsapp_text() and regex pipeline in main.py
Tester: Challenger 1 (critic/specialist)
"""

import os
import sys
import re
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import format_whatsapp_text, strip_conversational_preambles, strip_figure_citations


class TestMilestone1DeepStress(unittest.TestCase):
    """Deep adversarial stress testing on medical classifications, scores, and complex structures."""

    def test_deep_01_hypersensitivity_types_roman_numerals(self):
        raw = """📖 *IN-DEPTH EXPLANATION*

- *Type I Hypersensitivity:* IgE-mediated mast cell degranulation (e.g. anaphylaxis).
- *Type II Hypersensitivity:* IgG/IgM antibody-mediated cytotoxic reaction (e.g. Goodpasture syndrome).
- *Type III Hypersensitivity:* Immune complex deposition (e.g. SLE, serum sickness).
- *Type IV Hypersensitivity:* Cell-mediated delayed hypersensitivity (e.g. contact dermatitis, PPD test)."""
        res = format_whatsapp_text(raw)
        self.assertIn("Type I Hypersensitivity", res)
        self.assertIn("Type II Hypersensitivity", res)
        self.assertIn("Type III Hypersensitivity", res)
        self.assertIn("Type IV Hypersensitivity", res)

    def test_deep_02_coagulation_factors_and_antiarrhythmics(self):
        raw = """📖 *IN-DEPTH EXPLANATION*

- Factor VIIa initiates the extrinsic pathway with Tissue Factor.
- Factor IXa and Factor VIIIa form the tenase complex.
- Factor Xa converts prothrombin (Factor II) to thrombin (Factor IIa).
- Class I antiarrhythmics block Na+ channels.
- Class II antiarrhythmics are beta-blockers.
- Class III antiarrhythmics block K+ channels (Amiodarone).
- Class IV antiarrhythmics block Ca2+ channels (Verapamil, Diltiazem)."""
        res = format_whatsapp_text(raw)
        self.assertIn("Factor VIIa", res)
        self.assertIn("Factor IXa", res)
        self.assertIn("Factor VIIIa", res)
        self.assertIn("Factor Xa", res)
        self.assertIn("Factor IIa", res)
        self.assertIn("Class I antiarrhythmics", res)
        self.assertIn("Class II antiarrhythmics", res)
        self.assertIn("Class III antiarrhythmics", res)
        self.assertIn("Class IV antiarrhythmics", res)

    def test_deep_03_clinical_scoring_systems(self):
        raw = """📖 *IN-DEPTH EXPLANATION*

- *CURB-65 score:* Confusion, Urea >7 mmol/L, RR >=30, BP <90/60, Age >=65.
- *Wells score for DVT:* Assesses clinical probability of deep vein thrombosis.
- *Child-Pugh Class A/B/C:* Evaluates cirrhosis severity based on Bilirubin, Albumin, INR, Ascites, and Encephalopathy.
- *NYHA Class I-IV:* Functional classification of heart failure.
- *APGAR score:* Appearance, Pulse, Grimace, Activity, Respiration."""
        res = format_whatsapp_text(raw)
        self.assertIn("CURB-65 score", res)
        self.assertIn("Wells score for DVT", res)
        self.assertIn("Child-Pugh Class A/B/C", res)
        self.assertIn("NYHA Class I-IV", res)
        self.assertIn("APGAR score", res)

    def test_deep_04_cranial_nerves_and_anatomical_structures(self):
        raw = """📖 *IN-DEPTH EXPLANATION*

- Cranial Nerve III (Oculomotor): Innervates superior rectus, medial rectus, inferior rectus, and inferior oblique.
- Cranial Nerve IV (Trochlear): Innervates superior oblique (SO4).
- Cranial Nerve VI (Abducens): Innervates lateral rectus (LR6).
- Cranial Nerve VII (Facial): Muscles of facial expression and anterior 2/3 taste.
- Cranial Nerve X (Vagus): Parasympathetic innervation to thoracic and abdominal viscera."""
        res = format_whatsapp_text(raw)
        self.assertIn("Cranial Nerve III", res)
        self.assertIn("Cranial Nerve IV", res)
        self.assertIn("Cranial Nerve VI", res)
        self.assertIn("Cranial Nerve VII", res)
        self.assertIn("Cranial Nerve X", res)

    def test_deep_05_complex_table_with_bold_cells(self):
        raw = """| **Stage** | **Morphology** | **Clinical Significance** |
| :---: | :--- | ---: |
| Stage I | Localized tumor <= 2cm | High 5-year survival (>90%) |
| Stage II | Local spread / nodal involvement | Adjuvant chemotherapy indicated |
| Stage III | Regional nodal metastasis | Multimodal therapy |
| Stage IV | Distant metastasis | Palliative systemic therapy |"""
        res = format_whatsapp_text(raw)
        self.assertNotIn("|", res)
        self.assertIn("- *Stage I*", res)
        self.assertIn("• *Morphology:* Localized tumor <= 2cm", res)
        self.assertIn("- *Stage IV*", res)
        self.assertIn("• *Clinical Significance:* Palliative systemic therapy", res)

    def test_deep_06_multiple_intext_figure_stripping(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nAs shown in Figure 1.1 and Fig. 1.2, the cardiac cycle is partitioned into systole and diastole (see Table 3-1 for pressures; also refer to Fig. 4.5)."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Figure 1.1", res)
        self.assertNotIn("Fig. 1.2", res)
        self.assertNotIn("Table 3-1", res)
        self.assertNotIn("Fig. 4.5", res)
        self.assertIn("cardiac cycle is partitioned into systole and diastole", res)

    def test_deep_07_bracketed_and_semicolon_citations(self):
        raw = "📖 *IN-DEPTH EXPLANATION*\n\nReceptor tyrosine kinases dimerize upon ligand binding (Figure 15.2; Table 15.1). Autophosphorylation follows."
        res = format_whatsapp_text(raw)
        self.assertNotIn("Figure 15.2", res)
        self.assertNotIn("Table 15.1", res)
        self.assertIn("Receptor tyrosine kinases dimerize upon ligand binding", res)
        self.assertIn("Autophosphorylation follows.", res)

    def test_deep_08_empty_or_whitespace_resilience(self):
        self.assertEqual(format_whatsapp_text(""), "")
        self.assertEqual(format_whatsapp_text("   "), "")
        self.assertEqual(strip_conversational_preambles(""), "")
        self.assertEqual(strip_figure_citations(""), "")


if __name__ == "__main__":
    unittest.main()
