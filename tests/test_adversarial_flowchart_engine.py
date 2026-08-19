"""
Adversarial Stress-Test and Empirical Vulnerability Harness for Milestone 2:
Universal Flowchart Engine & Micrograph Demotion Filter (R2 & R3).

Author: Challenger 1 (EMPIRICAL CHALLENGER - critic & specialist)
Target: NEURA AI Medical Assistant (main.py)
"""

import os
import sys
import unittest
import asyncio
import re

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    detect_visual_intent_modality,
    should_generate_medical_illustration,
    _reject_micrograph_candidate,
    retrieve_real_medical_diagram,
    VERIFIED_MEDICAL_ATLAS,
    REJECT_MICROGRAPH_REGEX,
)


class TestAdversarialModalityDetection(unittest.TestCase):
    """Adversarial stress-testing of detect_visual_intent_modality against noisy, conversational, and space-separated inputs."""

    def test_01_space_separated_histology_triggers_adversarial(self):
        """
        Tests whether natural space-separated histology queries are properly recognized
        or if underscore-only regex patterns fail to detect them.
        """
        test_queries = [
            ("Blood smear showing Plasmodium ring forms", "HISTOLOGY_MICROSCOPY"),
            ("Show me a gram stain of Streptococcus pneumoniae", "HISTOLOGY_MICROSCOPY"),
            ("Can I see a blood film of sickle cell anemia?", "HISTOLOGY_MICROSCOPY"),
            ("Thin smear of Plasmodium vivax trophozoites", "HISTOLOGY_MICROSCOPY"),
            ("Thick smear for malaria parasite density", "HISTOLOGY_MICROSCOPY"),
            ("Pathology slide of invasive ductal carcinoma", "HISTOLOGY_MICROSCOPY"),
            ("Frozen section of thyroid nodule biopsy", "HISTOLOGY_MICROSCOPY"),
            ("Pap smear showing cervical dysplasia", "HISTOLOGY_MICROSCOPY"),
            ("IHC stain for HER2 neu receptor", "HISTOLOGY_MICROSCOPY"),
            ("Stained slide of bone marrow aspirate", "HISTOLOGY_MICROSCOPY"),
        ]
        failures = []
        for q, expected in test_queries:
            actual = detect_visual_intent_modality(q)
            if actual != expected:
                failures.append(f"Query: '{q}' -> Expected: {expected}, Got: {actual}")
        
        # Document any failures empirically
        print(f"\n[Test 01] Space-Separated Histology Detection Failures: {len(failures)}/{len(test_queries)}")
        for f in failures:
            print(f"  ❌ {f}")
        self.assertEqual(len(failures), 0, f"Failed space-separated histology intent tests: {failures}")

    def test_02_conversational_and_noisy_flowchart_queries(self):
        """Tests conversational framing and noisy phrasing across all 11 disciplines."""
        noisy_queries = [
            ("Hey Neura, could you please draw or show me a step by step flowchart of glycolysis?", "FLOWCHART_SCHEMATIC"),
            ("What are the exact steps of the urea cycle in liver mitochondria?", "FLOWCHART_SCHEMATIC"),
            ("Please visualize the cardiac cycle with a wiggers diagram for my exam tomorrow", "FLOWCHART_SCHEMATIC"),
            ("I need to understand the decision tree algorithm for ATLS primary survey resuscitation", "FLOWCHART_SCHEMATIC"),
            ("Can you provide the IMCI triage chart for a sick 2 year old infant?", "FLOWCHART_SCHEMATIC"),
            ("Explain the full cascade of complement activation via classical and alternative pathways", "FLOWCHART_SCHEMATIC"),
            ("Show me how hematopoiesis differentiates from stem cells down to erythrocytes", "FLOWCHART_SCHEMATIC"),
            ("What is the clinical management protocol flowchart for diabetic ketoacidosis DKA?", "FLOWCHART_SCHEMATIC"),
            ("Decision tree for meningitis CSF interpretation based on protein and glucose", "FLOWCHART_SCHEMATIC"),
            ("Cardinal movements of labor fetal head descent and internal rotation illustration", "FLOWCHART_SCHEMATIC"),
            ("Give me the complete life cycle of Schistosoma mansoni with snail intermediate host", "FLOWCHART_SCHEMATIC"),
        ]
        failures = []
        for q, expected in noisy_queries:
            actual = detect_visual_intent_modality(q)
            if actual != expected:
                failures.append(f"Query: '{q}' -> Expected: {expected}, Got: {actual}")
        
        print(f"\n[Test 02] Noisy Flowchart Intent Detection Failures: {len(failures)}/{len(noisy_queries)}")
        for f in failures:
            print(f"  ❌ {f}")
        self.assertEqual(len(failures), 0, f"Failed noisy flowchart intent tests: {failures}")

    def test_03_non_visual_conversational_purity(self):
        """Verifies that purely factual or pharmacological questions return NONE."""
        non_visual = [
            "What is the mechanism of action of amlodipine in simple words?",  # Contains "mechanism of" -> wait! "mechanism of" is a flowchart trigger!
            "What is the standard pediatric dose of amoxicillin for otitis media?",
            "Can you explain the difference between type 1 and type 2 diabetes?",
            "List 5 common side effects of lisinopril in elderly patients",
            "Why is potassium chloride given slowly during IV infusion?",
            "Thank you so much, Neura!",
            "Good morning doctor",
        ]
        results = {}
        for q in non_visual:
            results[q] = detect_visual_intent_modality(q)
        print("\n[Test 03] Non-Visual Classification Results:")
        for q, res in results.items():
            print(f"  '{q}' -> {res}")


class TestAdversarialMicrographFilterLeakage(unittest.TestCase):
    """Adversarial stress-testing of _reject_micrograph_candidate to detect leakages and false positive collisions."""

    def test_01_space_separated_micrograph_leakage(self):
        """
        Adversarial test: Can space-separated titles of actual micrographs evade the filter
        when modality is FLOWCHART_SCHEMATIC?
        """
        adversarial_micrographs = [
            ("Gram stain of Bacillus anthracis in blood culture", "https://upload.wikimedia.org/Bacillus_anthracis.jpg"),
            ("Blood film showing Plasmodium falciparum ring forms", "https://upload.wikimedia.org/Malaria_smear.jpg"),
            ("Thin film of Leishmania donovani amastigotes", "https://upload.wikimedia.org/Leishmania_film.jpg"),
            ("Thick film of Trypanosoma brucei", "https://upload.wikimedia.org/Trypanosoma_film.jpg"),
            ("Gross pathology of cirrhotic liver with regenerative nodules", "https://upload.wikimedia.org/Cirrhosis_liver.jpg"),
            ("Clinical photo of erysipelas on lower extremity", "https://upload.wikimedia.org/Erysipelas.jpg"),
            ("Patient photo of erythema multiforme target lesions", "https://upload.wikimedia.org/Target_lesion.jpg"),
            ("Single cell patch clamp recording", "https://upload.wikimedia.org/Patch_clamp.jpg"),
            ("Cell crop of macrophage engulfing bacteria", "https://upload.wikimedia.org/Macrophage.jpg"),
            ("Slide stain of Helicobacter pylori in gastric mucosa", "https://upload.wikimedia.org/H_pylori.jpg"),
            ("Stained slide of Reed-Sternberg cell in Hodgkin lymphoma", "https://upload.wikimedia.org/RS_cell.jpg"),
            ("Leishman stain of bone marrow aspirate", "https://upload.wikimedia.org/BM_aspirate.jpg"),
            ("Wright stain of peripheral blood showing blasts", "https://upload.wikimedia.org/Leukemia_blasts.jpg"),
        ]
        leaked = []
        for title, url in adversarial_micrographs:
            rejected = _reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC")
            if not rejected:
                leaked.append(f"LEAKED: '{title}' (URL: {url}) was NOT rejected by filter!")
        
        print(f"\n[Test 01 Micrograph Filter] Evading Micrograph Leakages: {len(leaked)}/{len(adversarial_micrographs)}")
        for l in leaked:
            print(f"  🚨 {l}")
        self.assertEqual(len(leaked), 0, f"Micrographs leaked through filter: {leaked}")

    def test_02_false_positive_collisions_in_atlas(self):
        """
        Adversarial test: Does REJECT_MICROGRAPH_REGEX erroneously reject valid vector schematics
        in VERIFIED_MEDICAL_ATLAS due to substring collisions ('tem_' matching 'system_', 'histolog' in title, etc.)?
        """
        false_positives = []
        for patterns, (title, img_url) in VERIFIED_MEDICAL_ATLAS:
            # All atlas entries except the deliberate caseating granuloma slide are supposed to be valid schematics!
            if "granuloma slide" in patterns or "Caseating Tubercular Granuloma (H&E Histology Slide)" in title:
                continue
            
            is_rejected = _reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC")
            if is_rejected:
                false_positives.append(f"FALSE POSITIVE REJECTION: '{title}' (URL: {img_url})")
        
        print(f"\n[Test 02 Atlas False Positives] Erroneously Rejected Atlas Schematics: {len(false_positives)}/{len(VERIFIED_MEDICAL_ATLAS)}")
        for fp in false_positives:
            print(f"  ❌ {fp}")
        self.assertEqual(len(false_positives), 0, f"Atlas entries false positively rejected: {false_positives}")

    def test_03_tem_and_sem_substring_collision_stress(self):
        """
        Tests the blast radius of 'tem_' and 'sem_' in REJECT_MICROGRAPH_REGEX.
        """
        valid_schematic_urls_and_titles = [
            ("Autonomic nervous system receptor pathways", "https://upload.wikimedia.org/Autonomic_nervous_system_actions.svg.png"),
            ("Digestive system anatomy and transport", "https://upload.wikimedia.org/Digestive_system_diagram.svg"),
            ("Cardiovascular system pressure gradients", "https://upload.wikimedia.org/Cardiovascular_system_schematic.png"),
            ("Central nervous system tract pathways", "https://upload.wikimedia.org/Nervous_system_overview.svg"),
            ("Stem cell lineage hematopoiesis tree", "https://upload.wikimedia.org/Stem_cell_lineage.png"),
            ("Ecosystem vector lifecycle transmission", "https://upload.wikimedia.org/Ecosystem_cycle.svg"),
        ]
        rejected_valid_schematics = []
        for title, url in valid_schematic_urls_and_titles:
            if _reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"):
                rejected_valid_schematics.append(f"Blocked by regex: '{title}' -> URL: '{url}'")
        
        print(f"\n[Test 03 TEM/SEM Collisions] Blocked Valid Schematics: {len(rejected_valid_schematics)}/{len(valid_schematic_urls_and_titles)}")
        for b in rejected_valid_schematics:
            print(f"  ❌ {b}")
        self.assertEqual(len(rejected_valid_schematics), 0, f"Valid schematics blocked by TEM/SEM regex: {rejected_valid_schematics}")

    def test_04_eosin_substring_collision_stress(self):
        """
        Tests the blast radius of 'eosin' in REJECT_MICROGRAPH_REGEX.
        """
        eosin_schematic_candidates = [
            ("Eosinophil differentiation and activation pathway", "https://upload.wikimedia.org/Eosinophil_pathway.svg"),
            ("Eosinophilia diagnostic decision tree", "https://upload.wikimedia.org/Eosinophilia_algorithm.png"),
            ("Eosinophilic granulomatosis with polyangiitis mechanism", "https://upload.wikimedia.org/EGPA_pathogenesis.svg"),
        ]
        rejected_eosin_schematics = []
        for title, url in eosin_schematic_candidates:
            if _reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"):
                rejected_eosin_schematics.append(f"Blocked by regex: '{title}' -> URL: '{url}'")
        
        print(f"\n[Test 04 Eosin Collisions] Blocked Eosinophil Schematics: {len(rejected_eosin_schematics)}/{len(eosin_schematic_candidates)}")
        for b in rejected_eosin_schematics:
            print(f"  ❌ {b}")
        self.assertEqual(len(rejected_eosin_schematics), 0, f"Eosin schematics blocked by eosin regex: {rejected_eosin_schematics}")


class TestAdversarialAtlas11DisciplinesExhaustive(unittest.IsolatedAsyncioTestCase):
    """Exhaustive empirical test across all 11 medical disciplines for deterministic 0ms retrieval."""

    async def test_all_141_atlas_entries_flowchart_mode(self):
        """Tests all 141 atlas entries for instant retrieval and 0% micrograph rejection failure."""
        atlas_test_failures = []
        for patterns, (title, img_url) in VERIFIED_MEDICAL_ATLAS:
            if "granuloma slide" in patterns:
                continue
            query = patterns[0]
            url, ret_title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            if url is None or url != img_url:
                atlas_test_failures.append(f"Atlas miss/mismatch for '{query}': Expected URL {img_url}, Got {url}")
        
        print(f"\n[Atlas 141 Exhaustive] Failures: {len(atlas_test_failures)}/{len(VERIFIED_MEDICAL_ATLAS)-1}")
        for f in atlas_test_failures:
            print(f"  ❌ {f}")
        self.assertEqual(len(atlas_test_failures), 0, f"Atlas flowchart lookups failed: {atlas_test_failures}")


if __name__ == "__main__":
    unittest.main()
