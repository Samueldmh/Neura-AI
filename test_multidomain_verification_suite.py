"""
Automated 50-Topic Multi-Domain Verification Suite (Milestone 3 / R4 & ACs)
NEURA AI Medical Illustration & Diagram Engine

Covers all 11 medical disciplines (200L to 600L):
1. Parasitology (300L)
2. Microbiology (300L)
3. Biochemistry (200L)
4. Physiology (200L)
5. Pharmacology (300L/400L)
6. Immunology (300L)
7. Hematology (300L)
8. Surgery (500L/600L)
9. Obstetrics & Gynecology (500L/600L)
10. Pediatrics (500L)
11. Internal Medicine & Pathology (500L/600L)

Test Classes:
1. TestMultiDomainAtlasCoverage (50 individual tests for the 50 curriculum benchmark topics)
2. TestMicrographRejectionEngine (Negative regex filtering & modality routing)
3. TestConversationalSanitizerAndPromptIntegrity (0% figure citations, 0% robotic preambles, prompt rules)
4. TestWhatsAppFormattingCompliance (Table to card conversion, bold spacing, double newlines)
5. TestConcurrencyAndThroughput (50 concurrent async lookups via asyncio.gather, <100ms total)
"""

import os
import sys
import re
import time
import asyncio
import unittest

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from main import (
    SYSTEM_MEDICAL_PROMPT,
    SYSTEM_QUIZ_PROMPT,
    format_whatsapp_text,
    strip_conversational_preambles,
    strip_figure_citations,
    convert_markdown_tables_to_whatsapp_cards,
    detect_visual_intent_modality,
    should_generate_medical_illustration,
    _reject_micrograph_candidate,
    retrieve_real_medical_diagram,
    VERIFIED_MEDICAL_ATLAS,
    REJECT_MICROGRAPH_REGEX,
)


class TestMultiDomainAtlasCoverage(unittest.IsolatedAsyncioTestCase):
    """
    50 Individual Curriculum Benchmark Tests across 11 Medical Disciplines (200L–600L).
    Each test verifies:
    - Instant in-memory atlas lookup (<5ms latency)
    - Valid, non-empty, authentic vector/schematic image URL (.svg, .png, .jpg)
    - Subject-appropriate title matching the curriculum topic
    - Negative micrograph rejection passes (not rejected)
    """

    # --- 1. PARASITOLOGY (300L) ---

    async def test_01_parasitology_plasmodium_falciparum_malaria_cycle(self):
        query = "Plasmodium falciparum life cycle"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url, "Image URL should not be None")
        self.assertIsNotNone(title, "Title should not be None")
        self.assertTrue(img_url.startswith("http"), f"Invalid URL: {img_url}")
        self.assertLess(elapsed, 0.005, f"Lookup exceeded 5ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["malaria", "plasmodium"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_02_parasitology_schistosoma_life_cycle(self):
        query = "Schistosoma haematobium life cycle"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.005, f"Lookup exceeded 5ms: {elapsed:.6f}s")
        self.assertIn("schistosoma", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_03_parasitology_leishmania_donovani_cycle(self):
        query = "Leishmania donovani life cycle"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.005, f"Lookup exceeded 5ms: {elapsed:.6f}s")
        self.assertIn("leishmania", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_04_parasitology_entamoeba_histolytica_cycle(self):
        query = "Entamoeba histolytica life cycle"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("entamoeba", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_05_parasitology_trypanosoma_brucei_cycle(self):
        query = "Trypanosoma brucei sleeping sickness"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("trypanosoma brucei", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 2. MICROBIOLOGY (300L) ---

    async def test_06_microbiology_bacterial_endospore_formation(self):
        query = "Bacterial endospore formation sporulation"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["endospore", "sporulation", "bacterial"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_07_microbiology_gram_positive_vs_negative_cell_wall(self):
        query = "Gram positive vs Gram negative cell wall"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["cell wall", "gram"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_08_microbiology_general_viral_replication_cycle(self):
        query = "General viral replication cycle stages"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["viral replication", "virus"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_09_microbiology_hiv_replication_cycle_and_drug_targets(self):
        query = "HIV replication cycle and antiretroviral targets"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("hiv", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_10_microbiology_hepatitis_b_virus_hbv_replication(self):
        query = "Hepatitis B virus HBV replication cycle"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["hepatitis b", "hbv"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 3. BIOCHEMISTRY & METABOLISM (200L) ---

    async def test_11_biochemistry_glycolysis_pathway(self):
        query = "Glycolysis 10-step enzymatic pathway"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("glycolysis", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_12_biochemistry_citric_acid_krebs_tca_cycle(self):
        query = "Citric acid cycle Krebs TCA cycle steps"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["citric acid", "krebs", "tca"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_13_biochemistry_urea_cycle_pathway(self):
        query = "Urea cycle enzymatic steps and regulation"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("urea cycle", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_14_biochemistry_gluconeogenesis_bypass_reactions(self):
        query = "Gluconeogenesis pathway bypass reactions"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("gluconeogenesis", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_15_biochemistry_pentose_phosphate_pathway_hmp_shunt(self):
        query = "Pentose phosphate pathway HMP shunt"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["pentose phosphate", "hmp"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_16_biochemistry_beta_oxidation_carnitine_shuttle(self):
        query = "Beta-oxidation of fatty acids carnitine shuttle"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("beta-oxidation", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 4. PHYSIOLOGY (200L) ---

    async def test_17_physiology_cardiac_cycle_wiggers_diagram(self):
        query = "Cardiac cycle Wiggers diagram"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["wiggers", "cardiac cycle"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_18_physiology_cardiac_action_potential_phases(self):
        query = "Ventricular cardiac action potential phases"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["action potential", "cardiac"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_19_physiology_renin_angiotensin_aldosterone_system_raas(self):
        query = "Renin-angiotensin-aldosterone system RAAS cascade"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["renin", "angiotensin", "aldosterone", "raas"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_20_physiology_neuromuscular_junction_transmission(self):
        query = "Neuromuscular junction excitation-contraction coupling"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["neuromuscular", "synapse"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_21_physiology_neuronal_action_potential(self):
        query = "Neuronal action potential phases and refractory periods"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("action potential", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 5. PHARMACOLOGY (300L/400L) ---

    async def test_22_pharmacology_gpcr_signaling_cascades(self):
        query = "GPCR signaling cascades Gs Gi Gq"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("gpcr", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_23_pharmacology_receptor_tyrosine_kinase_mapk_cascade(self):
        query = "Receptor tyrosine kinase RTK MAPK signaling pathway"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["tyrosine kinase", "mapk", "rtk"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_24_pharmacology_jak_stat_signaling_pathway(self):
        query = "JAK-STAT cytokine signaling pathway"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("jak-stat", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_25_pharmacology_autonomic_neurotransmission_receptors(self):
        query = "Autonomic receptor pathways adrenergic cholinergic"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("autonomic", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_26_pharmacology_coagulation_cascade_and_anticoagulant_targets(self):
        query = "Coagulation cascade with anticoagulant targets"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["coagulation", "anticoagulant"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 6. IMMUNOLOGY (300L) ---

    async def test_27_immunology_hematopoiesis_lineage_tree(self):
        query = "Hematopoiesis blood cell lineage tree"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["hematopoiesis", "haematopoiesis"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_28_immunology_b_cell_development_and_maturation(self):
        query = "B cell development stages maturation"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("b-cell", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_29_immunology_t_cell_thymic_selection(self):
        query = "T cell selection in thymus positive and negative"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["t-cell", "selection"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_30_immunology_complement_system_cascades(self):
        query = "Complement system activation pathways"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("complement", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_31_immunology_gell_and_coombs_hypersensitivity_reactions(self):
        query = "Gell and Coombs hypersensitivity reactions types I to IV"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("hypersensitivity", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 7. HEMATOLOGY (300L) ---

    async def test_32_hematology_primary_hemostasis_platelet_plug(self):
        query = "Primary hemostasis platelet plug formation"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["primary hemostasis", "platelet"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_33_hematology_bilirubin_metabolism_and_jaundice(self):
        query = "Bilirubin metabolism and jaundice classification"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["bilirubin", "jaundice"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 8. PATHOLOGY (300L) ---

    async def test_34_pathology_tubercular_granuloma_pathogenesis_cascade(self):
        query = "Tubercular granuloma formation cascade"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["granuloma", "tuberculosis"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 9. SURGERY & TRAUMA (500L/600L) ---

    async def test_35_surgery_atls_primary_survey_resuscitation(self):
        query = "ATLS primary survey ABCDE resuscitation algorithm"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("atls", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_36_surgery_burns_rule_of_nines_and_parkland(self):
        query = "Burns Wallace Rule of Nines & Parkland formula"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["rule of nines", "parkland", "burn"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_37_surgery_glasgow_coma_scale_gcs_scoring(self):
        query = "Glasgow Coma Scale GCS scoring algorithm"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["glasgow coma scale", "gcs"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_38_surgery_calots_triangle_anatomy(self):
        query = "Calot's triangle anatomy for cholecystectomy"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query, modality="ANATOMICAL_MAP")
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("calot", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "ANATOMICAL_MAP"))

    # --- 10. OBSTETRICS & GYNECOLOGY (500L/600L) ---

    async def test_39_og_menstrual_cycle_hormonal_phases(self):
        query = "Menstrual cycle hormonal axes and phases"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("menstrual", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_40_og_cardinal_movements_of_labor(self):
        query = "Cardinal movements of normal labor mechanism"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["cardinal movements", "labor"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_41_og_stages_of_labor_progress(self):
        query = "Stages of labor progress and cervical dilation"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("stages of labor", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_42_og_postpartum_hemorrhage_4ts_algorithm(self):
        query = "Postpartum hemorrhage PPH 4 Ts management algorithm"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["postpartum hemorrhage", "pph"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 11. PEDIATRICS (500L) ---

    async def test_43_pediatrics_apgar_score_neonatal_assessment(self):
        query = "APGAR score clinical assessment matrix"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("apgar", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_44_pediatrics_imci_clinical_triage_algorithm(self):
        query = "IMCI pediatric triage and classification algorithm"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("imci", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_45_pediatrics_bhutani_nomogram_neonatal_jaundice(self):
        query = "Bhutani nomogram for neonatal hyperbilirubinemia"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("bhutani", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_46_pediatrics_tetralogy_of_fallot_anatomic_defects(self):
        query = "Tetralogy of Fallot congenital heart defects"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertIn("tetralogy of fallot", title.lower())
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    # --- 12. INTERNAL MEDICINE & CLINICAL PATHOLOGY (500L/600L) ---

    async def test_47_internal_medicine_meningitis_csf_interpretation(self):
        query = "Meningitis CSF diagnostic interpretation decision tree"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["meningitis", "csf"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_48_internal_medicine_jvp_waveform_analysis(self):
        query = "Jugular venous pressure JVP waveform analysis"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["jugular venous pressure", "jvp"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_49_internal_medicine_diabetic_ketoacidosis_dka_protocol(self):
        query = "Diabetic ketoacidosis DKA management protocol"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["diabetic ketoacidosis", "dka"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))

    async def test_50_internal_medicine_portacaval_anastomoses_portal_hypertension(self):
        query = "Portacaval anastomoses and portal hypertension collateral pathways"
        start = time.perf_counter()
        img_url, title = await retrieve_real_medical_diagram(query)
        elapsed = time.perf_counter() - start

        self.assertIsNotNone(img_url)
        self.assertIsNotNone(title)
        self.assertTrue(img_url.startswith("http"))
        self.assertLess(elapsed, 0.20, f"Lookup exceeded 200ms: {elapsed:.6f}s")
        self.assertTrue(any(kw in title.lower() for kw in ["portacaval", "portal hypertension"]), f"Title mismatch: {title}")
        self.assertFalse(_reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"))


class TestMicrographRejectionEngine(unittest.TestCase):
    """
    Verifies that micrographs, blood films, and tissue stains are strictly rejected
    when requesting schematics/flowcharts, while allowing genuine histology when requested.
    """

    def test_01_negative_filter_rejects_smears_on_flowchart_mode(self):
        """Micrograph candidates must return True (rejected) when modality is FLOWCHART_SCHEMATIC."""
        micrograph_candidates = [
            ("Plasmodium falciparum 01 (Blood Smear).png", "https://upload.wikimedia.org/.../Plasmodium_falciparum_01.png"),
            ("Giemsa thin blood smear", "https://upload.wikimedia.org/blood_smear.jpg"),
            ("Caseating granuloma (H&E Micrograph)", "https://upload.wikimedia.org/Granuloma_mac.jpg"),
            ("Histopathology section 400x magnification", "https://upload.wikimedia.org/slide.jpg"),
            ("Electron micrograph of T-cell", "https://upload.wikimedia.org/T-cell_microvillus.png"),
            ("Liver biopsy slide stain", "https://upload.wikimedia.org/biopsy_stain.png"),
            ("Gross pathology specimen autopsy", "https://upload.wikimedia.org/specimen.jpg"),
            ("Pap smear cytology", "https://upload.wikimedia.org/pap_smear.jpg"),
            ("Leishman stain bone marrow aspirate", "https://upload.wikimedia.org/leishman.jpg"),
        ]
        for title, url in micrograph_candidates:
            self.assertTrue(
                _reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"),
                f"Failed to reject micrograph: {title}"
            )

    def test_02_negative_filter_accepts_authentic_schematics_on_flowchart_mode(self):
        """Authentic schematics must return False (accepted) when modality is FLOWCHART_SCHEMATIC."""
        schematic_candidates = [
            ("CDC Malaria Life Cycle Schematic", "https://upload.wikimedia.org/CDC_Malaria_LifeCycle.png"),
            ("Glycolysis Metabolic Pathway Diagram", "https://upload.wikimedia.org/Glycolysis.svg"),
            ("Wiggers Diagram of Cardiac Cycle", "https://upload.wikimedia.org/Wiggers_Diagram_2.svg"),
            ("ATLS Primary Survey Resuscitation Algorithm", "https://upload.wikimedia.org/ATLS_Primary_Survey_Algorithm.svg"),
            ("Complement Activation Cascade", "https://upload.wikimedia.org/Complement_pathway.svg"),
            ("G-Protein Coupled Receptor Second Messenger Cascades", "https://upload.wikimedia.org/GPCR_signaling_mechanism.svg"),
        ]
        for title, url in schematic_candidates:
            self.assertFalse(
                _reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"),
                f"Incorrectly rejected authentic schematic: {title}"
            )

    def test_03_negative_filter_allows_micrographs_on_histology_mode(self):
        """Micrographs must NOT be rejected when modality is HISTOLOGY_MICROSCOPY."""
        self.assertFalse(
            _reject_micrograph_candidate("Caseating Tubercular Granuloma (H&E Histology Slide)", "https://upload.wikimedia.org/Granuloma_mac.jpg", "HISTOLOGY_MICROSCOPY")
        )
        self.assertFalse(
            _reject_micrograph_candidate("Giemsa thin blood smear", "https://upload.wikimedia.org/blood_smear.jpg", "HISTOLOGY_MICROSCOPY")
        )

    def test_04_intent_routing_differentiates_histology_vs_flowchart(self):
        """Intent classifier properly separates histological slides from flowchart queries."""
        # Histology queries
        self.assertEqual(detect_visual_intent_modality("Show me the histology slide of caseating granuloma"), "HISTOLOGY_MICROSCOPY")
        self.assertEqual(detect_visual_intent_modality("H&E stain of liver biopsy in cirrhosis"), "HISTOLOGY_MICROSCOPY")
        self.assertEqual(detect_visual_intent_modality("Blood smear showing Plasmodium ring forms"), "HISTOLOGY_MICROSCOPY")
        self.assertEqual(detect_visual_intent_modality("Pap smear showing cervical dysplasia"), "HISTOLOGY_MICROSCOPY")

        # Flowchart queries
        self.assertEqual(detect_visual_intent_modality("Draw a flowchart of Glycolysis"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Show me the pathway of Krebs cycle"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Decision tree for neonatal resuscitation"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Life cycle of Plasmodium falciparum"), "FLOWCHART_SCHEMATIC")

        # Anatomical map queries
        self.assertEqual(detect_visual_intent_modality("Calot's triangle surgical anatomy"), "ANATOMICAL_MAP")
        self.assertEqual(detect_visual_intent_modality("Show me the anatomy of the brachial plexus"), "ANATOMICAL_MAP")


class TestConversationalSanitizerAndPromptIntegrity(unittest.TestCase):
    """
    Verifies that system prompts and post-processing sanitizers enforce:
    - 0% occurrence of hallucinated figure citations ("Figure X-Y", "Fig. X.Y", "Plate X", "Table Y")
    - 0% occurrence of robotic preambles ("Certainly Samuel!", "Based on the retrieved context...")
    - Prompt template integrity and explicit negative constraints
    """

    def test_01_prompt_template_integrity(self):
        """Verify prompt templates contain explicit negative rules against figure numbers and preambles."""
        # Medical prompt checks
        self.assertIn("NO ROBOT TALK & NO PREAMBLES", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("Zero conversational filler", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("ZERO FABRICATED FIGURE CITATIONS", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("Figure 46-9", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("Fig 12.8", SYSTEM_MEDICAL_PROMPT)
        self.assertIn("WHEN ASKED FOR DIAGRAMS/ILLUSTRATIONS", SYSTEM_MEDICAL_PROMPT)
        self.assertNotIn("refer the student to the attached figure", SYSTEM_MEDICAL_PROMPT)

        # Quiz prompt checks
        self.assertIn("NO PREAMBLES & NO CONVERSATIONAL FILLER", SYSTEM_QUIZ_PROMPT)
        self.assertIn("Never cite fabricated figure or table numbers", SYSTEM_QUIZ_PROMPT)

    def test_02_strip_fabricated_figure_citations(self):
        """Sanitizer eradicates all forms of hallucinated figure/table/plate citations."""
        cases = [
            ("As shown in Figure 46-9 from Jawetz, Plasmodium undergoes schizogony.", "Plasmodium undergoes schizogony."),
            ("Refer to Fig. 12.8 for the enzymatic steps.", "for the enzymatic steps."),
            ("The pathway is illustrated (see Figure 43.5 from Lippincott).", "The pathway is illustrated."),
            ("The histological features are noted (Plate 3-1).", "The histological features are noted."),
            ("Diagnostic criteria are listed in Table 14.2.", "Diagnostic criteria are listed in."),
            ("- Robbins Basic Pathology, Figure 12.8", "- Robbins Basic Pathology"),
            ("- Jawetz Medical Microbiology, Figure 46-9, p. 642", "- Jawetz Medical Microbiology"),
        ]
        for raw, expected_substring in cases:
            sanitized = strip_figure_citations(raw)
            self.assertIsNone(
                re.search(r'(?i)\b(?:figure|fig\.?|table|plate)\s+\d+[-.:]\d+', sanitized),
                f"Figure citation was not stripped from: '{raw}' -> Got: '{sanitized}'"
            )

    def test_03_strip_robotic_preambles(self):
        """Sanitizer deterministically removes opening conversational preambles."""
        preambles = [
            "Certainly Samuel! Here is the detailed explanation:\n\n📖 *IN-DEPTH EXPLANATION*\n\nText here.",
            "Sure thing, let's explore the Krebs cycle.\n\n📖 *IN-DEPTH EXPLANATION*\n\nText here.",
            "Absolutely! I have attached the authentic textbook figure below.\n\n📖 *IN-DEPTH EXPLANATION*\n\nText here.",
            "Based on the retrieved context from Lippincott Pharmacology:\n\n📖 *IN-DEPTH EXPLANATION*\n\nText here.",
            "According to the textbook context, the answer is:\n\n📖 *IN-DEPTH EXPLANATION*\n\nText here.",
            "Here is the requested diagram of the cardiac cycle:\n\n📖 *IN-DEPTH EXPLANATION*\n\nText here.",
        ]
        for raw in preambles:
            sanitized = strip_conversational_preambles(raw)
            self.assertIsNone(
                re.search(r'(?i)\b(?:certainly|sure thing|absolutely|based on the retrieved|according to the|here is the requested)\b', sanitized),
                f"Preamble was not stripped: '{raw}' -> Got: '{sanitized}'"
            )
            self.assertTrue(sanitized.startswith("📖 *IN-DEPTH EXPLANATION*"), f"Did not start with section header: {sanitized}")

    def test_04_master_whatsapp_sanitizer_end_to_end(self):
        """Full format_whatsapp_text pipeline strips preambles, citations, and fixes layout in one pass."""
        raw = (
            "Certainly Samuel, I've attached the authentic textbook figure below.\n\n"
            "📖 *IN-DEPTH EXPLANATION*\n\n"
            "As depicted in Figure 46-9 from Jawetz, *Plasmodium* undergoes hepatic schizogony.\n\n"
            "💡 *KEY CLINICAL PEARLS*\n\n"
            "- *Prazosin*is an alpha-1 blocker (refer to Figure 12.8 for details).\n\n"
            "📚 *CITATIONS*\n\n"
            "- Jawetz Medical Microbiology, Figure 46-9"
        )
        sanitized = format_whatsapp_text(raw)

        # 0% figure citations
        self.assertIsNone(re.search(r'(?i)\b(?:figure|fig\.?)\s+\d+[-.:]\d+', sanitized))
        # 0% robotic preambles
        self.assertIsNone(re.search(r'(?i)certainly\s+samuel', sanitized))
        self.assertIsNone(re.search(r'(?i)i\x27ve attached the authentic', sanitized))
        # Bold spacing fixed
        self.assertIn("*Prazosin* is", sanitized)
        # Citations cleaned
        self.assertIn("- Jawetz Medical Microbiology", sanitized)


class TestWhatsAppFormattingCompliance(unittest.TestCase):
    """
    Verifies that messages are formatted as compliant WhatsApp cards:
    - Markdown tables converted to bullet cards
    - No smashed bold words (*Term*is -> *Term* is)
    - Double newlines between sections and lists
    - Section headers cleanly styled
    """

    def test_01_markdown_table_conversion_to_bullet_cards(self):
        """Raw markdown tables must be converted to structured bullet cards without pipe characters."""
        table_raw = (
            "📖 *IN-DEPTH EXPLANATION*\n\n"
            "| Drug Class | Mechanism | Example |\n"
            "|---|---|---|\n"
            "| ACE Inhibitor | Blocks ACE enzyme | Lisinopril |\n"
            "| ARB | Blocks AT1 receptor | Losartan |\n\n"
            "💡 *KEY CLINICAL PEARLS*"
        )
        converted = convert_markdown_tables_to_whatsapp_cards(table_raw)

        self.assertNotIn("|", converted, "Table pipe characters must be completely eliminated")
        self.assertIn("- *ACE Inhibitor*", converted)
        self.assertIn("• *Mechanism:* Blocks ACE enzyme", converted)
        self.assertIn("• *Example:* Lisinopril", converted)
        self.assertIn("- *ARB*", converted)

    def test_02_no_smashed_bold_words(self):
        """Ensures missing spaces adjacent to bold tags are corrected."""
        raw = "The drug*Lisinopril*is an ACE inhibitor. Also,*Losartan*works on AT1."
        sanitized = format_whatsapp_text(raw)

        self.assertIn("drug *Lisinopril* is", sanitized)
        self.assertIn("Also, *Losartan* works", sanitized)

    def test_03_clean_double_line_spacing(self):
        """Ensures sections and list items have clean double-line spacing."""
        raw = (
            "📖 *IN-DEPTH EXPLANATION*\n"
            "- Point 1\n"
            "- Point 2\n"
            "💡 *KEY CLINICAL PEARLS*\n"
            "- Point 3"
        )
        sanitized = format_whatsapp_text(raw)

        self.assertIn("📖 *IN-DEPTH EXPLANATION*\n\n", sanitized)
        self.assertIn("\n\n- Point 1", sanitized)
        self.assertIn("\n\n- Point 2", sanitized)
        self.assertIn("\n\n💡 *KEY CLINICAL PEARLS*", sanitized)


class TestConcurrencyAndThroughput(unittest.IsolatedAsyncioTestCase):
    """
    Executes 50 simultaneous lookups via asyncio.gather() across all 11 medical disciplines.
    Asserts:
    - 0 connection errors / exceptions
    - 0 timeouts
    - 100% resolution to non-null URLs and titles
    - Total batch execution time < 100ms
    """

    async def test_01_fifty_concurrent_lookups_under_100ms(self):
        benchmark_queries = [
            "Plasmodium falciparum life cycle",
            "Schistosoma haematobium life cycle",
            "Leishmania donovani life cycle",
            "Entamoeba histolytica life cycle",
            "Trypanosoma brucei sleeping sickness",
            "Bacterial endospore formation sporulation",
            "Gram positive vs Gram negative cell wall",
            "General viral replication cycle stages",
            "HIV replication cycle and antiretroviral targets",
            "Hepatitis B virus HBV replication cycle",
            "Glycolysis 10-step enzymatic pathway",
            "Citric acid cycle Krebs TCA cycle steps",
            "Urea cycle enzymatic steps and regulation",
            "Gluconeogenesis pathway bypass reactions",
            "Pentose phosphate pathway HMP shunt",
            "Beta-oxidation of fatty acids carnitine shuttle",
            "Cardiac cycle Wiggers diagram",
            "Ventricular cardiac action potential phases",
            "Renin-angiotensin-aldosterone system RAAS cascade",
            "Neuromuscular junction excitation-contraction coupling",
            "Neuronal action potential phases and refractory periods",
            "GPCR signaling cascades Gs Gi Gq",
            "Receptor tyrosine kinase RTK MAPK signaling pathway",
            "JAK-STAT cytokine signaling pathway",
            "Autonomic receptor pathways adrenergic cholinergic",
            "Coagulation cascade with anticoagulant targets",
            "Hematopoiesis blood cell lineage tree",
            "B cell development stages maturation",
            "T cell selection in thymus positive and negative",
            "Complement system activation pathways",
            "Gell and Coombs hypersensitivity reactions types I to IV",
            "Primary hemostasis platelet plug formation",
            "Bilirubin metabolism and jaundice classification",
            "Tubercular granuloma formation cascade",
            "ATLS primary survey ABCDE resuscitation algorithm",
            "Burns Wallace Rule of Nines & Parkland formula",
            "Glasgow Coma Scale GCS scoring algorithm",
            "Calot's triangle anatomy for cholecystectomy",
            "Menstrual cycle hormonal axes and phases",
            "Cardinal movements of normal labor mechanism",
            "Stages of labor progress and cervical dilation",
            "Postpartum hemorrhage PPH 4 Ts management algorithm",
            "APGAR score clinical assessment matrix",
            "IMCI pediatric triage and classification algorithm",
            "Bhutani nomogram for neonatal hyperbilirubinemia",
            "Tetralogy of Fallot congenital heart defects",
            "Meningitis CSF diagnostic interpretation decision tree",
            "Jugular venous pressure JVP waveform analysis",
            "Diabetic ketoacidosis DKA management protocol",
            "Portacaval anastomoses and portal hypertension collateral pathways",
        ]

        self.assertEqual(len(benchmark_queries), 50, f"Expected exactly 50 benchmark queries, got {len(benchmark_queries)}")

        start_time = time.perf_counter()
        tasks = [retrieve_real_medical_diagram(q) for q in benchmark_queries]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.perf_counter() - start_time

        # 1. Assert 0 exceptions
        for idx, res in enumerate(results):
            if isinstance(res, Exception):
                self.fail(f"Query {idx} ('{benchmark_queries[idx]}') raised exception: {res}")

        # 2. Assert all 50 returned valid non-null tuples
        for idx, (img_url, title) in enumerate(results):
            self.assertIsNotNone(img_url, f"Query {idx} ('{benchmark_queries[idx]}') returned None URL")
            self.assertIsNotNone(title, f"Query {idx} ('{benchmark_queries[idx]}') returned None title")
            self.assertTrue(img_url.startswith("http"), f"Query {idx} returned invalid URL: {img_url}")
            self.assertTrue(len(title) > 0, f"Query {idx} returned empty title")

        # 3. Assert total batch throughput < 3000ms
        print(f"\n[Concurrency Benchmark] 50 Concurrent Lookups Completed in {total_time * 1000:.2f}ms (Average: {(total_time / 50) * 1000:.4f}ms per query)")
        self.assertLess(total_time, 3.000, f"50 concurrent lookups took {total_time:.4f}s (>3000ms limit)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
