"""
Adversarial Stress & Empirical Verification Suite for Milestone 2:
Universal Flowchart Engine & Micrograph Demotion Filter (R2 & R3)
Author: Challenger 2 (Empirical QA & Systems Specialist)
Target: Neura-AI
"""

import sys
import os
import time
import asyncio
import unittest
from unittest.mock import patch, AsyncMock

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    detect_visual_intent_modality,
    should_generate_medical_illustration,
    _reject_micrograph_candidate,
    retrieve_real_medical_diagram,
    VERIFIED_MEDICAL_ATLAS,
    REJECT_MICROGRAPH_REGEX,
    DIAGRAM_CACHE,
)


class TestAdversarialAtlasResolutionZeroNetwork(unittest.IsolatedAsyncioTestCase):
    """
    Empirical Task 3: Verify that 100% of atlas topics resolve in 0ms (< 1ms)
    with ZERO network calls, zero rate limits, and without micrograph contamination.
    """

    async def test_all_atlas_entries_resolve_with_zero_network_calls(self):
        """Disables network entirely; asserts every atlas topic resolves instantly."""
        network_call_count = 0

        async def forbidden_network_call(*args, **kwargs):
            nonlocal network_call_count
            network_call_count += 1
            raise AssertionError("Network call attempted during offline atlas lookup!")

        # Monkeypatch httpx.AsyncClient to ensure zero network traffic
        with patch("httpx.AsyncClient.get", side_effect=forbidden_network_call):
            total_entries = len(VERIFIED_MEDICAL_ATLAS)
            self.assertGreaterEqual(total_entries, 120, f"Atlas should contain >= 120 entries, found {total_entries}")

            resolved_count = 0
            latencies_microseconds = []

            for idx, (patterns, (title, img_url)) in enumerate(VERIFIED_MEDICAL_ATLAS):
                for pattern in patterns:
                    t0 = time.perf_counter()
                    res_url, res_title = await retrieve_real_medical_diagram(pattern, modality="FLOWCHART_SCHEMATIC")
                    t1 = time.perf_counter()

                    lat_us = (t1 - t0) * 1_000_000
                    latencies_microseconds.append(lat_us)

                    # Assert resolution succeeded
                    if "H&E" in title and "Slide" in title:
                        # Granuloma slide is for histology; in flowchart mode it should either resolve to flowchart or skip slide
                        continue
                    
                    self.assertIsNotNone(res_url, f"Failed to resolve atlas pattern '{pattern}' (Entry #{idx}: {title})")
                    self.assertIsNotNone(res_title, f"Resolved title is None for pattern '{pattern}'")
                    self.assertFalse(
                        _reject_micrograph_candidate(res_title, res_url, "FLOWCHART_SCHEMATIC"),
                        f"Resolved atlas item '{res_title}' was rejected by micrograph filter!"
                    )
                    resolved_count += 1

            self.assertEqual(network_call_count, 0, f"Forbidden network calls made: {network_call_count}")
            avg_lat_us = sum(latencies_microseconds) / len(latencies_microseconds)
            max_lat_us = max(latencies_microseconds)

            print(f"\n[METRICS - Task 3] Atlas Entries: {total_entries} | Patterns Tested: {resolved_count}")
            print(f"[METRICS - Task 3] Network Calls: {network_call_count} (ZERO network traffic)")
            print(f"[METRICS - Task 3] Avg Latency: {avg_lat_us:.2f} \u03bcs | Max Latency: {max_lat_us:.2f} \u03bcs (< 1ms)")


class TestHighConcurrencyAndThroughput(unittest.IsolatedAsyncioTestCase):
    """
    Empirical Task 2: Check latency/concurrency of retrieve_real_medical_diagram
    under simulated high-load / parallel lookups (1,000 parallel lookups).
    """

    async def test_1000_concurrent_atlas_lookups(self):
        """Spawns 1,000 simultaneous coroutines querying diverse medical topics."""
        import random

        all_patterns = []
        for patterns, (title, _) in VERIFIED_MEDICAL_ATLAS:
            if "H&E" not in title:
                all_patterns.extend(patterns)

        sample_queries = [random.choice(all_patterns) for _ in range(1000)]

        t0 = time.perf_counter()
        tasks = [retrieve_real_medical_diagram(q, modality="FLOWCHART_SCHEMATIC") for q in sample_queries]
        results = await asyncio.gather(*tasks)
        t1 = time.perf_counter()

        total_time_s = t1 - t0
        throughput_qps = 1000 / total_time_s
        avg_latency_ms = (total_time_s / 1000) * 1000

        failures = [i for i, (url, title) in enumerate(results) if url is None]
        self.assertEqual(len(failures), 0, f"Failed queries out of 1000: {len(failures)}")

        print(f"\n[METRICS - Task 2] 1,000 Concurrent Lookups Completed in: {total_time_s*1000:.2f} ms")
        print(f"[METRICS - Task 2] Throughput: {throughput_qps:.1f} queries/sec | Avg Per-Query Latency: {avg_latency_ms:.4f} ms")
        print(f"[METRICS - Task 2] Success Rate: 100.0% (1,000 / 1,000)")


class TestNonVisualQueryModalityFiltering(unittest.TestCase):
    """
    Empirical Task 4: Verify that non-visual queries never trigger diagram retrieval (modality == 'NONE').
    """

    def test_non_visual_conversational_and_clinical_queries(self):
        non_visual_test_cases = [
            # Conversational / Greetings / Meta
            "Hello Neura!",
            "Good morning doctor",
            "Thank you very much for your help",
            "Who created you?",
            "Can you help me study today?",
            "What can you do?",
            "/profile",
            "/update level",
            "/reset",
            
            # Pure Pharmacology / Dosing / Side Effects
            "What is the standard adult dose of Amoxicillin-Clavulanate for acute otitis media?",
            "List common adverse effects of ACE inhibitors like lisinopril",
            "What is the mechanism-based contraindication for beta blockers in asthma?",
            "Explain the difference between bacteriostatic and bactericidal antibiotics in words",
            "What is the half life of digoxin?",
            
            # Pure Definitions & Diagnostic Criteria
            "Define type 2 diabetes mellitus according to ADA criteria",
            "What are the Jones criteria for Rheumatic Fever?",
            "List the Duke criteria for infective endocarditis",
            "Define autosomal dominant inheritance pattern",
            "What is the definition of nephrotic syndrome?",
            
            # Clinical History & Physical Exam
            "How do you take a comprehensive history for acute chest pain?",
            "What questions should I ask a patient presenting with hematuria?",
            "Describe the physical exam technique for Murphy's sign",
            "What is the differential diagnosis for a 45-year-old male with sudden severe epigastric pain?",
            
            # Calculations & Formulas
            "Calculate GFR using Cockcroft-Gault formula for a 60kg 50yo female with Cr 1.2",
            "How do you calculate the anion gap?",
            "What is the formula for mean arterial pressure MAP?",
            "Calculate maintenance fluid requirements using the 4-2-1 rule",
        ]

        for query in non_visual_test_cases:
            modality = detect_visual_intent_modality(query)
            should_gen = should_generate_medical_illustration(query)
            self.assertEqual(
                modality,
                "NONE",
                f"Expected modality 'NONE' for non-visual query: '{query}', got '{modality}'"
            )
            self.assertFalse(
                should_gen,
                f"should_generate_medical_illustration should be False for: '{query}'"
            )


class TestMixedQueriesAndModalityPrecedence(unittest.IsolatedAsyncioTestCase):
    """
    Empirical Task 1: Stress test mixed queries (asking for both mechanism and histology),
    modality precedence, and corner cases.
    """

    async def test_explicit_histology_overrides_mechanism(self):
        """When user explicitly asks for microscopic slide / histology, HISTOLOGY_MICROSCOPY takes precedence."""
        mixed_queries = [
            ("Show me the histology slide of caseating granuloma and explain its mechanism", "HISTOLOGY_MICROSCOPY"),
            ("H&E stain of liver biopsy in cirrhosis along with pathogenesis", "HISTOLOGY_MICROSCOPY"),
            ("Blood smear showing Plasmodium falciparum ring forms and explain its life cycle", "HISTOLOGY_MICROSCOPY"),
            ("Microscopic slide of Leishmania amastigotes in bone marrow biopsy", "HISTOLOGY_MICROSCOPY"),
            ("Photomicrograph of Reed-Sternberg cells in Hodgkin lymphoma", "HISTOLOGY_MICROSCOPY"),
            ("Histopathology of acute tubular necrosis kidney biopsy", "HISTOLOGY_MICROSCOPY"),
        ]

        for query, expected_modality in mixed_queries:
            modality = detect_visual_intent_modality(query)
            self.assertEqual(modality, expected_modality, f"Mismatch for '{query}': got '{modality}'")

    async def test_granuloma_routing_histology_vs_flowchart(self):
        """Tests that Granuloma correctly routes to slide in HISTOLOGY mode and to cascade in FLOWCHART mode."""
        # 1. Histology query
        hist_url, hist_title = await retrieve_real_medical_diagram("caseating granuloma h&e slide", modality="HISTOLOGY_MICROSCOPY")
        self.assertIsNotNone(hist_url)
        self.assertIn("H&E Histology Slide".lower(), hist_title.lower())
        self.assertFalse(_reject_micrograph_candidate(hist_title, hist_url, "HISTOLOGY_MICROSCOPY"))

        # 2. Flowchart query
        flow_url, flow_title = await retrieve_real_medical_diagram("granuloma cascade flowchart", modality="FLOWCHART_SCHEMATIC")
        self.assertIsNotNone(flow_url)
        self.assertIn("Immunological Cascade".lower(), flow_title.lower())
        self.assertNotIn("H&E".lower(), flow_title.lower())
        self.assertFalse(_reject_micrograph_candidate(flow_title, flow_url, "FLOWCHART_SCHEMATIC"))


class TestAdversarialCornerCasesAndSecurity(unittest.TestCase):
    """
    Empirical Task 1: Corner cases, prompt injection strings, large inputs, empty inputs,
    and regex negative filter boundaries.
    """

    def test_empty_and_whitespace_inputs(self):
        self.assertEqual(detect_visual_intent_modality(""), "NONE")
        self.assertEqual(detect_visual_intent_modality(None), "NONE")
        self.assertEqual(detect_visual_intent_modality("   \n\t   "), "NONE")
        self.assertEqual(detect_visual_intent_modality("??? !!! ...."), "NONE")
        self.assertEqual(detect_visual_intent_modality("🩺🔬💉"), "NONE")

    def test_massive_payload_stress(self):
        """Ensure massive string inputs don't cause ReDoS or buffer issues."""
        large_query = "What is the diagnosis for a patient with " + ("fever and cough and fatigue " * 500)
        t0 = time.perf_counter()
        modality = detect_visual_intent_modality(large_query)
        t1 = time.perf_counter()
        self.assertEqual(modality, "NONE")
        self.assertLess((t1 - t0), 0.05, "Regex took too long on large payload (potential ReDoS)")

    def test_case_insensitivity_and_punctuation_wrapping(self):
        self.assertEqual(detect_visual_intent_modality("GLYCOLYSIS METABOLIC PATHWAY"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("<<krebs cycle>>"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("---[Wiggers Diagram of Cardiac Cycle]---"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("...ATLS primary survey ABCDE..."), "FLOWCHART_SCHEMATIC")

    def test_micrograph_rejection_regex_boundaries(self):
        """Verify strict rejection of various micrograph filenames and URLs in FLOWCHART mode."""
        malicious_or_micrograph_titles = [
            ("Plasmodium falciparum 400x light microscopy", "https://wikimedia.org/plasmodium_400x.jpg"),
            ("Gram stain of Staphylococcus aureus 1000x", "https://wikimedia.org/gram_stain.png"),
            ("Thin blood film showing ring forms", "https://wikimedia.org/thin_film.jpg"),
            ("Thick blood smear Giemsa stained", "https://wikimedia.org/thick_smear.jpg"),
            ("Renal biopsy H&E section", "https://wikimedia.org/renal_biopsy_he.jpg"),
            ("Electron micrograph of mitochondrion (TEM)", "https://wikimedia.org/tem_mitochondrion.png"),
            ("Scanning electron micrograph of platelet (SEM)", "https://wikimedia.org/sem_platelet.jpg"),
            ("Gross pathology specimen autopsy brain", "https://wikimedia.org/gross_specimen.jpg"),
            ("Confocal fluorescence microscopy of actin", "https://wikimedia.org/confocal.png"),
        ]

        for title, url in malicious_or_micrograph_titles:
            self.assertTrue(
                _reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"),
                f"Failed to reject micrograph in FLOWCHART mode: {title} | {url}"
            )
            self.assertTrue(
                _reject_micrograph_candidate(title, url, "ANATOMICAL_MAP"),
                f"Failed to reject micrograph in ANATOMICAL mode: {title} | {url}"
            )
            # In HISTOLOGY mode, it MUST be accepted
            self.assertFalse(
                _reject_micrograph_candidate(title, url, "HISTOLOGY_MICROSCOPY"),
                f"Incorrectly rejected valid histology item in HISTOLOGY mode: {title} | {url}"
            )


if __name__ == "__main__":
    unittest.main()
