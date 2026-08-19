"""
Adversarial Verification Suite for Milestone 3: 50-Topic Verification & Multi-Branch Git Sync
Author: Challenger 2 (Empirical QA & Systems Specialist)
Target: Neura-AI Medical Illustration & Diagram Engine (WhatsApp Cloud API)

Adversarial Stress Test Matrix:
1. Modality Routing across Ambiguous & Boundary Queries (FLOWCHART_SCHEMATIC vs HISTOLOGY_MICROSCOPY vs ANATOMICAL_MAP vs NONE)
2. Markdown Table Conversion to WhatsApp Bullet Cards under Complex/Nested/Malformed Table Structures
3. Bold Word Spacing and Punctuation Edge Cases (*bold* glued to punctuation, brackets, quotes, colons, hyphens)
4. Dynamic Wikimedia Fallback Negative Micrograph Filtering (100% rejection of smears/biopsies on flowchart requests)
5. Simulated Dynamic Wikimedia Pipeline with Micrograph/Schematic Mixed Candidates
6. End-to-End Adversarial Multi-Flaw Response Sanitization
"""

import sys
import os
import re
import asyncio
import unittest
from unittest.mock import patch, AsyncMock

# Add repository root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    detect_visual_intent_modality,
    should_generate_medical_illustration,
    _reject_micrograph_candidate,
    retrieve_real_medical_diagram,
    convert_markdown_tables_to_whatsapp_cards,
    strip_conversational_preambles,
    strip_figure_citations,
    format_whatsapp_text,
    VERIFIED_MEDICAL_ATLAS,
    REJECT_MICROGRAPH_REGEX,
    DIAGRAM_CACHE,
)


class TestAdversarialModalityRoutingAmbiguous(unittest.TestCase):
    """
    Test 1: Modality routing across ambiguous, hybrid, and subtle query formulations.
    Ensures correct classification between FLOWCHART_SCHEMATIC, HISTOLOGY_MICROSCOPY,
    ANATOMICAL_MAP, and NONE.
    """

    def test_explicit_histology_and_microscopy_priority(self):
        """Histology/microscopy queries must route to HISTOLOGY_MICROSCOPY even if they mention steps or diagrams."""
        histology_queries = [
            ("Show me a histology slide of caseating granuloma in TB", "HISTOLOGY_MICROSCOPY"),
            ("H&E stain of renal biopsy in minimal change disease", "HISTOLOGY_MICROSCOPY"),
            ("Giemsa stained thin blood smear showing Plasmodium ring forms", "HISTOLOGY_MICROSCOPY"),
            ("Light microscopy photomicrograph of liver cirrhosis biopsy", "HISTOLOGY_MICROSCOPY"),
            ("Frozen section histopathology of breast carcinoma", "HISTOLOGY_MICROSCOPY"),
            ("Pap smear cytology microscopic slide of cervical intraepithelial neoplasia", "HISTOLOGY_MICROSCOPY"),
            ("Gram stain microscopic slide showing Gram-positive cocci in clusters", "HISTOLOGY_MICROSCOPY"),
            ("Electron micrograph of glomerular basement membrane thickening", "HISTOLOGY_MICROSCOPY"),
            ("Thick blood film microscopy for microfilaria detection", "HISTOLOGY_MICROSCOPY"),
            ("Immunohistochemistry IHC stain for HER2 positive breast tissue", "HISTOLOGY_MICROSCOPY"),
        ]

        for query, expected_modality in histology_queries:
            with self.subTest(query=query):
                modality = detect_visual_intent_modality(query)
                self.assertEqual(
                    modality,
                    expected_modality,
                    f"Query '{query}' expected modality '{expected_modality}', got '{modality}'"
                )

    def test_flowchart_pathway_cycle_algorithm_routing(self):
        """Pathway, cycle, algorithm, and mechanism queries must route to FLOWCHART_SCHEMATIC."""
        flowchart_queries = [
            ("Explain the full enzymatic pathway of glycolysis with all regulatory steps", "FLOWCHART_SCHEMATIC"),
            ("Show me the citric acid cycle metabolic schematic and ATP yield", "FLOWCHART_SCHEMATIC"),
            ("Schistosoma mansoni multi-host life cycle diagram from snail to human", "FLOWCHART_SCHEMATIC"),
            ("ATLS primary survey resuscitation triage flowchart algorithm", "FLOWCHART_SCHEMATIC"),
            ("Cardiac action potential and Wiggers diagram phases", "FLOWCHART_SCHEMATIC"),
            ("Complement cascade classical and alternative activation pathway", "FLOWCHART_SCHEMATIC"),
            ("GPCR Gs/Gq signaling cascade second messenger flowchart", "FLOWCHART_SCHEMATIC"),
            ("IMCI pediatric integrated management algorithm decision tree", "FLOWCHART_SCHEMATIC"),
            ("Cardinal movements of labor mechanism flowchart", "FLOWCHART_SCHEMATIC"),
            ("DKA management protocol and insulin fluid resuscitation algorithm", "FLOWCHART_SCHEMATIC"),
            ("Bacterial endospore 7-stage sporulation cycle", "FLOWCHART_SCHEMATIC"),
            ("HIV replication cycle and antiretroviral drug targets schematic", "FLOWCHART_SCHEMATIC"),
        ]

        for query, expected_modality in flowchart_queries:
            with self.subTest(query=query):
                modality = detect_visual_intent_modality(query)
                self.assertEqual(
                    modality,
                    expected_modality,
                    f"Query '{query}' expected modality '{expected_modality}', got '{modality}'"
                )

    def test_anatomical_map_routing(self):
        """Gross anatomical relationships and spatial cross-sections must route to ANATOMICAL_MAP."""
        anatomy_queries = [
            ("Circle of Willis arterial blood supply anatomical relations", "ANATOMICAL_MAP"),
            ("Brachial plexus cords trunks and terminal branches cross section anatomy", "ANATOMICAL_MAP"),
            ("Inguinal canal boundaries deep ring and contents anatomy", "ANATOMICAL_MAP"),
            ("Cranial meninges dura arachnoid and pia mater layers structure", "ANATOMICAL_MAP"),
            ("Femoral triangle boundaries relations and neurovascular bundle", "ANATOMICAL_MAP"),
            ("Calot cystohepatic triangle boundaries in cholecystectomy anatomy", "ANATOMICAL_MAP"),
            ("Popliteal fossa anatomy and neurovascular relations", "ANATOMICAL_MAP"),
            ("Spinal cord ascending and descending tracts cross section anatomy", "ANATOMICAL_MAP"),
        ]

        for query, expected_modality in anatomy_queries:
            with self.subTest(query=query):
                modality = detect_visual_intent_modality(query)
                self.assertEqual(
                    modality,
                    expected_modality,
                    f"Query '{query}' expected modality '{expected_modality}', got '{modality}'"
                )

    def test_non_visual_conversational_queries_route_to_none(self):
        """Factual, conversational, pharmacological dosing, or conceptual queries must return NONE."""
        non_visual_queries = [
            "What is the starting dose of lisinopril in hypertensive diabetic nephropathy?",
            "Explain the difference between Type 1 and Type 2 statistical errors in epidemiology",
            "Who discovered penicillin and what year was the Nobel prize awarded?",
            "Give me 5 study strategies for MBBS 300L pharmacology exams",
            "What is the definition of odds ratio versus relative risk?",
            "Why is amoxicillin contraindicated in infectious mononucleosis?",
        ]

        for query in non_visual_queries:
            with self.subTest(query=query):
                modality = detect_visual_intent_modality(query)
                self.assertEqual(
                    modality,
                    "NONE",
                    f"Non-visual query '{query}' should return 'NONE', got '{modality}'"
                )
                self.assertFalse(
                    should_generate_medical_illustration(query),
                    f"Non-visual query '{query}' should not trigger illustration generation"
                )


class TestAdversarialMarkdownTableConversion(unittest.TestCase):
    """
    Test 2: Markdown table conversion to WhatsApp bullet cards under complex,
    nested, and malformed table structures.
    """

    def test_4_column_pharmacology_comparison_table(self):
        """Converts a 4-column drug comparison table with alignments into clean bullet cards."""
        raw_table = (
            "| Drug Class | Prototype | Mechanism of Action | Clinical Adverse Effects |\n"
            "|:-----------|:----------|:--------------------|:-------------------------|\n"
            "| ACE Inhibitors | Enalapril | Inhibits ACE decreasing Ang II | Dry cough, Angioedema, Hyperkalemia |\n"
            "| ARBs | Losartan | Selective AT1 receptor antagonist | Hyperkalemia, Hypotension |\n"
            "| Beta Blockers | Metoprolol | Selective Beta-1 adrenergic blocker | Bradycardia, Bronchospasm |"
        )

        formatted = convert_markdown_tables_to_whatsapp_cards(raw_table)

        # Assert no table pipes remaining
        self.assertNotIn("|", formatted, "Raw markdown pipe characters found in output!")
        self.assertNotIn("|:---", formatted, "Alignment row delimiter leaked into output!")

        # Assert primary cards
        self.assertIn("- *ACE Inhibitors*", formatted)
        self.assertIn("  • *Prototype:* Enalapril", formatted)
        self.assertIn("  • *Mechanism of Action:* Inhibits ACE decreasing Ang II", formatted)
        self.assertIn("  • *Clinical Adverse Effects:* Dry cough, Angioedema, Hyperkalemia", formatted)

        self.assertIn("- *ARBs*", formatted)
        self.assertIn("  • *Prototype:* Losartan", formatted)

        self.assertIn("- *Beta Blockers*", formatted)
        self.assertIn("  • *Prototype:* Metoprolol", formatted)

    def test_5_column_microbiology_diagnostic_table(self):
        """Converts a 5-column microbiology diagnostic table with standard headers."""
        raw_table = (
            "| Organism | Gram Stain | Morphology | Catalase | Coagulase |\n"
            "|---|---|---|---|---|\n"
            "| Staphylococcus aureus | Gram-positive | Cocci in clusters | Positive | Positive |\n"
            "| Staphylococcus epidermidis | Gram-positive | Cocci in clusters | Positive | Negative |\n"
            "| Streptococcus pyogenes | Gram-positive | Cocci in chains | Negative | Negative |"
        )

        formatted = convert_markdown_tables_to_whatsapp_cards(raw_table)

        self.assertNotIn("|", formatted)
        self.assertIn("- *Staphylococcus aureus*", formatted)
        self.assertIn("  • *Gram Stain:* Gram-positive", formatted)
        self.assertIn("  • *Morphology:* Cocci in clusters", formatted)
        self.assertIn("  • *Catalase:* Positive", formatted)
        self.assertIn("  • *Coagulase:* Positive", formatted)

        self.assertIn("- *Staphylococcus epidermidis*", formatted)
        self.assertIn("  • *Coagulase:* Negative", formatted)

    def test_table_with_empty_cells_and_missing_values(self):
        """Handles tables where some cells are blank without crashing or generating broken bullets."""
        raw_table = (
            "| Hormone | Source Organ | Target Tissue | Primary Action |\n"
            "|---|---|---|---|\n"
            "| Oxytocin | Posterior Pituitary | Uterus & Breast | |\n"
            "| ADH | Posterior Pituitary | | Promotes water reabsorption in collecting ducts |\n"
            "| Prolactin | Anterior Pituitary | Mammary glands | Stimulates milk production |"
        )

        formatted = convert_markdown_tables_to_whatsapp_cards(raw_table)

        self.assertNotIn("|", formatted)
        self.assertIn("- *Oxytocin*", formatted)
        self.assertIn("  • *Source Organ:* Posterior Pituitary", formatted)
        self.assertIn("  • *Target Tissue:* Uterus & Breast", formatted)

        self.assertIn("- *ADH*", formatted)
        self.assertIn("  • *Source Organ:* Posterior Pituitary", formatted)
        self.assertIn("  • *Primary Action:* Promotes water reabsorption in collecting ducts", formatted)

    def test_multiple_tables_embedded_in_explanatory_prose(self):
        """Correctly converts multiple distinct tables interspersed with clinical prose and headings."""
        full_text = (
            "📖 *IN-DEPTH EXPLANATION*\n\n"
            "Here is the comparison between shock classifications:\n\n"
            "| Shock Type | CVP / JVP | Cardiac Output | SVR |\n"
            "|---|---|---|---|\n"
            "| Hypovolemic | Low | Decreased | Increased |\n"
            "| Cardiogenic | High | Decreased | Increased |\n"
            "| Septic / Distributive | Low/Normal | Increased | Decreased |\n\n"
            "💡 *KEY CLINICAL PEARLS*\n\n"
            "Below is the fluid management summary:\n\n"
            "| Condition | Crystalloid | Colloid | Inotrope |\n"
            "|---|---|---|---|\n"
            "| Sepsis | 30 mL/kg Ringer's | Albumin if refractory | Norepinephrine |\n"
            "| Hemorrhagic | Packed RBCs | FFP + Platelets (1:1:1) | Avoid early |\n\n"
            "📚 *CITATIONS*\n"
            "- Schwartz's Principles of Surgery"
        )

        formatted = format_whatsapp_text(full_text)

        self.assertNotIn("|", formatted)
        self.assertIn("- *Hypovolemic*", formatted)
        self.assertIn("  • *CVP / JVP:* Low", formatted)
        self.assertIn("- *Sepsis*", formatted)
        self.assertIn("  • *Crystalloid:* 30 mL/kg Ringer's", formatted)
        self.assertIn("📖 *IN-DEPTH EXPLANATION*", formatted)
        self.assertIn("💡 *KEY CLINICAL PEARLS*", formatted)
        self.assertIn("📚 *CITATIONS*", formatted)


class TestAdversarialBoldSpacingAndPunctuation(unittest.TestCase):
    """
    Test 3: Bold word spacing edge cases (punctuation, brackets, quotes, colons, orphan asterisks).
    """

    def test_bold_glued_to_parentheses_and_brackets(self):
        """Tokens inside or adjacent to parentheses/brackets must preserve spacing without breaking bold syntax."""
        test_inputs = [
            ("(e.g., *Prazosin* is an alpha blocker)", "(e.g., *Prazosin* is an alpha blocker)"),
            ("First-line therapy (*Metformin*) reduces hepatic gluconeogenesis", "First-line therapy (*Metformin*) reduces hepatic gluconeogenesis"),
            ("The drug of choice is [*Epinephrine*] in anaphylaxis", "The drug of choice is [*Epinephrine*] in anaphylaxis"),
            ("Treatment includes (*Vancomycin* + *Ceftriaxone*) for bacterial meningitis", "Treatment includes (*Vancomycin* + *Ceftriaxone*) for bacterial meningitis"),
        ]

        for raw, expected_pattern in test_inputs:
            with self.subTest(raw=raw):
                formatted = format_whatsapp_text(raw)
                # Verify that bold marks are balanced and not broken
                self.assertIn("*", formatted)
                bold_tokens = re.findall(r'\*[^\*\n]+\*', formatted)
                self.assertGreaterEqual(len(bold_tokens), 1, f"Expected valid bold token in: {formatted}")
                # Ensure no smashed tokens like )is or (e.g.,*
                self.assertNotRegex(formatted, r'\*[a-zA-Z0-9]+\*[a-zA-Z0-9]')

    def test_bold_glued_to_punctuation_and_colons(self):
        """Words glued to asterisks (e.g. word*bold*word, *Heading:*Text) must be cleanly separated."""
        test_cases = [
            ("- *Mechanism of Action:*Inhibits bacterial cell wall synthesis", "- *Mechanism of Action:* Inhibits bacterial cell wall synthesis"),
            ("Administer*Prazosin*for hypertension", "Administer *Prazosin* for hypertension"),
            ("Use*Metformin*,a biguanide agent", "Use *Metformin*, a biguanide agent"),
            ("Key features:*Fever*,*Headache*, and *Neck stiffness*", "Key features: *Fever*, *Headache*, and *Neck stiffness*"),
        ]

        for raw, expected_sub in test_cases:
            with self.subTest(raw=raw):
                formatted = format_whatsapp_text(raw)
                # Assert no glued colons or glued words
                self.assertNotRegex(formatted, r'\*[^\*\n]+\*:[a-zA-Z0-9]', f"Glued colon after bold found in: {formatted}")
                self.assertNotRegex(formatted, r'[a-zA-Z0-9]\*[^\*\n]+\*', f"Word glued to leading asterisk found in: {formatted}")

    def test_orphan_asterisks_cleanup(self):
        """Mathematical asterisks (e.g. 5 * 5 = 25) or stray asterisks are cleaned safely without damaging valid bold."""
        raw_text = (
            "📖 *IN-DEPTH EXPLANATION*\n\n"
            "The calculation is 5 * 10 = 50 mg/kg dose.\n\n"
            "- *Primary Route:* Oral administration with *water*."
        )

        formatted = format_whatsapp_text(raw_text)

        # Valid bold must remain intact
        self.assertIn("📖 *IN-DEPTH EXPLANATION*", formatted)
        self.assertIn("- *Primary Route:*", formatted)
        self.assertIn("*water*", formatted)
        # Math text is preserved cleanly
        self.assertIn("5", formatted)
        self.assertIn("10", formatted)


class TestAdversarialWikimediaNegativeMicrographFilter(unittest.TestCase):
    """
    Test 4: Deterministic rejection of unannotated micrographs, stains, smears,
    biopsies, and clinical photos when modality is FLOWCHART_SCHEMATIC or ANATOMICAL_MAP.
    """

    def test_rejection_of_all_micrograph_and_smear_patterns_for_flowcharts(self):
        """Rejects histology slides, blood smears, biopsy sections, and microscope magnifications."""
        micrograph_candidates = [
            ("Plasmodium_falciparum_ring_form_blood_smear_1000x.jpg", "https://commons.wikimedia.org/wiki/File:smear.jpg"),
            ("Kidney_biopsy_minimal_change_disease_electron_microscopy_TEM.png", "https://commons.wikimedia.org/wiki/File:tem.png"),
            ("Leishmania_donovani_amastigotes_bone_marrow_aspirate_leishman_stain.jpg", "https://commons.wikimedia.org/wiki/File:stain.jpg"),
            ("Schistosoma_mansoni_egg_wet_mount_light_microscopy_400x.jpg", "https://commons.wikimedia.org/wiki/File:egg_400x.jpg"),
            ("Adenocarcinoma_colon_histopathology_h&e_stain_photomicrograph.jpg", "https://commons.wikimedia.org/wiki/File:colon_he.jpg"),
            ("Endoscopic_view_gastric_ulcer_patient_clinical_photo.jpg", "https://commons.wikimedia.org/wiki/File:endoscopy_ulcer.jpg"),
            ("Gross_pathology_specimen_myocardial_infarction_autopsy_macroscopic.jpg", "https://commons.wikimedia.org/wiki/File:autopsy.jpg"),
            ("Trypanosoma_brucei_thin_film_giemsa_stain_100x.jpg", "https://commons.wikimedia.org/wiki/File:giemsa.jpg"),
            ("Bacterial_spore_staining_schaeffer_fulton_stained_slide.jpg", "https://commons.wikimedia.org/wiki/File:spore_slide.jpg"),
            ("Scanning_electron_microscope_SEM_erythrocyte_sickle_cell.jpg", "https://commons.wikimedia.org/wiki/File:sem_cell.jpg"),
        ]

        for title, url in micrograph_candidates:
            with self.subTest(title=title):
                rejected = _reject_micrograph_candidate(title, url, modality="FLOWCHART_SCHEMATIC")
                self.assertTrue(
                    rejected,
                    f"Micrograph candidate '{title}' was NOT rejected under FLOWCHART_SCHEMATIC modality!"
                )

                # Must also be rejected under ANATOMICAL_MAP modality
                rejected_anatomy = _reject_micrograph_candidate(title, url, modality="ANATOMICAL_MAP")
                self.assertTrue(
                    rejected_anatomy,
                    f"Micrograph candidate '{title}' was NOT rejected under ANATOMICAL_MAP modality!"
                )

    def test_acceptance_of_genuine_flowcharts_and_schematics(self):
        """Allows authentic diagrams, vector schematics, and clinical flowcharts under FLOWCHART_SCHEMATIC."""
        flowchart_candidates = [
            ("Citric_acid_cycle_metabolic_pathway_enzymes.svg", "https://commons.wikimedia.org/wiki/File:krebs.svg"),
            ("Malaria_Plasmodium_life_cycle_CDC_vector_diagram.png", "https://commons.wikimedia.org/wiki/File:cdc_malaria.png"),
            ("ATLS_trauma_primary_survey_resuscitation_algorithm.svg", "https://commons.wikimedia.org/wiki/File:atls.svg"),
            ("Cardiac_action_potential_wiggers_diagram.svg", "https://commons.wikimedia.org/wiki/File:wiggers.svg"),
            ("Coagulation_cascade_intrinsic_extrinsic_pathway_schematic.png", "https://commons.wikimedia.org/wiki/File:coag.png"),
            ("Gram_positive_and_negative_cell_wall_diagram.svg", "https://commons.wikimedia.org/wiki/File:cell_wall.svg"),
        ]

        for title, url in flowchart_candidates:
            with self.subTest(title=title):
                rejected = _reject_micrograph_candidate(title, url, modality="FLOWCHART_SCHEMATIC")
                self.assertFalse(
                    rejected,
                    f"Authentic flowchart candidate '{title}' was falsely rejected under FLOWCHART_SCHEMATIC modality!"
                )

    def test_acceptance_of_micrographs_when_histology_modality_is_explicit(self):
        """When user explicitly requests histology/microscopy, genuine histology slides MUST be accepted."""
        histology_candidates = [
            ("Caseating_tubercular_granuloma_h&e_histology_slide.jpg", "https://commons.wikimedia.org/wiki/File:granuloma.jpg"),
            ("Kidney_biopsy_light_microscopy_h&e.jpg", "https://commons.wikimedia.org/wiki/File:biopsy.jpg"),
            ("Plasmodium_falciparum_blood_smear_giemsa.jpg", "https://commons.wikimedia.org/wiki/File:smear.jpg"),
        ]

        for title, url in histology_candidates:
            with self.subTest(title=title):
                rejected = _reject_micrograph_candidate(title, url, modality="HISTOLOGY_MICROSCOPY")
                self.assertFalse(
                    rejected,
                    f"Histology candidate '{title}' was incorrectly rejected under HISTOLOGY_MICROSCOPY modality!"
                )


class TestAdversarialDynamicWikimediaPipeline(unittest.IsolatedAsyncioTestCase):
    """
    Test 5: Simulated dynamic Wikimedia fallback lookup under adversarial API responses.
    """

    async def test_dynamic_fallback_rejects_micrographs_and_picks_schematic(self):
        """
        When Wikipedia returns both a micrograph and a schematic,
        the retriever rejects the micrograph and successfully selects the schematic.
        """
        mock_opensearch_res = AsyncMock()
        mock_opensearch_res.status_code = 200
        mock_opensearch_res.json.return_value = [
            "Test Pathway",
            ["Test Pathway Histology Slide 400x", "Test Pathway Metabolic Schematic Diagram"],
            ["Desc 1", "Desc 2"],
            ["url1", "url2"]
        ]

        def summary_side_effect(url, headers=None):
            mock_res = AsyncMock()
            mock_res.status_code = 200
            if "Histology_Slide" in url:
                mock_res.json.return_value = {
                    "originalimage": {"source": "https://upload.wikimedia.org/wikipedia/commons/1/11/Test_histology_smear_400x.jpg"},
                    "thumbnail": {"source": "https://upload.wikimedia.org/wikipedia/commons/thumb/1/11/Test_histology_smear_400x.jpg/300px.jpg"}
                }
            else:
                mock_res.json.return_value = {
                    "originalimage": {"source": "https://upload.wikimedia.org/wikipedia/commons/2/22/Test_pathway_schematic_diagram.svg"},
                    "thumbnail": {"source": "https://upload.wikimedia.org/wikipedia/commons/thumb/2/22/Test_pathway_schematic_diagram.svg/300px.svg.png"}
                }
            return mock_res

        # Clear test cache
        test_topic = "Uncataloged Novel Biochemical Pathway 2026"
        async def mock_get(url, **kwargs):
            if "opensearch" in url or kwargs.get("params", {}).get("action") == "opensearch":
                return mock_opensearch_res
            if "Histology_Slide" in url:
                return summary_side_effect("Histology_Slide")
            return summary_side_effect("Schematic")

        with patch("httpx.AsyncClient.get", side_effect=mock_get):
            img_url, title = await retrieve_real_medical_diagram(test_topic, modality="FLOWCHART_SCHEMATIC")

            # Assert that the schematic was picked, NOT the micrograph
            self.assertIsNotNone(img_url, "Failed to resolve dynamic schematic candidate!")
            self.assertIn(".svg", img_url, f"Expected vector schematic, got: {img_url}")
            self.assertFalse(
                _reject_micrograph_candidate(title, img_url, "FLOWCHART_SCHEMATIC"),
                f"Retrieved candidate '{title}' failed micrograph check!"
            )


class TestAdversarialFullPipelineSanitization(unittest.TestCase):
    """
    Test 6: Stress-tests format_whatsapp_text against multi-flaw adversarial text
    containing conversational preambles, hallucinated figure citations, raw markdown tables,
    smashed words, unspaced bolding, and trailing page citations simultaneously.
    """

    def test_multi_flaw_adversarial_response_sanitization(self):
        """Sanitizes an adversarial response containing 6 concurrent formatting violations."""
        adversarial_raw_output = (
            "**Certainly Samuel!** Here is the authentic textbook figure and explanation below:\n\n"
            "As shown in Figure 46-9 from Jawetz, the replication cycle proceeds in distinct stages.\n\n"
            "📖 *IN-DEPTH EXPLANATION*\n\n"
            "The initial step involves viral attachment to the host receptor (*CD4*and*CCR5*), as depicted in Fig. 12.8 in Robbins.\n\n"
            "| Drug Class | Target Enzyme | Prototype Drug | Clinical Note |\n"
            "|---|---|---|---|\n"
            "| NRTIs | Reverse Transcriptase | *Zidovudine* | Causes bone marrow suppression |\n"
            "| NNRTIs | Reverse Transcriptase (allosteric) | *Efavirenz* | CNS side effects & vivid dreams |\n"
            "| PIs | HIV Protease | *Ritonavir* | Potent CYP3A4 inhibitor |\n\n"
            "💡 *KEY CLINICAL PEARLS*\n\n"
            "- *Triple Therapy:*Consists of 2 NRTIs + 1 NNRTI or Integrase Inhibitor (see Plate 3-1 for regimen details).\n"
            "- *Monitoring:*Regular viral load checks (refer to Table 14.2 for frequency).\n\n"
            "📚 *CITATIONS*\n"
            "- Jawetz, Melnick, & Adelberg's Medical Microbiology, Figure 46-9, p. 612\n"
            "- Robbins Basic Pathology 10th Edition, Figure 12.8, page 430"
        )

        cleaned = format_whatsapp_text(adversarial_raw_output)

        # 1. Assert preambles completely eliminated
        self.assertNotIn("Certainly Samuel", cleaned)
        self.assertNotIn("Here is the authentic textbook figure", cleaned)
        self.assertNotIn("below:", cleaned[:50])

        # 2. Assert all figure citations completely eradicated
        self.assertNotIn("Figure 46-9", cleaned)
        self.assertNotIn("Fig. 12.8", cleaned)
        self.assertNotIn("Plate 3-1", cleaned)
        self.assertNotIn("Table 14.2", cleaned)
        self.assertNotIn("p. 612", cleaned)
        self.assertNotIn("page 430", cleaned)

        # 3. Assert raw markdown tables converted to bullet cards
        self.assertNotIn("|", cleaned)
        self.assertIn("- *NRTIs*", cleaned)
        self.assertIn("  • *Target Enzyme:* Reverse Transcriptase", cleaned)
        self.assertIn("  • *Prototype Drug:* *Zidovudine*", cleaned)
        self.assertIn("- *NNRTIs*", cleaned)
        self.assertIn("- *PIs*", cleaned)

        # 4. Assert clean bold spacing
        self.assertIn("- *Triple Therapy:* Consists of", cleaned)
        self.assertIn("- *Monitoring:* Regular", cleaned)
        self.assertIn("📖 *IN-DEPTH EXPLANATION*", cleaned)
        self.assertIn("💡 *KEY CLINICAL PEARLS*", cleaned)
        self.assertIn("📚 *CITATIONS*", cleaned)

        # 5. Assert citations are clean textbook titles without figure numbers
        self.assertIn("- Jawetz, Melnick, & Adelberg's Medical Microbiology", cleaned)
        self.assertIn("- Robbins Basic Pathology 10th Edition", cleaned)


if __name__ == "__main__":
    unittest.main(verbosity=2)
