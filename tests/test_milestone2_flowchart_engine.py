"""
Comprehensive Unit & Adversarial Test Suite for Milestone 2:
Universal Flowchart Engine & Micrograph Demotion Filter (R2 & R3).

Tests 4 core subsystems:
1. Visual Intent Modality Detection (detect_visual_intent_modality & should_generate_medical_illustration)
2. Micrograph Demotion & Rejection Filter (_reject_micrograph_candidate)
3. 100% Deterministic Resolution across 11 Medical Disciplines in VERIFIED_MEDICAL_ATLAS (140+ topics)
4. Dynamic Wikimedia Query Decoration & Modality-Aware Retrieval
"""

import os
import sys
import unittest
import asyncio

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from main import (
    detect_visual_intent_modality,
    should_generate_medical_illustration,
    _reject_micrograph_candidate,
    retrieve_real_medical_diagram,
    VERIFIED_MEDICAL_ATLAS,
)


class TestVisualIntentModalityDetection(unittest.TestCase):
    """Verifies that visual requests are accurately classified into their respective modalities."""

    def test_01_flowchart_explicit_triggers(self):
        self.assertEqual(detect_visual_intent_modality("Draw a flowchart of Glycolysis"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Show me the pathway of Krebs cycle"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Can you illustrate the stages of labor?"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Decision tree for neonatal resuscitation"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Wiggers diagram of cardiac cycle"), "FLOWCHART_SCHEMATIC")

    def test_02_flowchart_curriculum_pathways(self):
        self.assertEqual(detect_visual_intent_modality("Life cycle of Plasmodium falciparum"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Schistosoma haematobium transmission cycle"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Renin angiotensin aldosterone cascade"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Coagulation cascade intrinsic and extrinsic"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("Complement activation classical and alternative"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("B cell maturation in bone marrow"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("DKA management protocol"), "FLOWCHART_SCHEMATIC")
        self.assertEqual(detect_visual_intent_modality("ATLS primary survey ABCDE"), "FLOWCHART_SCHEMATIC")

    def test_03_histology_microscopy_triggers(self):
        self.assertEqual(detect_visual_intent_modality("Show me the histology slide of caseating granuloma"), "HISTOLOGY_MICROSCOPY")
        self.assertEqual(detect_visual_intent_modality("H&E stain of liver biopsy in cirrhosis"), "HISTOLOGY_MICROSCOPY")
        self.assertEqual(detect_visual_intent_modality("Blood smear showing Plasmodium ring forms"), "HISTOLOGY_MICROSCOPY")
        self.assertEqual(detect_visual_intent_modality("Microscopic slide of Leishmania amastigotes"), "HISTOLOGY_MICROSCOPY")
        self.assertEqual(detect_visual_intent_modality("Photomicrograph of Reed-Sternberg cells"), "HISTOLOGY_MICROSCOPY")
        self.assertEqual(detect_visual_intent_modality("Histopathology of renal tubular necrosis"), "HISTOLOGY_MICROSCOPY")

    def test_04_anatomical_map_triggers(self):
        self.assertEqual(detect_visual_intent_modality("Show me the anatomy of the brachial plexus"), "ANATOMICAL_MAP")
        self.assertEqual(detect_visual_intent_modality("Circle of Willis arterial blood supply"), "ANATOMICAL_MAP")
        self.assertEqual(detect_visual_intent_modality("Inguinal canal walls and spermatic cord relations"), "ANATOMICAL_MAP")
        self.assertEqual(detect_visual_intent_modality("Femoral triangle boundaries and contents"), "ANATOMICAL_MAP")
        self.assertEqual(detect_visual_intent_modality("Calot's triangle surgical anatomy"), "ANATOMICAL_MAP")
        self.assertEqual(detect_visual_intent_modality("Cross section of spinal cord tracts"), "ANATOMICAL_MAP")

    def test_05_non_visual_queries_return_none(self):
        self.assertEqual(detect_visual_intent_modality("What is the standard adult dose of Amoxicillin?"), "NONE")
        self.assertEqual(detect_visual_intent_modality("Define type 2 diabetes mellitus"), "NONE")
        self.assertEqual(detect_visual_intent_modality("List the side effects of lisinopril"), "NONE")
        self.assertEqual(detect_visual_intent_modality("How do you manage hypertension in pregnancy?"), "NONE")
        self.assertEqual(detect_visual_intent_modality("Hello!"), "NONE")

    def test_06_should_generate_medical_illustration_wrapper(self):
        self.assertTrue(should_generate_medical_illustration("Flowchart of Glycolysis"))
        self.assertTrue(should_generate_medical_illustration("Histology slide of lung granuloma"))
        self.assertTrue(should_generate_medical_illustration("Brachial plexus anatomy"))
        self.assertFalse(should_generate_medical_illustration("What is the dose of paracetamol?"))


class TestMicrographRejectionFilter(unittest.TestCase):
    """Verifies that micrographs, blood films, and stains are strictly rejected when requesting schematics."""

    def test_01_rejection_of_micrographs_on_flowchart_mode(self):
        self.assertTrue(_reject_micrograph_candidate("Plasmodium falciparum 01 (Blood Smear).png", "https://upload.wikimedia.org/.../Plasmodium_falciparum_01.png", "FLOWCHART_SCHEMATIC"))
        self.assertTrue(_reject_micrograph_candidate("Giemsa thin blood smear", "https://upload.wikimedia.org/blood_smear.jpg", "FLOWCHART_SCHEMATIC"))
        self.assertTrue(_reject_micrograph_candidate("Caseating granuloma (H&E Micrograph)", "https://upload.wikimedia.org/Granuloma_mac.jpg", "FLOWCHART_SCHEMATIC"))
        self.assertTrue(_reject_micrograph_candidate("Histopathology section 400x magnification", "https://upload.wikimedia.org/slide.jpg", "FLOWCHART_SCHEMATIC"))
        self.assertTrue(_reject_micrograph_candidate("Electron micrograph of T-cell", "https://upload.wikimedia.org/T-cell_microvillus.png", "FLOWCHART_SCHEMATIC"))
        self.assertTrue(_reject_micrograph_candidate("Liver biopsy slide stain", "https://upload.wikimedia.org/biopsy_stain.png", "FLOWCHART_SCHEMATIC"))
        self.assertTrue(_reject_micrograph_candidate("Gross pathology specimen autopsy", "https://upload.wikimedia.org/specimen.jpg", "FLOWCHART_SCHEMATIC"))

    def test_02_acceptance_of_schematics_on_flowchart_mode(self):
        self.assertFalse(_reject_micrograph_candidate("CDC Malaria Life Cycle Schematic", "https://upload.wikimedia.org/CDC_Malaria_LifeCycle.png", "FLOWCHART_SCHEMATIC"))
        self.assertFalse(_reject_micrograph_candidate("Glycolysis Metabolic Pathway Diagram", "https://upload.wikimedia.org/Glycolysis.svg", "FLOWCHART_SCHEMATIC"))
        self.assertFalse(_reject_micrograph_candidate("Wiggers Diagram of Cardiac Cycle", "https://upload.wikimedia.org/Wiggers_Diagram_2.svg", "FLOWCHART_SCHEMATIC"))
        self.assertFalse(_reject_micrograph_candidate("ATLS Primary Survey Resuscitation Algorithm", "https://upload.wikimedia.org/ATLS_Primary_Survey_Algorithm.svg", "FLOWCHART_SCHEMATIC"))
        self.assertFalse(_reject_micrograph_candidate("Complement Activation Cascade", "https://upload.wikimedia.org/Complement_pathway.svg", "FLOWCHART_SCHEMATIC"))

    def test_03_acceptance_of_micrographs_on_histology_mode(self):
        # In histology mode, genuine micrographs and histology plates must NOT be rejected!
        self.assertFalse(_reject_micrograph_candidate("Caseating Tubercular Granuloma (H&E Histology Slide)", "https://upload.wikimedia.org/Granuloma_mac.jpg", "HISTOLOGY_MICROSCOPY"))
        self.assertFalse(_reject_micrograph_candidate("Giemsa thin blood smear", "https://upload.wikimedia.org/blood_smear.jpg", "HISTOLOGY_MICROSCOPY"))
        self.assertFalse(_reject_micrograph_candidate("Liver biopsy H&E section", "https://upload.wikimedia.org/biopsy.jpg", "HISTOLOGY_MICROSCOPY"))


class TestVerifiedMedicalAtlasDisciplineCoverage(unittest.IsolatedAsyncioTestCase):
    """Verifies 100% deterministic resolution across all 11 medical disciplines in VERIFIED_MEDICAL_ATLAS."""

    async def test_01_parasitology_coverage(self):
        topics = [
            ("malaria life cycle", "Malaria Plasmodium"),
            ("plasmodium falciparum life cycle", "Plasmodium falciparum"),
            ("plasmodium vivax hypnozoite cycle", "Plasmodium vivax"),
            ("schistosoma life cycle flowchart", "Schistosoma"),
            ("leishmania promastigote amastigote stages", "Leishmania"),
            ("entamoeba histolytica life cycle", "Entamoeba histolytica"),
            ("trypanosoma brucei sleeping sickness", "Trypanosoma brucei"),
            ("trypanosoma cruzi chagas life cycle", "Trypanosoma cruzi"),
            ("ascaris lumbricoides pulmonary migration", "Ascaris lumbricoides"),
            ("taenia solium cysticercosis cycle", "Taenia solium"),
            ("taenia saginata beef tapeworm", "Taenia saginata"),
            ("echinococcus granulosus hydatid disease", "Echinococcus granulosus"),
            ("wuchereria bancrofti elephantiasis cycle", "Wuchereria bancrofti"),
            ("giardia lamblia trophozoite cyst cycle", "Giardia lamblia"),
            ("toxoplasma gondii life cycle", "Toxoplasma gondii"),
            ("hookworm skin penetration life cycle", "Hookworm"),
            ("strongyloides stercoralis autoinfection cycle", "Strongyloides"),
            ("enterobius vermicularis pinworm life cycle", "Enterobius"),
            ("dracunculus medinensis guinea worm cycle", "Dracunculus"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_02_microbiology_virology_coverage(self):
        topics = [
            ("bacterial endospore formation sporulation", "Endospore Formation"),
            ("gram positive vs gram negative cell wall architecture", "Cell Wall"),
            ("viral replication cycle stages", "Viral Replication"),
            ("hiv replication cycle drug targets", "HIV Replication"),
            ("hepatitis b virus hbv replication", "Hepatitis B"),
            ("bacteriophage lytic vs lysogenic cycle", "Bacteriophage"),
            ("influenza virus replication cycle", "Influenza"),
            ("cholera toxin mechanism of action", "Cholera Toxin"),
            ("bacterial conjugation transformation transduction", "Genetic Exchange"),
            ("tuberculosis pathogenesis granuloma cascade", "Tuberculosis Pathogenesis"),
            ("tetanospasmin tetanus toxin mechanism", "Tetanus Toxin"),
            ("botulinum toxin snare cleavage mechanism", "Botulinum Toxin"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_03_biochemistry_metabolism_coverage(self):
        topics = [
            ("glycolysis metabolic pathway 10 steps", "Glycolysis"),
            ("krebs citric acid cycle tca steps", "Citric Acid Cycle"),
            ("gluconeogenesis pathway bypass reactions", "Gluconeogenesis"),
            ("pentose phosphate pathway hmp shunt", "Pentose Phosphate"),
            ("urea cycle ammonia detoxification", "Urea Cycle"),
            ("beta oxidation of fatty acids carnitine shuttle", "Beta-Oxidation"),
            ("electron transport chain oxidative phosphorylation", "Electron Transport Chain"),
            ("purine synthesis salvage hgprt pathway", "Purine"),
            ("pyrimidine synthesis pathway cad complex", "Pyrimidine"),
            ("glycogenolysis and glycogenesis metabolism", "Glycogen Metabolism"),
            ("cholesterol synthesis mevalonate pathway", "Cholesterol Biosynthesis"),
            ("adrenal steroidogenesis pathway 21-hydroxylase", "Steroidogenesis"),
            ("cori cycle lactate gluconeogenesis", "Cori Cycle"),
            ("glucose-alanine cycle cahill cycle", "Glucose-Alanine"),
            ("ethanol metabolism alcohol dehydrogenase", "Ethanol Metabolism"),
            ("heme synthesis pathway porphyria", "Heme Biosynthesis"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_04_physiology_coverage(self):
        topics = [
            ("cardiac ventricular action potential phases", "Ventricular Myocyte"),
            ("cardiac pacemaker sa node action potential", "Pacemaker"),
            ("wiggers diagram cardiac cycle", "Wiggers Diagram"),
            ("cardiac electrical conduction pathway", "Cardiac Electrical Conduction"),
            ("raas renin angiotensin aldosterone cascade", "Renin-Angiotensin-Aldosterone"),
            ("neuromuscular junction excitation-contraction coupling", "Neuromuscular Junction"),
            ("countercurrent multiplier loop of henle", "Countercurrent Multiplier"),
            ("oxyhemoglobin dissociation curve bohr effect", "Oxyhemoglobin Dissociation"),
            ("neuronal action potential depolarization repolarization", "Neuronal Action Potential"),
            ("glomerular filtration tubular transport nephron", "Glomerular Filtration"),
            ("baroreceptor reflex arc blood pressure", "Baroreceptor Reflex"),
            ("brainstem respiratory control centers", "Respiratory Control"),
            ("sarcomere sliding filament mechanism cross-bridge", "Sarcomere"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_05_pharmacology_coverage(self):
        topics = [
            ("gpcr gs gi gq second messenger cascades", "GPCR"),
            ("receptor tyrosine kinase rtk mapk signaling", "Tyrosine Kinase"),
            ("jak stat cytokine signaling pathway", "JAK-STAT"),
            ("coagulation cascade anticoagulant targets heparin warfarin", "Coagulation Cascade with Anticoagulant"),
            ("autonomic receptor pathways adrenergic cholinergic", "Autonomic Nervous System"),
            ("beta-blocker mechanism of action", "Beta-Adrenergic Blockers"),
            ("diuretic nephron sites of action", "Diuretics Nephron Sites"),
            ("proton pump inhibitor ppi parietal cell acid secretion", "Proton Pump Inhibitors"),
            ("insulin signaling pi3k akt glut4 translocation", "Insulin Receptor Signaling"),
            ("nitric oxide cgmp pde5 inhibitor vasodilation", "Nitric Oxide"),
            ("local anesthetic sodium channel blockade mechanism", "Local Anesthetics"),
            ("raas inhibitors acei arb pharmacology targets", "Pharmacological Inhibition of RAAS"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_06_immunology_coverage(self):
        topics = [
            ("hematopoiesis blood cell lineage differentiation tree", "Hematopoiesis"),
            ("b cell development vdj recombination maturation", "B-Cell Maturation"),
            ("t cell thymic selection positive negative cortex medulla", "T-Cell Thymic Selection"),
            ("helper t cell th1 th2 th17 treg differentiation", "Helper T-Cell"),
            ("complement system classical alternative lectin pathways", "Complement System Cascades"),
            ("gell and coombs hypersensitivity reactions type i-iv", "Hypersensitivity Reactions"),
            ("mhc class i vs mhc class ii antigen presentation", "MHC Class I vs. MHC Class II"),
            ("tcr activation immunological synapse costimulation", "T-Cell Receptor"),
            ("immunoglobulin igg molecular structure", "Immunoglobulin (IgG)"),
            ("mast cell ige degranulation cascade", "Mast Cell"),
            ("phagocytic respiratory burst nadph oxidase", "Respiratory Burst"),
            ("cytokine storm systemic inflammatory cascade", "Cytokine Storm"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_07_hematology_coverage(self):
        topics = [
            ("primary hemostasis platelet adhesion aggregation plug", "Primary Hemostasis"),
            ("secondary hemostasis coagulation waterfall", "Secondary Hemostasis"),
            ("fibrinolysis cascade tpa plasmin d-dimer", "Fibrinolysis Cascade"),
            ("hemoglobin structure allosteric oxygen binding", "Hemoglobin"),
            ("iron metabolism absorption hepcidin regulation", "Iron Metabolism"),
            ("bilirubin degradation pathway jaundice classification", "Bilirubin Degradation"),
            ("abo and rh blood group compatibility matrix", "ABO and Rh"),
            ("erythropoiesis stages red blood cell maturation", "Erythropoiesis Stages"),
            ("hemolytic anemia diagnostic classification algorithm", "Diagnostic Algorithm for Hemolytic Anemias"),
            ("sickle cell disease pathophysiology vaso-occlusion", "Sickle Cell Disease"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_08_surgery_trauma_coverage(self):
        topics = [
            ("atls primary survey abcde trauma resuscitation", "ATLS Primary Survey"),
            ("burns wallace rule of nines parkland formula", "Burns Wallace Rule of Nines"),
            ("glasgow coma scale gcs triage algorithm", "Glasgow Coma Scale"),
            ("calots triangle cystohepatic triangle cholecystectomy", "Calot's Triangle"),
            ("brachial plexus roots trunks cords branches", "Brachial Plexus"),
            ("inguinal canal direct vs indirect hernia hesselbach", "Inguinal Canal"),
            ("femoral triangle navel anatomy", "Femoral Triangle"),
            ("acute abdomen surgical triage decision algorithm", "Acute Abdomen"),
            ("acute appendicitis alvarado scoring algorithm", "Alvarado Scoring"),
            ("surviving sepsis campaign hour-1 bundle flowchart", "Surviving Sepsis"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_09_obstetrics_gynecology_coverage(self):
        topics = [
            ("menstrual cycle hormonal axis endometrial phases", "Menstrual Cycle"),
            ("cardinal movements of normal labor mechanism", "Cardinal Movements"),
            ("stages of labor progress cervical dilation delivery", "Stages of Labor"),
            ("apgar score neonatal vitality assessment", "APGAR Score"),
            ("bishop score cervical ripening labor induction", "Bishop Score"),
            ("postpartum hemorrhage pph 4 ts algorithm", "Postpartum Hemorrhage"),
            ("preeclampsia eclampsia magnesium sulfate protocol", "Preeclampsia & Eclampsia"),
            ("fetal circulation neonatal transitional shunt closures", "Fetal Circulation"),
            ("who partograph labor monitoring flowchart", "Partograph"),
            ("rh isoimmunization hemolytic disease newborn", "Rhesus Isoimmunization"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_10_pediatrics_coverage(self):
        topics = [
            ("imci childhood illness triage algorithm", "IMCI"),
            ("tetralogy of fallot 4 defects schematic", "Tetralogy of Fallot"),
            ("pediatric developmental milestones timeline 0-5", "Developmental Milestones"),
            ("neonatal resuscitation program nrp algorithm", "Neonatal Resuscitation"),
            ("pediatric dehydration who plan a b c algorithm", "Dehydration"),
            ("bhutani neonatal jaundice phototherapy nomogram", "Bhutani"),
            ("congenital heart defects classification cyanotic acyanotic", "Congenital Heart Defects"),
            ("pediatric advanced life support pals algorithm", "PALS"),
            ("febrile seizure clinical assessment triage", "Febrile Seizure"),
            ("pediatric epi vaccination schedule", "Vaccination Schedule"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_11_internal_medicine_pathology_coverage(self):
        topics = [
            ("meningitis csf diagnostic interpretation algorithm", "Meningitis CSF"),
            ("jvp jugular venous pressure waveform clinical", "Jugular Venous Pressure"),
            ("diabetic ketoacidosis dka protocol management", "Diabetic Ketoacidosis"),
            ("acid-base disorder diagnostic algorithm anion gap", "Acid-Base Disturbance"),
            ("acls adult cardiac arrest algorithm", "ACLS Adult Cardiac Arrest"),
            ("acls tachycardia with pulse algorithm", "ACLS Tachycardia"),
            ("acls bradycardia with pulse algorithm", "ACLS Bradycardia"),
            ("shock classification hemodynamic diagnostic flowchart", "Classification of Shock"),
            ("portacaval anastomoses portal hypertension shunts", "Portacaval Anastomoses"),
            ("tubercular granuloma formation immunological cascade", "Tubercular Granuloma Immunological Cascade"),
            ("atherosclerosis pathogenesis plaque formation cascade", "Atherosclerosis Pathogenesis"),
            ("acute coronary syndrome acs stemi nstemi triage", "Acute Coronary Syndrome"),
            ("circle of willis cerebral arterial network", "Circle of Willis"),
            ("spinal cord cross-section ascending descending tracts", "Spinal Cord"),
            ("cranial meninges dura arachnoid pia layers", "Cranial Meninges"),
            ("epidermis strata cellular layers cross-section", "Epidermis Strata"),
        ]
        for query, expected_snippet in topics:
            url, title = await retrieve_real_medical_diagram(query, modality="FLOWCHART_SCHEMATIC")
            self.assertIsNotNone(url, f"Failed for {query}")
            self.assertIn(expected_snippet.lower(), title.lower(), f"Title mismatch for {query}: got {title}")
            self.assertFalse(_reject_micrograph_candidate(title, url, "FLOWCHART_SCHEMATIC"))

    async def test_12_histology_slide_routing(self):
        # Caseating granuloma histology slide query must route to H&E Histology Slide in HISTOLOGY mode!
        url, title = await retrieve_real_medical_diagram("caseating granuloma h&e slide", modality="HISTOLOGY_MICROSCOPY")
        self.assertIsNotNone(url)
        self.assertIn("H&E Histology Slide".lower(), title.lower())

        # But in FLOWCHART mode, granuloma must route to the Immunological Cascade diagram!
        url_flow, title_flow = await retrieve_real_medical_diagram("granuloma cascade flowchart", modality="FLOWCHART_SCHEMATIC")
        self.assertIsNotNone(url_flow)
        self.assertIn("Immunological Cascade".lower(), title_flow.lower())
        self.assertNotIn("Micrograph".lower(), title_flow.lower())


if __name__ == "__main__":
    unittest.main()
