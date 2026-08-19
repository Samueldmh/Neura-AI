"""
Automated 50-Topic Multi-Domain Verification Suite (Milestone 3 / R4 & ACs)
NEURA AI Medical Illustration & Diagram Engine
Co-located inside tests/ directory.
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Re-export and execute the main verification suite
from test_multidomain_verification_suite import (
    TestMultiDomainAtlasCoverage,
    TestMicrographRejectionEngine,
    TestConversationalSanitizerAndPromptIntegrity,
    TestWhatsAppFormattingCompliance,
    TestConcurrencyAndThroughput,
)

if __name__ == "__main__":
    import unittest
    unittest.main(verbosity=2)
