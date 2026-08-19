"""
Master Test Runner for Neura-AI Medical Illustration & Diagram Engine
Executes all verification suites across Milestones 1, 2, and 3:
- Milestone 1: Natural Conversational Delivery & Prompt Sanitization (M1 unit & adversarial suites)
- Milestone 2: Universal Flowchart Engine & Micrograph Demotion Filter (M2 unit & adversarial suites)
- Milestone 3: 50-Topic Multi-Domain Verification Suite (R4 & ACs across 11 medical disciplines)
"""

import os
import sys
import subprocess
import time
import unittest

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Define all test suites to execute
TEST_SUITES = [
    "tests/test_sanitizer_m1.py",
    "tests/test_adversarial_deep_stress_m1.py",
    "tests/test_adversarial_m1_challenger2.py",
    "tests/test_adversarial_preambles_m1.py",
    "tests/test_milestone2_flowchart_engine.py",
    "tests/test_adversarial_flowchart_engine.py",
    "tests/test_adversarial_m2_challenger2.py",
    "tests/test_adversarial_m3_challenger1.py",
    "tests/test_adversarial_m3_challenger2.py",
    "test_multidomain_verification_suite.py",
]

def run_via_subprocess(test_file: str) -> tuple[bool, str, float]:
    """Runs a test file in a separate python process, capturing stdout, stderr, and timing."""
    start_t = time.perf_counter()
    try:
        res = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=os.path.abspath(os.path.dirname(__file__))
        )
        elapsed = time.perf_counter() - start_t
        output = res.stdout + ("\n" + res.stderr if res.stderr else "")
        return (res.returncode == 0, output.strip(), elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - start_t
        return (False, f"Execution failed: {e}", elapsed)

def run_via_unittest_loader(test_file: str) -> tuple[bool, str, float]:
    """Fallback in-process unittest runner if subprocess execution is restricted."""
    start_t = time.perf_counter()
    suite = unittest.defaultTestLoader.discover(
        start_dir=os.path.dirname(test_file) or ".",
        pattern=os.path.basename(test_file)
    )
    from io import StringIO
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_t
    return (result.wasSuccessful(), stream.getvalue(), elapsed)

def main():
    print("=" * 80)
    print("NEURA AI — MASTER VERIFICATION TEST RUNNER (MILESTONES 1-3)")
    print("=" * 80)

    results = {}
    timings = {}
    all_passed = True

    for test_path in TEST_SUITES:
        full_path = os.path.join(os.path.dirname(__file__), test_path)
        if not os.path.exists(full_path):
            print(f"\n⚠️  Skipping {test_path} (file not found)")
            continue

        print(f"\n>>> Running: {test_path} ...")
        passed, output, elapsed = run_via_subprocess(test_path)
        
        # If subprocess returned error or empty, try in-process runner fallback
        if not passed and "permission" in output.lower():
            print("  ℹ️ Subprocess restricted, falling back to in-process unittest runner...")
            passed, output, elapsed = run_via_unittest_loader(full_path)

        status_str = "✅ PASS" if passed else "❌ FAIL"
        results[test_path] = status_str
        timings[test_path] = elapsed

        print(output)
        print(f"--- Result: {status_str} in {elapsed:.3f}s ---")

        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    print("TEST SUITE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"{'Test Suite':<50} | {'Status':<10} | {'Duration':<10}")
    print("-" * 80)
    for test_path, status in results.items():
        dur = f"{timings.get(test_path, 0):.3f}s"
        print(f"{test_path:<50} | {status:<10} | {dur:<10}")
    print("=" * 80)

    if all_passed:
        print("🎉 ALL TEST SUITES PASSED (100% SUCCESS)")
        sys.exit(0)
    else:
        print("💥 SOME TEST SUITES FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
