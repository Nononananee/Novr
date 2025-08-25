#!/usr/bin/env python3
"""
Test runner script for different testing phases.
Usage: python scripts/run_tests.py [phase] [options]
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, env=None):
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.stdout:
        print("STDOUT:", result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode == 0


def setup_test_environment():
    """Setup test environment variables."""
    test_env = os.environ.copy()
    test_env.update({
        "APP_ENV": "testing",
        "DATABASE_URL": "postgresql://test:test@localhost:5432/test_novel_rag",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_PASSWORD": "test_password",
        "LLM_API_KEY": "test_api_key",
        "EMBEDDING_API_KEY": "test_embedding_key"
    })
    return test_env


def run_phase_1_foundation(args):
    """Run Phase 1: Foundation & Critical Path Testing."""
    print("🎯 Phase 1: Foundation & Critical Path Testing")
    
    env = setup_test_environment()
    
    # Run critical tests first
    print("Running critical priority tests...")
    critical_cmd = [
        "pytest", 
        "tests/critical/",
        "-v",
        "--tb=short",
        "-m", "critical"
    ]
    
    if args.parallel:
        critical_cmd.extend(["-n", "auto"])
    
    critical_success = run_command(critical_cmd, env)
    
    if not critical_success:
        print("❌ Critical tests failed!")
        if args.fail_fast:
            return False
    
    # Run unit tests
    print("Running unit tests...")
    unit_cmd = [
        "pytest", 
        "tests/unit/",
        "-v",
        "--cov=agent",
        "--cov-report=html",
        "--cov-report=term-missing",
        f"--cov-fail-under={args.coverage_threshold}",
        "-m", "unit"
    ]
    
    if args.parallel:
        unit_cmd.extend(["-n", "auto"])
    
    unit_success = run_command(unit_cmd, env)
    
    overall_success = critical_success and unit_success
    
    if not overall_success:
        print("❌ Phase 1 tests failed!")
        return False
    
    print("✅ Phase 1 tests passed!")
    return True


def run_phase_2_integration(args):
    """Run Phase 2: Integration Testing."""
    print("🎯 Phase 2: Integration Testing")
    
    env = setup_test_environment()
    
    cmd = [
        "pytest",
        "tests/integration/",
        "-v",
        "--tb=short",
        "-m", "integration"
    ]
    
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    success = run_command(cmd, env)
    
    if not success:
        print("��� Phase 2 tests failed!")
        return False
    
    print("✅ Phase 2 tests passed!")
    return True


def run_phase_3_performance(args):
    """Run Phase 3: Performance & Load Testing."""
    print("🎯 Phase 3: Performance & Load Testing")
    
    env = setup_test_environment()
    
    cmd = [
        "pytest",
        "tests/performance/",
        "-v",
        "--tb=short",
        "-m", "performance"
    ]
    
    if not args.include_slow:
        cmd.extend(["-m", "performance and not slow"])
    
    success = run_command(cmd, env)
    
    if not success:
        print("❌ Phase 3 tests failed!")
        return False
    
    print("✅ Phase 3 tests passed!")
    return True


def run_phase_4_error_scenarios(args):
    """Run Phase 4: Error Scenario Testing."""
    print("🎯 Phase 4: Error Scenario Testing")
    
    env = setup_test_environment()
    
    cmd = [
        "pytest",
        "tests/error_scenarios/",
        "-v",
        "--tb=short",
        "-m", "error_scenarios"
    ]
    
    success = run_command(cmd, env)
    
    if not success:
        print("❌ Phase 4 tests failed!")
        return False
    
    print("✅ Phase 4 tests passed!")
    return True


def run_phase_5_e2e(args):
    """Run Phase 5: End-to-End Testing."""
    print("🎯 Phase 5: End-to-End Testing")
    
    env = setup_test_environment()
    
    if args.real_dependencies:
        env["RUN_E2E_TESTS"] = "true"
        env["RUN_PERFORMANCE_E2E"] = "true"
    
    cmd = [
        "pytest",
        "tests/e2e/",
        "-v",
        "--tb=short",
        "-m", "e2e"
    ]
    
    if not args.real_dependencies:
        cmd.extend(["-k", "not real"])
    
    success = run_command(cmd, env)
    
    if not success:
        print("❌ Phase 5 tests failed!")
        return False
    
    print("✅ Phase 5 tests passed!")
    return True


def run_all_phases(args):
    """Run all testing phases in sequence."""
    print("🚀 Running all testing phases...")
    
    phases = [
        ("Phase 1", run_phase_1_foundation),
        ("Phase 2", run_phase_2_integration),
        ("Phase 3", run_phase_3_performance),
        ("Phase 4", run_phase_4_error_scenarios),
        ("Phase 5", run_phase_5_e2e)
    ]
    
    results = {}
    
    for phase_name, phase_func in phases:
        print(f"\n{'='*50}")
        print(f"Starting {phase_name}")
        print(f"{'='*50}")
        
        success = phase_func(args)
        results[phase_name] = success
        
        if not success and args.fail_fast:
            print(f"❌ {phase_name} failed and --fail-fast is enabled. Stopping.")
            break
    
    # Print summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print(f"{'='*50}")
    
    for phase_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{phase_name}: {status}")
    
    overall_success = all(results.values())
    print(f"\nOverall: {'✅ ALL PASSED' if overall_success else '❌ SOME FAILED'}")
    
    return overall_success


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Novel RAG Testing Framework")
    
    parser.add_argument(
        "phase",
        choices=["1", "2", "3", "4", "5", "all", "foundation", "integration", "performance", "error", "e2e"],
        help="Testing phase to run"
    )
    
    parser.add_argument(
        "--coverage-threshold",
        type=int,
        default=70,
        help="Coverage threshold for unit tests (default: 70)"
    )
    
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)"
    )
    
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow performance tests"
    )
    
    parser.add_argument(
        "--real-dependencies",
        action="store_true",
        help="Use real dependencies for E2E tests (requires proper setup)"
    )
    
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on first phase failure when running all phases"
    )
    
    args = parser.parse_args()
    
    # Map phase names to functions
    phase_map = {
        "1": run_phase_1_foundation,
        "foundation": run_phase_1_foundation,
        "2": run_phase_2_integration,
        "integration": run_phase_2_integration,
        "3": run_phase_3_performance,
        "performance": run_phase_3_performance,
        "4": run_phase_4_error_scenarios,
        "error": run_phase_4_error_scenarios,
        "5": run_phase_5_e2e,
        "e2e": run_phase_5_e2e,
        "all": run_all_phases
    }
    
    # Check if pytest is available
    try:
        subprocess.run(["pytest", "--version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ pytest not found. Please install pytest and required dependencies.")
        print("Run: pip install pytest pytest-cov pytest-asyncio")
        return 1
    
    # Run the selected phase
    phase_func = phase_map[args.phase]
    success = phase_func(args)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())