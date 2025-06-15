#!/usr/bin/env python3
"""
Test runner script for squid-control project.
Supports different test types and coverage reporting.
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path


def run_command(cmd, env=None):
    """Run a shell command and return the exit code."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, env=env, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Run tests for squid-control")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--unit-only",
        action="store_true",
        help="Run only unit tests (squid_controller tests)"
    )
    parser.add_argument(
        "--integration-only",
        action="store_true",
        help="Run only integration tests (requires network access and tokens)"
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="Skip integration tests (recommended for CI/CD)"
    )
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML coverage report"
    )
    parser.add_argument(
        "--open-html",
        action="store_true",
        help="Open HTML coverage report in browser after generation"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose test output"
    )
    parser.add_argument(
        "--simulation",
        action="store_true",
        help="Run simulation tests only"
    )
    
    args = parser.parse_args()

    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    
    # Select test files and markers
    if args.unit_only:
        cmd.append("tests/test_squid_controller.py")
        print("🧪 Running UNIT TESTS only")
    elif args.integration_only:
        cmd.extend(["-m", "integration"])
        cmd.append("tests/")
        print("🌐 Running INTEGRATION TESTS only (requires network and tokens)")
        # Check for required tokens
        if not os.environ.get("AGENT_LENS_WORKSPACE_TOKEN"):
            print("⚠️  WARNING: AGENT_LENS_WORKSPACE_TOKEN not set - integration tests may fail")
            print("   Set the token with: export AGENT_LENS_WORKSPACE_TOKEN=your_token")
    elif args.skip_integration:
        cmd.extend(["-m", "not integration"]) 
        cmd.append("tests/")
        print("🧪 Running ALL TESTS except integration tests")
    else:
        cmd.append("tests/")
        print("🔄 Running ALL TESTS (including integration tests)")
    
    # Add simulation marker if requested
    if args.simulation:
        cmd.extend(["-m", "simulation"])
    
    # Add coverage options if requested
    if args.coverage:
        cmd.extend([
            "--cov=squid_control",
            "--cov=start_hypha_service.py",
            "--cov-config=pyproject.toml",
            "--cov-report=xml:coverage.xml",
            "--cov-report=html:htmlcov",
            "--cov-report=term-missing"
        ])
    
    # Always add junit XML output for CI/CD
    cmd.extend(["--junitxml=pytest-results.xml"])
    
    # Set environment
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    # Run tests
    return_code = run_command(cmd, env=env)
    
    # Open HTML coverage report if requested
    if args.coverage and args.html and args.open_html and return_code == 0:
        html_path = Path("htmlcov/index.html")
        if html_path.exists():
            print(f"\nOpening coverage report: {html_path}")
            try:
                subprocess.run(["xdg-open", str(html_path)], check=False)
            except FileNotFoundError:
                print("Could not open browser. Please open htmlcov/index.html manually.")
    
    # Print coverage info
    if args.coverage and return_code == 0:
        print("\nCoverage report generated:")
        print("- XML: coverage.xml")
        print("- HTML: htmlcov/index.html")
        print("- Terminal: shown above")
    
    print("\nTest results saved to: pytest-results.xml")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main()) 