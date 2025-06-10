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
        help="Run only integration tests (hypha service tests)"
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
    
    # Select test files
    if args.unit_only:
        cmd.append("tests/test_squid_controller.py")
    elif args.integration_only:
        cmd.append("tests/test_hypha_service.py")
    else:
        cmd.append("tests/")
    
    # Add simulation marker if requested
    if args.simulation:
        cmd.extend(["-m", "simulation"])
    
    # Add coverage options if requested
    if args.coverage:
        cmd.extend([
            "--cov=squid_control",
            "--cov-report=xml:coverage.xml",
            "--cov-report=term-missing"
        ])
        
        if args.html:
            cmd.append("--cov-report=html:htmlcov")
    
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
        if args.html:
            print("- HTML: htmlcov/index.html")
        print("- Terminal: shown above")
    
    return return_code


if __name__ == "__main__":
    sys.exit(main()) 