#!/usr/bin/env python3
"""
Helper script to run ZIP64 upload and endpoint tests.

This script provides an easy way to run the ZIP64 tests with appropriate
configuration and timeout settings.

Usage:
    python run_zip64_tests.py                    # Run standard tests (100MB-3.2GB)
    python run_zip64_tests.py --include-large    # Include 10GB test (very slow)
    python run_zip64_tests.py --size 400         # Run only 400MB test
    python run_zip64_tests.py --help            # Show help
"""

import argparse
import sys
import subprocess
import os
from pathlib import Path


def run_tests(include_large=False, specific_size=None, verbose=False, capture_output=True):
    """Run the ZIP64 tests with appropriate configuration."""
    
    # Check for required environment variable
    if not os.environ.get("AGENT_LENS_WORKSPACE_TOKEN"):
        print("❌ Error: AGENT_LENS_WORKSPACE_TOKEN environment variable not set")
        print("Please set your workspace token:")
        print("export AGENT_LENS_WORKSPACE_TOKEN='your_token_here'")
        return 1
    
    # Base pytest command
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/test_zip64_upload_endpoint.py",
        "--timeout=3600",  # 1 hour timeout for large tests
        "-v" if verbose else "-q",
        "--tb=short",
        "--strict-markers"
    ]
    
    # Add output capture control
    if capture_output:
        cmd.append("-s")  # Don't capture output for real-time feedback
    
    # Configure which tests to run
    if specific_size:
        # Run specific size test
        cmd.extend(["-k", f"test_zip64_upload_and_access[{specific_size}]"])
        print(f"🧪 Running ZIP64 test for {specific_size}MB dataset...")
    elif include_large:
        # Run all tests including the 10GB one
        print("🧪 Running ALL ZIP64 tests (including 10GB - this will take a LONG time)...")
        print("⚠️ The 10GB test can take 30+ minutes and use significant disk space and bandwidth")
    else:
        # Run standard tests (exclude the very large one)
        cmd.extend(["-m", "not slow"])
        print("🧪 Running standard ZIP64 tests (100MB - 3.2GB)...")
        print("ℹ️ Use --include-large to also run the 10GB test")
    
    print(f"📁 Working directory: {Path.cwd()}")
    print(f"🔧 Command: {' '.join(cmd)}")
    print()
    
    # Run the tests
    try:
        result = subprocess.run(cmd, check=False)
        return result.returncode
    except KeyboardInterrupt:
        print("\n⚠️ Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run ZIP64 upload and endpoint tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_zip64_tests.py                     # Standard tests (100MB-3.2GB)
  python run_zip64_tests.py --include-large     # Include 10GB test
  python run_zip64_tests.py --size 800          # Test only 800MB
  python run_zip64_tests.py --verbose           # Verbose output
  
Note: You must set AGENT_LENS_WORKSPACE_TOKEN environment variable before running.
        """
    )
    
    parser.add_argument(
        "--include-large", 
        action="store_true",
        help="Include the 10GB test (very slow, can take 30+ minutes)"
    )
    
    parser.add_argument(
        "--size", 
        type=int,
        choices=[100, 200, 400, 800, 1600, 3200],
        help="Run test for specific size only (MB)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Capture output (less real-time feedback)"
    )
    
    args = parser.parse_args()
    
    # Check for conflicting arguments
    if args.include_large and args.size:
        print("❌ Error: Cannot specify both --include-large and --size")
        return 1
    
    print("🚀 ZIP64 Test Runner")
    print("=" * 50)
    
    # Run tests
    exit_code = run_tests(
        include_large=args.include_large,
        specific_size=args.size,
        verbose=args.verbose,
        capture_output=not args.quiet
    )
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main()) 