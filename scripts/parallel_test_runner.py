#!/usr/bin/env python3
"""
Parallel test runner for Python and PyPy3
Executes pytest with both interpreters simultaneously and aggregates results
"""
import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


class ParallelTestRunner:
    def __init__(
        self,
        test_path: str = "tests",
        args: List[str] = None,
        live_mode: bool = False,
        progress_mode: bool = False,
    ):
        self.test_path = test_path
        self.args = args or []
        self.results = {}
        self.live_mode = live_mode
        self.progress_mode = progress_mode
        self.test_counters = {}  # Track progress for each interpreter

    def clear_json_reports(self):
        """Clear old JSON reports to ensure fresh results"""
        cache_dir = Path(".pytest_cache")
        if cache_dir.exists():
            for report_file in cache_dir.glob("*_report.json"):
                try:
                    report_file.unlink()
                    print(f"🗑️  Cleared old report: {report_file.name}")
                except Exception as e:
                    print(f"⚠️  Could not clear {report_file.name}: {e}")

    async def pypy_warmup(self, pypy_path: str) -> bool:
        """Warm up PyPy JIT compiler with a small computational task"""
        warmup_code = """
import time
import math

def fibonacci_recursive(n):
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def prime_sieve(limit):
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    return [i for i in range(2, limit + 1) if sieve[i]]

def matrix_multiply(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b)))
             for j in range(len(b[0]))] for i in range(len(a))]

# JIT Warmup exercises
print("🏃‍♂️ PyPy JIT Warmup - Running computational exercises...")

# 1. Recursive computation (tests function call optimization)
start = time.time()
fib_25 = fibonacci_recursive(25)
fib_time = time.time() - start

# 2. Loop-heavy computation (tests loop optimization)
start = time.time()
primes = prime_sieve(10000)
prime_time = time.time() - start

# 3. Nested loops (tests nested loop optimization)
start = time.time()
matrix_a = [[i+j for j in range(20)] for i in range(20)]
matrix_b = [[i*j+1 for j in range(20)] for i in range(20)]
result = matrix_multiply(matrix_a, matrix_b)
matrix_time = time.time() - start

print(f"   📈 Fibonacci(25): {fib_25} in {fib_time:.3f}s")
print(f"   🔢 Primes to 10k: {len(primes)} found in {prime_time:.3f}s")
print(f"   🔢 Matrix multiply: 20x20 in {matrix_time:.3f}s")
print("✅ PyPy JIT warmup completed!")
"""

        try:
            print("🔥 Warming up PyPy JIT compiler...")

            process = await asyncio.create_subprocess_exec(
                pypy_path,
                "-c",
                warmup_code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **dict(os.environ),
                    "PYTHONPATH": os.getcwd(),
                    "PATH": f"/Users/ilpinto/micromamba/envs/alc-pypy/bin:{os.environ.get('PATH', '')}",
                },
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Print warmup output
                for line in stdout.decode().strip().split("\n"):
                    if line.strip():
                        print(f"   {line}")
                return True
            else:
                print(f"⚠️  PyPy warmup failed: {stderr.decode()}")
                return False

        except Exception as e:
            print(f"⚠️  PyPy warmup error: {e}")
            return False

    async def process_test_output_live(self, process, name: str) -> Tuple[str, str]:
        """Process test output line by line for live progress display"""
        stdout_lines = []
        stderr_lines = []

        # Initialize test counter for this interpreter
        self.test_counters[name] = {
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "total": 0,
            "current": "",
        }

        # Colors for different interpreters
        colors = {
            "CPython": "\033[94m",  # Blue
            "PyPy3": "\033[92m",  # Green
            "reset": "\033[0m",
        }
        color = colors.get(name, "")
        reset = colors["reset"]

        async def read_stream(stream, lines_list, is_stderr=False):
            while True:
                try:
                    line = await stream.readline()
                    if not line:
                        break

                    line_text = line.decode("utf-8", errors="replace").rstrip()
                    lines_list.append(line_text)

                    if self.live_mode or self.progress_mode:
                        self.process_test_line(line_text, name, color, reset, is_stderr)

                except Exception as e:
                    if self.live_mode:
                        print(f"{color}[{name}] Stream error: {e}{reset}")
                    break

        # Read stdout and stderr concurrently
        await asyncio.gather(
            read_stream(process.stdout, stdout_lines),
            read_stream(process.stderr, stderr_lines, True),
        )

        return "\n".join(stdout_lines), "\n".join(stderr_lines)

    def process_test_line(
        self, line: str, name: str, color: str, reset: str, is_stderr: bool = False
    ):
        """Process individual test output lines for progress display"""
        # Look for pytest test result patterns
        if " PASSED " in line or "::test_" in line and " PASSED" in line:
            self.test_counters[name]["passed"] += 1
            if self.live_mode:
                test_name = self.extract_test_name(line)
                count = (
                    self.test_counters[name]["passed"]
                    + self.test_counters[name]["failed"]
                    + self.test_counters[name]["skipped"]
                )
                print(f"{color}[{name}] ✓ {test_name} [{count}]{reset}")

        elif " FAILED " in line or "::test_" in line and " FAILED" in line:
            self.test_counters[name]["failed"] += 1
            if self.live_mode:
                test_name = self.extract_test_name(line)
                count = (
                    self.test_counters[name]["passed"]
                    + self.test_counters[name]["failed"]
                    + self.test_counters[name]["skipped"]
                )
                print(f"{color}[{name}] ✗ {test_name} [{count}]{reset}")

        elif " SKIPPED " in line or "::test_" in line and " SKIPPED" in line:
            self.test_counters[name]["skipped"] += 1
            if self.live_mode:
                test_name = self.extract_test_name(line)
                count = (
                    self.test_counters[name]["passed"]
                    + self.test_counters[name]["failed"]
                    + self.test_counters[name]["skipped"]
                )
                print(f"{color}[{name}] ⊝ {test_name} [{count}]{reset}")

        elif "::test_" in line and (" ... " in line or " ..." in line):
            # Test currently running
            test_name = self.extract_test_name(line)
            self.test_counters[name]["current"] = test_name
            if self.live_mode:
                print(f"{color}[{name}] ⏳ {test_name}...{reset}")

        # Update progress display periodically
        if self.progress_mode and not self.live_mode:
            total_tests = (
                self.test_counters[name]["passed"]
                + self.test_counters[name]["failed"]
                + self.test_counters[name]["skipped"]
            )
            if total_tests > 0 and total_tests % 5 == 0:  # Update every 5 tests
                self.display_progress_once()

    def extract_test_name(self, line: str) -> str:
        """Extract test name from pytest output line"""
        # Look for patterns like "tests/test_file.py::TestClass::test_method"
        if "::" in line:
            parts = line.split("::")
            if len(parts) >= 2:
                return parts[-1].split()[0]  # Get the test method name
        return "unknown"

    def display_progress(self):
        """Display current progress for both interpreters"""
        if not self.progress_mode:
            return

        cpython_stats = self.test_counters.get(
            "CPython", {"passed": 0, "failed": 0, "skipped": 0}
        )
        pypy_stats = self.test_counters.get(
            "PyPy3", {"passed": 0, "failed": 0, "skipped": 0}
        )

        cpython_total = (
            cpython_stats["passed"] + cpython_stats["failed"] + cpython_stats["skipped"]
        )
        pypy_total = pypy_stats["passed"] + pypy_stats["failed"] + pypy_stats["skipped"]

        print(
            f"\r📊 Progress: CPython {cpython_total} tests | PyPy3 {pypy_total} tests",
            end="",
            flush=True,
        )

    def display_progress_once(self):
        """Display progress update only when meaningful changes occur"""
        if not hasattr(self, "_last_progress"):
            self._last_progress = {"CPython": 0, "PyPy3": 0}

        cpython_stats = self.test_counters.get(
            "CPython", {"passed": 0, "failed": 0, "skipped": 0}
        )
        pypy_stats = self.test_counters.get(
            "PyPy3", {"passed": 0, "failed": 0, "skipped": 0}
        )

        cpython_total = (
            cpython_stats["passed"] + cpython_stats["failed"] + cpython_stats["skipped"]
        )
        pypy_total = pypy_stats["passed"] + pypy_stats["failed"] + pypy_stats["skipped"]

        # Only update if there's meaningful change
        if (
            cpython_total != self._last_progress["CPython"]
            or pypy_total != self._last_progress["PyPy3"]
        ):

            print(
                f"\r📊 Progress: CPython {cpython_total} tests | PyPy3 {pypy_total} tests",
                end="",
                flush=True,
            )
            self._last_progress["CPython"] = cpython_total
            self._last_progress["PyPy3"] = pypy_total

    async def run_tests_with_interpreter(self, interpreter: str, name: str) -> Dict:
        """Run tests with specified interpreter"""
        # Handle different interpreter setups
        if name == "PyPy3":
            # Use full path to PyPy3 in micromamba environment
            pypy_path = "/Users/ilpinto/micromamba/envs/alc-pypy/bin/pypy3"
            cmd = [pypy_path, "-u", "-m", "pytest"]  # -u for unbuffered output

            # Set environment variables for PyPy3 environment
            env = {
                **dict(os.environ),
                "PYTHONPATH": os.getcwd(),
                "PATH": f"/Users/ilpinto/micromamba/envs/alc-pypy/bin:{os.environ.get('PATH', '')}",
                "PYTHONWARNINGS": "always",
            }
        else:
            # Regular Python interpreter
            cmd = [interpreter, "-u", "-m", "pytest"]  # -u for unbuffered output
            env = {
                **dict(os.environ),
                "PYTHONPATH": os.getcwd(),
                "PYTHONWARNINGS": "always",
            }

        # Add common pytest arguments
        test_args = [self.test_path, "-v"]

        # Add progress-friendly options for live mode
        if self.live_mode or self.progress_mode:
            test_args.extend(["--tb=line"])  # Shorter tracebacks for cleaner output

        test_args.extend(self.args)
        cmd.extend(test_args)

        # Only add JSON report if plugin is available
        try:
            if name == "CPython":
                import pytest_json_report

                cmd.extend(
                    [
                        "--json-report",
                        f"--json-report-file=.pytest_cache/{name}_report.json",
                    ]
                )
        except ImportError:
            pass

        if not (self.live_mode or self.progress_mode):
            print(f"🚀 Starting {name} tests...")
            if name == "PyPy3":
                print(f"   Using: {pypy_path}")

        start_time = time.time()

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )

            # Use live processing if enabled, otherwise use traditional communicate()
            if self.live_mode or self.progress_mode:
                stdout, stderr = await self.process_test_output_live(process, name)
                await process.wait()  # Wait for process to complete
            else:
                stdout, stderr = await process.communicate()
                stdout, stderr = stdout.decode(), stderr.decode()

            duration = time.time() - start_time

            # Parse JSON report with timestamp validation
            report_file = Path(f".pytest_cache/{name}_report.json")
            json_report_valid = False

            if report_file.exists():
                try:
                    with open(report_file) as f:
                        report = json.load(f)

                    # Validate report timestamp - should be created during this run
                    report_created = report.get("created", 0)
                    time_diff = time.time() - report_created

                    # If report is older than our run duration + 30s buffer, it's stale
                    if time_diff > duration + 30:
                        print(
                            f"⚠️  {name}: JSON report is stale ({time_diff:.1f}s old), using stdout parsing"
                        )
                        json_report_valid = False
                    else:
                        json_report_valid = True
                        print(f"✅ {name}: Using fresh JSON report")

                except (json.JSONDecodeError, KeyError) as e:
                    print(f"⚠️  {name}: Invalid JSON report ({e}), using stdout parsing")
                    json_report_valid = False

            if not json_report_valid:
                # If no JSON report, use our live counters if available, otherwise parse stdout
                if name in self.test_counters and (
                    self.live_mode or self.progress_mode
                ):
                    # Use live counters from streaming
                    stats = self.test_counters[name]
                    passed = stats["passed"]
                    failed = stats["failed"]
                    skipped = stats["skipped"]
                    errors = 0
                    total = passed + failed + skipped

                    if not (self.live_mode or self.progress_mode):
                        print(
                            f"📊 {name}: Live counters -> {passed} passed, {failed} failed, {skipped} skipped"
                        )
                else:
                    # Fallback to stdout parsing
                    import re

                    # Look for pytest final summary line with improved patterns
                    passed_match = re.search(r"=+ (\d+) passed", stdout)
                    failed_match = re.search(r"(\d+) failed", stdout)
                    skipped_match = re.search(r"(\d+) skipped", stdout)
                    error_match = re.search(r"(\d+) error", stdout)

                    passed = int(passed_match.group(1)) if passed_match else 0
                    failed = int(failed_match.group(1)) if failed_match else 0
                    skipped = int(skipped_match.group(1)) if skipped_match else 0
                    errors = int(error_match.group(1)) if error_match else 0
                    total = passed + failed + skipped + errors

                    if not (self.live_mode or self.progress_mode):
                        print(
                            f"📊 {name}: Parsed stdout -> {passed} passed, {failed} failed, {skipped} skipped, {errors} errors"
                        )

                # If we found test results, use them; otherwise check return code
                if total > 0 or process.returncode == 0:
                    report = {
                        "summary": {
                            "passed": passed,
                            "failed": failed,
                            "skipped": skipped,
                            "total": total,
                        }
                    }
                else:
                    report = {
                        "summary": {
                            "failed": 1 if process.returncode != 0 else 0,
                            "passed": 0,
                            "skipped": 0,
                            "total": 0,
                        }
                    }

            return {
                "name": name,
                "interpreter": cmd[0],
                "returncode": process.returncode,
                "duration": duration,
                "stdout": stdout,  # Already decoded in live mode
                "stderr": stderr,  # Already decoded in live mode
                "report": report,
            }

        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Failed to start {name} interpreter: {str(e)}"
            print(f"❌ {error_msg}")

            return {
                "name": name,
                "interpreter": cmd[0] if cmd else interpreter,
                "returncode": -1,
                "duration": duration,
                "stdout": "",
                "stderr": error_msg,
                "report": {"summary": {"failed": -1, "passed": 0, "total": 0}},
            }

    async def run_parallel(self, skip_warmup: bool = False) -> Tuple[Dict, Dict]:
        """Run tests with both Python and PyPy3 in parallel"""
        print("🔄 Clearing old JSON reports to ensure fresh results...")
        self.clear_json_reports()
        print()

        # Warm up PyPy JIT compiler before running tests
        if not skip_warmup:
            pypy_path = "/Users/ilpinto/micromamba/envs/alc-pypy/bin/pypy3"
            warmup_success = await self.pypy_warmup(pypy_path)
            if warmup_success:
                print("🚀 PyPy JIT is now warmed up and ready for optimal performance!")
            else:
                print("⚠️  PyPy warmup failed, but continuing with tests...")
            print()

        # Start both test suites
        print("🏁 Starting parallel test execution...")
        tasks = [
            self.run_tests_with_interpreter("python", "CPython"),
            self.run_tests_with_interpreter("pypy3", "PyPy3"),
        ]

        results = await asyncio.gather(*tasks)
        return results[0], results[1]

    def print_results(self, cpython_result: Dict, pypy_result: Dict):
        """Print formatted test results comparison"""
        print("\n" + "=" * 80)
        print("📊 PARALLEL TEST RESULTS COMPARISON")
        print("=" * 80)

        for result in [cpython_result, pypy_result]:
            name = result["name"]
            report = result["report"]
            summary = report.get("summary", {})

            status = "✅" if result["returncode"] == 0 else "❌"

            print(f"\n{status} {name} Results:")
            print(f"   Duration: {result['duration']:.2f}s")
            print(f"   Passed:   {summary.get('passed', 0)}")
            print(f"   Failed:   {summary.get('failed', 0)}")
            print(f"   Skipped:  {summary.get('skipped', 0)}")
            print(f"   Total:    {summary.get('total', 0)}")

        # Performance comparison
        if cpython_result["duration"] > 0 and pypy_result["duration"] > 0:
            speedup = cpython_result["duration"] / pypy_result["duration"]
            if speedup > 1:
                print(f"\n⚡ PyPy Performance: {speedup:.2f}x faster than CPython")
            elif speedup < 1:
                slowdown = pypy_result["duration"] / cpython_result["duration"]
                print(
                    f"\n🐢 PyPy Performance: {slowdown:.2f}x slower than CPython (JIT warmup?)"
                )
            else:
                print(f"\n⚖️  PyPy Performance: Similar to CPython")

        # Overall test comparison
        cpython_success = cpython_result["returncode"] == 0
        pypy_success = pypy_result["returncode"] == 0

        if cpython_success and pypy_success:
            print(f"\n🎉 All tests pass on both interpreters!")
        elif not cpython_success and not pypy_success:
            print(f"\n⚠️  Tests failed on both interpreters")
        elif cpython_success and not pypy_success:
            print(f"\n⚠️  Tests pass on CPython but fail on PyPy3")
        else:
            print(f"\n⚠️  Tests pass on PyPy3 but fail on CPython")

        print("=" * 80)


async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Run tests in parallel with Python and PyPy3",
        epilog="""
Examples:
  %(prog)s tests                        # Run all tests with PyPy JIT warmup
  %(prog)s tests --live                 # Show real-time test progress with colors
  %(prog)s tests --progress             # Show periodic progress updates
  %(prog)s tests --skip-warmup          # Skip PyPy warmup for faster startup
  %(prog)s tests/test_file.py -k test   # Run specific tests with warmup
  %(prog)s tests --warmup-only          # Only run PyPy warmup (no tests)
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "test_path", nargs="?", default="tests", help="Test path or file"
    )
    parser.add_argument("-k", "--keyword", help="Run tests matching keyword")
    parser.add_argument(
        "-x", "--exitfirst", action="store_true", help="Exit on first failure"
    )
    parser.add_argument("--tb", default="short", help="Traceback print mode")
    parser.add_argument(
        "--skip-warmup",
        action="store_true",
        help="Skip PyPy JIT warmup for faster startup",
    )
    parser.add_argument(
        "--warmup-only",
        action="store_true",
        help="Only run PyPy warmup, skip tests (useful for JIT preparation)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Show real-time test progress with colors for each interpreter",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Show periodic progress updates (less verbose than --live)",
    )

    args, unknown = parser.parse_known_args()

    pytest_args = []
    if args.keyword:
        pytest_args.extend(["-k", args.keyword])
    if args.exitfirst:
        pytest_args.append("-x")
    if not (args.live or args.progress):  # Only add --tb if not in live modes
        pytest_args.extend(["--tb", args.tb])
    pytest_args.extend(unknown)

    runner = ParallelTestRunner(
        args.test_path, pytest_args, live_mode=args.live, progress_mode=args.progress
    )

    if args.warmup_only:
        # Only run warmup
        print("🔥 Running PyPy JIT warmup only...")
        pypy_path = "/Users/ilpinto/micromamba/envs/alc-pypy/bin/pypy3"
        success = await runner.pypy_warmup(pypy_path)
        sys.exit(0 if success else 1)

    # Show live mode header
    if args.live:
        print("🎬 Live test progress mode enabled!")
        print("═══════════════════════════════════════════════════════════════")

    # Run tests with optional warmup
    cpython_result, pypy_result = await runner.run_parallel(
        skip_warmup=args.skip_warmup
    )

    # Clear progress line if using progress mode
    if args.progress and not args.live:
        print()  # New line after progress updates

    # Show final results
    runner.print_results(cpython_result, pypy_result)

    # Exit with failure if either interpreter failed
    sys.exit(max(cpython_result["returncode"], pypy_result["returncode"]))


if __name__ == "__main__":
    asyncio.run(main())
