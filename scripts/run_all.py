"""Run the analysis scripts in order, or only the ones named (task 0.5).

Usage:
    python scripts/run_all.py                  # everything, S01 to S11
    python scripts/run_all.py s02              # one script
    python scripts/run_all.py s06 s07 s09 s11  # several, always in the standard order

Each script runs in its own Python process. The run stops at the first script
that fails or has not been written yet, and says which one and why.
The runner's own log is logs/run_all.log.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

import common

# Task 0.5: the standard run order. check_db (task 3.2) checks the database
# that S01-S05 build, so it runs straight after S05.
RUN_ORDER = ["s01", "s02", "s03", "s04", "s05", "check_db",
             "s06", "s07", "s08", "s09", "s10", "s11"]

SCRIPTS_DIR = Path(__file__).resolve().parent


def parse_args():
    """Read the script names (optional) and the test-only scripts folder."""
    parser = argparse.ArgumentParser(description="Run S01-S11 in order, or only the scripts named.")
    parser.add_argument("names", nargs="*",
                        help=f"scripts to run, any of: {' '.join(RUN_ORDER)} (default: all)")
    parser.add_argument("--scripts-dir", type=Path, default=SCRIPTS_DIR,
                        help="folder to look for scripts in (for testing the runner only)")
    return parser.parse_args()


def choose_scripts(names):
    """Return (names to run in the standard order, names not recognised).

    No names means all scripts. Repeated names run once.
    """
    if not names:
        return list(RUN_ORDER), []
    wanted = {name.lower() for name in names}
    unknown = sorted(wanted - set(RUN_ORDER))
    return [name for name in RUN_ORDER if name in wanted], unknown


def find_script(name, scripts_dir):
    """Find the file for a short name: "s02" -> s02_*.py, "check_db" -> check_db.py.

    Returns the path, or None if the script has not been written yet.
    """
    if name.startswith("s") and name[1:].isdigit():
        matches = sorted(scripts_dir.glob(f"{name}_*.py"))
    else:
        matches = sorted(scripts_dir.glob(f"{name}.py"))
    if len(matches) > 1:
        raise RuntimeError(f"More than one file matches {name}: {', '.join(m.name for m in matches)}")
    return matches[0] if matches else None


def run_script(path):
    """Run one script in its own Python process.

    Its normal output goes straight to the screen. Its error output is shown
    on screen too and kept, so the last line (the error message) can be reported.
    Returns (exit code, last line of error output).
    """
    process = subprocess.Popen([sys.executable, str(path)], cwd=common.ROOT,
                               stderr=subprocess.PIPE, text=True,
                               encoding="utf-8", errors="replace")
    last_error_line = ""
    for line in process.stderr:
        sys.stderr.write(line)
        if line.strip():
            last_error_line = line.strip()
    process.wait()
    return process.returncode, last_error_line


def main():
    args = parse_args()
    settings = common.load_settings()
    log = common.start_log("run_all", settings)
    scripts_dir = args.scripts_dir.resolve()
    log_dir = common.get_path(settings, "logs")

    # Task 0.5: work out which scripts to run, in the standard order.
    chosen, unknown = choose_scripts(args.names)
    if unknown:
        log.error("Unknown script name(s): %s. Valid names: %s", ", ".join(unknown), " ".join(RUN_ORDER))
        return 1
    log.info("Requested: %s", "all scripts" if not args.names else " ".join(args.names))
    log.info("Will run, in order: %s", " ".join(chosen))
    if scripts_dir != SCRIPTS_DIR:
        log.info("Scripts folder (test override): %s", scripts_dir)

    passed = []
    for i, name in enumerate(chosen):
        not_run = chosen[i + 1:]
        path = find_script(name, scripts_dir)
        if path is None:
            log.error("STOPPED: %s has not been written yet (no %s in %s)",
                      name, f"{name}_*.py" if name != "check_db" else "check_db.py", scripts_dir)
            summarise(log, passed, f"{name} (not written yet)", not_run)
            return 1

        log.info("---- %s started", path.name)
        start = time.perf_counter()
        exit_code, error_line = run_script(path)
        seconds = time.perf_counter() - start

        if exit_code != 0:
            log.error("STOPPED: %s failed after %.1f s (exit code %d): %s",
                      path.name, seconds, exit_code, error_line or "no error message")
            script_log = log_dir / f"{path.stem}.log"
            if script_log.exists():
                log.error("Details: %s", script_log)
            else:
                log.error("Details: %s (no log was written)", script_log)
            summarise(log, passed, f"{path.name} (failed)", not_run)
            return 1
        log.info("---- %s finished OK in %.1f s", path.name, seconds)
        passed.append(path.name)

    summarise(log, passed, None, [])
    return 0


def summarise(log, passed, stopped_at, not_run):
    """Log the closing summary: what passed, where the run stopped, what did not run."""
    log.info("Summary | passed: %s", ", ".join(passed) or "none")
    if stopped_at:
        log.info("Summary | stopped at: %s", stopped_at)
        log.info("Summary | not run: %s", ", ".join(not_run) or "none")
    else:
        log.info("Summary | all %d script(s) finished OK", len(passed))


if __name__ == "__main__":
    sys.exit(main())
