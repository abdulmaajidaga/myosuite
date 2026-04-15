"""
run_phase.py — Sequential queue runner for a full experimental phase.

Runs every experiment in the given phase one by one (train → generate → evaluate),
writing progress to a log file so you can tail it.

Usage:
  python test/run_phase.py --phase A            # run all Phase A experiments
  python test/run_phase.py --phase B            # run all Phase B experiments
  python test/run_phase.py --phase A --epochs 200
  python test/run_phase.py --phase A --skip A0_smote A0_dtw  # resume after failures
  python test/run_phase.py --phase A --only A3_smote         # single experiment

Background usage (recommended):
  nohup python test/run_phase.py --phase A > test/logs/phase_a.log 2>&1 &
  tail -f test/logs/phase_a.log
"""

import os, sys, time, argparse, subprocess

TEST_DIR  = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR  = os.path.dirname(TEST_DIR)
PYTHON    = sys.executable
RUNNER    = os.path.join(TEST_DIR, "run_experiment.py")
LOGS_DIR  = os.path.join(TEST_DIR, "logs")

sys.path.insert(0, TEST_DIR)
sys.path.insert(0, ROOT_DIR)
from experiments import EXPERIMENTS


def phase_experiments(phase: str) -> list[str]:
    """Return all experiment IDs starting with the given phase letter."""
    return [k for k in EXPERIMENTS if k.startswith(phase)]


def run_one(exp_name: str, epochs: int, fma: str) -> bool:
    cmd = [
        PYTHON, RUNNER,
        "--experiment", exp_name,
        "--phases",    "train,generate,evaluate",
        "--fma",       fma,
        "--epochs",    str(epochs),
    ]
    print(f"\n{'='*60}", flush=True)
    print(f"  Starting: {exp_name}", flush=True)
    print(f"  {EXPERIMENTS[exp_name]['desc']}", flush=True)
    print(f"{'='*60}", flush=True)

    t = time.time()
    result = subprocess.run(cmd, cwd=ROOT_DIR)
    elapsed = (time.time() - t) / 60

    ok = result.returncode == 0
    status = "DONE" if ok else "FAILED"
    print(f"\n  [{status}] {exp_name} — {elapsed:.1f} min", flush=True)
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase",   required=True, help="Phase letter: A or B")
    parser.add_argument("--epochs",  type=int, default=200)
    parser.add_argument("--fma",     default="16-66",
                        help="FMA range for generation (default: 16-66)")
    parser.add_argument("--skip",    nargs="*", default=[],
                        help="Experiment IDs to skip (already done)")
    parser.add_argument("--only",    nargs="*", default=[],
                        help="Run only these experiment IDs")
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)

    experiments = phase_experiments(args.phase.upper())
    if not experiments:
        print(f"No experiments found for phase '{args.phase}'")
        sys.exit(1)

    if args.only:
        experiments = [e for e in experiments if e in args.only]
    if args.skip:
        experiments = [e for e in experiments if e not in args.skip]

    total = len(experiments)
    print(f"\nPhase {args.phase.upper()} — {total} experiments queued")
    print(f"Epochs: {args.epoch if hasattr(args, 'epoch') else args.epochs}  |  FMA: {args.fma}")
    print(f"Queue:")
    for e in experiments:
        print(f"  {e:20s}  {EXPERIMENTS[e]['desc']}")

    results = {}
    wall_start = time.time()

    for i, exp_name in enumerate(experiments, 1):
        print(f"\n[{i}/{total}] {exp_name}", flush=True)
        ok = run_one(exp_name, args.epochs, args.fma)
        results[exp_name] = "OK" if ok else "FAILED"

    # ── Final summary ──────────────────────────────────────────────────────────
    wall_elapsed = (time.time() - wall_start) / 60
    passed  = [k for k, v in results.items() if v == "OK"]
    failed  = [k for k, v in results.items() if v == "FAILED"]

    print(f"\n{'='*60}")
    print(f"Phase {args.phase.upper()} complete — {wall_elapsed:.0f} min total")
    print(f"  Passed : {len(passed)}/{total}")
    if failed:
        print(f"  Failed : {len(failed)}")
        for f in failed:
            print(f"    - {f}")
    print(f"{'='*60}")
    print(f"\nResults logged to: test/results/results_log.csv")
    print(f"View summary with:")
    print(f"  python test/run_experiment.py --summary")


if __name__ == "__main__":
    main()
