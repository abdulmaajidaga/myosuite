"""
Generate Full FMA Range: produce multi-sample generated motions across 13 FMA levels.

Calls cvae_generate.py for each score, then run_generated_pipeline.py for IK/ID.
Skips scores that already have all samples.

FMA scores: [10, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 66]
  - 7 new levels (10, 15, 25, 35, 45, 55, 60) + 6 existing (18, 20, 30, 40, 50, 66)
  - 11 samples each = 143 total generated sessions

Usage:
  python scripts/generate_full_fma_range.py                 # Full run
  python scripts/generate_full_fma_range.py --n-samples 5   # Fewer samples per score
  python scripts/generate_full_fma_range.py --skip-id       # Only generate CSVs + IK (no ID)
  python scripts/generate_full_fma_range.py --scores 10 15 25   # Only specific scores
"""
import os
import sys
import subprocess
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.utils.config import get_path, get_project_root, load_config

WORKSPACE = get_project_root()
CONDA_PYTHON = load_config()["conda_python"]

ALL_FMA_SCORES = [10, 15, 18, 20, 25, 30, 35, 40, 45, 50, 55, 60, 66]
DEFAULT_N_SAMPLES = 11


def count_existing_samples(fma_score, csv_dir):
    """Count how many samples already exist for this FMA score."""
    count = 0
    for i in range(100):
        path = os.path.join(csv_dir, f"FMA_{fma_score}_s{i:02d}.csv")
        if os.path.exists(path):
            count += 1
        else:
            break
    # Also check the base file (FMA_{score}.csv)
    base = os.path.join(csv_dir, f"FMA_{fma_score}.csv")
    if os.path.exists(base):
        count = max(count, 1)
    return count


def generate_samples(fma_score, n_samples, csv_dir):
    """Run cvae_generate.py to produce N samples for a given FMA score."""
    print(f"\n  Generating {n_samples} samples for FMA {fma_score}...")

    env = os.environ.copy()
    env["PYTHONPATH"] = WORKSPACE + os.pathsep + env.get("PYTHONPATH", "")

    result = subprocess.run(
        [CONDA_PYTHON, "src/generation/cvae_generate.py",
         "--fma", str(fma_score),
         "--n-samples", str(n_samples),
         "--no-viz"],
        capture_output=True, text=True, cwd=WORKSPACE, env=env
    )

    if result.returncode != 0:
        print(f"  GENERATION FAILED for FMA {fma_score}:")
        print(result.stderr[-500:])
        return False

    print(f"  Generated {n_samples} CSVs for FMA {fma_score}")
    return True


def run_pipeline_for_score(fma_score, n_samples, skip_id=False):
    """Run the generated pipeline (IK + ID) for all samples of a given FMA score."""
    csv_dir = get_path("output_generated_csv")
    output_dir = get_path("output_generated")

    # Collect all CSV files for this FMA score
    files = []
    for i in range(n_samples):
        path = os.path.join(csv_dir, f"FMA_{fma_score}_s{i:02d}.csv")
        if os.path.exists(path):
            files.append(f"FMA_{fma_score}_s{i:02d}.csv")

    # Also process base file
    base = os.path.join(csv_dir, f"FMA_{fma_score}.csv")
    if os.path.exists(base) and f"FMA_{fma_score}.csv" not in files:
        files.append(f"FMA_{fma_score}.csv")

    if not files:
        print(f"  No CSV files found for FMA {fma_score}")
        return

    print(f"\n  Running pipeline for {len(files)} files (FMA {fma_score})...")

    env = os.environ.copy()
    env["PYTHONPATH"] = WORKSPACE + os.pathsep + env.get("PYTHONPATH", "")

    cmd = [CONDA_PYTHON, "scripts/run_generated_pipeline.py"] + files
    if skip_id:
        cmd.append("--skip-id")

    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=WORKSPACE, env=env,
        timeout=3600  # 1 hour timeout per score
    )

    if result.returncode != 0:
        print(f"  PIPELINE FAILED for FMA {fma_score}:")
        print(result.stderr[-500:])
    else:
        print(f"  Pipeline complete for FMA {fma_score}")


def main():
    parser = argparse.ArgumentParser(description="Generate full FMA range with multi-sample pipeline")
    parser.add_argument("--scores", type=int, nargs="+", default=None,
                        help=f"Specific FMA scores to generate (default: {ALL_FMA_SCORES})")
    parser.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                        help=f"Samples per FMA score (default: {DEFAULT_N_SAMPLES})")
    parser.add_argument("--skip-id", action="store_true",
                        help="Skip inverse dynamics (only generate CSV + IK)")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate CSVs (skip IK and ID pipeline)")
    parser.add_argument("--pipeline-only", action="store_true",
                        help="Only run pipeline on existing CSVs (skip generation)")
    args = parser.parse_args()

    fma_scores = args.scores or ALL_FMA_SCORES
    n_samples = args.n_samples
    csv_dir = get_path("output_generated_csv")

    print("=" * 60)
    print("Full FMA Range Generation")
    print("=" * 60)
    print(f"FMA scores: {fma_scores}")
    print(f"Samples per score: {n_samples}")
    print(f"Total target: {len(fma_scores) * n_samples} sessions")
    print(f"CSV dir: {csv_dir}")

    # Phase 1: Generate CSVs
    if not args.pipeline_only:
        print(f"\n{'=' * 60}")
        print("Phase 1: CVAE Generation")
        print(f"{'=' * 60}")

        for fma in fma_scores:
            existing = count_existing_samples(fma, csv_dir)
            if existing >= n_samples:
                print(f"\n  FMA {fma}: already has {existing} samples, skipping generation")
                continue
            generate_samples(fma, n_samples, csv_dir)

    # Phase 2: Run pipeline (IK + ID)
    if not args.generate_only:
        print(f"\n{'=' * 60}")
        print("Phase 2: IK/ID Pipeline")
        print(f"{'=' * 60}")

        for fma in fma_scores:
            run_pipeline_for_score(fma, n_samples, skip_id=args.skip_id)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")

    id_dir = os.path.join(get_path("output_generated"), "id")
    total = 0
    for fma in fma_scores:
        count = 0
        for name in os.listdir(id_dir) if os.path.isdir(id_dir) else []:
            if name.startswith(f"FMA_{fma}") and os.path.isfile(
                    os.path.join(id_dir, name, 'effort_metrics.json')):
                count += 1
        total += count
        status = "OK" if count >= n_samples else f"INCOMPLETE ({count}/{n_samples})"
        print(f"  FMA {fma:3d}: {count:3d} sessions  [{status}]")

    print(f"\n  Total: {total} sessions in {id_dir}")


if __name__ == '__main__':
    main()
