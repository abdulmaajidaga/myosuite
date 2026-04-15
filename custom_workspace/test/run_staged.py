"""
run_staged.py — Phased pipeline for all 12 CVAE configs × 51 FMA scores.

USAGE
-----
# Phase 1a: generate one sample video per config (FMA 40) for visual confirmation
python test/run_staged.py --phase sample

# Phase 1b: after green-lighting samples, generate ALL 51 FMA scores (CSV+TRC+MOT+video)
python test/run_staged.py --phase generate

# Phase 2: inverse dynamics on all confirmed MOTs
python test/run_staged.py --phase id

# Phase 3: paper figures
python test/run_staged.py --phase plots

# Run a single config only (useful for debugging)
python test/run_staged.py --phase sample --stage 3 --aug smote

# Run from a specific FMA score (resume interrupted run)
python test/run_staged.py --phase generate --stage 3 --aug smote --fma-start 30

OUTPUT
------
test/output/final/
  stage{N}_{aug}/
    csv/          FMA_16.csv … FMA_66.csv
    trc/          FMA_16.trc … FMA_66.trc
    mot/          FMA_16.mot … FMA_66.mot
    videos/       FMA_16.mp4 … FMA_66.mp4  (+ muscle videos)
    id/           FMA_16/ … FMA_66/
    plots/        paper figures
    samples/      FMA_40.mp4  (sample confirmation video)
"""

import os
import sys
import subprocess
import argparse
import time

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))          # test/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                           # custom_workspace/
CONDA_PY    = "/home/abdul/miniconda3/envs/MyoSuite/bin/python"
OUTPUT_ROOT = os.path.join(BASE_DIR, "output", "final")

GENERATE_SCRIPT  = os.path.join(BASE_DIR, "src", "generation", "cvae_generate.py")
PIPELINE_SCRIPT  = os.path.join(BASE_DIR, "scripts", "run_generated_pipeline.py")

# ── Config grid ───────────────────────────────────────────────────────────────
ALL_STAGES = [0, 1, 2, 3]
ALL_AUGS   = ["dtw", "smote", "linear"]
FMA_SCORES = list(range(16, 67))          # 51 scores
SAMPLE_FMA = 40                           # used for Phase 1a confirmation

# ── Helpers ───────────────────────────────────────────────────────────────────

def run(cmd, label, cwd=PROJECT_ROOT, loud=False):
    """Run a subprocess, streaming output if loud=True, else quiet."""
    start = time.time()
    if loud:
        print(f"  >> {label}")
        result = subprocess.run(cmd, cwd=cwd)
    else:
        result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    elapsed = time.time() - start
    ok = result.returncode == 0
    status = "OK" if ok else "FAILED"
    print(f"  [{status}] {label} ({elapsed:.1f}s)")
    if not ok and not loud:
        # Print last few lines of stderr so failures are diagnosable
        lines = (result.stderr or "").strip().split("\n")
        for l in lines[-5:]:
            print(f"    {l}")
    return ok


def config_dir(stage, aug):
    return os.path.join(OUTPUT_ROOT, f"stage{stage}_{aug}")


def ensure_dirs(stage, aug):
    base = config_dir(stage, aug)
    for sub in ["csv", "trc", "mot", "videos", "id", "plots"]:
        os.makedirs(os.path.join(base, sub), exist_ok=True)
    return base


# ── Phase 1a: sample ──────────────────────────────────────────────────────────

def phase_sample(stages, augs):
    """Generate one FMA_40 video per config for visual confirmation."""
    print(f"\n{'='*70}")
    print(f"PHASE 1a — Sample videos (FMA {SAMPLE_FMA}) for all {len(stages)*len(augs)} configs")
    print(f"{'='*70}")
    failed = []

    for stage in stages:
        for aug in augs:
            label = f"stage{stage}_{aug}"
            base  = ensure_dirs(stage, aug)
            csv_dir = os.path.join(base, "csv")
            print(f"\n[{label}]")

            # 1. Generate CSV
            ok = run([CONDA_PY, GENERATE_SCRIPT,
                      "--fma", str(SAMPLE_FMA),
                      "--stage", str(stage),
                      "--data-source", aug,
                      "--output-dir", csv_dir],
                     f"generate FMA {SAMPLE_FMA}")
            if not ok:
                failed.append(f"{label} generate"); continue

            # 2. CSV → TRC → MOT → video (single file, no ID)
            ok = run([CONDA_PY, PIPELINE_SCRIPT,
                      f"FMA_{SAMPLE_FMA}.csv",
                      "--input-dir", csv_dir,
                      "--output-dir", base,
                      "--skip-id"],
                     f"IK + video FMA {SAMPLE_FMA}")
            if not ok:
                failed.append(f"{label} pipeline"); continue


    print(f"\n{'='*70}")
    if failed:
        print(f"SAMPLE PHASE DONE — {len(failed)} failures: {failed}")
    else:
        print(f"SAMPLE PHASE DONE — all {len(stages)*len(augs)} configs generated OK")
    print(f"\nSample videos are in:")
    for stage in stages:
        for aug in augs:
            p = os.path.join(config_dir(stage, aug), "videos", f"FMA_{SAMPLE_FMA}.mp4")
            exists = "✓" if os.path.exists(p) else "✗"
            print(f"  {exists}  test/output/final/stage{stage}_{aug}/videos/FMA_{SAMPLE_FMA}.mp4")
    print(f"\nReview the videos above, then run:")
    print(f"  python test/run_staged.py --phase generate")


# ── Phase 1b: generate ────────────────────────────────────────────────────────

def phase_generate(stages, augs, fma_start=16):
    """Generate all 51 FMA scores per config: CSV + TRC + MOT + video."""
    fma_range = [f for f in FMA_SCORES if f >= fma_start]
    total = len(stages) * len(augs) * len(fma_range)
    print(f"\n{'='*70}")
    print(f"PHASE 1b — Full generation: {len(stages)} stages × {len(augs)} augs × {len(fma_range)} FMA scores = {total} files")
    print(f"{'='*70}")
    failed = []
    done = 0

    for stage in stages:
        for aug in augs:
            label = f"stage{stage}_{aug}"
            base  = ensure_dirs(stage, aug)
            csv_dir = os.path.join(base, "csv")
            print(f"\n{'─'*50}")
            print(f"Config: {label}")
            print(f"{'─'*50}")

            for fma in fma_range:
                csv_path = os.path.join(csv_dir, f"FMA_{fma}.csv")
                mot_path = os.path.join(base, "mot", f"FMA_{fma}.mot")
                vid_path = os.path.join(base, "videos", f"FMA_{fma}.mp4")

                # Skip if video already exists (resume support)
                if os.path.exists(vid_path) and os.path.exists(mot_path):
                    done += 1
                    print(f"  [SKIP] FMA {fma} — already complete")
                    continue

                # Step 1: generate CSV (skip if exists)
                if not os.path.exists(csv_path):
                    ok = run([CONDA_PY, GENERATE_SCRIPT,
                              "--fma", str(fma),
                              "--stage", str(stage),
                              "--data-source", aug,
                              "--output-dir", csv_dir],
                             f"generate FMA {fma}")
                    if not ok:
                        failed.append(f"{label} FMA {fma} generate"); continue

                # Step 2: CSV → TRC → MOT → video (no ID yet)
                ok = run([CONDA_PY, PIPELINE_SCRIPT,
                          f"FMA_{fma}.csv",
                          "--input-dir", csv_dir,
                          "--output-dir", base,
                          "--skip-id"],
                         f"IK + video FMA {fma}")
                if not ok:
                    failed.append(f"{label} FMA {fma} pipeline")
                    continue

                done += 1
                print(f"  [{done}/{total}] {label} FMA {fma} done")

    print(f"\n{'='*70}")
    if failed:
        print(f"GENERATE PHASE DONE — {len(failed)} failures:")
        for f in failed:
            print(f"  - {f}")
    else:
        print(f"GENERATE PHASE DONE — {done} files complete")
    print(f"\nReview videos in test/output/final/stage*/videos/")
    print(f"When satisfied, run:")
    print(f"  python test/run_staged.py --phase id")


# ── Phase 2: id ───────────────────────────────────────────────────────────────

def phase_id(stages, augs, fma_start=16):
    """Run inverse dynamics on all MOTs that exist."""
    fma_range = [f for f in FMA_SCORES if f >= fma_start]
    print(f"\n{'='*70}")
    print(f"PHASE 2 — Inverse dynamics: {len(stages)*len(augs)} configs × {len(fma_range)} FMA scores")
    print(f"{'='*70}")
    failed = []
    done = 0

    for stage in stages:
        for aug in augs:
            label = f"stage{stage}_{aug}"
            base  = config_dir(stage, aug)
            csv_dir = os.path.join(base, "csv")
            print(f"\n{'─'*50}")
            print(f"Config: {label}")
            print(f"{'─'*50}")

            for fma in fma_range:
                mot_path = os.path.join(base, "mot", f"FMA_{fma}.mot")
                id_dir   = os.path.join(base, "id", f"FMA_{fma}")

                if not os.path.exists(mot_path):
                    print(f"  [MISSING MOT] FMA {fma} — run generate phase first")
                    continue

                # Skip if ID already complete (has torques.csv)
                if os.path.exists(os.path.join(id_dir, "torques.csv")):
                    done += 1
                    print(f"  [SKIP] FMA {fma} — ID already complete")
                    continue

                ok = run([CONDA_PY, PIPELINE_SCRIPT,
                          f"FMA_{fma}.csv",
                          "--input-dir", csv_dir,
                          "--output-dir", base,
                          "--skip-video"],
                         f"ID FMA {fma}")
                if not ok:
                    failed.append(f"{label} FMA {fma}")
                    continue

                done += 1
                print(f"  [done] {label} FMA {fma}")

    print(f"\n{'='*70}")
    if failed:
        print(f"ID PHASE DONE — {len(failed)} failures:")
        for f in failed: print(f"  - {f}")
    else:
        print(f"ID PHASE DONE — {done} complete")
    print(f"\nWhen ready, run:")
    print(f"  python test/run_staged.py --phase plots")


# ── Phase 3: plots ────────────────────────────────────────────────────────────

def phase_plots(stages, augs):
    """Generate paper figures for all configs."""
    print(f"\n{'='*70}")
    print(f"PHASE 3 — Paper figures for {len(stages)*len(augs)} configs")
    print(f"{'='*70}")

    for stage in stages:
        for aug in augs:
            label     = f"stage{stage}_{aug}"
            base      = config_dir(stage, aug)
            id_dir    = os.path.join(base, "id")
            plots_dir = os.path.join(base, "plots")
            csv_dir   = os.path.join(base, "csv")
            os.makedirs(plots_dir, exist_ok=True)
            print(f"\n[{label}]")

            # Cross-FMA comparison plots (torques, synergies, activations)
            run([CONDA_PY, "-c",
                 f"import sys; sys.path.insert(0, '{PROJECT_ROOT}');"
                 f"from src.visualization.plot_id_comparison import generate_all;"
                 f"generate_all(id_base_dir='{id_dir}', output_dir='{plots_dir}')"],
                "cross-FMA comparison plots")

            # Training verification dashboard
            run([CONDA_PY,
                 os.path.join(BASE_DIR, "src", "visualization", "plot_training_verification.py"),
                 "--model-version", f"stage{stage}",
                 "--data-source", aug,
                 "--gen-dir", csv_dir,
                 "--output-dir", plots_dir],
                "verification dashboard")

            # Literature validation
            run([CONDA_PY,
                 os.path.join(BASE_DIR, "scripts", "literature_validation.py"),
                 "--input-id-dir", id_dir,
                 "--output-dir", plots_dir],
                "literature validation")

    print(f"\n{'='*70}")
    print(f"PLOTS PHASE DONE — figures in test/output/final/stage*/plots/")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Staged pipeline for 12-config CVAE sweep")
    parser.add_argument("--phase", required=True,
                        choices=["sample", "generate", "id", "plots"],
                        help="Which phase to run")
    parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3],
                        help="Limit to one stage (default: all 4)")
    parser.add_argument("--aug", type=str, choices=["dtw", "smote", "linear"],
                        help="Limit to one augmentation (default: all 3)")
    parser.add_argument("--fma-start", type=int, default=16,
                        help="Resume from this FMA score (default: 16)")
    args = parser.parse_args()

    stages = [args.stage] if args.stage is not None else ALL_STAGES
    augs   = [args.aug]   if args.aug   is not None else ALL_AUGS

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    if args.phase == "sample":
        phase_sample(stages, augs)
    elif args.phase == "generate":
        phase_generate(stages, augs, args.fma_start)
    elif args.phase == "id":
        phase_id(stages, augs, args.fma_start)
    elif args.phase == "plots":
        phase_plots(stages, augs)


if __name__ == "__main__":
    main()
