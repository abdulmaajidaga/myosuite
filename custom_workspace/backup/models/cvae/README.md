# Part-1 CVAE weights (archive)

These 20 checkpoints are the trained output of the part-1 augmentation comparison.
They existed in **only two places** — this repo's working tree and the unpushed local
branch `backup/pre-push-fix-20260415-151613` — and were absent from every pushed ref,
verified by blob hash against all objects reachable from `main` and `test`. This
branch exists so they survive deletion of the local clone.

## The grid

A 3 x 3 sweep of model version against augmentation method, each saved twice
(final epoch and best validation loss):

|          | DTW                            | Linear                            | SMOTE                            |
|----------|--------------------------------|-----------------------------------|----------------------------------|
| **v1**   | `cvae_cutoff_fma_v1_dtw.pth`   | `cvae_cutoff_fma_v1_linear.pth`   | `cvae_cutoff_fma_v1_smote.pth`   |
| **v2**   | `cvae_cutoff_fma_v2_dtw.pth`   | `cvae_cutoff_fma_v2_linear.pth`   | `cvae_cutoff_fma_v2_smote.pth`   |
| **v3**   | `cvae_cutoff_fma_v3_dtw.pth`   | `cvae_cutoff_fma_v3_linear.pth`   | `cvae_cutoff_fma_v3_smote.pth`   |

Each also has a `_best` sibling. Plus `cvae_v1_best.pth` and `cvae_v2_best.pth`,
which predate the augmentation sweep.

Architectures: `model_v2.py` and `model_v3.py` (already on `main`).

## Loading one

A checkpoint is a bare `state_dict`, so the matching architecture must be
instantiated first. **The StandardScaler is not stored here** — a checkpoint is
useless without the scaler it was trained with, since inputs must be normalised
identically at generation time.

## Why these are worth keeping

They back the augmentation-selection result (SMOTE over DTW over linear). Having
the trained weights rather than only a results table is the difference between
answering a reviewer's question and re-running the sweep.

## Not here

The 9.4 GB of augmented training data (`data/kinematic/cutoff`, 174,049 files) was
deliberately not pushed. It is generated, reproducible from the 82 MB of source
mocap via the `generate_augmented_*.py` scripts on the `test` branch.
