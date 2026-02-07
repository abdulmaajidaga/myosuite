# Codebase Restructuring Plan

Saved for later implementation. See conversation for full details.

## Proposed Structure

```
custom_workspace/
├── config/
│   └── settings.yaml            # All hyperparams & paths
├── data/kinematic/
│   ├── healthy/
│   ├── stroke/
│   └── augmented/
├── models/myo_sim/arm/
├── src/
│   ├── data_processing/         # CSV → TRC
│   ├── inverse_kinematics/      # TRC → MOT
│   ├── inverse_dynamics/        # MOT → torques
│   ├── generation/              # CVAE training & inference
│   ├── rl/                      # Imitation learning
│   ├── visualization/           # All rendering/plotting
│   └── utils/                   # Shared I/O, transforms, config
├── scripts/                     # Thin CLI entry points
├── tests/
├── output/
├── requirements.txt
└── CLAUDE.md
```

## Key Changes
- Split by domain (IK, ID, generation, viz) not by tool
- Delete IK/processed/ (legacy duplication)
- Absorb IK/modular/ into src/
- Move CVAE from IK/cutoff/ to src/generation/
- Centralize config in settings.yaml
- Never mix visualization into solvers
- Add tests/ and requirements.txt
