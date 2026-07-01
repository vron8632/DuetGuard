# DuetGuard: Dual-Branch Active-Passive Fusion for Robust Image Tamper Detection

**Active watermarking + passive SPN fingerprint fusion for image forensics.**

DuetGuard jointly optimizes an OmniGuard-based watermarking branch and a lightweight CNN-based SPN extractor through a learned fusion layer, achieving 99.4% detection accuracy on synthetic tampering benchmarks.

## Quick Start

```bash
# Install dependencies
conda create -n apjf python=3.10
conda activate apjf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# Run full CVPR evaluation suite
python -u apjf_src/run_cvpr.py
```

## Results

| Experiment | Accuracy |
|-----------|----------|
| Watermark only | 50.0% |
| SPN only (no finetune) | 50.6% |
| **DuetGuard (P3)** | **99.4%** |

## Repository Structure

```
apjf_src/           # Source code
  ├── train.py           # Joint training
  ├── run_cvpr.py        # Full evaluation suite
  ├── run_experiments.py # Experiment runner
  ├── figures.py         # Paper figure generation
  ├── models/            # Network definitions
  ├── data/              # Dataset classes
  └── losses.py          # Loss functions
AAAI_PLAN.md        # AAAI submission plan
lessons/            # Course materials
指南.md              # Documentation
```
