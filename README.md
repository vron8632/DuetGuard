<h1 align="center">🛡️ DuetGuard</h1>
<h3 align="center">Dual-Branch Active-Passive Fusion for Robust AI-Generated Content Detection</h3>

<p align="center">
  <b>Active watermarking (OmniGuard)</b> + <b>Passive SPN fingerprinting</b> →
  <b>99.4%</b> detection accuracy via joint end-to-end training
</p>

<p align="center">
  <a href="#-overview">Overview</a> •
  <a href="#-datasets">Datasets</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-results">Results</a> •
  <a href="#-repository-structure">Structure</a>
</p>

---

<p align="center">
  <img src="fig/architecture.jpg" alt="DuetGuard Architecture" width="85%">
</p>

---

## 📦 Datasets

All datasets used in the paper are available at:  
**Baidu NetDisk:** https://pan.baidu.com/s/1tWSeEi1dOG7oZjLSguI0Xw  
**Password:** `bkww`

| Content | Description |
|---------|-------------|
| `aigc_test/` | AIGC-generated test set (SD Inpaint) |
| `casia2/` | CASIA v2 forensic benchmark (splicing, copy-move) |
| `coco_mini/` | COCO 2017 subset for training |
| `columbia/` | Columbia splicing dataset |

After downloading, place in `data/`:

```
DuetGuard/data/
├── coco/train2017/       # COCO 2017 train
├── coco/val2017/         # COCO 2017 val
├── aigc_test/
├── casia2/
└── columbia/
```

---

## 📋 Overview

DuetGuard is an end-to-end framework for image tamper detection that synergistically combines **two physically independent forensic signals**:

- **🔵 Active Branch (Watermarking):** A frozen OmniGuard decoder extracts a pre-embedded secret watermark. Tampering disrupts the watermark, creating a detectable spatial signal.
- **🟢 Passive Branch (SPN):** A lightweight CNN (268K params) extracts the camera's Sensor Pattern Noise fingerprint—an intrinsic hardware-level characteristic.
- **🔄 Fusion Layer:** A compact MLP (50K params) integrates both signals into a 3-class verdict: **Authentic / Suspicious / Confirmed Forgery**, with uncertainty estimation.

### Key Innovation: Dual-Mask Consistency Loss

A novel loss function enforces **spatial alignment** between the active branch's watermark difference map and the passive branch's noise residual map. This creates cross-modal self-supervision where two independent signals guide each other during training.

---

## 🚀 Quick Start

### Requirements
- Python 3.10+, PyTorch 2.6+, CUDA 12.4
- NVIDIA GPU with ≥8GB VRAM (tested on RTX 4060 Ti 16GB)

### Setup

```bash
git clone https://github.com/vron8632/DuetGuard.git
cd DuetGuard
conda create -n apjf python=3.10
conda activate apjf
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

### Training

```bash
# Stage 1: SPN contrastive pretraining
python apjf_src/train_spn.py

# Stage 2: Joint P3 fine-tuning
python -u apjf_src/train.py
```

### Evaluation

```bash
# Full evaluation suite
python -u apjf_src/run_cvpr.py

# Supplementary experiments (zero-shot, strong attacks)
bash run_supplement_experiments.sh
```

---

## 📊 Results

### Core Detection Accuracy

| Configuration | Accuracy |
|:---|---:|
| Watermark Only (P0) | 50.0% |
| SPN Only (No Finetune) | 50.6% |
| **DuetGuard (P3)** | **99.4%** |

**Extended (4,000 samples):** 99.38%, 95% Wilson CI [99.08%, 99.58%]
**Multi-seed (3 runs):** 99.29% ± 0.16%

### Ablation Study

| Configuration | Accuracy |
|:---|---:|
| Watermark Only (P0) | 50.0% |
| SPN Only (No Finetune) | 50.6% |
| **DuetGuard (P3)** | **99.4%** |

### Confusion Matrix

|  | Pred: Auth | Pred: Susp | Pred: Forg |
|---|:---:|:---:|:---:|
| **GT: Authentic** | 399 | 0 | 1 |
| **GT: Suspicious** | 0 | 0 | 0 |
| **GT: Forgery** | 5 | 0 | 395 |

### Robustness

| Attack | Accuracy |
|:---|---:|
| Clean | 99.0% |
| JPEG q=70 | 94.8% |
| JPEG q=50 | 90.8% |
| JPEG q=30 | 85.2% |
| JPEG q=15 | 53.5% |
| Gaussian σ=0.01 | 99.8% |
| Gaussian σ=0.05 | 87.0% |
| Gaussian σ=0.10 | 76.2% |
| Gaussian σ=0.20 | 64.2% |

### Zero-Shot Generalization

| Dataset | Samples | Accuracy |
|:---|---:|---:|
| CASIA v2 | 9,501 | **78.0%** |
| Columbia | 1,845 | 50.6% |
| AIGC Test (SD Inpaint) | 400 | 50.0% |

### Image Quality

| PSNR | SSIM | LPIPS |
|:---:|:---:|:---:|
| 48.86 dB | 0.991 | 0.0002 |

---


---

## 📁 Repository Structure

```
DuetGuard/
├── apjf_src/                  # Core source code
│   ├── train.py               # Joint training entry point
│   ├── train_spn.py           # SPN contrastive pretraining
│   ├── run_cvpr.py            # Evaluation suite
│   ├── losses.py              # 5-term joint loss
│   ├── figures.py             # Figure generation
│   ├── exp_aaai_supp.py       # Supplementary experiments
│   ├── fig_qualitative.py     # Qualitative results
│   ├── models/
│   │   ├── adapter.py         # OmniGuard wrapper
│   │   ├── spn_extractor.py   # SPN CNN (268K params)
│   │   └── fusion_layer.py    # MLP classifier (50K params)
│   └── data/
│       ├── dataset.py         # Joint training dataset
│       └── spn_dataset.py     # SPN pretraining dataset
├── results/                   # Experiment results & figures
├── run_supplement_experiments.sh
├── EXPERIMENTS.md             # Full experiment log
└── README.md
```

---

