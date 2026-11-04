# Surgical Desmoking with ViT + DPT 🩺💨

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2603.25867-b31b1b.svg)](https://arxiv.org/abs/2603.25867)
[![Task](https://img.shields.io/badge/Task-Surgical%20Smoke%20Removal-orange.svg)]()

PyTorch reimplementation of the model from:  
**"Seeing Through Smoke: Surgical Desmoking for Improved Visual Perception" (arXiv:2603.25867)**, built from the paper text. Details not specified in the paper are implemented as marked assumptions in the codebase.

---

## 📌 Overview

During minimally invasive laparoscopic and robotic surgery, electrosurgical instruments (cautery hooks, harmonic ultrasonic shears) produce thick plumes of smoke that degrade visual perception and obscure critical anatomical boundaries.

This project implements an end-to-end surgical desmoking framework uniting:
- **ViT-B (MAE) Backbone**: Pretrained Vision Transformer extracting multi-scale hierarchical representations from blocks {3, 6, 9, 12}.
- **DPT Decoder**: Dense Prediction Transformer decoder reassembling 1D patch tokens into 2D feature pyramids via cascading residual fusion blocks.
- **`extra_block` Hook**: Pluggable hook in every fusion block for custom attention modules (e.g. prior-gated, frequency-split, specular-aware attention).
- **Physics-Inspired DesmokeHead**: Predicts $O = [K, B]$ in $\mathbb{R}^{H \times W \times 4}$ where:
  $$J = K \cdot I - B + I$$
  $$\text{Smoke Map } S = \text{conv}\left(\frac{B}{K + 1}\right)$$
- **Joint Output**: Simultaneously recovers the clean surgical field $J$ and the dense smoke density map $S$.
- **Dual Data Pipeline**:
  - **Synthetic Mode**: Generates realistic electrocautery smoke on-the-fly directly on PyTorch GPU tensors (fractal Brownian turbulence, tool-tip plume dispersion, spotlight vignetting).
  - **Paired Mode**: Supports training and evaluation on pre-paired surgical benchmarks.

---

## 🔬 Mathematical Formulation

The paper's head prediction formulation $O = [K, B]$ is a linear reparameterization of **Koschmieder's physical atmospheric scattering law**:
$$I = J \odot t + A(1 - t)$$

Where:
- $I$: Observed smoky frame
- $J$: De-hazed clean surgical scene
- $t$: Medium transmission map
- $A(1 - t)$: Airlight / smoke haze veil

Rearranging the paper's restoration equation:
$$J = K \cdot I - B + I = I \cdot (K + 1) - B$$
$$\implies I = J \cdot \left(\frac{1}{K + 1}\right) + \frac{B}{K + 1}$$

Comparing with Koschmieder's law reveals exact physical correspondence:
- **Transmission map**: $t = \frac{1}{K + 1}$
- **Scattered Airlight**: $A(1 - t) = \frac{B}{K + 1}$
- **Smoke Density Map**: $S = \text{conv}\left(\frac{B}{K + 1}\right)$ is directly extracted from the physical airlight.

---

## 🗂 Project Structure

```text
├── models/
│   ├── __init__.py         # Exports SurgicalDesmoker & TotalDesmokingLoss
│   ├── desmoker.py         # ViT-B (MAE) + DPTDecoder + FusionBlock + DesmokeHead
│   └── losses.py           # TotalDesmokingLoss: L1(J) + 0.1*L1(S) + Perceptual
├── data/
│   ├── __init__.py         # Exports CleanFrameDataset & GPUSmokeSynthesizer
│   ├── dataset.py          # CleanFrameDataset (512x640 bicubic, random flip)
│   └── synthesizer.py      # GPUSmokeSynthesizer (on-the-fly GPU fractal noise & alpha blending)
├── utils/
│   ├── __init__.py         # Exports calc_psnr
│   └── metrics.py          # PSNR & evaluation helpers
├── option.py               # Argument parser & hyperparameter configuration
├── main.py                 # Mixed-precision training orchestrator (AdamW + CosineAnnealing)
├── test.py                 # Single/batch inference saving clean and smoke maps
├── surgical_desmoker.py    # Standalone single-file paper reimplementation
└── requirements.txt
```

---

## 🚀 Installation

```bash
git clone https://github.com/bugfreepranav/surgical-desmoker.git
cd surgical-desmoker
pip install torch torchvision timm pillow numpy
```

---

## ⚡ Usage

### 1. Architecture & Smoke Test
```bash
python surgical_desmoker.py test
```

### 2. Training (Synthetic On-the-Fly GPU Smoke)
```bash
# Using the standalone script
python surgical_desmoker.py train --clean_dir /path/to/clean_frames --epochs 300 --bs 8

# Or using the modular training orchestrator
python main.py --clean_dir /path/to/clean_frames --epochs 300 --batch_size 4
```

### 3. Inference
```bash
# Standalone inference
python surgical_desmoker.py infer --ckpt desmoker.pt --image smoky.png --out clean.png

# Modular inference
python test.py --ckpt trained_models/latest.pt --test_image smoky.png --out_dir ./results
```
Outputs both:
- `{image}_clean.png`: Restored smoke-free surgical scene.
- `{image}_smoke.png`: Estimated smoke density map.

---

## 📊 Status
Work in progress. Reimplementation built directly from paper specifications.

## 📄 Citation
If you use this work, please cite the original paper:
```bibtex
@article{seeing_through_smoke_2026,
  title   = {Seeing Through Smoke: Surgical Desmoking for Improved Visual Perception},
  journal = {arXiv preprint arXiv:2603.25867},
  year    = {2026}
}
```
