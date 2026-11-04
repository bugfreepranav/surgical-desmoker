# Surgical-Desmoker 🩺💨

**Dense Prediction Transformer (DPT) with Physics-Guided Reconstruction & GPU Smoke Synthesis for Laparoscopic Surgery**

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Domain](https://img.shields.io/badge/Domain-Surgical%20Vision%20%26%20Medical%20AI-success.svg)]()
[![Task](https://img.shields.io/badge/Task-Electrocautery%20Smoke%20Removal-orange.svg)]()

---

## 📌 Overview

During minimally invasive laparoscopic and robotic surgery, electrosurgical instruments (cautery hooks, harmonic ultrasonic shears, bipolar graspers) produce thick plumes of surgical smoke. This smoke degrades visibility, obscures anatomical tissue boundaries, hampers robotic tracking, and jeopardizes patient safety.

**Surgical-Desmoker** is an end-to-end deep learning framework designed specifically for real-time surgical smoke removal. It unites:
1. **Vision Transformer (ViT) Multi-Stage Backbone**: Extracts semantic representations across 4 resolution hierarchies ($1/4, 1/8, 1/16, 1/32$).
2. **Dense Prediction Transformer (DPT) Decoder**: Reassembles 1D patch tokens into 2D feature pyramids connected through residual feature fusion blocks.
3. **Physics-Guided DesmokeHead**: Incorporates Koschmieder's atmospheric scattering model to estimate the optical transmission map $t(x)$ and endoscopic spotlight illumination $A(x)$, supplemented by residual micro-vascular texture refinement $\Delta J(x)$.
4. **On-the-Fly GPU Smoke Synthesizer**: Overcomes the lack of paired clinical data by simulating realistic electrosurgical smoke (multi-octave fractal turbulence, tool-tip plume dispersion, spotlight vignetting, chromatic absorption) directly on PyTorch GPU tensors.

---

## 🗂 Project Structure

```
surgical-desmoker/
├── models/
│   ├── __init__.py
│   ├── desmoker.py         # ViT, DPTDecoder, FusionBlock, DesmokeHead
│   └── losses.py           # PerceptualLoss, SSIMLoss, L1, EdgeLoss
├── data/
│   ├── __init__.py
│   ├── dataset.py          # CleanFrameDataset, PairedDesmokeDataset
│   └── synthesizer.py      # GPUSmokeSynthesizer (vectorized GPU physics)
├── trained_models/         # Checkpoints (.pt)
├── utils/
│   ├── __init__.py
│   └── metrics.py          # PSNR, SSIM, RMSE evaluation helpers
├── option.py               # Argument parser & config hyperparams
├── main.py                 # Training loop & validation orchestrator
├── test.py                 # Inference & batch testing script
├── requirements.txt
└── README.md
```

---

## 🔬 Mathematical Formulation

### 1. Endoscopic Optical Scattering Model
Under Koschmieder's atmospheric scattering law adapted for endoscopic cavities:
$$I(x) = J(x) \odot t(x) + A(x) \odot (1 - t(x))$$

Where:
* $I(x)$: Observed smoky surgical frame.
* $J(x)$: De-hazed, clean underlying surgical scene.
* $t(x) = \exp(-\beta(x))$: Medium transmission map indicating light penetration ($t(x) \in (0, 1]$).
* $A(x)$: Spatially modulated endoscopic illumination vector reflecting radial spotlight vignetting.

### 2. Dual-Branch Physics-Guided Reconstruction
The `DesmokeHead` predicts the transmission map $t(x)$, endoscopic illumination vector $A$, and residual micro-texture correction $\Delta J(x)$:
$$J_{phys}(x) = \frac{I(x) - A}{\max(t(x), \epsilon)} + A$$
$$J_{clean}(x) = \text{clamp}\left(J_{phys}(x) + \Delta J(x), 0.0, 1.0\right)$$

### 3. Multi-Objective Composite Loss
$$\mathcal{L}_{total} = \lambda_1 \mathcal{L}_{Charbonnier} + \lambda_{ssim} (1 - \text{SSIM}) + \lambda_{perc} \mathcal{L}_{VGG} + \lambda_{edge} \mathcal{L}_{Sobel} + \lambda_t \mathcal{L}_{trans}$$

* **Charbonnier Loss**: $\sqrt{\|J - \hat{J}\|^2 + \epsilon^2}$ ensures outlier-robust pixel fidelity.
* **SSIM Loss**: Preserves structural and luminance coherence across organ surfaces.
* **Perceptual Loss**: Deep feature distance computed from VGG-16 layers (`relu1_2`, `relu2_2`, `relu3_3`, `relu4_3`).
* **Sobel Edge Loss**: Preserves surgical instrument edges, sutures, and micro-vascular tissue networks.

---

## 🚀 Installation & Environment Setup

### 1. Prerequisites
* Python 3.8+
* CUDA 11.8+ or 12.0+ (for GPU acceleration)

### 2. Install Dependencies
```bash
cd surgical-desmoker
pip install -r requirements.txt
```

---

## ⚡ Quick Start & Demo

### 1. Self-Contained Synthetic Demo
Run inference on a procedural laparoscopic tissue scene with on-the-fly GPU smoke synthesis (requires zero external datasets):
```bash
python test.py --demo --output results/demo
```
This generates:
* `results/demo/demo_surgical_comparison.png` displaying `[Smoky Input | Desmoked Output | Ground Truth Clean | Transmission Map]`.
* Restoration metrics report (PSNR, SSIM, RMSE, and Latency in ms).

---

## 🏋️ Training

### Mode A: Self-Supervised with GPU Smoke Synthesizer (Recommended)
Train on any folder of clean surgical frames without needing paired ground truth. The `GPUSmokeSynthesizer` generates randomized electrocautery smoke on-the-fly on GPU:
```bash
python main.py \
    --data_dir /path/to/clean_frames \
    --val_dir /path/to/val_clean_frames \
    --img_size 256 \
    --batch_size 8 \
    --epochs 100 \
    --lr 1e-4 \
    --amp
```

### Mode B: Paired Benchmark Training
If using pre-paired clinical datasets (e.g., Cholec80-smoke benchmark with matching `smoky/` and `clean/` subdirectories):
```bash
python main.py \
    --paired \
    --data_dir /path/to/paired_dataset \
    --img_size 256 \
    --batch_size 8 \
    --epochs 100
```

### Mode C: Testing / Self-Contained Procedural Training
```bash
python main.py --demo --epochs 10 --batch_size 4
```

---

## 🔍 Inference & Evaluation

### 1. Single Image Desmoking
```bash
python test.py \
    --input sample_smoky.png \
    --checkpoint trained_models/best_checkpoint.pt \
    --output results/inferences \
    --save_comparison
```

### 2. Batch Directory Evaluation
Evaluate against clean ground truth to compute mean PSNR, SSIM, and FPS:
```bash
python test.py \
    --input /path/to/smoky_frames \
    --gt_dir /path/to/clean_frames \
    --checkpoint trained_models/best_checkpoint.pt \
    --output results/eval_benchmark
```
Outputs `evaluation_summary.json` containing comprehensive benchmark statistics.

### 3. Surgical Video Desmoking (.mp4 / .avi)
```bash
python test.py \
    --input laparoscopy_surgery.mp4 \
    --checkpoint trained_models/best_checkpoint.pt \
    --output results/videos
```

---

## ⚙️ Hyperparameter Configuration (`option.py`)

| Parameter | Type | Default | Description |
|:---|:---:|:---:|:---|
| `--img_size` | `int` | `256` | Input resolution ($H \times W$) |
| `--patch_size` | `int` | `16` | Vision Transformer patch embedding size |
| `--embed_dim` | `int` | `384` | ViT token feature dimension |
| `--depth` | `int` | `12` | Number of ViT encoder transformer blocks |
| `--num_heads` | `int` | `6` | Self-attention heads in each transformer block |
| `--decoder_features` | `int` | `256` | DPT decoder intermediate channel dimension |
| `--use_physics` | `flag` | `True` | Enables Koschmieder physical scattering head |
| `--smoke_density_min` | `float` | `0.25` | Lower bound for synthetic smoke optical density |
| `--smoke_density_max` | `float` | `0.95` | Upper bound for synthetic smoke optical density |
| `--lambda_l1` | `float` | `1.0` | Weight for Charbonnier L1 loss |
| `--lambda_ssim` | `float` | `0.2` | Weight for differentiable SSIM loss |
| `--lambda_perceptual` | `float` | `0.1` | Weight for VGG perceptual loss |
| `--lambda_edge` | `float` | `0.05` | Weight for Sobel boundary preservation loss |
| `--lr` | `float` | `1e-4` | Initial learning rate (AdamW) |
| `--amp` | `flag` | `True` | Automatic mixed precision (AMP) |

---

## 📊 Benchmark Metrics Reference

Surgical desmoking quality is evaluated using standard medical computer vision metrics:
* **PSNR (Peak Signal-to-Noise Ratio)**: Higher is better ($>30\text{ dB}$ indicates high-fidelity reconstruction).
* **SSIM (Structural Similarity Index)**: Measures preservation of organ structural details ($>0.92$ is clinically superior).
* **Throughput (FPS)**: Target $>30\text{ FPS}$ for real-time surgical video integration.

---

## 📄 License
This project is released under the MIT License. See [LICENSE](LICENSE) for details.
