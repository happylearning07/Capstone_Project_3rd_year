# Adversarial-Aware IoT Intrusion Detection System

## Acknowledgement

This project was made possible under the guidance of Prof. Somnath Tripathy. I sincerely thank him for his valuable support, guidance, and encouragement throughout the development of this project.

## Introduction

This repository implements an **end-to-end research pipeline** for **network intrusion detection** on **IoT-style traffic** (features aligned with the **IoT-23** family of datasets). Traffic is represented as tabular features after preprocessing; models learn to separate **benign** connections from several **attack classes** (for example DDoS, port scans, and malware-related flows).

The project’s distinctive focus is **not only accuracy on clean test data**, but **behavior under adversarial perturbation**. A **Gateway** module synthesizes evaluation scenarios: realistic mixed streams, attack-heavy batches, and **white-box FGSM** perturbations aimed at fooling each detector. Three detector families are trained and compared:

| Model | Role |
|--------|------|
| **BiGAN** | Binary anomaly detector (encoder + generator + discriminator). |
| **AAE + RF** | Adversarial autoencoder latent features + **Random Forest** for multi-class labels. |
| **Robust VAE-BiGAN** (`RobustVAEBiGAN`) | VAE-style encoder (μ, log σ) + generator + discriminator; calibrated for binary anomaly detection. |

Across the scripted comparisons—especially in **adversarial mode**—the **Robust VAE-BiGAN** is designed and observed to **resist evasion better** than BiGAN and AAE+RF: it maintains higher **detection rate (DR)** and lower **successful evasion** when inputs are perturbed to fool the model. Use `comparison.py` as the primary side-by-side benchmark (see [What to run for conclusions](#what-to-run-for-conclusions)).

---

## Goals

1. **Train and calibrate** multiple IDS backends on a shared feature space so metrics are comparable.
2. **Simulate operational conditions** (mixed benign/attack streams) and **targeted attack** settings via the Gateway.
3. **Stress-test with adversarial examples** (FGSM on the active detector’s own graph) to measure **robustness** and **evasion rate**.
4. **Demonstrate** that the **robust** architecture remains the strongest under attack among the three, while documenting trade-offs (e.g., multi-class granularity on AAE+RF).

---

## How it works (high level)

1. **Data & features**  
   CSV data (e.g. `data/iot23_combined_new.csv`) is processed into a fixed set of features; `saved_state/feature_cols.pkl` and scalers define the canonical input shape.

2. **Generative Gateway**  
   `gateway/` builds synthetic evaluation traffic by combining:
   - **GMM per class** (`fit_gmm.py`, weights under `gateway/weights/`),
   - optional **cGAN** or **TVAE** refiners (`gateway/trainer.py`, `gateway/train_tvae.py`),
   - blending controlled by **`--gan-blend`** (0 = pure GMM, 1 = full learned refiner).

3. **Modes** (`run_pipeline.py`, `Gateway.generate`):
   - **`stream`**: mixed benign/attack samples (default benign ratio 70%).
   - **`attack`**: attack-focused sampling.
   - **`adversarial`**: **FGSM** perturbations using the loaded IDS model’s gradients; reports **evasion** (attack classified as benign) vs **detection held**.

4. **Metrics**  
   For binary detectors: **Detection Rate (DR)**, **False Alarm Rate (FAR)**, accuracy; for adversarial runs, extra **evasion counts** on perturbed attack samples. `comparison.py` prints aligned columns for **AAE+RF**, **BiGAN**, and **Robust VAE-BiGAN**.

---

## Repository layout (important paths)

| Path | Purpose |
|------|---------|
| `data/iot23_combined_new.csv` | Source dataset (train/fit generators). |
| `test_data.npz` | Held-out `X_test` / `y_test` for evaluation scripts. |
| `saved_state/` | Scalers, feature columns, label maps, robust checkpoints. |
| `gateway/` | GMM, cGAN/TVAE loading, `Gateway` orchestration. |
| `models/` | `aae_model.py`, `bigan_model.py`, `robust_model.py`, … |
| `utils/` | Preprocessing, model loading registry. |

**Pretrained artifacts** (after training or as shipped) typically include: `aae_final.pth`, `encoder_final.pth`, `aae_classifier.pkl`, `bigan_final.pth`, `bigan_calibration.pkl`, `saved_state/robust_vae_bigan_model.pth`, `robust_calibration.pkl`, `test_data.npz`, and files under `gateway/weights/`.

---

## Installation

From the **`final_capstone`** directory (the folder that contains `run_pipeline.py`):

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

Dependencies include **PyTorch**, **NumPy**, **pandas**, **scikit-learn**, **imbalanced-learn**, and **joblib**. If you use `evaluate_robust.py` plotting, ensure **matplotlib** is available (install if missing).

---

## Data: where it comes from, what you need

- **Training / GMM & GANs**: point scripts at `data/iot23_combined_new.csv` (or your IoT-23–compatible export with the same feature pipeline).
- **Quick evaluation**: `test_data.npz` must exist in the project root (same directory as `comparison.py`). It is produced by the training/evaluation split logic in your train scripts (see training order below).

If you only run comparisons with **shipped** `test_data.npz` and checkpoints, you do not need to re-download IoT-23 as long as paths match.

---

## Training order (full reproducibility)

Run from **`final_capstone`**:

```bash
python train_aae.py
python train_aae_classifier.py
python train_bigan.py
python train_robust_ids.py
python fit_gmm.py --data data/iot23_combined_new.csv --rows 300000
python -m gateway.trainer --data data/iot23_combined_new.csv --epochs 150 --batch 256 --latent 64
python -m gateway.train_tvae --data data/iot23_combined_new.csv --rows 300000
```

Optional robust recalibration:

```bash
python calibrate_robust.py
```


---

## What to run for conclusions

### Main three-way benchmark (recommended)

```bash
python comparison.py
```

**Console output** includes:

1. **Baseline** – metrics on real `test_data.npz` for all three models.  
2. **Stream mode** – DR / FAR / accuracy-style summaries for varying `n`.  
3. **Adversarial FGSM (ε = 0.15)** – for stress attacks such as DDoS, Okiru, PartOfAHorizontalPortScan; **Robust VAE-BiGAN** typically shows the **best resistance** (highest DR / lowest evasion among the three).

This is the script that directly supports the claim that the **robust model performs best among the comparing models** under adversarial stress.

### Per-model evaluation scripts

```bash
python evaluate.py
python evaluate_bigan.py
python evaluate_robust.py
```


### Interactive pipeline (single model, all Gateway modes)

```bash
python run_pipeline.py [OPTIONS]
```

| Option | Values / notes |
|--------|----------------|
| `--mode` | `stream` · `attack` · `adversarial` (default `stream`) |
| `--model` | `bigan` · `aae` · `robust` (default `bigan`) |
| `--generator` | `cgan` · `tvae` · `none` |
| `--n` | Sample count (default `100`) |
| `--attack` | Attack class name (defaults e.g. to DDoS for attack/adversarial modes) |
| `--epsilon` | FGSM strength; default **0.05** for stream/attack, **0.15** for adversarial if omitted |
| `--benign-ratio` | Stream mode benign fraction (default `0.70`) |
| `--gan-blend` | 0 = GMM-only, higher = more cGAN/TVAE influence |
| `--export-csv` | Path to save features + labels + predictions |
| `--save` | Path to save per-sample **JSON** results |

Examples:

```bash
python run_pipeline.py --mode stream --model robust --generator none --n 300
python run_pipeline.py --mode adversarial --model robust --generator cgan --attack PortScan --epsilon 0.15 --gan-blend 0.1 --n 200
python run_pipeline.py --mode stream --model bigan --generator cgan --gan-blend 0.1 --n 500 --export-csv out.csv --save out.json
```

### Generator ablation (GMM vs TVAE vs cGAN)

```bash
python compare_generators.py
```

Default CSV output: **`generator_comparison_results.csv`** (matrix over generators, blend values, and modes). Use `--csv` to change path.

---

## Summary

This project is a **comparative IoT IDS study** with a **Gateway-driven evaluation harness** and explicit **adversarial (FGSM) testing**. Train or restore checkpoints, then run **`python comparison.py`** to see all three models on the same scenarios; the **Robust VAE-BiGAN** is the intended **strongest under adversarial attack** among the three compared detectors. For single-model deep dives and exportable artifacts, use **`run_pipeline.py`** and the **`evaluate_*.py`** scripts as described above.

---

## Dataset - https://www.kaggle.com/datasets/engraqeel/iot23preprocesseddata

## How to get full code 
In order to get full code for our robust-model which is in a private repo you can mail us at juhisahni07@gmail.com or shefalibishnoisb@gmail.com to get the access.
