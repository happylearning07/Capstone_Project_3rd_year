# IoT IDS Pipeline

Local, script-based IDS workflow for IoT-23 style data.

## Project Scope

- IDS models:
  - `bigan` (binary anomaly detector)
  - `aae` (multi-class via AAE encoder + RF)
  - `robust` (robust VAE-BiGAN binary detector)
- Generators:
  - `none` (pure GMM)
  - `cgan` (GMM + cGAN blend)
  - `tvae` (GMM + TVAE blend)
- Main run modes:
  - `stream`
  - `attack`
  - `adversarial`

## Current Files (important)

Top-level scripts:

- `run_pipeline.py`
- `compare_generators.py`
- `comparison.py`
- `evaluate.py`
- `evaluate_bigan.py`
- `evaluate_robust.py`
- `train_aae.py`
- `train_aae_classifier.py`
- `train_bigan.py`
- `train_robust_ids.py`
- `calibrate_robust.py`
- `fit_gmm.py`

Gateway modules:

- `gateway/gateway.py`
- `gateway/gmm_generator.py`
- `gateway/generator_model.py`
- `gateway/trainer.py` (cGAN training)
- `gateway/train_tvae.py`
- `gateway/load_tvae.py`
- `gateway/tvae_model.py`
- `gateway/utils.py`

Core model/util modules:

- `models/aae_model.py`
- `models/bigan_model.py`
- `models/robust_model.py`
- `utils/preprocessing.py`
- `utils/model_loader.py`
- `utils/feature_engineering.py`

## Install

```bash
pip install -r requirements.txt
```

## Typical Training Order

Run from repository root:

```bash
python train_aae.py
python train_aae_classifier.py
python train_bigan.py
python train_robust_ids.py
python fit_gmm.py --data data/iot23_combined_new.csv --rows 300000
python -m gateway.trainer --data data/iot23_combined_new.csv --epochs 150 --batch 256 --latent 64
python -m gateway.train_tvae --data data/iot23_combined_new.csv --rows 300000
```

Optional robust-only recalibration:

```bash
python calibrate_robust.py
```

## Quick Evaluation

```bash
python evaluate.py
python evaluate_bigan.py
python evaluate_robust.py
python comparison.py
```

## `run_pipeline.py` CLI

```bash
python run_pipeline.py [OPTIONS]
```

Options:

- `--mode`: `stream | attack | adversarial` (default: `stream`)
- `--model`: `bigan | aae | robust` (default: `bigan`)
- `--generator`: `cgan | tvae | none` (default: `cgan`)
- `--n`: sample count (default: `100`)
- `--attack`: attack class for `attack`/`adversarial`
- `--stage`: `early | full` (default: `full`)
- `--epsilon`: FGSM epsilon (auto-selected when omitted)
- `--benign-ratio`: stream-mode benign ratio (default: `0.70`)
- `--gan-blend`: blend alpha between GMM and learned generator
- `--quiet`: hide sample-level rows
- `--export-csv`: save generated samples + predictions CSV
- `--save`: save result JSON

Examples:

```bash
python run_pipeline.py --mode stream --model bigan --generator none --n 300
python run_pipeline.py --mode attack --model aae --generator tvae --attack DDoS --gan-blend 0.1 --n 200
python run_pipeline.py --mode adversarial --model robust --generator cgan --attack PortScan --epsilon 0.15 --gan-blend 0.1 --n 200
python run_pipeline.py --mode stream --model bigan --generator cgan --gan-blend 0.1 --n 500 --export-csv out.csv --save out.json
```

Blend reference:

- `--gan-blend 0.0`: pure GMM behavior
- `--gan-blend 0.1`: mostly GMM + small learned refinement
- `--gan-blend 1.0`: pure learned generator output

## `compare_generators.py` CLI

```bash
python compare_generators.py [OPTIONS]
```

Options:

- `--attack`: optional attack class focus
- `--n`: sample budget (default: `500`)
- `--no-detection`: skip detection section
- `--skip-matrix`: skip full matrix section
- `--csv`: output path (default: `generator_comparison_results.csv`)

Matrix dimensions:

- generators: `none`, `tvae`, `cgan`
- blends: `0.0` to `1.0` in steps of `0.1`
- modes: `stream`, `attack`, `adversarial`

## Training Script CLIs

### `fit_gmm.py`

```bash
python fit_gmm.py --data data/iot23_combined_new.csv --rows 300000
```

### `gateway/trainer.py` (cGAN)

```bash
python -m gateway.trainer --data data/iot23_combined_new.csv --model cgan --epochs 300 --rows 300000 --batch 256 --latent 64 --device cpu
```

### `gateway/train_tvae.py`

```bash
python -m gateway.train_tvae --data data/iot23_combined_new.csv --rows 300000 --epochs 100 --batch 512 --latent 32 --embed 32 --lr 1e-3 --beta-max 0.1 --device cpu --log-every 5
```

## Artifacts You Should Expect

- `saved_state/`
  - `scaler.pkl`
  - `feature_cols.pkl`
  - `reverse_label_map.pkl`
  - robust artifacts (from robust training flow)
- root artifacts
  - `aae_final.pth`, `encoder_final.pth`, `aae_classifier.pkl`
  - `bigan_final.pth`, `bigan_calibration.pkl`
  - `robust_calibration.pkl`
  - `test_data.npz`
- `gateway/weights/`
  - `gmm_per_class.pkl`
  - `cgan_weights.pth` and GAN metadata pkls
  - `tvae_weights.pth` and TVAE metadata pkls

## Troubleshooting

- Missing generator files:
  - run `fit_gmm.py`
  - run `gateway.trainer` for cGAN
  - run `gateway.train_tvae` for TVAE
- Feature/state mismatch:
  - re-run `train_aae.py` first to regenerate canonical preprocessing artifacts
- For baseline reproducibility:
  - use `--generator none --gan-blend 0.0`
