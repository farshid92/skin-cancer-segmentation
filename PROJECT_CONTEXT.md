PROJECT CONTEXT — Skin Cancer Segmentation: Evolutionary Ensemble + MLOps
0. HOW TO USE THIS FILE
This is the single source of truth for this repo. Read it fully before generating any code.If a user request conflicts with this file, follow this file and state why.Do not propose changes to structure/stack/deps unless explicitly asked.

1. GOAL
Build a research-grade dermoscopic lesion segmentation system that:

Uses evolutionary algorithms (NSGA-II, Differential Evolution, PSO) to optimizearchitectures, losses, and ensemble weights.
Ensembles heterogeneous segmentation networks (CNN + Transformer) with TTA anduncertainty estimation.
Ships full MLOps: MLflow tracking/registry, FastAPI + ONNX serving, Docker,GitHub Actions CI, Prometheus/Grafana monitoring.Deliverable = reproducible paper-ready experiments AND a deployable API service.
2. HARD RULES (always obey)
Python 3.10+. Type hints on ALL public functions. Google-style docstrings, 1-line summary first.
No hardcoded paths or hyperparameters. Everything comes from configs/*.yaml via OmegaConf.
Images: float32 [B,3,H,W] ImageNet-normalized. Masks: float32 [B,1,H,W] in {0,1}. Never silently mix layouts.
All training runs MUST log to MLflow (params, metrics per epoch, artifacts). No bare print() — use src/utils/logging.py.
Determinism: every entry point calls src.utils.seed.set_seed(cfg.seed).
Prefer existing deps. Add a new dependency ONLY if unavoidable; add it to pyproject.toml and flag it.
Functions ≤ 50 lines. No logic in notebooks — notebooks are for EDA/figures only.
Every new module gets a pytest test file in tests/.
Use pathlib.Path. Never os.path string concatenation.
Losses and metrics must be numerically safe (clamp logits, avoid log(0), handle empty masks → DSC=1.0 if both empty, 0.0 if pred-only).
3. DIRECTORY STRUCTURE (authoritative)
skin-cancer-segmentation/├── data/│ ├── raw//images/ + masks/ # ISIC2016/17/18, HAM10000, PH2, BCN20000│ ├── processed/ # cleaned, resized, hair-removed│ ├── masks/ # aligned binary masks│ └── synthetic/ # GAN augmentation samples (optional, phase 2)├── src/│ ├── data/│ │ ├── dataset.py # LesionDataset, build_loaders()│ │ ├── preprocessing.py # hair removal (DullRazor), CLAHE, normalize│ │ ├── augmentation.py # Albumentations pipelines (levels 0/1/2)│ │ └── synthetic.py # GAN-based generation (optional)│ ├── models/│ │ ├── unetpp.py # SMP UnetPlusPlus│ │ ├── deeplabv3plus.py # SMP DeepLabV3Plus│ │ ├── manet.py # SMP MAnet│ │ ├── transunet.py # custom hybrid CNN-ViT (build on timm/torch)│ │ ├── swinunet.py # custom pure-transformer U-Net│ │ └── factory.py # build_model(cfg) -> nn.Module (single entry)│ ├── losses/│ │ ├── combined.py # BCE + Dice + Tversky, weighted, configurable│ │ ├── boundary.py # boundary loss on distance maps (scipy EDT)│ │ └── lovasz.py # Lovász-Softmax│ ├── evolution/│ │ ├── chromosome.py # gene <-> pymoo variable encoding/decoding│ │ ├── nsga2.py # NSGA-II (pymoo Problem)│ │ ├── differential_evo.py # DE/rand/1/bin│ │ ├── pso.py # PSO for ensemble weights│ │ └── surrogate.py # XGBoost fitness proxy + eval cache (parquet)│ ├── ensemble/│ │ ├── weighted_avg.py # PSO weights, logit-space averaging│ │ ├── stacking.py # per-pixel 1x1-conv meta-learner on stacked logits│ │ ├── tta.py # 8 geometric transforms, logit-space average│ │ └── uncertainty.py # MC Dropout (20 passes) + temperature scaling│ ├── training/│ │ ├── trainer.py # Trainer class: AMP, clip 1.0, ckpt best Dice│ │ ├── callbacks.py # early stopping (patience), checkpointing│ │ └── scheduler.py # cosine with 5-epoch linear warmup│ ├── evaluation/│ │ ├── metrics.py # DSC, IoU, Sens, Spec, HD95, AUC, ECE│ │ ├── ablation.py # ablation runner (loops configs, 3 seeds)│ │ └── calibration.py # temperature scaling fit on val│ ├── serving/│ │ ├── main.py # FastAPI app + Prometheus instrumentator│ │ ├── schemas.py # Pydantic request/response models│ │ └── inference.py # ONNX Runtime sessions, ensemble predict│ ├── monitoring/│ │ └── metrics.py # custom Prometheus counters/histograms│ └── utils/│ ├── seed.py, logging.py, device.py, config.py, io.py├── configs/│ ├── base.yaml # seed, image_size, paths, split, training│ ├── model_zoo.yaml # per-architecture defaults│ └── ea_config.yaml # EA population/generations/objectives├── scripts/│ ├── download_data.sh # ISIC API / figshare / zenodo downloads│ ├── run_eda.py, train.py, run_ea.py, train_ensemble.py, evaluate.py, export_onnx.py├── experiments/ # MLflow artifacts, eval_cache.parquet├── notebooks/ # EDA + paper figures only├── tests/ # pytest, one file per src module├── frontend/app.py # Gradio demo -> calls API├── monitoring/ # prometheus.yml, alert_rules.yml, grafana/├── .github/workflows/ci.yml # lint + test + docker build on PR├── Dockerfile, docker-compose.yml, pyproject.toml, README.md

4. TECH STACK (do not substitute)
Layer	Tool
DL framework	PyTorch ≥ 2.1 (AMP fp16, torch.compile optional)
Seg models	segmentation-models-pytorch (SMP): UnetPlusPlus, DeepLabV3Plus, MAnet
Medical utils	MONAI (metrics: DiceMetric, HausdorffDistanceMetric(pct=95); transforms where useful)
EA	pymoo (NSGA-II, DE); custom PSO in numpy
Surrogate	XGBoost + scikit-learn
Augmentation	Albumentations (mask interp = nearest)
Tracking	MLflow ≥ 2.9 (tracking + registry), backend = Postgres in compose
Serving	FastAPI + uvicorn + ONNX Runtime (CUDA provider → CPU fallback)
Infra	Docker, docker-compose, GitHub Actions
Monitoring	prometheus-client, prometheus-fastapi-instrumentator, Grafana
Demo	Gradio
Config	OmegaConf + dataclasses (src/utils/config.py: load_config(path) -> Dataclass)
Quality	ruff, black, mypy, pytest (dev deps in pyproject)
5. RESEARCH SPEC
5.1 Data & Splits
Sources: ISIC 2016/2017/2018 Task 1, HAM10000 (10k), PH2 (200), BCN20000 (20k).
Unified file naming: {dataset}_{original_id}.png for image and mask.
Split: 70/10/20 train/val/test, stratified by dataset source. PH2 reserved asexternal cross-dataset test. Splits saved to data/processed/splits.json, never recomputed silently.
5.2 Preprocessing pipeline (fixed order)
Load image + mask → resize to cfg.image_size (default 256) bilinear/nearest.
Hair removal: grayscale → black-hat morphology → threshold → inpaint (cv2.inpaint).
CLAHE on LAB L-channel.
Normalize ImageNet mean/std (all encoders are ImageNet-pretrained).
Mask: binarize > 0.5, morphological open+close (3×3) to remove specks.
5.3 Augmentation levels (Albumentations)
L0: HorizontalFlip, VerticalFlip, Rotate(90)
L1: + ShiftScaleRotate(0.1), RandomBrightnessContrast(0.2), HueSaturationValue
L2: + ElasticTransform, CoarseDropout, GridDistortion
5.4 Architectures (5 candidates, one per file in src/models/)
Model	Encoder	Source
U-Net++	efficientnet-b4	SMP
DeepLabV3+	resnet101	SMP
MAnet	efficientnet-b4	SMP
TransUNet	resnet50 + 12-layer ViT on 16×16 patches	custom
SwinUNet	swin-tiny encoder/decoder	custom
All builders return nn.Module with forward(x[B,3,H,W]) -> logits[B,1,H,W] (NO sigmoid in forward).		
Register each in src/models/factory.py: build_model(cfg.model).		
SMP models: set activation=None, classes=1.		
5.5 Losses (all operate on logits + binary mask)
combined: w_bce*BCEWithLogits + w_dice*SoftDice + w_tversky*Tversky(α=0.7,β=0.3), defaults (0.5, 0.25, 0.25), weights configurable.
boundary: signed boundary loss on scipy distance-transform maps.
lovasz: Lovász-Softmax on sigmoid probabilities.
Total loss = configurable weighted sum of any subset → encoded in chromosome.
5.6 Evolutionary search (pymoo)
Chromosome genes (chromosome.py — implement encode/decode both directions):

Gene	Type	Range
decoder	choice	unetpp, deeplabv3plus, manet, transunet, swinunet
encoder	choice	efficientnet-b0/b4, resnet50, resnet101
decoder_channels_scale	int	16, 32, 64
loss_recipe	choice	combined / +boundary / +lovasz / all-three
loss_weights	real ×3	[0,1], normalized
lr	log-real	[1e-5, 1e-3]
image_size	int	224, 256, 320
augment_level	int	0, 1, 2
NSGA-II (nsga2.py): bi-objective — maximize val DSC, minimize param count (M).pop=20, gen=30, SBX(η=10), polynomial mutation(η=20), binary tournament, duplicate elimination.Differential Evolution (differential_evo.py): DE/rand/1/bin, F=0.5, CR=0.9 — single-objective DSC variant.PSO (pso.py): search K ensemble weights over val; 50 particles, 40 iters,inertia 0.7, c1=c2=1.5; weights → softmax → sum to 1.Surrogate (surrogate.py): XGBoost regressor, features = one-hot genes,target = val DSC; pre-screen each generation (top 30% proceed to full training).Cache every evaluated (gene_hash → DSC, params) row in experiments/eval_cache.parquet;skip re-training of duplicate genes (gene_hash = sha1 of sorted gene dict).

5.7 Ensemble (ensemble/)
weighted_avg: PSO weights, average in logit space.
stacking: input = concat of K model logits [B,K,H,W] → 1×1 conv → [B,1,H,W]; trained on val set only.
tta.py: 8 transforms (id, hflip, vflip, hv, rot90/180/270, transpose), average logits.
uncertainty.py: MC Dropout — 20 stochastic forward passes (dropout active),report mean prob + variance map; calibrate with temperature scaling fit on val (minimize ECE).
5.8 Evaluation (threshold 0.5, pixel-wise; report mean±std over seeds 0,1,2)
Metric	Definition / Tool
DSC	MONAI DiceMetric
IoU	2·DSC/(1+DSC) equivalently or MONAI MeanIoU
Sensitivity / Specificity	confusion counts
HD95	MONAI HausdorffDistanceMetric(percentile=95)
AUC-ROC	on probability maps, pixel-level (sklearn)
ECE	15-bin expected calibration error
Report protocol: in-distribution test set + external PH2 + per-dataset breakdown.	
5.9 Training defaults (overridable via configs)
image_size 256 · batch 16 · AdamW(lr 1e-4, wd 1e-4) · cosine schedule, 5-ep warmup ·epochs 100 · early stop patience 20 on val DSC · AMP fp16 · grad clip 1.0 · seed 42.

6. MLFLOW CONTRACT (must implement exactly)
Experiment name: skin-cancer-segmentation
Run name: {decoder}_{encoder}_{date}_{short_hash}
Params: all chromosome genes + dataset, split, seed, epochs, batch, optimizer.
Metrics per epoch: train_loss, val_loss, val_dsc, val_iou, val_hd95, lr (use step=epoch).
Final metrics: test_dsc, test_iou, test_hd95, test_auc, ece_pre, ece_post, params_M, latency_ms.
Artifacts: config.yaml, best_model.pt, sample_predictions/grid_epoch{N}.png (every 10 epochs), onnx/model.onnx (after export).
Registry: model name lesion-segmenter; stage Staging on new best val DSC, promote to Production only after external PH2 eval.
Trainer receives mlflow client via constructor — do NOT scatter mlflow.* calls through the codebase; only training/trainer.py and scripts/* touch MLflow.
7. SERVING API SPEC (src/serving/)
Method	Path	Body	Response
POST	/v1/predict	multipart image (png/jpg)	{mask_png_b64, overlay_png_b64, latency_ms, model_version}
POST	/v1/predict/batch	list of images	array of above + total_latency_ms
GET	/v1/health	—	{status, model_loaded, model_version, device}
GET	/metrics	—	Prometheus (via instrumentator)
Inference: ONNX Runtime sessions (one per ensemble member), CUDAExecutionProvider → CPU fallback.
Optional query param tta=bool (default false) and uncertainty=bool (adds variance map).
Ensemble = weighted average with Production-stage weights; weights stored in experiments/ensemble_weights.json.
API never loads PyTorch — ONNX only. Model files cached at startup, reload endpoint /v1/admin/reload (guarded by API key from env).
8. DOCKER & CI
docker-compose.yml services (all on network ml-net):

Service	Image/Build	Port	Notes
api	Dockerfile (python:3.10-slim)	8000	healthcheck on /v1/health
mlflow	mlflow image + Postgres backend	5000	artifacts volume
postgres	postgres:15-alpine	5432	MLflow backend store
prometheus	prom/prometheus	9090	mounts monitoring/prometheus.yml
grafana	grafana/grafana	3000	provisioned dashboards
demo	Dockerfile.demo (gradio)	7860	calls http://api:8000
.github/workflows/ci.yml (on PR to main):

ruff check + black --check + mypy src
pytest -q (coverage ≥ 70% gate)
docker build api image (no push)CPU-only CI — tests must run without GPU (device auto-detects).
9. TESTING REQUIREMENTS (tests/)
test_dataset.py: shapes [3,H,W]/[1,H,W], dtypes, mask binarity, transform alignment.
test_preprocessing.py: output range, idempotence of CLAHE, hair removal doesn't blank image.
test_losses.py: outputs finite, per-pixel range valid, empty-mask edge case (DSC=1 rule).
test_chromosome.py: encode→decode roundtrip identity; gene_hash stable.
test_pso.py: weights sum to 1, beats uniform-weight baseline on toy preds.
test_metrics.py: DSC/HD95 match MONAI reference on a 2-image fixture.
test_api.py: httpx AsyncClient → /v1/health 200, /v1/predict returns valid mask PNG.
test_calibration.py: temperature scaling reduces ECE on synthetic overconfident logits.
Fixtures: generate tiny random arrays/images in-memory; no dataset downloads in CI.
10. CONFIG SYSTEM
YAML files in configs/ merged in order: base.yaml ← model_zoo.yaml[select] ← ea_config.yaml ← CLI overrides.src/utils/config.py provides load_config(paths: list[str], overrides: list[str]) -> OmegaConf.Schema enforced by dataclasses in the same file. Unknown keys = error, never ignore.

Example — configs/ea_config.yaml keys:

algorithm: nsga2            # nsga2 | depopulation: 20generations: 30objectives: [val_dsc, params_M]   # max, minsurrogate: {enabled: true, keep_top_pct: 30, retrain_every_gen: 1}cache_path: experiments/eval_cache.parquet
11. COMMANDS
bash

bash scripts/download_data.sh --dataset ham10000     # also: isic2018, ph2, bcn20000
python scripts/train.py --config configs/base.yaml configs/model_zoo.yaml model=unetpp
python scripts/run_ea.py --config configs/base.yaml configs/ea_config.yaml
python scripts/train_ensemble.py --run_ids <id1> <id2> --method pso
python scripts/evaluate.py --run_id <id> --external ph2
python scripts/export_onnx.py --run_id <id> --opset 17
uvicorn src.serving.main:app --host 0.0.0.0 --port 8000
docker compose up -d
pytest -q
mlflow ui --backend-store-uri postgresql://mlflow:mlflow@localhost:5432/mlflow
12. STATUS (update as work progresses — Copilot: check before suggesting)
 data download + preprocessing
 dataset + augmentation
 5 model files + factory
 losses
 trainer + scheduler + callbacks
 metrics + calibration
 EA (chromosome → nsga2/de → pso → surrogate)
 ensemble (weighted/stacking/tta/uncertainty)
 ablation runner
 MLflow integration
 ONNX export + FastAPI serving
 Docker compose + monitoring
 CI + tests ≥70%
 README + paper figures