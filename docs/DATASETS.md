# Datasets

Download data only after completing the dataset-loader test milestone. The
current repository has reached that point, so the first download should be
**ISIC 2018 Challenge, Task 1: Lesion Boundary Segmentation**.

## First Dataset: ISIC 2018 Task 1

1. Create an account and accept the data terms at
   [ISIC Archive](https://www.isic-archive.com/).
2. Download the Task 1 training images and their ground-truth segmentation
   masks from the 2018 challenge page.
3. Preserve the original downloads under `data/raw/isic2018/`.
4. Do not commit downloaded data. The project `.gitignore` already excludes it.

After download, run the preparation command, adapting the two source folders to
match the extracted ISIC archive:

```powershell
python scripts/prepare_dataset.py `
  --images data/raw/isic2018/images `
  --masks data/raw/isic2018/masks `
  --output data/processed `
  --dataset isic2018 `
  --seed 42
```

The preprocessing command will later produce the canonical project layout:

```text
data/processed/
  images/isic2018_<original_id>.png
  masks/isic2018_<original_id>.png
  splits.json
```

`splits.json` contains records shaped as follows and is committed after it is
generated, so all experiments use the same split:

```json
{
  "train": [{"image": "images/isic2018_ISIC_0000000.png", "mask": "masks/isic2018_ISIC_0000000.png"}],
  "val": [],
  "test": []
}
```

## Download Roadmap

| When | Dataset | Purpose |
| --- | --- | --- |
| Now | ISIC 2018 Task 1 | Baseline training and development |
| After the baseline is reproducible | ISIC 2016 and ISIC 2017 Task 1 | Historical benchmark comparison |
| After multi-source ingestion is tested | HAM10000 | Image diversity and source-stratified training |
| Only after final-model selection | PH2 | Held-out external evaluation; never tune on it |
| Optional final expansion | BCN20000 | Scale and distribution-shift study |

Never download all sources at once. Establish the ISIC 2018 baseline, record
the result, then add one dataset at a time with a documented split and ablation.
