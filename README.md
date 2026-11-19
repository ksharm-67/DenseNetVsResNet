# DenseNet121 vs ResNet50 on ChestX-ray14

A controlled comparison of two ImageNet-pretrained backbones for multilabel classification of 14 thoracic diseases (plus No Finding) on the NIH ChestX-ray14 dataset. Everything is held fixed except the architecture: same patient-level split, same preprocessing, same hyperparameters, three seeds each.

## Results

Test AUC, mean ± standard deviation over 3 seeds. The mean row averages the 14 disease labels (No Finding excluded).

| Label | ResNet50 | DenseNet121 | Difference |
| --- | --- | --- | --- |
| Atelectasis | 0.7672 ± 0.0019 | 0.7679 ± 0.0061 | +0.0007 |
| Cardiomegaly | 0.8593 ± 0.0110 | 0.8657 ± 0.0097 | +0.0064 |
| Consolidation | 0.7358 ± 0.0036 | 0.7440 ± 0.0094 | +0.0082 |
| Edema | 0.8349 ± 0.0017 | 0.8403 ± 0.0100 | +0.0054 |
| Effusion | 0.8223 ± 0.0032 | 0.8259 ± 0.0009 | +0.0037 |
| Emphysema | 0.8989 ± 0.0077 | 0.9122 ± 0.0054 | +0.0133 |
| Fibrosis | 0.8120 ± 0.0075 | 0.8185 ± 0.0125 | +0.0065 |
| Hernia | 0.8335 ± 0.0069 | 0.8805 ± 0.0124 | +0.0471 |
| Infiltration | 0.6858 ± 0.0042 | 0.6882 ± 0.0121 | +0.0024 |
| Mass | 0.8080 ± 0.0056 | 0.8167 ± 0.0079 | +0.0087 |
| Nodule | 0.7426 ± 0.0080 | 0.7438 ± 0.0011 | +0.0012 |
| Pleural Thickening | 0.7592 ± 0.0029 | 0.7739 ± 0.0030 | +0.0147 |
| Pneumonia | 0.7127 ± 0.0101 | 0.7195 ± 0.0116 | +0.0068 |
| Pneumothorax | 0.8530 ± 0.0090 | 0.8614 ± 0.0025 | +0.0084 |
| **Mean, 14 diseases** | **0.7946 ± 0.0019** | **0.8042 ± 0.0019** | **+0.0095** |
| No Finding (not in mean) | 0.7251 ± 0.0015 | 0.7272 ± 0.0041 | +0.0021 |

- DenseNet121 is ahead by 0.0095 mean AUC (95% CI 0.0052 to 0.0139, Welch t-test p = 0.004). Every DenseNet121 run beat every ResNet50 run.
- DenseNet121 has the higher mean AUC on all 14 disease labels. Most single-label gaps are within seed variation; Hernia and Pleural Thickening are the clearest.
- Both models peak on validation at epoch 4 to 5 and overfit afterward.

| | ResNet50 | DenseNet121 |
| --- | --- | --- |
| Parameters | 23.5 M | 7.0 M |
| Peak GPU memory, batch 64 | 3.28 GB | 4.43 GB |
| Time per epoch (T4) | 326 s | 385 s |
| Best epoch by validation AUC, per seed | 4, 4, 4 | 5, 4, 5 |
| Best validation mean AUC | 0.8248 ± 0.0044 | 0.8311 ± 0.0042 |

 The per-run `results.json` files are in `results/`.

## Setup

- Data: ChestX-ray14 (112,120 frontal images, 15 labels). The folder you point `--data_root` at needs `Data_Entry_2017.csv`, `train_val_list.txt`, `test_list.txt` and `images_*/images/*.png`.
- Split: the official test list is the test set. The official train/val list is split by patient ID (10% of patients for validation), giving 77,988 train, 8,536 validation and 25,596 test images. The split seed is fixed at 42.
- Preprocessing: grayscale, resize to 224 × 224, three channels, ImageNet normalization. Random horizontal flip on training images only.
- Models: torchvision ResNet50 and DenseNet121 with ImageNet-1K V1 weights (V1 for both, since the default ResNet50 weights are a newer recipe). Final layer replaced with a 15-output linear layer, all layers fine-tuned.
- Training: BCE with logits, Adam, lr 1e-4, batch size 64, mixed precision, 10 epochs, no schedule. The checkpoint with the best validation mean AUC is used for the test set.

## Usage

```
pip install torch torchvision pandas scikit-learn Pillow matplotlib scipy kagglehub
```

Download the data:

```python
import kagglehub
path = kagglehub.dataset_download("nih-chest-xrays/data")
```

Cache the images once at 224 × 224 (about 15 to 30 minutes, makes training several times faster):

```
python cache.py --data_root <data folder> --out_dir <cache folder>
```

Train, one run per architecture and seed:

```
python train.py --arch resnet50 --seed 0 --epochs 10 --cache_dir <cache folder> --data_root <data folder> --out_dir runs
python train.py --arch densenet121 --seed 0 --epochs 10 --cache_dir <cache folder> --data_root <data folder> --out_dir runs
```

Each run writes `best.pth`, `results.json`, `roc.png` and `test_preds.npz` to `runs/<arch>_seed<N>/`.

Aggregate all runs into a mean ± std table and a per-label bar chart:

```
python compare.py runs
```

Flags for `train.py`: `--arch {resnet50,densenet121}`, `--seed`, `--epochs`, `--lr`, `--batch_size`, `--cache_dir` (omit to read the original PNGs), `--subset N` (quick smoke test on N images per split).

On Kaggle, the data lives under `/kaggle/input/...` and `--out_dir` defaults to `/kaggle/working/runs`. Long runs are safest as a Save & Run All commit.

## Files

- `dataset.py`: labels, patient-level split, dataset class
- `model.py`: builds either backbone with a 15-class head
- `train.py`: training, validation, checkpointing, test evaluation, ROC plot
- `cache.py`: one-time image resize to a 224 × 224 array
- `compare.py`: aggregates runs into tables and a chart
