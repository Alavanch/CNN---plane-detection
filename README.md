# CNN — plane detection

A small convolutional neural network that classifies 20×20 RGB satellite image
chips as **plane** / **no plane**, trained on the
[PlanesNet dataset](https://www.kaggle.com/datasets/rhammell/planesnet)
(32,000 chips: 8,000 planes, 24,000 no-planes, extracted from Planet imagery).

## Setup

```
pip install -r requirements.txt
```

Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/rhammell/planesnet) and extract it so
that the `.png` files end up in a `planesnet/` folder next to `CNN.py`
(the script also accepts the archive's nested `planesnet/planesnet/` layout).
Filenames encode the label: `1__...` = plane, `0__...` = no plane.

## Usage

```
python CNN.py
```

Options: `--data-dir` (default `planesnet`), `--epochs` (default 30, with early
stopping), `--batch-size` (default 32), `--out-dir` (default `outputs`),
`--no-show` (save the figures without opening windows).

## Model

A deliberately small CNN (~100k parameters):

```
Conv2D(32, 3x3) → Conv2D(64, 3x3) → MaxPooling2D
→ Conv2D(128, 3x3) → GlobalAveragePooling2D
→ Dense(64) → Dropout(0.3) → Dense(1, sigmoid)
```

Training details:

- stratified 72/8/20 train/validation/test split (the dataset is imbalanced
  1:3, so accuracy must be read against the 75% majority-class baseline that
  the script prints);
- balanced class weights so the rare *plane* class is not under-served;
- early stopping on validation loss (best weights restored);
- seeded runs (`tf.keras.utils.set_random_seed`) for comparability.

## Outputs

Everything is written to `outputs/`:

- `plane_cnn.keras` — the trained model;
- `metrics.json` — test accuracy/precision/recall + per-class report;
- `samples.png`, `training_curves.png`, `correct_predictions.png`,
  `misclassified.png`, `confusion_matrix.png`.

The console also prints a full `classification_report` (precision, recall, F1
per class).
