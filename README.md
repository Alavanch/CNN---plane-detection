# CNN plane detection

A small convolutional network that answers one question about a 20x20 satellite
image chip: plane, or no plane.

The training data is [PlanesNet](https://www.kaggle.com/datasets/rhammell/planesnet),
32,000 RGB chips cut from PlanetScope scenes over California at 3 m per pixel.
Only 8,000 chips contain a plane. The other 24,000 were picked to hurt: a third
are ordinary landcover (water, vegetation, bare earth, buildings), a third show
part of a plane without the body, and a third are "confusers", bright or
elongated objects the dataset author collected because detection models had
mislabeled them before. Each filename starts with its label, so
`1__20140723_181317_0905__-122.14_37.69.png` is a plane and a `0__` prefix is
not.

## Setup

```
pip install -r requirements.txt
```

I tested with Python 3.9 and TensorFlow 2.15 on Windows 11. The dataset is on
[Kaggle](https://www.kaggle.com/datasets/rhammell/planesnet), or as a 25 MB
`planesnet.7z` in the author's [repo](https://github.com/rhammell/planesnet).
Extract it next to `train.py`. A flat `planesnet/` folder works, and so does
the nested `planesnet/planesnet/` layout that some unzip tools produce.

## Training

```
python train.py
```

The script loads the 32,000 chips and splits them 72/8/20 into train,
validation, and test sets, stratified and seeded. Random flips augment the
chips. A plane viewed from overhead has no preferred orientation, and the
flips cost nothing. The learning rate halves after 3 epochs without a new
best validation AUC; training stops after 8 and the weights roll back to the
best epoch. I first stopped on validation loss instead, and it jumped around
so much under the augmentation that one run ended at epoch 5, badly underfit.
After training, the script scans decision thresholds on the validation set
and stores the most accurate one in `metrics.json`, where `predict.py` picks
it up.

Flags: `--data-dir`, `--epochs` (ceiling, default 60), `--batch-size`,
`--out-dir`, and `--no-show` to save the figures without opening windows.

The network is a small two-block design, 270k parameters:

```
RandomFlip -> [Conv2D(32) x2 -> MaxPooling2D] -> [Conv2D(64) x2 -> MaxPooling2D]
-> Flatten -> Dense(128) -> Dropout -> Dense(1)
```

The first version of this script had no pooling at all and flattened straight
into a Dense layer. That one matrix held 3.2 million parameters, 97% of the
model. A 101k-parameter rewrite with a GlobalAveragePooling head fixed the
bloat and plateaued near 97% accuracy. The current network sits in between
and beats both.

## Results

One number means little on an imbalanced dataset. Always answering "no plane"
already scores 75%.

On the 6,400 held-out test chips the network reaches 98.2% accuracy, with a
precision of 0.97 and a recall of 0.96 on the plane class (AUC 0.998). The
stored threshold, 0.72 on this run, gives the same test accuracy as plain
0.5; when validation cannot separate two thresholds, the script keeps the
lower one for its recall. The threshold stays useful as a trade knob. Lower
it with `predict.py --threshold` when a missed plane costs more than a false
alarm.

Training runs in a few minutes on a laptop CPU. The learning rate dropped
twice on the way; early stopping ended the run at epoch 21 and kept the
weights from epoch 13, where validation AUC peaked at 0.998.

![Accuracy, loss, and AUC per epoch, training vs validation](docs/img/training_curves.png)

<img src="docs/img/confusion_matrix.png" width="420" alt="Confusion matrix on the 6,400 test chips">


Everything a run produces lands in `outputs/`: the trained model
(`plane_cnn.keras`), the test metrics and per-class report in `metrics.json`,
the per-epoch numbers in `history.json`, the training curves, a confusion
matrix, and grids of sample, correct, and misclassified chips. The
misclassified grid is worth a look. The 114 test errors split almost evenly:
61 missed planes, half-cut or blended into the tarmac, against 53 false
alarms on plane-shaped white blobs.

![Misclassified test chips, true and predicted label above each](docs/img/misclassified.png)

## Predicting

```
python predict.py path/to/chip.png
python predict.py some_folder/
```

Loads `outputs/plane_cnn.keras` and prints a label and probability per file,
using the stored tuned threshold (override with `--threshold`).

## Data license

The PlanesNet imagery comes from Planet's Open California program and is
distributed under CC-BY-SA 4.0. The chips are not included in this repo.
