"""Plane detection on the PlanesNet dataset with a small CNN.

Dataset: https://www.kaggle.com/datasets/rhammell/planesnet (32,000 RGB chips
of 20x20 pixels; 8,000 planes / 24,000 no-planes). Filenames encode the label:
"1__..." = plane, "0__..." = no plane.

Usage:
    python train.py                   # expects the images in ./planesnet
    python train.py --data-dir path/to/planesnet --epochs 30 --no-show
"""

import argparse
import json
import os
import sys

import numpy as np

import figures

SEED = 42
IMG_SIZE = 20
CLASS_NAMES = ["no plane", "plane"]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a plane/no-plane CNN on PlanesNet.")
    parser.add_argument("--data-dir", default="planesnet",
                        help="Folder containing the PlanesNet .png files (default: planesnet)")
    parser.add_argument("--epochs", type=int, default=60,
                        help="Maximum number of epochs; early stopping usually ends sooner (default: 60)")
    parser.add_argument("--batch-size", type=int, default=32, help="Training batch size (default: 32)")
    parser.add_argument("--out-dir", default="outputs",
                        help="Folder for the saved model and figures (default: outputs)")
    parser.add_argument("--no-show", action="store_true",
                        help="Only save the figures, do not open matplotlib windows")
    return parser.parse_args()


def set_seeds(seed):
    """Seed python, numpy and tensorflow so runs are comparable."""
    from tensorflow.keras.utils import set_random_seed
    set_random_seed(seed)  # seeds python random, numpy and tensorflow at once


def load_dataset(data_dir):
    """Load every PlanesNet chip as a (N, 20, 20, 3) float array in [0, 1]."""
    from tensorflow.keras.utils import img_to_array, load_img

    if not os.path.isdir(data_dir):
        sys.exit(
            "Data folder '%s' not found.\n"
            "Download the dataset from https://www.kaggle.com/datasets/rhammell/planesnet\n"
            "and extract the .png files into that folder (or pass --data-dir)." % data_dir
        )

    # The Kaggle archive extracts as planesnet/planesnet: descend if needed.
    nested = os.path.join(data_dir, "planesnet")
    if os.path.isdir(nested) and not any(
        f.lower().endswith(".png") for f in os.listdir(data_dir)
    ):
        data_dir = nested

    images = []
    labels = []
    skipped = 0
    for fname in sorted(os.listdir(data_dir)):
        if not fname.lower().endswith(".png"):
            continue
        # Filenames look like "1__20140723_181317_0905__-122.143_37.697.png";
        # the part before the first "__" is the label.
        prefix = fname.split("__", 1)[0]
        if prefix not in ("0", "1"):
            skipped += 1
            continue
        img = load_img(os.path.join(data_dir, fname), target_size=(IMG_SIZE, IMG_SIZE))
        images.append(img_to_array(img) / 255.0)
        labels.append(int(prefix))

    if skipped:
        print("Warning: skipped %d .png files without a 0/1 label prefix" % skipped)
    if not images:
        sys.exit("No labelled .png files found in '%s'." % data_dir)

    images = np.array(images, dtype=np.float32)  # (N, 20, 20, 3)
    labels = np.array(labels)                    # (N,) with values 0/1
    print("Loaded %d images (%d planes, %d no-planes)"
          % (len(labels), int(labels.sum()), int((labels == 0).sum())))
    return images, labels


def build_model():
    """Small CNN: ~100k parameters instead of the ~3.2M a Flatten head would give."""
    from tensorflow.keras.layers import (Conv2D, Dense, Dropout, GlobalAveragePooling2D,
                                         Input, MaxPooling2D, RandomFlip)
    from tensorflow.keras.metrics import AUC, Precision, Recall
    from tensorflow.keras.models import Sequential

    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        # Satellite chips have no preferred orientation: free augmentation
        # (active during training only).
        RandomFlip("horizontal_and_vertical", seed=SEED),
        Conv2D(32, (3, 3), activation="relu"),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        GlobalAveragePooling2D(),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy", Precision(name="precision"), Recall(name="recall"),
                 AUC(name="auc")],
    )
    return model


def main():
    args = parse_args()
    set_seeds(SEED)

    import matplotlib.pyplot as plt
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split
    from sklearn.utils.class_weight import compute_class_weight
    from tensorflow.keras.callbacks import EarlyStopping

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(SEED)

    images, labels = load_dataset(args.data_dir)

    # Three-way split, stratified so the 1:3 plane/no-plane ratio is preserved:
    # 20% test, then 10% of the remainder as validation (72/8/20 overall).
    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=0.2, random_state=SEED, stratify=labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train
    )
    print("Split: %d train / %d val / %d test" % (len(y_train), len(y_val), len(y_test)))

    # The dataset is imbalanced (3 no-planes per plane): weight the rare class up
    # so the model is not rewarded for ignoring planes.
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
    class_weight = {0: weights[0], 1: weights[1]}
    print("Class weights:", class_weight)

    model = build_model()
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        class_weight=class_weight,
        # val_loss is noisy under class weights + augmentation and stops far
        # too early; val_auc climbs smoothly and is threshold-independent.
        callbacks=[EarlyStopping(monitor="val_auc", mode="max", patience=5,
                                 restore_best_weights=True)],
    )

    # Final evaluation on the held-out test set. Accuracy alone is misleading
    # here: always answering "no plane" already scores the majority share.
    baseline = max(1 - y_test.mean(), y_test.mean())
    test_metrics = model.evaluate(X_test, y_test, return_dict=True)
    print("Majority-class baseline accuracy: %.4f" % baseline)
    print("Test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})

    probs = model.predict(X_test).ravel()
    pred_classes = (probs >= 0.5).astype(int)
    print(classification_report(y_test, pred_classes, target_names=CLASS_NAMES,
                                zero_division=0))

    model_path = os.path.join(args.out_dir, "plane_cnn.keras")
    model.save(model_path)
    print("Saved model to", model_path)

    metrics_path = os.path.join(args.out_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "majority_baseline_accuracy": float(baseline),
            "test": {k: float(v) for k, v in test_metrics.items()},
            "report": classification_report(y_test, pred_classes, target_names=CLASS_NAMES,
                                            zero_division=0, output_dict=True),
            "epochs_run": len(history.history["loss"]),
        }, f, indent=2)
    print("Saved metrics to", metrics_path)

    history_path = os.path.join(args.out_dir, "history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.history.items()},
                  f, indent=2)
    print("Saved per-epoch history to", history_path)

    # --- Figures (all saved to out_dir, shown once at the end) ---

    # 20 random chips from the dataset.
    idxs = rng.choice(len(images), min(20, len(images)), replace=False)
    figures.save_chip_grid(
        [images[i] for i in idxs],
        [CLASS_NAMES[labels[i]] for i in idxs],
        [figures.GREEN if labels[i] else figures.DARK for i in idxs],
        os.path.join(args.out_dir, "samples.png"))

    figures.save_training_curves(history.history,
                                 os.path.join(args.out_dir, "training_curves.png"))

    # Random correct and incorrect test predictions.
    good = np.where(pred_classes == y_test)[0]
    bad = np.where(pred_classes != y_test)[0]
    if len(good):
        picks = rng.choice(good, min(20, len(good)), replace=False)
        figures.save_chip_grid(
            [X_test[i] for i in picks],
            [CLASS_NAMES[y_test[i]] for i in picks],
            [figures.GREEN if y_test[i] else figures.DARK for i in picks],
            os.path.join(args.out_dir, "correct_predictions.png"))
    if len(bad):
        picks = rng.choice(bad, min(20, len(bad)), replace=False)
        figures.save_chip_grid(
            [X_test[i] for i in picks],
            ["T:%s / P:%s" % (CLASS_NAMES[y_test[i]], CLASS_NAMES[pred_classes[i]])
             for i in picks],
            [figures.DARK] * len(picks),
            os.path.join(args.out_dir, "misclassified.png"))
    else:
        print("No misclassified test images.")

    cm = confusion_matrix(y_test, pred_classes)
    figures.save_confusion_matrix(cm, CLASS_NAMES,
                                  os.path.join(args.out_dir, "confusion_matrix.png"))

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
