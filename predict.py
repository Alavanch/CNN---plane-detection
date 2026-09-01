"""Classify PNG chips with the model trained by train.py.

Usage:
    python predict.py image1.png image2.png ...
    python predict.py --model outputs/plane_cnn.keras some_folder/
"""

import argparse
import json
import os
import sys

import numpy as np

IMG_SIZE = 20


def collect_paths(inputs):
    """Expand folder arguments into their .png files."""
    paths = []
    for item in inputs:
        if os.path.isdir(item):
            paths.extend(os.path.join(item, f) for f in sorted(os.listdir(item))
                         if f.lower().endswith(".png"))
        else:
            paths.append(item)
    return paths


def main():
    parser = argparse.ArgumentParser(description="Plane/no-plane prediction on 20x20 chips.")
    parser.add_argument("inputs", nargs="+", help="PNG files or folders of PNG files")
    parser.add_argument("--model", default=os.path.join("outputs", "plane_cnn.keras"),
                        help="Path to the trained model (default: outputs/plane_cnn.keras)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold; defaults to the tuned value stored in "
                             "metrics.json next to the model, or 0.5")
    parser.add_argument("--tta", action="store_true",
                        help="Average the prediction over the 8 rotations/flips of "
                             "each chip (slower, slightly more accurate)")
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit("Model '%s' not found. Train one first with: python train.py" % args.model)

    threshold = args.threshold
    if threshold is None:
        metrics_path = os.path.join(os.path.dirname(args.model), "metrics.json")
        try:
            with open(metrics_path) as f:
                threshold = float(json.load(f)["tuned_threshold"])
        except (OSError, KeyError, ValueError):
            threshold = 0.5
    print("Decision threshold: %.3f" % threshold)

    paths = collect_paths(args.inputs)
    if not paths:
        sys.exit("No .png files found in the given inputs.")

    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import img_to_array, load_img

    model = load_model(args.model)

    batch = np.stack([
        img_to_array(load_img(p, target_size=(IMG_SIZE, IMG_SIZE))) / 255.0
        for p in paths
    ]).astype(np.float32)
    if args.tta:
        probs = np.zeros(len(batch), dtype=np.float64)
        for k in range(4):
            rot = np.rot90(batch, k, axes=(1, 2))
            probs += model.predict(rot, verbose=0).ravel()
            probs += model.predict(rot[:, :, ::-1, :], verbose=0).ravel()
        probs /= 8.0
    else:
        probs = model.predict(batch, verbose=0).ravel()

    for path, prob in zip(paths, probs):
        label = "plane" if prob >= threshold else "no plane"
        print("%-60s %s (p=%.3f)" % (path, label, prob))


if __name__ == "__main__":
    main()
