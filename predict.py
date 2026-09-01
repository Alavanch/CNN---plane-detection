"""Classify PNG chips with the model trained by train.py.

Usage:
    python predict.py image1.png image2.png ...
    python predict.py --model outputs/plane_cnn.keras some_folder/
"""

import argparse
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
    args = parser.parse_args()

    if not os.path.isfile(args.model):
        sys.exit("Model '%s' not found. Train one first with: python train.py" % args.model)

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
    probs = model.predict(batch, verbose=0).ravel()

    for path, prob in zip(paths, probs):
        label = "plane" if prob >= 0.5 else "no plane"
        print("%-60s %s (p=%.3f)" % (path, label, prob))


if __name__ == "__main__":
    main()
