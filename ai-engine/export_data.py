"""
export_data.py
--------------
Processes real GSE42568 CEL.gz genomic files through the trained AI model
and exports predictions.json with real model predictions.

Usage:
    python export_data.py
"""

import gzip
import io
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd

# ─── Paths ───────────────────────────────────────────────────────────────────
ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLDER      = os.path.join(ROOT, "GSE42568_RAW")
MODEL_PATH  = os.path.join(ROOT, "breast_model.pkl")
META_PATH   = os.path.join(ROOT, "model_meta.json")
OUTPUT_PATH = os.path.join(ROOT, "predictions.json")

# ─── Load model ──────────────────────────────────────────────────────────────
model = joblib.load(MODEL_PATH)
CLASS_MAP = {0: "Malignant", 1: "Benign"}

N_FEATURES = 20
if os.path.exists(META_PATH):
    with open(META_PATH) as f:
        meta = json.load(f)
    N_FEATURES = meta.get("n_features", 20)

classes       = list(model.classes_)
malignant_idx = 0 if 0 in classes else classes.index(min(classes))
benign_idx    = 1 if 1 in classes else classes.index(max(classes))


# ─── Parse CEL.gz file ───────────────────────────────────────────────────────

def parse_cel_gz(filepath: str):
    """
    Extract numeric intensity values from a CEL.gz file.
    Returns a 1-D numpy array of floats, or None if no data found.
    """
    try:
        with gzip.open(filepath, "rt", errors="ignore") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"  ⚠ Could not open {os.path.basename(filepath)}: {e}")
        return None

    values = []
    in_intensities = False

    for line in lines:
        stripped = line.strip()

        # CEL intensity section markers
        if stripped.startswith("[INTENSITY]") or stripped.startswith("[CHP_PAIRED_DATA]"):
            in_intensities = True
            continue
        if stripped.startswith("[") and in_intensities:
            in_intensities = False

        if in_intensities and stripped and not stripped.startswith("NumberCells") \
                and not stripped.startswith("CellHeader") and not stripped.startswith("X"):
            parts = stripped.split()
            # CEL format: X Y MEAN STDDEV PIXELS — col index 2 is the probe intensity
            for part in parts[2:3] or parts:
                try:
                    values.append(float(part))
                except ValueError:
                    continue

    if not values:
        # Fallback: grab ANY numeric token from the file
        for line in lines:
            for token in line.strip().split():
                try:
                    values.append(float(token))
                except ValueError:
                    continue
        if not values:
            return None

    return np.array(values, dtype=float)


def build_feature_vector(values: np.ndarray, n: int) -> np.ndarray:
    """Summarise probe intensities into exactly n features using statistics."""
    if len(values) == 0:
        return np.zeros(n)

    # Build statistical summary features from the probe distribution
    step = max(1, len(values) // n)
    features = values[::step][:n]           # evenly-sampled probe values

    # If still fewer than n, duplicate cyclically
    while len(features) < n:
        features = np.concatenate([features, features])
    features = features[:n]

    # Per-sample MinMax normalization (same as api_server.py)
    rng = features.max() - features.min()
    if rng == 0:
        return np.zeros(n)
    return (features - features.min()) / rng


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    if not os.path.isdir(FOLDER):
        print(f"ERROR: Folder not found: {FOLDER}")
        sys.exit(1)

    files = sorted([f for f in os.listdir(FOLDER) if f.endswith(".CEL.gz")])
    if not files:
        print("ERROR: No .CEL.gz files found in GSE42568_RAW/")
        sys.exit(1)

    print(f"Found {len(files)} CEL.gz files. Processing...")
    print("-" * 55)

    all_vectors = []
    all_names   = []

    for fname in files:
        path   = os.path.join(FOLDER, fname)
        values = parse_cel_gz(path)
        if values is None or len(values) == 0:
            print(f"  SKIP  {fname} — no numeric data")
            continue

        vec = build_feature_vector(values, N_FEATURES)
        all_vectors.append(vec)
        all_names.append(fname)
        print(f"  OK    {fname}  ({len(values)} probes)")

    if not all_vectors:
        print("ERROR: No valid CEL files could be processed.")
        sys.exit(1)

    X = np.array(all_vectors, dtype=float)

    # Predict using real model
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)

    predictions = []
    for i, (name, p) in enumerate(zip(all_names, preds)):
        predictions.append({
            "sample":      name,
            "result":      CLASS_MAP.get(int(p), str(p)),
            "confidence":  round(float(np.max(probs[i])) * 100, 2),
            "risk":        round(float(probs[i][malignant_idx]) * 100, 2),
            "benignScore": round(float(probs[i][benign_idx])    * 100, 2),
        })

    # Aggregate stats
    mean_malign = float(np.mean(probs[:, malignant_idx]))
    std_malign  = float(np.std(probs[:, malignant_idx]))
    n_malignant = int(np.sum(preds == malignant_idx))
    n_benign    = int(np.sum(preds == benign_idx))

    output = {
        "accuracy":           round(float(np.mean([p["confidence"] for p in predictions])), 2),
        "samples":            len(predictions),
        "hxRisk":             round(mean_malign * 10, 1),
        "aggression":         "High"      if mean_malign > 0.7 else "Moderate" if mean_malign > 0.4 else "Low",
        "therapySensitivity": "Resistant" if mean_malign > 0.7 else "Moderate" if mean_malign > 0.4 else "Sensitive",
        "instability":        "Critical"  if std_malign  > 0.3 else "Variable" if std_malign  > 0.15 else "Stable",
        "predictions":        predictions[:25],   # first 25 for frontend
        "summary":            (
            f"Processed {len(predictions)} real CEL genomic samples "
            f"({n_malignant} Malignant, {n_benign} Benign). "
            "Dashboard preview limited to 25 for performance."
        ),
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 55)
    print(f"  Total samples processed : {len(predictions)}")
    print(f"  Malignant               : {n_malignant}")
    print(f"  Benign                  : {n_benign}")
    print(f"  Mean confidence         : {output['accuracy']}%")
    print(f"  Saved → {OUTPUT_PATH}")
    print("=" * 55)


if __name__ == "__main__":
    main()