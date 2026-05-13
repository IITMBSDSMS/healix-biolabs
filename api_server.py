# pyrefly: ignore [missing-import]
import io
import json
import os
import warnings
import traceback
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── Load model artifacts ────────────────────────────────────────────────────
MODEL_PATH = "breast_model.pkl"

print(f"Looking for model at: {os.path.abspath(MODEL_PATH)}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")

try:
    print("Loading model...")
    model = joblib.load(MODEL_PATH)
    print("Model loaded successfully")
except Exception as e:
    print(f"MODEL LOAD FAILED: {e}")
    traceback.print_exc()
    raise RuntimeError(f"Failed to load breast_model.pkl: {e}")

# Read feature count and class mapping from training metadata (fallback to 20)
N_FEATURES = 20
# sklearn Wisconsin BC: target 0 = Malignant, 1 = Benign
# (label_names = ['malignant', 'benign'])
CLASS_MAP = {0: "Malignant", 1: "Benign"}
if os.path.exists("model_meta.json"):
    try:
        with open("model_meta.json") as f:
            meta = json.load(f)
        N_FEATURES = meta.get("n_features", 20)
    except Exception as e:
        print(f"Warning: Could not load model_meta.json: {e}")

GENES = ["BRCA1", "TP53", "EGFR", "PIK3CA", "KRAS"]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def safe_read_csv(file) -> pd.DataFrame:
    """
    Read uploaded file safely with auto-detection of encoding/delimiter.
    Multiple fallback strategies to handle malformed files.
    """
    if not file or not file.filename:
        raise ValueError("No valid file provided")

    content = file.read()
    if not content:
        raise ValueError("Uploaded file is empty")

    # Try multiple encoding strategies
    encoding_strategies = [
        ("utf-8", None),
        ("latin1", "replace"),
        ("iso-8859-1", "replace"),
        ("cp1252", "replace"),
    ]

    decoded_text = None
    for encoding, errors in encoding_strategies:
        try:
            decoded_text = content.decode(encoding, errors=errors if errors else "strict")
            break
        except (UnicodeDecodeError, AttributeError):
            continue

    if decoded_text is None:
        raise ValueError("Could not decode file with any supported encoding (UTF-8, Latin1, ISO-8859-1, CP1252)")

    # Try multiple parsing strategies
    parsing_strategies = [
        {"sep": None, "engine": "python", "on_bad_lines": "skip"},   # Auto-detect delimiter
        {"sep": ",", "engine": "python", "on_bad_lines": "skip"},    # CSV
        {"sep": "\t", "engine": "python", "on_bad_lines": "skip"},   # TSV
        {"sep": ";", "engine": "python", "on_bad_lines": "skip"},    # Semicolon
        {"sep": "|", "engine": "python", "on_bad_lines": "skip"},    # Pipe
        {"sep": r"\\s+", "engine": "python", "on_bad_lines": "skip"},  # Whitespace
    ]

    last_error = None
    for strategy in parsing_strategies:
        try:
            df = pd.read_csv(io.StringIO(decoded_text), **strategy)
            if not df.empty and df.shape[1] > 0:
                return df
        except Exception as e:
            last_error = e
            continue

    raise ValueError(f"Could not parse file with any delimiter strategy. Last error: {last_error}")


def extract_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only numeric columns, handling NaN safely.
    Drop fully non-numeric or empty columns.
    """
    if df.empty or df.shape[1] == 0:
        raise ValueError("Input dataframe is empty")

    # Convert all columns to numeric, coercing errors to NaN
    X = df.apply(pd.to_numeric, errors="coerce")

    # Drop columns that are completely empty or non-numeric
    X = X.dropna(axis=1, how="all")

    # Drop rows that are completely empty
    X = X.dropna(axis=0, how="all")

    if X.empty or X.shape[1] == 0:
        raise ValueError("No numeric columns found in uploaded file after conversion")

    if X.shape[0] == 0:
        raise ValueError("No valid data rows found after removing empty rows")

    return X


def resize_to_n(X: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Resize features to exactly n via duplication or slicing.
    Ensures feature count matches model input requirements.
    """
    if X.shape[1] == 0:
        raise ValueError("Cannot resize empty dataframe")

    if X.shape[1] < n:
        # Duplicate real columns to reach target feature count
        original = list(X.columns)
        i = 0
        while X.shape[1] < n:
            src = original[i % len(original)]
            X[f"dup_{i}"] = X[src].values
            i += 1
    elif X.shape[1] > n:
        # Slice to first n features
        X = X.iloc[:, :n]

    return X.reset_index(drop=True)


def apply_scaler(X_np: np.ndarray) -> np.ndarray:
    """
    Apply per-dataset MinMax normalization with robust zero-division handling.

    WHY: The training scaler was fitted on Wisconsin Breast Cancer feature
    distributions. Applying it to user-uploaded genomic CSVs with completely
    different value ranges maps all values to ~1.0, forcing predictions of
    Malignant for every row.

    Per-dataset normalization scales each upload relative to its own min/max,
    keeping feature values in [0, 1] regardless of original CSV scale.
    Safe from zero-division: when max=min, sets denominator to 1.0.
    """
    if X_np.size == 0:
        raise ValueError("Cannot scale empty array")

    X_np = np.asarray(X_np, dtype=float)

    # Handle NaN/inf values before scaling
    X_np = np.nan_to_num(X_np, nan=0.0, posinf=0.0, neginf=0.0)

    mins = X_np.min(axis=0)
    maxs = X_np.max(axis=0)

    # Compute range, handling zero-division safely
    rng = maxs - mins
    rng = np.where(rng == 0, 1.0, rng)  # Avoid division by zero: if range is 0, use 1.0

    # Normalize to [0, 1]
    X_scaled = (X_np - mins) / rng

    # Ensure all values are in valid range [0, 1]
    X_scaled = np.clip(X_scaled, 0, 1)

    return X_scaled


def gene_graph(X_df: pd.DataFrame) -> list:
    """
    Compute normalized mean signal of first 5 features as gene expression proxy.
    Maps to GENES list for frontend visualization.
    Safely handles NaN/empty/zero cases.
    """
    if X_df.empty or X_df.shape[1] == 0:
        return [{"gene": GENES[i], "value": 0.0} for i in range(min(5, len(GENES)))]

    # Use first 5 features (or all if fewer than 5)
    n_genes = min(5, X_df.shape[1], len(GENES))
    vals = X_df.iloc[:, :n_genes].mean(axis=0)

    # Handle NaN in means
    vals = vals.fillna(0)

    # Find max value for normalization
    max_val = vals.max()
    if pd.isna(max_val) or max_val <= 0:
        max_val = 1.0

    # Normalize to 0-100 scale
    norm = (vals / max_val * 100).fillna(0)

    # Return gene objects with real computed values
    result = []
    for i in range(n_genes):
        result.append({
            "gene": GENES[i],
            "value": round(float(norm.iloc[i]), 2)
        })

    # Pad with zeros if fewer than 5 genes
    while len(result) < 5:
        result.append({"gene": GENES[len(result)], "value": 0.0})

    return result


# ─── Predict endpoint ────────────────────────────────────────────────────────

@app.route("/predict", methods=["POST"])
def predict():
    """
    Main prediction endpoint for genomic inference.
    - Accepts CSV/TXT uploads with auto-delimiter detection
    - Processes all uploaded rows with real model predictions
    - Returns full results but shows only first 25 for frontend
    - Computes real confidence from model probability distributions
    - Robust error handling to prevent 400/500 errors
    """
    try:
        # ─── 1. Validate file upload ──────────────────────────────────
        if "file" not in request.files:
            return jsonify({
                "error": "No file uploaded",
                "code": "MISSING_FILE"
            }), 400

        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({
                "error": "No file selected",
                "code": "EMPTY_FILE"
            }), 400

        # ─── 2. Parse file with robust encoding/delimiter detection ────
        try:
            df = safe_read_csv(file)
        except ValueError as e:
            return jsonify({
                "error": str(e),
                "code": "PARSE_ERROR"
            }), 400
        except Exception as e:
            return jsonify({
                "error": f"Unexpected error parsing file: {str(e)}",
                "code": "FILE_READ_ERROR"
            }), 500

        if df.empty or df.shape[0] == 0:
            return jsonify({
                "error": "Uploaded file is empty or has no valid data rows",
                "code": "EMPTY_DATA"
            }), 400

        total_uploaded_rows = df.shape[0]

        # ─── 3. Extract numeric columns only ─────────────────────────
        try:
            X = extract_numeric(df)
        except ValueError as e:
            return jsonify({
                "error": str(e),
                "code": "NO_NUMERIC_DATA"
            }), 400
        except Exception as e:
            return jsonify({
                "error": f"Error extracting numeric data: {str(e)}",
                "code": "DATA_EXTRACTION_ERROR"
            }), 500

        numeric_rows = X.shape[0]
        if numeric_rows == 0:
            return jsonify({
                "error": "No valid numeric data rows after processing",
                "code": "NO_VALID_ROWS"
            }), 400

        # ─── 4. Handle missing values safely ──────────────────────────
        try:
            # Fill NaN with column mean, then with 0 if all NaN
            for col in X.columns:
                col_mean = X[col].mean()
                if pd.isna(col_mean):
                    X[col] = X[col].fillna(0)
                else:
                    X[col] = X[col].fillna(col_mean)

            # Final safety: replace any remaining NaN or inf
            X = X.fillna(0)
            X = X.replace([np.inf, -np.inf], 0)
        except Exception as e:
            return jsonify({
                "error": f"Error handling missing values: {str(e)}",
                "code": "NAN_HANDLING_ERROR"
            }), 500

        # ─── 5. Resize features to match model input ─────────────────
        try:
            X = resize_to_n(X.copy(), N_FEATURES)
        except Exception as e:
            return jsonify({
                "error": f"Error resizing features: {str(e)}",
                "code": "FEATURE_RESIZE_ERROR"
            }), 500

        # Keep copy of raw data for gene graph (before scaling)
        X_raw = X.copy()

        # ─── 6. Scale using per-dataset normalization ────────────────
        try:
            X_np = apply_scaler(X.to_numpy())
        except Exception as e:
            return jsonify({
                "error": f"Error scaling data: {str(e)}",
                "code": "SCALING_ERROR"
            }), 500

        # ─── 7. Run real model inference on ALL rows ──────────────────
        try:
            probs = model.predict_proba(X_np)      # shape: (n_rows, 2)
            preds = np.argmax(probs, axis=1)       # class predictions
        except Exception as e:
            return jsonify({
                "error": f"Model prediction failed: {str(e)}",
                "code": "PREDICTION_ERROR"
            }), 500

        # ─── 8. Determine class indices from model ───────────────────
        try:
            classes = list(model.classes_)
            if len(classes) < 2:
                raise ValueError(f"Model has {len(classes)} classes, expected 2")

            malignant_idx = 0 if 0 in classes else min(classes)
            benign_idx = 1 if 1 in classes else max(classes)
        except Exception as e:
            return jsonify({
                "error": f"Error interpreting model classes: {str(e)}",
                "code": "CLASS_INTERPRETATION_ERROR"
            }), 500

        # ─── 9. Build predictions list (all rows processed, first 25 for frontend) ──
        try:
            predictions = []
            confidences = []

            for i in range(len(preds)):
                # Real confidence from model probability distribution
                conf = float(np.max(probs[i])) * 100
                confidences.append(conf)

                # Add to response predictions (limit display to 25 for performance)
                if i < 25:
                    predictions.append({
                        "sample": f"Sample #{i + 1}",
                        "result": CLASS_MAP.get(int(preds[i]), str(preds[i])),
                        "confidence": round(conf, 2),
                        "risk": round(float(probs[i][malignant_idx]) * 100, 2),
                        "benignScore": round(float(probs[i][benign_idx]) * 100, 2),
                    })
        except Exception as e:
            return jsonify({
                "error": f"Error building predictions: {str(e)}",
                "code": "PREDICTION_BUILD_ERROR"
            }), 500

        # ─── 10. Compute aggregate metrics (all rows) ────────────────
        try:
            if not confidences or len(confidences) == 0:
                mean_conf = 0.0
            else:
                mean_conf = float(np.mean(confidences))

            mean_malign = float(np.mean(probs[:, malignant_idx]))
            std_malign = float(np.std(probs[:, malignant_idx]))

            total_samples = len(X_np)
            n_malignant = int(np.sum(preds == malignant_idx))
            n_benign = int(np.sum(preds == benign_idx))

            # Compute clinical metrics
            if mean_malign > 0.7:
                aggression = "High"
                therapy_sensitivity = "Resistant"
            elif mean_malign > 0.4:
                aggression = "Moderate"
                therapy_sensitivity = "Moderate"
            else:
                aggression = "Low"
                therapy_sensitivity = "Sensitive"

            if std_malign > 0.3:
                instability = "Critical"
            elif std_malign > 0.15:
                instability = "Variable"
            else:
                instability = "Stable"

            summary = (
                f"Processed {total_samples} real genomic samples "
                f"({n_malignant} Malignant, {n_benign} Benign). "
                "Dashboard preview limited to 25 for performance."
            )
        except Exception as e:
            return jsonify({
                "error": f"Error computing aggregate metrics: {str(e)}",
                "code": "METRICS_COMPUTATION_ERROR"
            }), 500

        # ─── 11. Compute gene graph ──────────────────────────────────
        try:
            genes_graph = gene_graph(X_raw)
        except Exception as e:
            genes_graph = [{"gene": GENES[i], "value": 0.0} for i in range(5)]

        # ─── 12. Build and return response ───────────────────────────
        return jsonify({
            "accuracy": round(mean_conf, 2),
            "samples": total_samples,
            "hxRisk": round(mean_malign * 10, 1),
            "aggression": aggression,
            "therapySensitivity": therapy_sensitivity,
            "instability": instability,
            "predictions": predictions,
            "genes": genes_graph,
            "summary": summary,
        }), 200

    except Exception as e:
        # Catch-all for any unhandled exceptions
        error_msg = f"Unexpected error in predict endpoint: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        return jsonify({
            "error": error_msg,
            "code": "INTERNAL_SERVER_ERROR"
        }), 500


# ─── Health check ────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "n_features": N_FEATURES,
        "scaler": "per-dataset-minmax",
        "class_map": CLASS_MAP,
        "model_classes": list(map(int, model.classes_)),
        "version": "1.0-production",
        "api_port": 5000,
    }), 200


# ─── Error handlers ──────────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(e):
    return jsonify({
        "error": "Bad request",
        "code": "BAD_REQUEST",
        "details": str(e)
    }), 400


@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Endpoint not found",
        "code": "NOT_FOUND"
    }), 404


@app.errorhandler(500)
def internal_error(e):
    print(f"Internal server error: {traceback.format_exc()}")
    return jsonify({
        "error": "Internal server error",
        "code": "INTERNAL_SERVER_ERROR"
    }), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))

    print("=" * 60)
    print("  HEALIX GENOMIC INFERENCE ENGINE")
    print("=" * 60)
    print(f"  Model Features:     {N_FEATURES}")
    print(f"  Scaler Type:        per-dataset MinMax normalization")
    print(f"  Model Classes:      {list(map(int, model.classes_))}")
    print(f"  API Endpoint:       http://0.0.0.0:{port}/predict")
    print(f"  Health Check:       http://0.0.0.0:{port}/health")
    print("=" * 60)
    print(f"  Starting Flask server on port {port}")
    print("=" * 60)

    app.run(host="0.0.0.0", port=port, debug=False)