import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import io
import traceback


def clean_genomic_dataset(input_file, output_file="cleaned_genomic_data.csv"):
    """
    Comprehensively clean genomic dataset:
    1. Load CSV/TXT safely (UTF-8, Latin1, CP1252 with malformed row recovery)
    2. Convert numeric-looking strings to floats
    3. Remove duplicate rows/columns
    4. Handle NaN values: fill numeric with mean, categorical with mode
    5. Remove rows with >50% missing data
    6. Remove columns >70% missing
    7. Cap extreme outliers using IQR without dropping rows
    8. Remove zero-variance columns
    9. Feature Engineering: Interaction features and sample statistics
    10. Normalize numeric features using StandardScaler
    11. Save cleaned CSV for model training
    """
    print(f"Loading dataset: {input_file}...")

    # ─── 1. Load CSV/TXT safely with multiple encoding strategies ──────────
    try:
        with open(input_file, "rb") as f:
            content = f.read()

        if not content:
            print("ERROR: Input file is empty")
            return None

        # Try multiple encoding strategies
        text = None
        for encoding in ["utf-8", "latin1", "iso-8859-1", "cp1252"]:
            try:
                text = content.decode(encoding, errors="replace")
                if text:
                    break
            except Exception:
                continue

        if not text:
            print("ERROR: Could not decode file with any supported encoding")
            return None

        # Try multiple parsing strategies
        df = None
        for delimiter in [None, ",", "\t", ";", "|", "\s+"]:
            try:
                kwargs = {
                    "on_bad_lines": "skip",
                    "engine": "python"
                }
                if delimiter is not None:
                    kwargs["sep"] = delimiter
                else:
                    kwargs["sep"] = None  # Auto-detect
                    
                df_attempt = pd.read_csv(io.StringIO(text), **kwargs)
                if not df_attempt.empty and df_attempt.shape[1] > 0:
                    df = df_attempt
                    break
            except Exception:
                continue

        if df is None or df.empty:
            print("ERROR: Could not parse file with any delimiter strategy")
            return None

    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load dataset: {e}")
        traceback.print_exc()
        return None

    original_rows, original_cols = df.shape
    print(f"✓ Successfully loaded: {original_rows} rows × {original_cols} columns")

    # ─── 2. Convert numeric-looking strings to real floats ────────────────
    print("Converting columns to numeric...")
    for col in df.columns:
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            # If >50% of column is numeric, convert permanently
            if converted.notna().sum() > (len(df) * 0.5) or df[col].dtype.kind in "bifc":
                df[col] = converted
        except Exception:
            continue

    # ─── 3. Remove duplicate rows ────────────────────────────────────────
    df = df.drop_duplicates()

    # ─── 4. Remove duplicate columns ─────────────────────────────────────
    df = df.loc[:, ~df.columns.duplicated(keep="first")]

    # ─── 5. Drop fully empty rows/columns and low-quality rows ────────────
    df = df.dropna(axis=0, how="all")
    df = df.dropna(axis=1, how="all")

    # Remove rows with >50% missing values (malformed or low quality)
    threshold_row = int(df.shape[1] * 0.5)
    initial_rows = df.shape[0]
    df = df.dropna(axis=0, thresh=threshold_row)
    rows_dropped_na = initial_rows - df.shape[0]
    if rows_dropped_na > 0:
        print(f"  → Dropped {rows_dropped_na} malformed/sparse rows")

    if df.empty or df.shape[0] == 0:
        print("ERROR: No valid data after removing duplicates and empty rows")
        return None

    # ─── 6. Track NaN statistics ────────────────────────────────────────
    initial_nans = df.isna().sum().sum()

    # ─── 7. Remove columns with >70% missing data ──────────────────────
    threshold = int(len(df) * 0.3)  # Require at least 30% non-NA values
    initial_cols = df.shape[1]
    df = df.dropna(axis=1, thresh=threshold)
    cols_dropped_na = initial_cols - df.shape[1]

    # ─── 8. Fill missing numeric values using column mean ─────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isna().any():
            col_mean = df[col].mean()
            if pd.isna(col_mean):
                # All NaN column - fill with 0
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(col_mean)

    # ─── 9. Fill missing categorical values using mode ─────────────────
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_cols:
        if df[col].isna().any():
            col_mode = df[col].mode()
            if not col_mode.empty:
                df[col] = df[col].fillna(col_mode[0])
            else:
                df[col] = df[col].fillna("unknown")

    # ─── 10. Final NaN/inf cleanup ─────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], 0)

    # ─── 11. Remove extreme outliers using IQR (cap instead of drop) ───
    if len(numeric_cols) > 0:
        Q1 = df[numeric_cols].quantile(0.25)
        Q3 = df[numeric_cols].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[numeric_cols] = df[numeric_cols].clip(lower=lower_bound, upper=upper_bound, axis=1)

    # ─── 12. Remove zero-variance columns ──────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        variances = df[numeric_cols].var()
        low_var_cols = variances[variances < 1e-5].index
        if len(low_var_cols) > 0:
            df = df.drop(columns=low_var_cols)

    # ─── 13. Feature Engineering ───────────────────────────────────────
    print("Performing feature engineering...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 2:
        # 1. Sample statistics
        df['sample_mean'] = df[numeric_cols].mean(axis=1)
        df['sample_std'] = df[numeric_cols].std(axis=1)
        df['sample_skew'] = df[numeric_cols].skew(axis=1)
        
        # 2. Top interaction features (interactions between first few features)
        # In a real scenario, we'd pick biologically relevant ones, here we use proxies
        top_cols = numeric_cols[:min(5, len(numeric_cols))]
        for i in range(len(top_cols)):
            for j in range(i + 1, len(top_cols)):
                col_name = f"inter_{top_cols[i]}_{top_cols[j]}"
                df[col_name] = df[top_cols[i]] * df[top_cols[j]]

    # ─── 14. Normalize numeric columns using StandardScaler ────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        try:
            scaler = StandardScaler()
            # Handle any remaining NaN or inf
            X_numeric = df[numeric_cols].fillna(0).replace([np.inf, -np.inf], 0)
            df[numeric_cols] = scaler.fit_transform(X_numeric)
            print(f"  → Normalized {len(numeric_cols)} features using StandardScaler")
        except Exception as e:
            print(f"WARNING: Could not normalize numeric columns: {e}")
            # Continue without normalization

    cleaned_rows, cleaned_cols = df.shape
    cols_dropped_total = original_cols - cleaned_cols

    # ─── 14. Save cleaned file ────────────────────────────────────────
    try:
        df.to_csv(output_file, index=False)
    except Exception as e:
        print(f"ERROR: Could not save cleaned dataset: {e}")
        return None

    # ─── 15. Print cleaning report ─────────────────────────────────────
    cleaned_rows, cleaned_cols = df.shape
    cols_dropped_total = original_cols - cleaned_cols

    print("\n" + "=" * 60)
    print("  GENOMIC DATA CLEANING REPORT")
    print("=" * 60)
    print(f"  Original:            {original_rows} rows × {original_cols} columns")
    print(f"  Cleaned:             {cleaned_rows} rows × {cleaned_cols} columns")
    print(f"  Rows removed:        {original_rows - cleaned_rows}")
    print(f"  Columns dropped:     {cols_dropped_total}")
    print(f"  Initial NaNs:        {initial_nans}")
    print(f"  Numeric features:    {len(numeric_cols)}")
    print("=" * 60)
    print(f"  ✓ Saved to: {output_file}")
    print("=" * 60)

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        target_file = sys.argv[1]
        result = clean_genomic_dataset(target_file)
        if result is not None:
            print("\n✓ Dataset cleaning successful!")
        else:
            print("\n✗ Dataset cleaning failed!")
            sys.exit(1)
    else:
        print("Usage: python clean_dataset.py <path_to_genomic_file.csv>")
