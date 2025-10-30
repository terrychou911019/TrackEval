from pathlib import Path
import sys
import csv
from typing import List, Dict, Optional
import pandas as pd

# --- Configuration (Mostly unchanged, but some flags are now ignored) ---
# NOTE: SHOW_ALL_SEQUENCES and WRITE_SUMMARY_CSV are effectively ignored
# as we are now focusing only on the COMBINED row for the final table.
SHOW_ALL_SEQUENCES = False # Set to False to enforce COMBINED only logic
WRITE_SUMMARY_CSV = False

DATASET_NAME = "SportsMOT-val"

# List all CSV paths here (each CSV contains multiple seq rows and one COMBINED row)
SUMMARY_CSV_PATHS: List[Path] = [
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/botsort/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/bytetrack/pedestrian_detailed.csv"),
]

# The key metrics you want to display (you can add or remove as needed)
KEEP_METRICS: List[str] = [
    "HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDs", "Purity",
]

# Column aliases: map logical metric names to actual CSV column names
# (HOTA uses HOTA___AUC or HOTA(0); Purity uses Purity___AUC or Purity(0);
# others usually match directly without mapping)
KEY_ALIASES: Dict[str, List[str]] = {
    "HOTA":  ["HOTA___AUC", "HOTA(0)", "HOTA___0", "HOTA0"],
    "AssA":  ["AssA___AUC", "AssA(0)", "AssA___0", "AssA0"],
    "DetA":  ["DetA___AUC", "DetA(0)", "DetA___0", "DetA0"],
    "Purity": ["Purity___AUC", "Purity(0)", "Purity___0", "Purity0"],
    # "IDF1": ["IDF1"], "MOTA": ["MOTA"], "IDs": ["IDs"]
}

# Whether to output a merged CSV (all trackers × seq)
SUMMARY_OUT_PATH = Path("key_metrics_summary_all.csv")

# --- Helper Functions (Unchanged) ---

def _pick_column(cols: List[str], logical_name: str) -> Optional[str]:
    """
    Map a logical metric name in KEEP_METRICS to an actual CSV column.
    First check KEY_ALIASES; if not found, fall back to the same name.
    """
    aliases = KEY_ALIASES.get(logical_name, [])
    for cand in aliases + [logical_name]:
        if cand in cols:
            return cand
    return None


def load_tracker_df(path: Path, keep_metrics: List[str]) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"❌ File not found: {path}")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"❌ Failed to read CSV: {path} -> {e}")
        return None

    if "seq" not in df.columns:
        print(f"❌ CSV missing 'seq' column: {path}")
        return None

    # Resolve actual column names
    resolved_cols: Dict[str, str] = {}
    for k in keep_metrics:
        actual = _pick_column(list(df.columns), k)
        if actual is None:
            # Note: This print is fine to keep, helps debug missing columns
            print(f"⚠️ {path.parent.name} missing column: {k} (no matching name found), will mark as N/A")
        resolved_cols[k] = actual # could be None

    # Build output DataFrame: keep seq + selected metrics (missing filled with N/A)
    out = pd.DataFrame()
    out["seq"] = df["seq"].astype(str)

    for k, actual in resolved_cols.items():
        if actual is None:
            out[k] = "N/A"
        else:
            out[k] = df[actual]

    # Add tracker name column
    out.insert(0, "Tracker", path.parent.name)
    
    # *** KEY CHANGE: Filter immediately to only keep the 'COMBINED' row ***
    combined_row = out[out["seq"].str.upper() == "COMBINED"]
    
    if combined_row.empty:
        print(f"❌ {path.parent.name} CSV does not contain a 'COMBINED' row.")
        return None # Return None if combined row is missing

    return combined_row.iloc[[0]].copy() # Ensure it's a single-row DataFrame


def _fmt_cell(x) -> str:
    """Format a cell nicely: numbers with 3 decimals, integers without decimals, others unchanged."""
    if isinstance(x, str):
        return x
    try:
        if pd.isna(x):
            return "N/A"
        # MOTA, HOTA, etc. are usually between 0 and 1, so we multiply by 100
        # IDs is an integer count. Check if it's close to an integer.
        if abs(float(x) - round(float(x))) < 1e-6:
             return f"{int(round(float(x)))}"
        else:
             return f"{(100 * float(x)):.3f}"
    except Exception:
        return str(x)

# --- Core Logic (Modified to print consolidated table) ---

def print_combined_results_table(df_combined: pd.DataFrame, keep_metrics: List[str]):
    """
    Prints a consolidated table of COMBINED results.
    Rows are trackers, columns are metrics.
    """
    if df_combined.empty:
        print("No valid COMBINED results to display.")
        return

    print(f"📊 Consolidated COMBINED Tracking Metrics on {DATASET_NAME} 📊")
    print("Rows: Trackers | Columns: Metrics")

    # Define column widths
    tracker_col_name = "Tracker"
    tracker_width = max(len(tracker_col_name), max((len(t) for t in df_combined[tracker_col_name]), default=0)) + 2
    metric_width = 12 # Sufficient width for formatted percentage + padding

    # Create Header
    header = f"{tracker_col_name:<{tracker_width}}" + "".join(f"{k:>{metric_width}}" for k in keep_metrics)
    sep = "=" * (tracker_width + metric_width * len(keep_metrics))

    print(sep)
    print(header)
    print(sep)

    # Print data rows
    for index, row in df_combined.iterrows():
        tracker_name = row[tracker_col_name]
        vals = [_fmt_cell(row[k]) for k in keep_metrics]
        print(f"{tracker_name:<{tracker_width}}" + "".join(f"{v:>{metric_width}}" for v in vals))

    print(sep)


def main(paths: List[Path]):
    all_combined_rows: List[pd.DataFrame] = []
    
    # 1. Load data, filter for COMBINED row, and collect
    for p in paths:
        # load_tracker_df is now modified to return ONLY the COMBINED row (or None)
        df_combined_row = load_tracker_df(p, KEEP_METRICS)
        if df_combined_row is not None:
            all_combined_rows.append(df_combined_row)

    if not all_combined_rows:
        print("No valid CSV files or 'COMBINED' rows found, exiting.")
        return

    # 2. Concatenate all single-row DataFrames into one table
    full_combined_df = pd.concat(all_combined_rows, ignore_index=True)
    
    # Remove the redundant 'seq' column (which is "COMBINED" for all rows)
    if 'seq' in full_combined_df.columns:
        full_combined_df = full_combined_df.drop(columns=['seq'])

    # 3. Sort by Tracker name for consistency
    full_combined_df = full_combined_df.sort_values(by="Tracker").reset_index(drop=True)

    # 4. Print the final consolidated table
    print_combined_results_table(full_combined_df, KEEP_METRICS)

    # The WRITE_SUMMARY_CSV logic is removed as per the simplified requirement,
    # but could be re-added if needed.
    if WRITE_SUMMARY_CSV:
         print("\nNOTE: WRITE_SUMMARY_CSV is False, skipping summary output.")


if __name__ == "__main__":
    main(SUMMARY_CSV_PATHS)