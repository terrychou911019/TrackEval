from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd

DATASET_NAME = "TGB-test"

# List all CSV paths here (each CSV contains multiple seq rows and one COMBINED row)
SUMMARY_CSV_PATHS: List[Path] = [
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/mcbyte/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/mcbyte+gta_split+gta_merge/pedestrian_detailed.csv"),
]

# The key metrics you want to display (you can add or remove as needed)
KEEP_METRICS: List[str] = [
    "HOTA", "AssA", "DetA", "IDs", "Purity",
    # "HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDs", "Purity",
]

# Column aliases: map logical metric names to actual CSV column names
KEY_ALIASES: Dict[str, List[str]] = {
    "HOTA":  ["HOTA___AUC"],
    "AssA":  ["AssA___AUC"],
    "DetA":  ["DetA___AUC"],
    "IDF1": ["IDF1"],
    "MOTA": ["MOTA"],
    "IDs": ["IDs"],
    "Purity": ["Purity___AUC"],  
}

def _pick_column(cols: List[str], logical_name: str) -> Optional[str]:
    aliases = KEY_ALIASES.get(logical_name, [])
    for cand in aliases + [logical_name]:
        if cand in cols:
            return cand
    return None

def _fmt_cell(x) -> str:
    if isinstance(x, str):
        return x
    try:
        if pd.isna(x):
            return "N/A"
        if x == 1:
            return f"{(100 * float(x)):.3f}"
        if abs(float(x) - round(float(x))) < 1e-6:
            return f"{int(round(float(x)))}"
        else:
            return f"{(100 * float(x)):.3f}"
    except Exception:
        return str(x)

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
            print(f"⚠️ {path.parent.name} missing column: {k} (no matching name found), will mark as N/A")
        resolved_cols[k] = actual

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
    
    combined_row = out[out["seq"].str.upper() == "COMBINED"]
    
    if combined_row.empty:
        print(f"❌ {path.parent.name} CSV does not contain a 'COMBINED' row.")
        return None # Return None if combined row is missing

    return combined_row.iloc[[0]].copy() # Ensure it's a single-row DataFrame

def print_combined_results_table(df_combined: pd.DataFrame, keep_metrics: List[str]):
    if df_combined.empty:
        print("No valid COMBINED results to display.")
        return

    print(f"📊 Consolidated COMBINED Tracking Metrics on {DATASET_NAME} 📊")
    print("Rows: Trackers | Columns: Metrics")

    # Define column widths
    tracker_col_name = "Tracker"
    tracker_width = max(len(tracker_col_name), max((len(t) for t in df_combined[tracker_col_name]), default=0)) + 2
    metric_width = 12 

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
    
    # Load data, filter for COMBINED row, and collect
    for p in paths:
        df_combined_row = load_tracker_df(p, KEEP_METRICS)
        if df_combined_row is not None:
            all_combined_rows.append(df_combined_row)

    if not all_combined_rows:
        print("No valid CSV files or 'COMBINED' rows found, exiting.")
        return

    # Concatenate all single-row DataFrames into one table
    full_combined_df = pd.concat(all_combined_rows, ignore_index=True)
    
    # Remove the redundant "seq" column (which is "COMBINED" for all rows)
    if "seq" in full_combined_df.columns:
        full_combined_df = full_combined_df.drop(columns=["seq"])

    # Print the final consolidated table
    print_combined_results_table(full_combined_df, KEEP_METRICS)

if __name__ == "__main__":
    main(SUMMARY_CSV_PATHS)