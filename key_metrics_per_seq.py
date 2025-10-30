from pathlib import Path
import sys
import csv
from typing import List, Dict, Optional
import pandas as pd

SHOW_ALL_SEQUENCES = True
WRITE_SUMMARY_CSV = False

DATASET_NAME = "TGB-test"

# List all CSV paths here (each CSV contains multiple seq rows and one COMBINED row)
SUMMARY_CSV_PATHS: List[Path] = [
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/botsort/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/bytetrack/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/hybridsort/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/mcbyte/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/motip/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/ocsort/pedestrian_detailed.csv"),

    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/botsort+gta/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/bytetrack+gta/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/hybridsort+gta/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/mcbyte+gta/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/motip+gta/pedestrian_detailed.csv"),
    Path(f"data/trackers/mot_challenge/{DATASET_NAME}/ocsort+gta/pedestrian_detailed.csv"),
]

# The key metrics you want to display (you can add or remove as needed)
KEEP_METRICS: List[str] = [
    "HOTA", "AssA", "DetA", "IDF1", "MOTA", "IDs", "Purity",
]

# Column aliases: map logical metric names to actual CSV column names
# (HOTA uses HOTA___AUC or HOTA(0); Purity uses Purity___AUC or Purity(0); 
#  others usually match directly without mapping)
KEY_ALIASES: Dict[str, List[str]] = {
    "HOTA":   ["HOTA___AUC", "HOTA(0)", "HOTA___0", "HOTA0"],
    "AssA":   ["AssA___AUC", "AssA(0)", "AssA___0", "AssA0"],
    "DetA":   ["DetA___AUC", "DetA(0)", "DetA___0", "DetA0"],
    "Purity": ["Purity___AUC", "Purity(0)", "Purity___0", "Purity0"],
    # "IDF1": ["IDF1"], "MOTA": ["MOTA"], "IDs": ["IDs"]
}

# Whether to output a merged CSV (all trackers × seq)
SUMMARY_OUT_PATH = Path("key_metrics_summary_all.csv")


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
            print(f"⚠️  {path.parent.name} missing column: {k} (no matching name found), will mark as N/A")
        resolved_cols[k] = actual  # could be None

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

    return out


def _fmt_cell(x) -> str:
    """Format a cell nicely: numbers with 3 decimals, integers without decimals, others unchanged."""
    if isinstance(x, str):
        return x
    try:
        if pd.isna(x):
            return "N/A"
        if float(x).is_integer():
            return f"{int(x)}"
        else:
            return f"{(100 * float(x)):.3f}"
    except Exception:
        return str(x)


def print_table_for_tracker(df_tracker: pd.DataFrame, keep_metrics: List[str]):
    """
    Print results for one tracker (df contains only one tracker):
      1) one row per sequence (if SHOW_ALL_SEQUENCES is True)
      2) COMBINED row (if exists)
    """
    if df_tracker.empty:
        return

    tracker_name = df_tracker["Tracker"].iloc[0]
    # Sort sequences: put COMBINED at the end
    seqs = df_tracker["seq"].tolist()
    
    if SHOW_ALL_SEQUENCES:
        seq_order = sorted([s for s in seqs if s.upper() != "COMBINED"])
    else:
        seq_order = []

    if "COMBINED" in [s.upper() for s in seqs]:
        seq_order += ["COMBINED"]

    # Skip if we only want COMBINED and it's not present
    if not SHOW_ALL_SEQUENCES and not seq_order:
        print(f"\n{tracker_name}: No COMBINED results found")
        return

    # Column width design
    seq_width = max(len("Seq"), max((len(s) for s in seq_order), default=0)) + 2
    col_width = 10

    header = f"{'Seq':<{seq_width}}" + "".join(f"{k:>{col_width}}" for k in keep_metrics)
    sep = "-" * (seq_width + col_width * len(keep_metrics))

    print(f"\n=== {tracker_name} ===")
    print(header)
    print(sep)

    for s in seq_order:
        row = df_tracker[df_tracker["seq"].str.upper() == s.upper()]
        if row.empty:
            continue
        row = row.iloc[0]
        vals = [_fmt_cell(row[k]) for k in keep_metrics]
        print(f"{s:<{seq_width}}" + "".join(f"{v:>{col_width}}" for v in vals))

    print("=" * len(sep))


def main(paths: List[Path]):
    global SHOW_ALL_SEQUENCES

    all_rows: List[pd.DataFrame] = []
    for p in paths:
        df = load_tracker_df(p, KEEP_METRICS)
        if df is not None:
            all_rows.append(df)

    if not all_rows:
        print("No valid CSV files found, exiting.")
        return

    full = pd.concat(all_rows, ignore_index=True)

    # Print grouped by tracker
    for tracker, df_tracker in full.groupby("Tracker", sort=True):
        print_table_for_tracker(df_tracker, KEEP_METRICS)

        # Reminder: if COMBINED row is missing
        if not any(df_tracker["seq"].str.upper() == "COMBINED"):
            print(f"⚠️  {tracker} CSV does not contain a 'COMBINED' row. "
                  f"If you need combined metrics, make sure TrackEval outputs it "
                  f"or define your own weighted rule.")

    # Export merged summary file (optional)
    if WRITE_SUMMARY_CSV:
        try:
            full.to_csv(SUMMARY_OUT_PATH, index=False)
            print(f"\n✅ Merged summary written: {SUMMARY_OUT_PATH}")
        except Exception as e:
            print(f"❌ Failed to write merged CSV: {e}")


if __name__ == "__main__":
    main(SUMMARY_CSV_PATHS)
