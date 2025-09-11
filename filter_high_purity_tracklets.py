import pandas as pd
import os

# === Config: multiple trackers and sequences ===
tracker_names = [
    "botsort+gta_splitter",
    "bytetrack+gta_splitter",
    "deepeiou+gta_splitter",
    "hybridsort+gta_splitter",
    "ocsort+gta_splitter"
]
seq_names = ["seq01", "seq02", "seq03"]
filter_threshold = 10  # Minimum TP threshold

for tracker_name in tracker_names:
    for seq_name in seq_names:
        print(f"\n=== Processing tracker={tracker_name}, seq={seq_name} ===")

        # === Step 1: Read per_tracklet_purity/{seq}_high_purity.csv ===
        input_path = os.path.join(
            "data/trackers/mot_challenge/TGB-test",
            tracker_name,
            "per_tracklet_purity",
            f"{seq_name}_high_purity.csv"
        )
        if not os.path.exists(input_path):
            print(f"❌ per_tracklet_purity file not found: {input_path}")
            continue

        df = pd.read_csv(input_path)

        # Filter tracklets with tp >= threshold
        filtered_df = df[df['tp'] >= filter_threshold]
        filtered_ids = set(filtered_df['tracker_id'].astype(int))

        # Save filtered CSV
        output_path = os.path.join(
            "data/trackers/mot_challenge/TGB-test",
            tracker_name,
            "per_tracklet_purity",
            f"{seq_name}_high_purity_filtered.csv"
        )
        filtered_df.to_csv(output_path, index=False)
        print(f"✅ Filtered CSV saved to {output_path}")

        # === Step 2: Filter original tracker output ===
        tracker_result_path = os.path.join(
            "data/trackers/mot_challenge/TGB-test",
            tracker_name,
            "data",
            f"{seq_name}.txt"
        )

        if os.path.exists(tracker_result_path):
            tracker_df = pd.read_csv(tracker_result_path, header=None)

            # Assume the 2nd column (index=1) is the tracklet ID
            # High purity tracklets (tp >= threshold)
            kept_df = tracker_df[tracker_df[1].isin(filtered_ids)]
            kept_output_path = os.path.join(
                "data/trackers/mot_challenge/TGB-test",
                tracker_name,
                f"high_purity_tracklets_filtered",
                f"{seq_name}.txt"
            )
            os.makedirs(os.path.dirname(kept_output_path), exist_ok=True)
            kept_df.to_csv(kept_output_path, index=False, header=False)
            print(f"✅ High purity tracklets saved to {kept_output_path}")

            # Unhigh purity tracklets (tp < threshold)
            removed_df = tracker_df[~tracker_df[1].isin(filtered_ids)]
            removed_output_path = os.path.join(
                "data/trackers/mot_challenge/TGB-test",
                tracker_name,
                f"unhigh_purity_tracklets_filtered",
                f"{seq_name}.txt"
            )
            os.makedirs(os.path.dirname(removed_output_path), exist_ok=True)
            removed_df.to_csv(removed_output_path, index=False, header=False)
            print(f"✅ Unhigh purity tracklets saved to {removed_output_path}")

        else:
            print(f"❌ Tracker result file not found: {tracker_result_path}")
