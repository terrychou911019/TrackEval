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
high_purity_threshold = 1.0

for tracker_name in tracker_names:
    for seq_name in seq_names:
        print(f"\n=== Processing tracker={tracker_name}, seq={seq_name} ===")

        # === Step 1: Read per_tracklet_purity/{seq}.csv ===
        input_path = os.path.join(
            "data/trackers/mot_challenge/TGB-test",
            tracker_name,
            "per_tracklet_purity",
            f"{seq_name}.csv"
        )
        if not os.path.exists(input_path):
            print(f"❌ per_tracklet_purity file not found: {input_path}")
            continue

        df = pd.read_csv(input_path)

        # Filter tracklets with high purity
        high_purity_df = df[df['purity'] >= high_purity_threshold]
        high_purity_ids = set(high_purity_df['tracker_id'].astype(int))

        # Save high_purity_df
        high_purity_output_path = os.path.join(
            "data/trackers/mot_challenge/TGB-test",
            tracker_name,
            "per_tracklet_purity",
            f"{seq_name}_high_purity.csv"
        )
        high_purity_df.to_csv(high_purity_output_path, index=False)
        print(f"✅ High purity saved to {high_purity_output_path}")

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
            # High purity tracklets
            high_purity_tracklets_df = tracker_df[tracker_df[1].isin(high_purity_ids)]
            high_purity_tracklets_output_path = os.path.join(
                "data/trackers/mot_challenge/TGB-test",
                tracker_name,
                f"high_purity_tracklets",
                f"{seq_name}.txt"
            )
            os.makedirs(os.path.dirname(high_purity_tracklets_output_path), exist_ok=True)
            high_purity_tracklets_df.to_csv(high_purity_tracklets_output_path, index=False, header=False)
            print(f"✅ High purity tracklets saved to {high_purity_tracklets_output_path}")

            # Unhigh purity tracklets
            unhigh_purity_tracklets_df = tracker_df[~tracker_df[1].isin(high_purity_ids)]
            unhigh_purity_tracklets_output_path = os.path.join(
                "data/trackers/mot_challenge/TGB-test",
                tracker_name,
                f"unhigh_purity_tracklets",
                f"{seq_name}.txt"
            )
            os.makedirs(os.path.dirname(unhigh_purity_tracklets_output_path), exist_ok=True)
            unhigh_purity_tracklets_df.to_csv(unhigh_purity_tracklets_output_path, index=False, header=False)
            print(f"✅ Unhigh purity tracklets saved to {unhigh_purity_tracklets_output_path}")

        else:
            print(f"❌ Tracker result file not found: {tracker_result_path}")
