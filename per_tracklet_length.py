import pandas as pd
import matplotlib.pyplot as plt
import os

# Lists to iterate over (add more as needed)
trackers = [
    "botsort+gta_splitter-high_purity",
    "bytetrack+gta_splitter-high_purity",
    "hybridsort+gta_splitter-high_purity",
    "ocsort+gta_splitter-high_purity"
]
sequences = ["seq01", "seq02", "seq03"]

for tracker_name in trackers:
    for seq_name in sequences:
        # Set input file and output image paths
        input_path = rf"data/trackers/mot_challenge/TGB-test/{tracker_name}/data/{seq_name}.txt"
        png_path   = rf"data/trackers/mot_challenge/TGB-test/{tracker_name}/per_tracklet_length/{seq_name}_length.png"

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(png_path), exist_ok=True)

        # Read tracking results (MOT format)
        df = pd.read_csv(input_path, header=None)
        df.columns = ["frame", "track_id", "x", "y", "w", "h", "conf", "a", "b", "c"]

        # Count the number of frames per track_id
        track_lengths = df.groupby("track_id")["frame"].count().reset_index()
        track_lengths.columns = ["tracker_id", "frame_count"]

        # Sort by frame_count (desc) then tracker_id (asc)
        df_sorted = track_lengths.sort_values(
            by=["frame_count", "tracker_id"],
            ascending=[False, True]
        ).reset_index(drop=True)

        # Plot bar chart
        plt.figure(figsize=(10, 5))
        x_labels = df_sorted["tracker_id"].astype(str)
        y_vals = df_sorted["frame_count"]

        plt.bar(x_labels, y_vals)
        plt.xlabel("tracker_id")
        plt.ylabel("frame_count")
        plt.title(f"Frame Count by tracker_id (sorted) — {tracker_name} / {seq_name}")

        # To show values on top of bars, uncomment below:
        # for i, v in enumerate(y_vals):
        #     plt.text(i, v, str(v), ha="center", va="bottom")

        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"Saved plot to {png_path}")
