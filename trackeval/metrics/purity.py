
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from ._base_metric import _BaseMetric
from .. import _timing


class Purity(_BaseMetric):
    """Tracklet-level Purity (multi-alpha)
    
    """

    def __init__(self, config=None):
        super().__init__()
        self.plottable = True     
        self.array_labels = np.arange(0.05, 0.99, 0.05)
        self.integer_array_fields = []                        
        self.float_array_fields = ['Purity']                  
        self.float_fields = ['Purity(0)']                    
        self.fields = self.float_array_fields + self.integer_array_fields + self.float_fields
        self.summary_fields = self.float_array_fields + self.float_fields

    @_timing.time
    def eval_sequence(self, data):
        """Calculates the Purity metrics for one sequence"""
       
        # Initialise results
        res = {}
        for field in self.float_array_fields + self.integer_array_fields:
            res[field] = np.zeros((len(self.array_labels)), dtype=np.float)
        for field in self.float_fields:
            res[field] = 0

        # Return result quickly if tracker or gt sequence is empty
        if data['num_tracker_dets'] == 0 or data['num_gt_dets'] == 0:
            return res

        # Variables counting global association
        potential_matches_count = np.zeros((data['num_gt_ids'], data['num_tracker_ids']))
        gt_id_count = np.zeros((data['num_gt_ids'], 1))
        tracker_id_count = np.zeros((1, data['num_tracker_ids']))

        # First loop through each timestep and accumulate global track information.
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            # Count the potential matches between ids in each timestep
            # These are normalised, weighted by the match similarity.
            similarity = data['similarity_scores'][t]
            sim_iou_denom = similarity.sum(0)[np.newaxis, :] + similarity.sum(1)[:, np.newaxis] - similarity
            sim_iou = np.zeros_like(similarity)
            sim_iou_mask = sim_iou_denom > 0 + np.finfo('float').eps
            sim_iou[sim_iou_mask] = similarity[sim_iou_mask] / sim_iou_denom[sim_iou_mask]
            potential_matches_count[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] += sim_iou

            # Calculate the total number of dets for each gt_id and tracker_id.
            gt_id_count[gt_ids_t] += 1
            tracker_id_count[0, tracker_ids_t] += 1

        # Calculate overall jaccard alignment score (before unique matching) between IDs
        global_alignment_score = potential_matches_count / (gt_id_count + tracker_id_count - potential_matches_count)
        matches_counts = [np.zeros_like(potential_matches_count) for _ in self.array_labels]
        
        # Initialize total TP counts for each alpha
        total_tp = np.zeros(len(self.array_labels))

        # Calculate matches for each timestep  
        for t, (gt_ids_t, tracker_ids_t) in enumerate(zip(data['gt_ids'], data['tracker_ids'])):
            if len(gt_ids_t) == 0 or len(tracker_ids_t) == 0:
                continue

            # Get matching scores between pairs of dets for optimizing HOTA
            similarity = data['similarity_scores'][t]
            score_mat = global_alignment_score[gt_ids_t[:, np.newaxis], tracker_ids_t[np.newaxis, :]] * similarity

            # Hungarian algorithm to find best matches
            match_rows, match_cols = linear_sum_assignment(-score_mat)

            # Accumulate TP counts for each alpha 
            for a, alpha in enumerate(self.array_labels):
                actually_matched_mask = similarity[match_rows, match_cols] >= alpha - np.finfo('float').eps
                alpha_match_rows = match_rows[actually_matched_mask]
                alpha_match_cols = match_cols[actually_matched_mask]
                num_matches = len(alpha_match_rows)

                # If no matches found, continue to next timestep
                if num_matches == 0:
                    continue
                matches_counts[a][gt_ids_t[alpha_match_rows], tracker_ids_t[alpha_match_cols]] += 1
                total_tp[a] += num_matches
                
        # Calculate Purity curve for each alpha 
        for a, alpha in enumerate(self.array_labels):
            if total_tp[a] == 0:
                res['Purity'][a] = 0.0
                continue
            
            # matches_count: show how many times each tracker_id matched with each gt_id 
            matches_count = matches_counts[a]         # shape: (num_gt_ids, num_tracker_ids)
            tp_per_trk = matches_count.sum(0)         # TP count for each tracker_id
            main_tp_per_trk = matches_count.max(0)    # main TP count for each tracker_id

            # Filter out tracker_ids with no TP
            valid_mask = tp_per_trk > 0

            # Calculate Purity for each tracker_id
            purity_per_trk = np.zeros_like(tp_per_trk, dtype=float)
            purity_per_trk[valid_mask] = main_tp_per_trk[valid_mask] / tp_per_trk[valid_mask]
            res['Purity'][a] = float(np.sum(purity_per_trk * tp_per_trk) / total_tp[a])

        # Store results
        res['Purity(0)'] = res['Purity'][0]
        res['total_tp'] = total_tp.copy()

        # Use the first alpha (0) to calculate per-tracklet results
        a0 = 0  # self.array_labels[0]
        matches_count_0   = matches_counts[a0]           # shape: (num_gt_ids, num_tracker_ids)
        tp_per_trk_0      = matches_count_0.sum(0)       # TP count for each tracker_id
        main_tp_per_trk_0 = matches_count_0.max(0)       # main TP count for each tracker_id
        main_gt_per_trk_0 = matches_count_0.argmax(0)    # main GT id for each tracker_id

        # Filter out tracker_ids with no TP
        valid_mask_0 = tp_per_trk_0 > 0

        # Calculate Purity for each tracker_id
        purity_per_trk_0 = np.zeros_like(tp_per_trk_0, dtype=float)
        purity_per_trk_0[valid_mask_0] = main_tp_per_trk_0[valid_mask_0] / tp_per_trk_0[valid_mask_0]

        # Store per-tracklet results
        res['per_trk_Purity'] = purity_per_trk_0
        res['per_trk_TP']     = tp_per_trk_0
        res['per_trk_MainGT'] = np.where(valid_mask_0, main_gt_per_trk_0, -1)
        res['per_trk_MainTP'] = main_tp_per_trk_0

        # Store original tracker IDs 
        num_trk = matches_count_0.shape[1]
        assert len(data['orig_trk_ids']) == num_trk
        res['orig_trk_ids'] = np.asarray(data['orig_trk_ids'], dtype=int)

        # Store original GT IDs
        num_gt = matches_count_0.shape[0]
        assert len(data['orig_gt_ids']) == num_gt, "orig_gt_ids length mismatch"
        res['orig_gt_ids'] = np.asarray(data['orig_gt_ids'], dtype=int)

        return res

    def combine_sequences(self, all_res):
        if not all_res:
            zeros = np.zeros(len(self.array_labels))
            return {'Purity': zeros, 'Purity(0)': 0.0}

        # Combine results across all sequences
        purity  = []
        total_tp = []
        for res in all_res.values():
            purity.append(res['Purity'])
            total_tp.append(res['total_tp'])

        purity         = np.stack(purity,  axis=0)     # (N_seq , len α)
        total_tp       = np.stack(total_tp, axis=0)    # (N_seq , len α)
        tracker_tp_sum = np.sum(total_tp, axis=0)
        main_gt_sum    = np.sum(purity * total_tp, axis=0)       
        purity_comb    = np.divide(main_gt_sum,
                                tracker_tp_sum,
                                out=np.zeros_like(main_gt_sum, dtype=float),
                                where=tracker_tp_sum > 0)

        return {'Purity': purity_comb, 'Purity(0)': purity_comb[0]}

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=False):
        vals = [v['Purity'] for v in all_res.values()
                if not ignore_empty_classes or np.any(v['Purity'] > 0)]
        if not vals:
            return {'Purity': np.zeros(len(self.array_labels)), 'Purity(0)': 0.0}
        arr = np.mean(vals, axis=0)
        return {'Purity': arr, 'Purity(0)': arr[0]}

    def combine_classes_det_averaged(self, all_res):
        return self.combine_classes_class_averaged(all_res, ignore_empty_classes=False)

    def plot_single_tracker_results(self, table_res, tracker, cls, output_folder):
        from matplotlib import pyplot as plt

        res = table_res['COMBINED_SEQ']
        plt.plot(self.array_labels, res['Purity'], 'r')
        plt.xlabel('alpha')
        plt.ylabel('Purity')
        plt.title(tracker + ' - ' + cls)
        plt.axis([0, 1, 0, 1])
        plt.legend(['Purity (' + str(np.round(np.mean(res['Purity']), 2)) + ')'], loc='lower left')

        out_file = os.path.join(output_folder, cls + '_purity_plot.pdf')
        os.makedirs(os.path.dirname(out_file), exist_ok=True)
        plt.savefig(out_file)
        plt.savefig(out_file.replace('.pdf', '.png'))
        plt.clf()

    def save_per_tracklet_csv(self, res, seq_name, tracker_name, output_folder):
        """
        CSV format: tracker_id, purity, tp, main_gt, main_tp
        Also saves a sorted bar chart of Purity per tracklet.
        """
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import os

        if ('per_trk_Purity' not in res) or ('per_trk_TP' not in res):
            return

        # Map internal tracker IDs to original tracker IDs
        num_trk = len(res['per_trk_Purity'])
        orig_trk_ids = res.get('orig_trk_ids', np.arange(num_trk, dtype=int))
        assert len(orig_trk_ids) == num_trk, "orig_trk_ids length mismatch"

        # Map internal GT IDs to original GT IDs
        internal_main_gt = res['per_trk_MainGT'].astype(int)
        orig_gt_ids = res['orig_gt_ids'].astype(int)
        orig_main_gt = np.full_like(internal_main_gt, fill_value=-1)
        valid_m = internal_main_gt >= 0
        orig_main_gt[valid_m] = orig_gt_ids[internal_main_gt[valid_m]]

        # Create DataFrame for results
        df = pd.DataFrame({
            'tracker_id': orig_trk_ids.astype(int),
            'purity':     res['per_trk_Purity'].astype(float),
            'tp':         res['per_trk_TP'].astype(int),
            'main_gt':    orig_main_gt.astype(int),
            'main_tp':    res['per_trk_MainTP'].astype(int),
            # (optional) if you want to keep the internal GT ID
            # 'internal_main_gt': internal_main_gt.astype(int),
        })

        # Filter out tracker_ids with no TP
        df = df[df['tp'] > 0].copy()

        # Save to CSV
        out_dir = os.path.join(output_folder)
        os.makedirs(out_dir, exist_ok=True)
        out_path_csv = os.path.join(out_dir, f'{seq_name}.csv')
        df.to_csv(out_path_csv, index=False)

        # Sort by purity
        df_sorted = df.sort_values('purity', ascending=True)

        # Plot Purity per tracklet
        plt.figure(figsize=(10, 4))
        plt.bar(df_sorted['tracker_id'].astype(str), df_sorted['purity'])
        plt.ylim(0, 1.05)
        plt.xlabel('Tracklet ID')
        plt.ylabel('Purity')
        plt.title(f'Purity per Tracklet - {seq_name}')
        plt.xticks(rotation=90)
        plt.tight_layout()

        # Save to PNG
        out_path_png = os.path.join(out_dir, f'{seq_name}.png')
        plt.savefig(out_path_png, bbox_inches='tight')
        plt.close()
