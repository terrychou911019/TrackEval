
from ._base_metric import _BaseMetric
from .. import _timing


class Count(_BaseMetric):
    """Class which simply counts the number of tracker and gt detections and ids."""
    def __init__(self, config=None):
        super().__init__()
        self.integer_fields = ['Dets', 'GT_Dets', 'IDs', 'GT_IDs']
        self.float_fields = ['ID_Ratio']
        self.fields = self.float_fields + self.integer_fields
        self.summary_fields = self.fields

    @staticmethod
    def _calc_id_ratio(num_tracker_ids, num_gt_ids):
        denom = num_tracker_ids + num_gt_ids
        if denom == 0:
            return 0.0
        return 1.0 - abs(num_tracker_ids - num_gt_ids) / denom

    @_timing.time
    def eval_sequence(self, data):
        """Returns counts for one sequence"""
        # Get results
        num_tracker_ids = data['num_tracker_ids']
        num_gt_ids = data['num_gt_ids']
        res = {'ID_Ratio': self._calc_id_ratio(num_tracker_ids, num_gt_ids),
               'Dets': data['num_tracker_dets'],
               'GT_Dets': data['num_gt_dets'],
               'IDs': num_tracker_ids,
               'GT_IDs': num_gt_ids,
               'Frames': data['num_timesteps']}
        return res

    def combine_sequences(self, all_res):
        """Combines metrics across all sequences"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res['ID_Ratio'] = self._calc_id_ratio(res['IDs'], res['GT_IDs'])
        return res

    def combine_classes_class_averaged(self, all_res, ignore_empty_classes=None):
        """Combines metrics across all classes by averaging over the class values"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res['ID_Ratio'] = self._calc_id_ratio(res['IDs'], res['GT_IDs'])
        return res

    def combine_classes_det_averaged(self, all_res):
        """Combines metrics across all classes by averaging over the detection values"""
        res = {}
        for field in self.integer_fields:
            res[field] = self._combine_sum(all_res, field)
        res['ID_Ratio'] = self._calc_id_ratio(res['IDs'], res['GT_IDs'])
        return res
