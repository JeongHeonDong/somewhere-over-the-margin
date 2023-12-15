import torch

from pytorch_metric_learning.reducers import AvgNonZeroReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction


class TripletMarginLoss(BaseMetricLossFunction):
    """
    Args:
        margin: The desired difference between the anchor-positive distance and the
                anchor-negative distance.
        swap: Use the positive-negative distance instead of anchor-negative distance,
              if it violates the margin more.
        margin_activation: Select last activation for loss.
            - relu (default)
            - soft_plus (log_sum_exp)
            - leaky_relu
            - hard_swish
            - selu
            - celu
            - gelu
            - silu
            - mish
    """

    def __init__(
        self,
        margin=0.05,
        swap=False,
        margin_activation="relu",
        triplets_per_anchor="all",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.margin = margin
        self.swap = swap
        self.margin_activation = margin_activation
        self.triplets_per_anchor = triplets_per_anchor
        self.add_to_recordable_attributes(
            list_of_names=["margin"], is_stat=False)

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        indices_tuple = lmu.convert_to_triplets(
            indices_tuple, labels, ref_labels, t_per_anchor=self.triplets_per_anchor
        )
        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        ap_dists = mat[anchor_idx, positive_idx]
        an_dists = mat[anchor_idx, negative_idx]
        if self.swap:
            pn_dists = mat[positive_idx, negative_idx]
            an_dists = self.distance.smallest_dist(an_dists, pn_dists)

        current_margins = self.distance.margin(ap_dists, an_dists)
        violation = current_margins + self.margin
        if self.margin_activation == "soft_plus":
            loss = torch.nn.functional.softplus(violation)
        elif self.margin_activation == "relu":
            loss = torch.nn.functional.relu(violation)
        elif self.margin_activation == "leaky_relu":
            loss = torch.nn.functional.leaky_relu(violation)
        elif self.margin_activation == "hard_swish":
            loss = torch.nn.functional.hardswish(violation)
        elif self.margin_activation == "selu":
            loss = torch.nn.functional.selu(violation)
        elif self.margin_activation == "celu":
            loss = torch.nn.functional.celu(violation)
        elif self.margin_activation == "gelu":
            loss = torch.nn.functional.gelu(violation)
        elif self.margin_activation == "silu":
            loss = torch.nn.functional.silu(violation)
        elif self.margin_activation == "mish":
            loss = torch.nn.functional.mish(violation)
        else:
            loss = violation

        return {
            "loss": {
                "losses": loss,
                "indices": indices_tuple,
                "reduction_type": "triplet",
            }
        }

    def get_default_reducer(self):
        return AvgNonZeroReducer()
