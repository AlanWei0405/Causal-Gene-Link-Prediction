import numpy as np
from torch.nn import Module
from torch_geometric.nn import GAE
from typing import Optional, Tuple, Any
import torch
from torch import Tensor
from torch_geometric.utils import negative_sampling
from sklearn.metrics import average_precision_score, roc_auc_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt


EPS = 1e-15
MAX_LOGSTD = 10


class HeteroDecoder(torch.nn.Module):

    def forward(
            self,
            z_0: Tensor,
            z_1: Tensor,
            edge_index: Tensor,
            sigmoid: bool = True,
    ) -> Tensor:
        value = (z_0[edge_index[0]] * z_1[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value

    # def forward_all(self, z: Tensor, sigmoid: bool = True) -> Tensor:
    #     r"""Decodes the latent variables :obj:`z` into a probabilistic dense
    #     adjacency matrix.
    #
    #     Args:
    #         z (torch.Tensor): The latent space :math:`\mathbf{Z}`.
    #         sigmoid (bool, optional): If set to :obj:`False`, does not apply
    #             the logistic sigmoid function to the output.
    #             (default: :obj:`True`)
    #     """
    #     adj = torch.matmul(z, z.t())
    #     return torch.sigmoid(adj) if sigmoid else adj

class HeteroGAE(GAE):
    def __init__(self, encoder, decoder: Optional[Module] = None):
        super().__init__(encoder)
        self.decoder = HeteroDecoder() if decoder is None else decoder
        GAE.reset_parameters(self)

    def decode(self, z_0, z_1, edge_label_index):
        return self.decoder(z_0, z_1, edge_label_index)

    def recon_loss(self, z: Tensor, pos_edge_index: Tensor,
                   neg_edge_index: Optional[Tensor] = None) -> Tensor:

        z_0 = z['gene']
        z_1 = z['disease']

        pos_loss = -torch.log(
            self.decoder(z_0, z_1, pos_edge_index, sigmoid=True) + EPS).mean()

        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index)
        neg_loss = -torch.log(1 -
                              self.decoder(z_0, z_1, neg_edge_index, sigmoid=True) +
                              EPS).mean()

        return pos_loss + neg_loss

    def test(self, z: Tensor, pos_edge_index: Tensor,
             neg_edge_index: Tensor) -> tuple[float, float, Any]:

        z_0 = z['gene']
        z_1 = z['disease']

        pos_y = z_0.new_ones(pos_edge_index.size(1))
        neg_y = z_1.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z_0, z_1, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z_0, z_1, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        # print(f"\n", classification_report(y, pred.round()))
        precision, recall, thresholds = precision_recall_curve(y, pred)
        auc = roc_auc_score(y, pred)
        ap = average_precision_score(y, pred)
        # Calculate F1-score for each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall)

        # Find the threshold that maximizes F1-score
        f1_max = np.max(f1_scores)
        best_threshold = thresholds[np.argmax(f1_scores)]
        print(f"Max F1-Score: {f1_max}")
        print(f"Best Threshold (Max F1-Score): {best_threshold}")

        return auc, ap, f1_max

    def test_top_k(self, z: Tensor, pos_edge_index: Tensor, neg_edge_index: Tensor, k=3) -> dict:
        z_0 = z['gene']
        z_1 = z['disease']

        # Get predictions for all positive and negative edges
        pos_pred = self.decoder(z_0, z_1, pos_edge_index, sigmoid=False)
        # neg_pred = self.decoder(z_0, z_1, neg_edge_index, sigmoid=False)

        # Combine predictions and indices for sorting
        pos_edges = list(zip(pos_edge_index[0].tolist(), pos_edge_index[1].tolist()))
        # neg_edges = list(zip(neg_edge_index[0].tolist(), neg_edge_index[1].tolist()))

        all_predictions = []
        for i in range(pos_pred.size(0)):
            all_predictions.append((pos_edges[i], pos_pred[i].item(), 1))
        # for i in range(neg_pred.size(0)):
        #     all_predictions.append((neg_edges[i], neg_pred[i].item(), 0))

        # Group predictions by disease
        disease_predictions = {}
        for (gene, disease), score, label in all_predictions:
            if disease not in disease_predictions:
                disease_predictions[disease] = []
            disease_predictions[disease].append((gene, score, label))

        precisions, recalls = [], []
        total_diseases = len(disease_predictions)

        # Calculate metrics for each disease
        for disease, predictions in disease_predictions.items():

            # Sort predictions by score in descending order
            predictions_sorted = sorted(predictions, key=lambda x: x[1], reverse=True)

            # Extract top k predictions
            top_k_predictions = predictions_sorted[:k]
            top_k_genes = set(gene for gene, _, _ in top_k_predictions)

            # Get true positive genes
            true_positives = set(gene for gene, _, label in predictions_sorted if label == 1)

            # Safeguard against division by zero
            num_true_positives_in_top_k = len(top_k_genes & true_positives)
            num_top_k = len(top_k_genes)
            num_true_positives = len(true_positives)

            # Compute precision and recall
            precision = (num_true_positives_in_top_k / num_top_k if num_top_k > 0 else 0)
            recall = (num_true_positives_in_top_k / num_true_positives if num_true_positives > 0 else 0)

            precisions.append(precision)
            recalls.append(recall)

        # Average metrics across diseases
        precision_avg = np.mean(precisions)
        recall_avg = np.mean(recalls)
        f1 = (
            2 * precision_avg * recall_avg / (precision_avg + recall_avg)
            if (precision_avg + recall_avg) > 0
            else 0
        )

        return {"Precision": precision_avg, "Recall": recall_avg, "F1-score": f1}



