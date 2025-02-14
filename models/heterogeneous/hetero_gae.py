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
             neg_edge_index: Tensor) -> tuple[float, float, Any, Any]:

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

        return auc, ap, f1_max, best_threshold

    def test_top_k(self, z: Tensor, test_edge_index: Tensor, pos_edge_label_index: Tensor, k) -> dict:

        disease_embeddings = z['disease']  # [num_diseases, latent_dim]
        gene_embeddings = z['gene']  # [num_genes, latent_dim]

        scores = torch.mm(disease_embeddings, gene_embeddings.T)

        # 3. Get test edge index for ('disease', 'associated_with', 'gene')
        num_diseases = disease_embeddings.shape[0]
        num_genes = gene_embeddings.shape[0]

        # 4. Create test mask
        test_mask = torch.zeros((num_diseases, num_genes), dtype=torch.bool)
        test_mask[test_edge_index[1], test_edge_index[0]] = True

        # 5. Initialize metrics
        precision, recall= 0.0, 0.0
        valid_diseases = []

        for disease in range(num_diseases):
            # True pathogenic genes for this disease
            true_genes = pos_edge_label_index[0][pos_edge_label_index[1] == disease].tolist()
            if len(true_genes) == 0:
                continue
            valid_diseases.append(disease)

        true_pos_list = []
        # 6. Perform Top@i evaluation for each disease
        for disease in valid_diseases:
            # Get scores for this disease
            disease_scores = scores[disease]  # [num_genes]
            # Use the test_mask to identify relevant gene indices
            valid_gene_indices = torch.where(test_mask[disease] == False)[0]

            # Get scores only for valid (test) gene indices
            valid_scores = disease_scores[valid_gene_indices]
            # Sort gene indices by score
            top_indices = torch.argsort(valid_scores, descending=True)[:k]

            # R(d)
            top_genes = valid_gene_indices[top_indices].tolist()
            # T(d) True pathogenic genes for this disease
            true_genes = pos_edge_label_index[0][pos_edge_label_index[1] == disease].tolist()

            # Compute the intersection between T(d) and R(d)
            true_positives = len(set(top_genes) & set(true_genes))

            # Precision and recall for this disease
            local_precision = true_positives / k
            local_recall = true_positives / len(true_genes)

            # Update global metrics
            precision += local_precision
            recall += local_recall

            # Association Precision (AP)
            true_pos_list.append(true_positives)

        # Compute the overall metrics
        num_d_set = float(len(valid_diseases))

        precision /= num_d_set
        recall /= num_d_set

        f1 = (2 * precision * recall) / (precision + recall + 1e-8)
        ap = np.sum(true_pos_list) / (num_d_set * k)

        return {
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "AP": ap
        }
