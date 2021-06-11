import torch

from torch import nn
import scipy.linalg as linalg


class PositionalScorer(nn.Module):
    def __init__(self, num_attention_heads: int = 12, clusters_k: int = 5):
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.clusters_k = clusters_k
        self.offset_param = nn.Embedding(self.clusters_k, self.num_attention_heads)
        self.width_param = nn.Embedding(self.clusters_k, self.num_attention_heads)
        self.amplitude_param = nn.Embedding(self.clusters_k, self.num_attention_heads)

    def create_indices(self, seq_len: int):
        """ Creates indices for all the relative distances between
        -seq_len + 1 to seq_len - 1. """
        return torch.arange(-seq_len, seq_len + 1)

    def compute_position_scores(self, seq_len: int):
        """ Takes seq_len and outputs position scores for each relative
        distance. """
        indices = self.create_indices(seq_len)
        interpretable_scores = (
            self.amplitude_param.weight.unsqueeze(-1)
            * torch.exp(
                -torch.abs(self.width_param.weight.unsqueeze(-1))
                * ((self.offset_param.weight.unsqueeze(-1) - indices) ** 2)
            )
        ).sum(axis=0)
        return interpretable_scores

    def expand_to_toeplitz_tensor(self, position_scores, seq_len: int):
        deformed_toeplitz = (
            torch.tensor(
                linalg.toeplitz(
                    range(seq_len - 1, 2 * seq_len - 1), range(seq_len)[::-1]
                )
            )
            .view(-1)
            .long()
            .to(device=position_scores.device)
        )
        expanded_position_scores = torch.stack(
            list(
                torch.gather(position_scores[i], 0, deformed_toeplitz)
                for i in range(self.num_attention_heads)
            )
        ).view(self.num_attention_heads, seq_len, seq_len)
        return expanded_position_scores

    def forward(self, seq_len: int):
        """ Predicts the translation-invariant positional contribution to the
        attention matrix in the self-attention module of transformer models. """
        if not self.clusters_k:
            return torch.zeros((self.num_attention_heads, seq_len, seq_len))
        position_scores_vector = self.compute_position_scores(seq_len)
        position_scores_matrix = self.expand_to_toeplitz_tensor(
            position_scores_vector, seq_len
        )
        return position_scores_matrix

    def visualize(self, seq_len: int, attention_heads=None):
        """Visualization of TISA interpretability by plotting position scores as a function of
        relative distance for each attention head."""
        if attention_heads is None:
            attention_heads = list(range(self.num_attention_heads))
        import matplotlib.pyplot as plt

        x = self.create_indices(seq_len).detach().numpy()
        y = self.compute_position_scores(seq_len).detach().numpy()
        for i in attention_heads:
            plt.plot(x, y[i])
        plt.show()
