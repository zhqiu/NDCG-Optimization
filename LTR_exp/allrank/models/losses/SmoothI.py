"""
    SmoothI
    source: https://github.com/ygcinar/SmoothI/blob/main/src/losses.py
"""

import torch
import torch.nn as nn

from data.dataset_loading import PADDED_Y_VALUE
from models.losses import DEFAULT_EPS
from models.model_utils import get_torch_device


class ListwiseSmoothINDCGKLoss(nn.Module):
    """
    SmoothI-NDCG@K or SmoothI-NDCG Loss
    """

    def __init__(self, alpha=None, delta=None, K=10, rank_list_length=10, 
                    ndcg_loss=True, stop_grad=True, epsilon=DEFAULT_EPS):
        """
        Args:
            alpha: (float) value of hyperparameter alpha
            delta: (float) value of hyperparameter delta
            K: (int) threshold value of top K documents
            rank_list_length: (int) max. number of documents for a query
            ndcg_loss: (bool) if True, returns ndcg_loss; otherwise ndcg estimate
            device: (string) cpu or cuda device
            stop_grad: (bool) if True, applies stop_grad: it is used to overcome
             the gradient flow issues
            epsilon: (float) very small value
        """
        super(ListwiseSmoothINDCGKLoss, self).__init__()
        self.stop_grad = stop_grad
        if K:  # NDCG@K
            self.K = K
            self.update_K = False
        else:  # NDCG if K is zero
            self.K = rank_list_length
            self.update_K = True
        self.device = get_torch_device()
        self.alpha = torch.tensor([alpha], device=self.device)  # alpha hyperparameter
        self.delta = torch.tensor([delta], device=self.device)  # delta hyperparameter
        self.fn_pos = self.make_pos_by_subtraction
        if ndcg_loss:  # ndcg loss
            self.return_loss = True
        else:  # ndcg estimate
            self.return_loss = False
        self.dcg_denominator = torch.log2(torch.arange(2., self.K + 2.)).to(self.device)  # log2(1+k)
        self.epsilon = epsilon

    def make_pos_by_subtraction(self, s):
        """
        Shift the scores to ensure positivity
        Args:
            s: torch tensor of shape (batch_size, max.#ofdocs for a query) corresponds
            to scores -- output of NN logits
        """
        min_vals, _ = torch.min(s, -1)  # min. logit values
        s = s.t() - min_vals  # .shape (ll, bs)
        return s.t()  # .shape (bs, ll)

    def ideal_rank(self, label):
        """
        sorts the relevance labels descending order, and returns relevance of
        the ideal (max.) top K items
        Args:
            label: torch tensor of shape (batch_size, max.#ofdocs for a query)

        Returns:
            sorted_label: torch tensor of shape (batch_size, K)
        """
        sorted_label, sort_indices = torch.sort(label, descending=True)  # sort the labels
        return sorted_label[:, :self.K]  # .shape: (bs, K) # returns relevance of ideal top K items

    def discounted_cum_gain(self, y):
        """
        computes discounted cumulative gain
        Args:
            y: torch tensor of shape (batch_size, K) - relevance labels

        Returns:
           a torch tensor of shape (batch_size, K) which corresponds discounted cumulative
            gain per query
        """
        numerator = torch.Tensor([2.0]).to(self.device).pow(y)
        dcg = numerator / self.dcg_denominator[:self.K]  # dcg.shape: (bs,K)
        return torch.sum(dcg, -1) + self.epsilon  # .shape: (bs,)

    def forward(self, s, label, padded_value_indicator=PADDED_Y_VALUE):
        """
        Args:
            s: torch tensor of shape (batch_size, max.#ofdocs for a query) corresponds to \\
            scores -- output of NN logits
            label: torch tensor of shape (batch_size, max.#ofdocs for a query) relevance labels

        Returns:
            NDCG(@K) loss for s and label
        """
        s = s.clone()
        label = label.clone()

        padded_mask = label == padded_value_indicator
        s[padded_mask] = float("-inf")
        label[padded_mask] = float("-inf")

        bs, ll = s.shape  # s.shape: (batchsize, listlength)
        if self.update_K:
            self.K = ll  # maximizing ndcg - whole list
        s = self.fn_pos(s)  # ensure that scores are positive
        # Compute the approximate rank indicators [I_j^{r,alpha}]_{j,r}
        prod = torch.ones(1, device=self.device)  # initialize prod with 1
        rel_k = torch.zeros((bs, self.K), device=self.device)
        for k in range(self.K):
            B_ = self.alpha * s * prod   #
            maxB_, _ = torch.max(B_, -1)
            diff_B = (B_.t() - maxB_).t()
            approx_inds = torch.softmax(diff_B, -1)  # I_j^{k,alpha} rank indicator estimate
            rel_k[:, k] = torch.sum(label * approx_inds, -1)  # rel_{q}(j_{k}) relevance
            # Update the logits [S_j * prod_{l=1}^k (1-I_j^{l,alpha}-delta)]_j
            if self.stop_grad:
                # Use the stop-gradient trick on the recursive component
                prod = prod * (1 - approx_inds - self.delta).detach()
            else:
                prod = prod * (1 - approx_inds - self.delta)
        idcg = self.discounted_cum_gain(self.ideal_rank(label))
        dcg = self.discounted_cum_gain(rel_k)
        ndcg = dcg / idcg  # NDCG@K_{q}^{alpha}
        if self.return_loss:
            return torch.sum(1 - ndcg)
        else:
            return torch.sum(ndcg)