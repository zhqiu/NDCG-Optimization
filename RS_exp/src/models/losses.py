"""
    pointwise, pairwise and listwise losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


"""
    useful functions
"""
def sinkhorn_scaling(mat, tol=1e-6, max_iter=10, eps=1e-10):
    """
    Sinkhorn scaling procedure.
    :param mat: a tensor of square matrices of shape N x M x M, where N is batch size
    :param tol: Sinkhorn scaling tolerance
    :param max_iter: maximum number of iterations of the Sinkhorn scaling
    :return: a tensor of (approximately) doubly stochastic matrices
    """
    for _ in range(max_iter):
        mat = mat / mat.sum(dim=1, keepdim=True).clamp(min=eps)
        mat = mat / mat.sum(dim=2, keepdim=True).clamp(min=eps)

        if torch.max(torch.abs(mat.sum(dim=2) - 1.)) < tol and torch.max(torch.abs(mat.sum(dim=1) - 1.)) < tol:
            break

    return mat


def deterministic_neural_sort(s, tau, device):
    """
    Deterministic neural sort.
    Code taken from "Stochastic Optimization of Sorting Networks via Continuous Relaxations", ICLR 2019.
    :param s: values to sort, shape [batch_size, slate_length, 1]
    :param tau: temperature for the final softmax function
    :return: approximate permutation matrices of shape [batch_size, slate_length, slate_length]
    """

    batch_size, n = s.size()[:2]
    one = torch.ones((n, 1), dtype=torch.float32, device=device)

    A_s = torch.abs(s - s.permute(0, 2, 1))
    B = torch.matmul(A_s, torch.matmul(one, torch.transpose(one, 0, 1)))
    scaling = (n + 1 - 2 * (torch.arange(n) + 1)).type(torch.float32).to(device)  # type: ignore
    C = torch.matmul(s, scaling.unsqueeze(0))

    P_max = (C - B).permute(0, 2, 1)
    sm = torch.nn.Softmax(-1)
    P_hat = sm(P_max / tau)
    return P_hat


# BPR loss (RankNet loss)
def bpr_loss(pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> torch.Tensor:
    """
    pos_pred: [batch_size * num_pos]
    neg_pred: [batch_size * num_pos, num_neg]
    """
    loss = -(pos_pred[:, None] - neg_pred).sigmoid().log().sum(dim=1).mean()
    return loss


# BPR_max loss 
# from 'Recurrent Neural Networks with Top-k Gains for Session-based Recommendations'
def bpr_max_loss(pos_pred: torch.Tensor, neg_pred: torch.Tensor) -> torch.Tensor:
    """
    pos_pred: [batch_size * num_pos]
    neg_pred: [batch_size * num_pos, num_neg]
    """
    neg_softmax = (neg_pred - neg_pred.max()).softmax(dim=1)
    loss = -((pos_pred[:, None] - neg_pred).sigmoid() * neg_softmax).sum(dim=1).log().mean()
    return loss


# ListNet loss
def listnet_loss(predictions: torch.Tensor, ratings: torch.Tensor, device, eps=1e-10) -> torch.Tensor:
    """
    predictions: [batch_size, num_pos + num_neg]
    ratings:     [batch_size, num_pos]
    """
    batch_size, slate_length = predictions.size()
    num_pos = ratings.size()[1]
    ratings = torch.cat([ratings.float(), torch.zeros(batch_size, slate_length-num_pos, device=device)], dim=1) # [batch_size, num_pos + num_neg]
    
    pred_smax = F.softmax(predictions, dim=1)
    ratings_smax = F.softmax(ratings, dim=1)

    return torch.mean(-torch.sum(ratings_smax * torch.log(pred_smax + eps), dim=1))


# ListMLE loss
def listmle_loss(predictions: torch.Tensor, ratings: torch.Tensor, device, eps=1e-10) -> torch.Tensor:
    """
    predictions: [batch_size, num_pos + num_neg]
    ratings:     [batch_size, num_pos]
    """
    batch_size, slate_length = predictions.size()
    num_pos = ratings.size()[1]
    ratings = torch.cat([ratings.float(), torch.zeros(batch_size, slate_length-num_pos, device=device)], dim=1) # [batch_size, num_pos + num_neg]

    pred_sorted = torch.gather(predictions, dim=1, index=ratings.sort(descending=True, dim=1)[1])

    max_pred_values, _ = pred_sorted.max(dim=1, keepdim=True)
    pred_sorted_minus_max = pred_sorted - max_pred_values

    cumsums = torch.cumsum(pred_sorted_minus_max.exp().flip(dims=[1]), dim=1).flip(dims=[1])
    observation_loss = torch.log(cumsums + eps) - pred_sorted_minus_max

    return torch.mean(torch.sum(observation_loss, dim=1))


# Lambda loss family
def lambda_loss(predictions: torch.Tensor, ratings: torch.Tensor, device, weighing_scheme=None, k=None, sigma=1.0):
    """
    predictions:      [batch_size, num_pos + num_neg]
    ratings:          [batch_size, num_pos]
    weighing_scheme:  a string corresponding to a name of one of the weighing schemes
    k:                rank at which the loss is truncated
    sigma:            score difference weight used in the sigmoid function
    """
    batch_size, slate_length = predictions.size()
    num_pos = ratings.size()[1]
    ratings = torch.cat([ratings.float(), torch.zeros(batch_size, slate_length-num_pos, device=device)], dim=1) # [batch_size, num_pos + num_neg]
    if k is None:
        k = slate_length

    pred_sorted, indices_pred = predictions.sort(descending=True, dim=1)
    ratings_sorted, _ = ratings.sort(descending=True, dim=1)

    ratings_sorted_by_pred = torch.gather(ratings, dim=1, index=indices_pred)
    ratings_diffs = ratings_sorted_by_pred[:, :, None] - ratings_sorted_by_pred[:, None, :]  # x_ij = x_i - x_j

    pairs_mask = torch.isfinite(ratings_diffs)
    if weighing_scheme != "ndcgLoss1_scheme":
        pairs_mask = pairs_mask & (ratings_diffs > 0)

    ndcg_at_k_mask = torch.zeros((slate_length, slate_length), dtype=torch.bool, device=device)
    ndcg_at_k_mask[:k, :k] = 1

    pos_idxs = torch.arange(1, slate_length+1).to(device)
    D = torch.log2(1.0 + pos_idxs.float())[None, :]
    maxDCGs = torch.sum(((2 ** ratings_sorted - 1) / D)[:, :k], dim=1)
    G = (2 ** ratings_sorted_by_pred - 1) / maxDCGs[:, None]

    if weighing_scheme is None:
        weights = 1.0
    else:
        weights = globals()[weighing_scheme](G, D)

    pred_diffs = pred_sorted[:, :, None] - pred_sorted[:, None, :]
    weighted_probs = torch.sigmoid(sigma * pred_diffs) ** weights
    losses = torch.log2(weighted_probs)
    loss = -torch.sum(losses[pairs_mask & ndcg_at_k_mask])

    return loss



"""
    weighing schemes used in lambda_loss
"""
def ndcgLoss1_scheme(G, D):
    return (G / D)[:, :, None]

def ndcgLoss2_scheme(G, D):
    pos_idxs = torch.arange(1, G.shape[1] + 1, device=G.device)
    delta_idxs = torch.abs(pos_idxs[:, None] - pos_idxs[None, :])
    deltas = torch.abs(torch.pow(torch.abs(D[0, delta_idxs - 1]), -1.) - torch.pow(torch.abs(D[0, delta_idxs]), -1.))
    deltas.diagonal().zero_()
    return deltas[None, :, :] * torch.abs(G[:, :, None] - G[:, None, :])

def lambdaRank_scheme(G, D):
    return torch.abs(torch.pow(D[:, :, None], -1.) - torch.pow(D[:, None, :], -1.)) * torch.abs(G[:, :, None] - G[:, None, :])



# PiRank / NeuralNDCG loss, both are based on NeuralSort
def neural_sort_loss(predictions: torch.Tensor, ratings: torch.Tensor, device, temperature=1.0, k=None) -> torch.Tensor:
    """
    predictions: [batch_size, num_pos + num_neg]
    ratings:     [batch_size, num_pos]
    temperature: temperature for NeuralSort algorithm
    k:           rank at which the loss is truncated
    """
    batch_size, slate_length = predictions.size()
    num_pos = ratings.size()[1]
    ratings = torch.cat([ratings.float(), torch.zeros(batch_size, slate_length-num_pos, device=device)], dim=1) # [batch_size, num_pos + num_neg]
    if k is None:
        k = slate_length

    P_hat = deterministic_neural_sort(predictions.unsqueeze(-1), temperature, device)
    P_hat = sinkhorn_scaling(P_hat)               # [batch_size, slate_len, slate_len]
    y_true = (2.0 ** ratings.unsqueeze(-1)) - 1   # [batch_size, slate_len, 1] 

    ground_truth = torch.matmul(P_hat, y_true).squeeze(-1)  # [batch_size, slate_len]
    sorted_ratings = torch.sort(ratings, dim=1, descending=True)[0]
    discounts = (1.0 / torch.log2(torch.arange(slate_length, dtype=torch.float) + 2.0)).to(device)
    discounted_gains = (ground_truth * discounts)
    normalizers = torch.sum((2.0 ** sorted_ratings -1) * discounts, dim=1)

    ndcg = (discounted_gains.sum(dim=1) / normalizers).mean()
    return -1.0 * ndcg


# Approximate NDCG loss
def approx_ndcg_loss(predictions: torch.Tensor, ratings: torch.Tensor, device, eps=1e-10, alpha=1.0) -> torch.Tensor:
    """
    predictions: [batch_size, num_pos + num_neg]
    ratings:     [batch_size, num_pos]
    alpha:       score difference weight used in the sigmoid function
    """
    batch_size, slate_length = predictions.size()
    num_pos = ratings.size()[1]
    ratings = torch.cat([ratings.float(), torch.zeros(batch_size, slate_length-num_pos, device=device)], dim=1) # [batch_size, num_pos + num_neg]
    
    predictions_sorted, indices_pred = predictions.sort(descending=True, dim=-1)
    ratings_sorted, _ = ratings.sort(descending=True, dim=-1)
    predictions_sorted_diffs = predictions_sorted[:, :, None] - predictions_sorted[:, None, :]

    D = torch.log2(torch.arange(slate_length, dtype=torch.float) + 2.0)[None, :].to(device)
    maxDCG = torch.sum((2.0 ** ratings_sorted - 1) / D, dim=-1)
    G = (2.0 ** torch.gather(ratings, dim=1, index=indices_pred) - 1) / maxDCG[:, None]

    approx_D = torch.log2(1.5 + torch.sum(torch.sigmoid(-alpha * predictions_sorted_diffs), dim=-1))
    approx_ndcg = torch.sum(G / approx_D, dim=-1)
    return -approx_ndcg.mean()


"""
    SmoothI
    source: https://github.com/ygcinar/SmoothI/blob/main/src/losses.py
"""
class ListwiseSmoothINDCGKLoss(nn.Module):
    """
    SmoothI-NDCG@K or SmoothI-NDCG Loss
    """

    def __init__(self, alpha=None, delta=None, K=10, rank_list_length=10, ndcg_loss=True,
                 device='cpu', stop_grad=True, epsilon=2.220446049250313e-16):
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
        self.alpha = torch.tensor([alpha], device=device)  # alpha hyperparameter
        self.delta = torch.tensor([delta], device=device)  # delta hyperparameter
        self.fn_pos = self.make_pos_by_subtraction
        if ndcg_loss:  # ndcg loss
            self.return_loss = True
        else:  # ndcg estimate
            self.return_loss = False
        self.dcg_denominator = torch.log2(torch.arange(2., self.K + 2.)).to(device)  # log2(1+k)
        self.device = device  # cpu or cuda device
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

    def forward(self, s, label):
        """
        Args:
            s: torch tensor of shape (batch_size, max.#ofdocs for a query) corresponds to \\
            scores -- output of NN logits
            label: torch tensor of shape (batch_size, max.#ofdocs for a query) relevance labels

        Returns:
            NDCG(@K) loss for s and label
        """
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

