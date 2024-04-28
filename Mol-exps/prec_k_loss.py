import torch
import torch.nn as nn
import torch.nn.functional as F
from libauc.losses.surrogate import get_surrogate_loss


"""
    SmoothI
    source: https://github.com/ygcinar/SmoothI/blob/main/src/losses.py
"""
class ListwiseSmoothIPKLoss(nn.Module):
    """
    SmoothI-P@K Loss
    """
    def __init__(self, alpha=None, delta=None, K=10, pk_loss=True, device='cpu', stop_grad=True):
        """
        Args:
            alpha: (float) value of hyperparameter alpha
            delta: (float) value of hyperparameter delta
            K: (int) threshold value of top K documents
            pk_loss: (bool) if True, returns P@K_loss; otherwise P@K estimate
            device: (string) cpu or cuda device
            stop_grad: (bool) if True, applies stop_grad: it is used to overcome
            the gradient flow issues
        """
        super(ListwiseSmoothIPKLoss, self).__init__()
        self.stop_grad = stop_grad
        self.K = K
        self.alpha = torch.tensor([alpha], device=device)
        self.delta = torch.tensor([delta], device=device)
        self.fn_pos = self.make_pos_by_subtraction
        if pk_loss:
            self.return_pk_loss = True
        else:
            self.return_pk_loss = False
        self.device = device
        #

    def make_pos_by_subtraction(self, s):
        """
        Shift the scores to ensure positivity
        Args:
            s: torch tensor of shape (batch_size, max.#ofdocs for a query) corresponds to scores
             (output of NN logits)
        """
        min_vals, _ = torch.min(s, -1)  # min. logit values
        s = s.t() - min_vals  # .shape (ll, bs)
        return s.t()  # .shape (bs, ll)

    def binarize_label(self, label):
        """
        Args:
            label: torch tensor of shape (batch_size, max.#ofdocs for a query) relevance labels

        Returns:
            binary label - 1 if label>0; 0 otherwise
        """
        label_bool = label >= 1
        label_new = torch.zeros_like(label, dtype=torch.bool)
        label_new[label_bool] = 1
        return label_new

    def forward(self, s, label):
        """
        Args:
            s: torch tensor of shape (batch_size, max.#ofdocs for a query) corresponds to scores
            (output of NN logits)
            label: torch tensor of shape (batch_size, max.#ofdocs for a query) relevance labels

        Returns:
            P@K loss for logits s and relevance label
        """
        bs, ll = s.shape  # s.shape: (batchsize, listlength)
        # ensure that scores are positive
        s = self.fn_pos(s)
        # Compute the approximate rank indicators I_jˆ{k,alpha} and relevance rel_{q}^{b}(j_{k})
        prod = torch.ones(1, device=self.device)
        rel_i = torch.zeros((bs, self.K), device=self.device)
        label_b = self.binarize_label(label) # binary labels {0,1}
        for k in range(self.K):
            B_ = self.alpha * s * prod
            maxB_, _ = torch.max(B_, -1)
            diff_B = (B_.t() - maxB_).t()
            approx_inds = torch.softmax(diff_B, -1)  # I_jˆ{k,alpha}
            rel_i[:, k] = torch.sum(label_b * approx_inds, -1)  # rel_{q}^{b}(j_{k})
            # Update the logits [S_j * prod_{l=1}ˆk (1-I_jˆ{l,alpha}-delta)]_j
            if self.stop_grad:
                # Use the stop-gradient trick on the recursive component
                prod = prod * (1 - approx_inds - self.delta).detach()
            else:
                prod = prod * (1 - approx_inds - self.delta)
        pk = torch.sum(rel_i, dim=-1) / self.K  # P@K_{q}^{alpha}
        if self.return_pk_loss:
            return torch.sum(1 - pk)
        else:
            return torch.sum(pk)



class Prec_at_K_Loss(torch.nn.Module):
    def __init__(self, data_len, margin=1.0, gamma=0.9,
                 top_k=-1, surr_loss='squared_hinge',
                 tau_1=0.001, tau_2=0.0001, eta_0 = 0.5,
                 device=None):
        super(Prec_at_K_Loss, self).__init__()
        self.N = data_len
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.s = torch.zeros([]).to(self.device).detach()
        self.lambada = torch.zeros([]).to(self.device).detach()
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = get_surrogate_loss(surr_loss)
        self.top_k = top_k
        self.tau_1 = tau_1
        self.tau_2 = tau_2
        self.eta_0 = eta_0

    def sqh_grad(self, x):
        return -2 * torch.max(self.margin - x, torch.zeros_like(x))

    def forward(self, y_pred, y_true):
        pos_mask = y_true == 1
        pos_pred = y_pred[pos_mask]
        top_k_selector = self.sqh_grad(pos_pred - self.lambada).detach()

        preds_lambda_diffs = y_pred - self.lambada
        grad_lambada = self.top_k/self.N + self.tau_2*self.lambada - torch.mean(torch.sigmoid(preds_lambda_diffs / self.tau_1))
        self.lambada = self.lambada - self.eta_0 * grad_lambada.detach()

        temp_term = torch.sigmoid(preds_lambda_diffs / self.tau_1) * (1 - torch.sigmoid(preds_lambda_diffs / self.tau_1)) / self.tau_1
        L_lambda_hessian = (self.tau_2 + torch.mean(temp_term)).detach()

        self.s = self.gamma * L_lambda_hessian + (1-self.gamma) * self.s
        hessian_term = grad_lambada / self.s

        loss = torch.mean(top_k_selector * (pos_pred + hessian_term))

        return loss

        

