import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from data.dataset_loading import PADDED_Y_VALUE
from models.losses import DEFAULT_EPS
from models.model_utils import get_torch_device


class NDCG_M(nn.Module):
    def __init__(self, query_num, longest_query_size, enable_u, gamma, topk):
        super(NDCG_M, self).__init__()
        self.enable_u = enable_u
        self.gamma = gamma
        self.topk = topk
        self.beta = 0.9
        self.eta = 0.01
        self.tau_1 = 0.001
        self.tau_2 = 0.0001
        self.device = get_torch_device()
        self.u_warmup = torch.zeros(query_num+2, longest_query_size+2)
        self.lambda_q = torch.zeros(query_num+2)
        self.v_q = torch.zeros(query_num+2)
        self.s_q = torch.zeros(query_num+2)
        self.psi_func = 'sigmoid'
        self.sigmoid_alpha = 2.0
        self.c_sigmoid = 2.0
        self.topk_version = 'theo'

    def _squared_hinge_loss(self, x, c=1.0):
        return torch.max(torch.zeros_like(x), x + c) ** 2

    def forward(self, y_pred, y_true, qid, indices, num_pos, num_item, ideal_dcg, padded_value_indicator=PADDED_Y_VALUE, eps=DEFAULT_EPS, alpha=1.):
        """
            y_pred:    [batch_size, slate_length]
            y_true:    [batch_size, slate_length], pad with -1
            qid:       [batch_size, slate_length]
            indices:   [batch_size, slate_length], pad with -1
            num_pos:   [batch_size]
            num_item:  [batch_size]
            ideal_dcg: [batch_size]
        """
        assert qid is not None, "qid cannot be None"
        assert indices is not None, "indices cannot be None"
        assert num_pos is not None, "num_pos cannot be None"
        assert num_item is not None, "num_item cannot be None"
        assert ideal_dcg is not None, "ideal_dcg cannot be None"

        device = y_pred.device
        y_pred = y_pred.clone()
        y_true = y_true.clone()

        padded_mask = y_true == padded_value_indicator
        y_pred[padded_mask] = float("-inf")
        y_true[padded_mask] = float("-inf")

        num_of_noninf = torch.isfinite(y_true).float().sum(dim=-1) # [batch_size]

        scores_diffs = y_pred[:, None, :] - y_pred[:, :, None]
        scores_diffs_mask = torch.isfinite(scores_diffs)
        scores_diffs[~scores_diffs_mask] = 0.0          # [batch_size, slate_length, slate_length]

        g = torch.sum(scores_diffs_mask.float() * self._squared_hinge_loss(scores_diffs), dim=-1) / num_of_noninf[:, None]  # [batch_size, slate_length]
        
        g += eps
        ideal_dcg += eps

        G = (2.0 ** y_true.clamp_(min=0.0) - 1).float()

        if self.enable_u:
            qid_flat, indices_flat = qid.reshape(-1)+1, indices.reshape(-1)+1
            self.u_warmup[qid_flat, indices_flat] = (1-self.gamma) * self.u_warmup[qid_flat, indices_flat] + \
                                                        self.gamma * g.clone().detach_().reshape(-1).cpu()
            g_u = self.u_warmup[qid_flat, indices_flat].reshape(g.size()).to(device)
            nabla_f_g = (G * num_item[:, None]) / (torch.log2(2 + num_item[:, None] * g_u)**2 * (2 + num_item[:, None] * g_u) * np.log(2))

            if self.topk > 0:
                qid_batch = qid[:, 0] + 1
                pred_lambda_diffs = y_pred.clone().detach_() - self.lambda_q[qid_batch][:, None].to(device)
                pred_lambda_diffs_mask = torch.isfinite(pred_lambda_diffs)
                pred_lambda_diffs[~pred_lambda_diffs_mask] = 0.0
                grad_lambda_q = self.topk/num_item.cpu() + self.tau_2*self.lambda_q[qid_batch] - \
                                    torch.sum(pred_lambda_diffs_mask.cpu() * torch.sigmoid(pred_lambda_diffs.cpu() / self.tau_1), dim=-1) / num_of_noninf.cpu()
                self.v_q[qid_batch] = self.beta * grad_lambda_q + (1-self.beta) * self.v_q[qid_batch]
                self.lambda_q[qid_batch] = self.lambda_q[qid_batch] - self.eta * self.v_q[qid_batch]

                if self.topk_version == 'prac':
                    nabla_f_g *= self.c_sigmoid * torch.sigmoid(pred_lambda_diffs * self.sigmoid_alpha)                

                elif self.topk_version == 'theo':
                    nabla_f_g *= self.c_sigmoid * torch.sigmoid(pred_lambda_diffs * self.sigmoid_alpha)
                    weight_1 = self.c_sigmoid * torch.sigmoid(pred_lambda_diffs * self.sigmoid_alpha) * (1 - torch.sigmoid(pred_lambda_diffs * self.sigmoid_alpha))
                                   
                    y_pred[padded_mask] = 0.0
                    temp_term = torch.sigmoid(pred_lambda_diffs / self.tau_1) * (1 - torch.sigmoid(pred_lambda_diffs / self.tau_1)) / self.tau_1
                    L_lambda_hessian = self.tau_2 + torch.sum(pred_lambda_diffs_mask * temp_term, dim=1) / num_of_noninf
                    self.s_q[qid_batch] = self.beta * L_lambda_hessian.cpu() + (1-self.beta) * self.s_q[qid_batch]
                    hessian_term = (torch.sum(pred_lambda_diffs_mask * temp_term * y_pred, dim=1) / num_of_noninf) / self.s_q[qid_batch].to(device)
                    f_g_u = -G / torch.log2(2 + num_item[:, None] * g_u)
                    loss = (num_pos[:, None] * torch.mean(nabla_f_g * g + weight_1 * f_g_u * (y_pred - hessian_term[:, None]), dim=-1) / ideal_dcg[:, None]).mean()
                    return loss

        else:
            nabla_f_g = (G * num_item[:, None]) / (torch.log2(2 + num_item[:, None] * g)**2 * (2 + num_item[:, None] * g) * np.log(2)).detach_()

        loss = (num_pos[:, None] * torch.mean(nabla_f_g * g, dim=-1) / ideal_dcg[:, None]).mean()
        return loss
