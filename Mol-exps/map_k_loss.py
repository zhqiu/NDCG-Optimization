import torch
import torch.nn.functional as F
from libauc.losses.surrogate import get_surrogate_loss
from libauc.utils.utils import check_tensor_shape


class meanAveragePrecisionLoss(torch.nn.Module):
    def __init__(self, 
                 data_len, 
                 num_labels, 
                 margin=1.0, 
                 gamma=0.9,
                 top_k=-1, 
                 surr_loss='squared_hinge',  
                 device=None):
        super(meanAveragePrecisionLoss, self).__init__()
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device   
        self.margin = margin
        self.num_labels = num_labels
        self.u_all = torch.zeros((num_labels, data_len, 1)).to(self.device).detach()
        self.u_pos = torch.zeros((num_labels, data_len, 1)).to(self.device).detach()
        self.margin = margin
        self.gamma = gamma
        self.surrogate_loss = get_surrogate_loss(surr_loss)
        self.top_k = top_k

    def forward(self, y_pred, y_true, index, task_id=[], **kwargs):
        y_pred = check_tensor_shape(y_pred, (-1, self.num_labels))
        y_true = check_tensor_shape(y_true, (-1, self.num_labels))
        index  = check_tensor_shape(index, (-1,))
        if len(task_id) == 0:
           task_id = list(range(self.num_labels))
        else:
           task_id = torch.unique(task_id)
        total_loss = 0
        for idx in task_id:
            y_pred_i, y_true_i = y_pred[:, idx].reshape(-1, 1),  y_true[:, idx].reshape(-1, 1)
            pos_mask = (1==y_true_i).squeeze()
            assert sum(pos_mask) > 0, 'input data contains no positive sample. To fix it, please use libauc.sampler.TriSampler to resampling data!'
            if len(index) == len(y_pred): 
                index_i = index[pos_mask]   # for positive samples only   
            f_ps  = y_pred_i[pos_mask]      # shape: (len(f_ps), 1)
            f_all = y_pred_i.squeeze()      # shape: (len(f_all), )
            sur_loss = self.surrogate_loss(self.margin, (f_ps - f_all)) # shape: (len(f_ps), len(f_all))
            pos_sur_loss = sur_loss * pos_mask
            self.u_all[idx][index_i] = (1 - self.gamma) * self.u_all[idx][index_i]  + self.gamma * (sur_loss.mean(1, keepdim=True)).detach()
            self.u_pos[idx][index_i] = (1 - self.gamma) * self.u_pos[idx][index_i]  + self.gamma * (pos_sur_loss.mean(1, keepdim=True)).detach()
            p_i = (self.u_pos[idx][index_i] - (self.u_all[idx][index_i]) * pos_mask) / (self.u_all[idx][index_i] ** 2) # size of p_i: len(f_ps)* len(y_pred)
            if self.top_k > -1:
                selector = torch.sigmoid(self.top_k - sur_loss.sum(dim=0, keepdim=True).clone())
                p_i *= selector
            p_i.detach_()
            loss = torch.mean(p_i * sur_loss)
            total_loss += loss
        return total_loss/len(task_id)