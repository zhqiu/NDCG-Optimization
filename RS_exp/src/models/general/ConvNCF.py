# -*- coding: UTF-8 -*-

""" NeuMF
Reference:
    "Outer Product-based Neural Collaborative Filtering"
    Xiangnan He et al., IJCAI'2018.
Reference code:
    https://github.com/gitgiter/ConvNCF-pytorch/blob/master/ConvNCF.py
"""

import torch
import torch.nn as nn

from models.BaseModel import GeneralModel


class ConvNCF(GeneralModel):
    extra_log_args = ['emb_size', 'channel_size', 'kernel_size', 'strides']

    @staticmethod
    def parse_model_args(parser):
        parser.add_argument('--emb_size', type=int, default=64,
                            help='Size of embedding vectors.')
        parser.add_argument('--channel_size', type=int, default=32,
                            help='CNN setting')
        parser.add_argument('--kernel_size', type=int, default=8,
                            help='CNN setting')
        parser.add_argument('--strides', type=int, default=8,
                            help='CNN setting')
        return GeneralModel.parse_model_args(parser)

    def __init__(self, args, corpus):
        self.emb_size = args.emb_size
        self.channel_size = args.channel_size
        self.kernel_size = args.kernel_size
        self.strides = args.strides
        super().__init__(args, corpus)

    def _define_params(self):
        self.P = nn.Embedding(self.user_num, self.emb_size)
        self.Q = nn.Embedding(self.item_num, self.emb_size)

        self.cnn = nn.Sequential(
            # batch_size * 1 * 64 * 64
            nn.Conv2d(1, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 8 * 8
            nn.Conv2d(self.channel_size, self.channel_size, self.kernel_size, stride=self.strides),
            nn.ReLU(),
            # batch_size * 32 * 1 * 1
        )

        # fully-connected layer, used to predict
        self.fc = nn.Linear(32, 1, bias=False)

    def reset_last_layer(self):
        self.fc.reset_parameters()

    def fix_embedding_layer(self):
        self.P.weight.requires_grad = False
        self.Q.weight.requires_grad = False

    def forward(self, feed_dict):
        self.check_list = []
        u_ids = feed_dict['user_id']  # [batch_size]
        i_ids = feed_dict['item_id']  # [batch_size, -1]

        u_ids = u_ids.unsqueeze(-1).repeat((1, i_ids.shape[1]))  # [batch_size, -1]

        u_ids = u_ids.view(-1)
        i_ids = i_ids.view(-1)

        user_embeddings = self.P(u_ids)
        item_embeddings = self.Q(i_ids)

        interaction_map = torch.bmm(user_embeddings.unsqueeze(2), item_embeddings.unsqueeze(1))
        interaction_map = interaction_map.view(-1, 1, self.emb_size, self.emb_size)

        feature_map = self.cnn(interaction_map).view(-1, 32)
        prediction = self.fc(feature_map).view(-1)
        return {'prediction': prediction.view(feed_dict['batch_size'], -1)}

