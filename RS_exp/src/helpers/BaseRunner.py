# -*- coding: UTF-8 -*-

import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, NoReturn

from utils import utils
from models.BaseModel import BaseModel


class BaseRunner(object):
    @staticmethod
    def parse_runner_args(parser):
        parser.add_argument('--epoch', type=int, default=120,
                            help='Number of epochs.')
        parser.add_argument('--check_epoch', type=int, default=1,
                            help='Check some tensors every check_epoch.')
        parser.add_argument('--test_epoch', type=int, default=-1,
                            help='Print test results every test_epoch (-1 means no print).')
        parser.add_argument('--early_stop', type=int, default=-1,
                            help='The number of epochs when dev results drop continuously.')
        parser.add_argument('--lr', type=float, default=0.0004,
                            help='Learning rate.')
        parser.add_argument('--schedule', default='[60]', type=str,
                            help='Learning rate schedule (when to drop lr by 4x)')
        parser.add_argument('--l2', type=float, default=0,
                            help='Weight decay in optimizer.')
        parser.add_argument('--batch_size', type=int, default=256,
                            help='Batch size during training.')
        parser.add_argument('--eval_batch_size', type=int, default=256,
                            help='Batch size during testing.')
        parser.add_argument('--optimizer', type=str, default='Adam',
                            help='optimizer: SGD, Adam, Adagrad, Adadelta')
        parser.add_argument('--clip_value', type=float, default=0.0,
                            help='value for gradient clipping')
        parser.add_argument('--num_workers', type=int, default=8,
                            help='Number of processors when prepare batches in DataLoader')
        parser.add_argument('--pin_memory', type=int, default=1,
                            help='pin_memory in DataLoader')
        parser.add_argument('--topk', type=str, default='5,10,20,50',
                            help='The number of items recommended to each user.')
        parser.add_argument('--metric', type=str, default='NDCG,MAP',
                            help='metrics: NDCG, MAP')
        return parser

    def _MAP_at_k(self, hit: np.ndarray, gt_rank: np.ndarray) -> float:
        ap_list = []
        hit_gt_rank = (hit * gt_rank).astype(float)
        sorted_hit_gt_rank = np.sort(hit_gt_rank)
        for idx, row in enumerate(sorted_hit_gt_rank):
            precision_list = []
            counter = 1
            for item in row:
                if item > 0:
                    precision_list.append(counter / item)
                    counter += 1
            ap = np.sum(precision_list) / np.sum(hit[idx]) if np.sum(hit[idx]) > 0 else 0
            ap_list.append(ap)
        return np.mean(ap_list)

    def _NDCG_at_k(self, ratings: np.ndarray, normalizer_mat: np.ndarray, hit: np.ndarray, gt_rank: np.ndarray, k: int) -> float:
        # calculate the normalizer first
        normalizer = np.sum(normalizer_mat[:, :k], axis=1)
        # calculate DCG
        DCG = np.sum(((np.exp2(ratings) - 1) / np.log2(gt_rank+1)) * hit.astype(float), axis=1)
        return np.mean(DCG / normalizer)

    def evaluate_method(self, predictions: np.ndarray, ratings: np.ndarray, topk: list, metrics: list) -> Dict[str, float]:
        """
        :param predictions: (-1, n_candidates) shape, the first column is the score for ground-truth item
        :param ratings: (# of users, # of pos items)
        :param topk: top-K value list
        :param metrics: metric string list
        :return: a result dict, the keys are metric@topk
        """
        evaluations = dict()

        num_of_users, num_pos_items = ratings.shape
        sorted_ratings = -np.sort(-ratings)            # descending order !!
        discounters = np.tile([np.log2(i+1) for i in range(1, 1+num_pos_items)], (num_of_users, 1))
        normalizer_mat = (np.exp2(sorted_ratings) - 1) / discounters

        sort_idx = (-predictions).argsort(axis=1)    # index of sorted predictions (max->min)
        gt_rank = np.array([np.argwhere(sort_idx == i)[:, 1]+1 for i in range(num_pos_items)]).T  # rank of the ground-truth (start from 1)
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'NDCG':
                    evaluations[key] = self._NDCG_at_k(ratings, normalizer_mat, hit, gt_rank, k)
                elif metric == 'MAP':
                    evaluations[key] = self._MAP_at_k(hit, gt_rank)
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        return evaluations

    def __init__(self, args):
        self.epoch = args.epoch
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.schedule = eval(args.schedule)
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.clip_value = args.clip_value
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(',')]
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[0])  # early stop based on main_metric

        self.time = None  # will store [start_time, last_step_time]

    def _adjust_lr(self, optimizer, epoch):
        lr = self.learning_rate
        for milestone in self.schedule:
            lr *= 0.25 if epoch >= milestone else 1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        logging.info('Optimizer: ' + self.optimizer_name)
        optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(
            model.customize_parameters(), lr=self.learning_rate, weight_decay=self.l2)
        return optimizer

    def train(self, data_dict: Dict[str, BaseModel.Dataset]) -> NoReturn:
        model = data_dict['train'].model
        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)
        try:
            for epoch in range(self.epoch):
                # Fit
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                training_time = self._check_time()

                # Observe selected tensors
                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                # Record dev results
                dev_result = self.evaluate(data_dict['dev'], self.topk[:1], self.metrics)
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                logging_str = 'Epoch {:<5} loss={:<.4f} [{:<3.1f} s]    dev=({})'.format(
                    epoch + 1, loss, training_time, utils.format_metric(dev_result))

                # Test
                if self.test_epoch > 0 and epoch % self.test_epoch  == 0:
                    test_result = self.evaluate(data_dict['test'], self.topk[:1], self.metrics)
                    logging_str += ' test=({})'.format(utils.format_metric(test_result))
                testing_time = self._check_time()
                logging_str += ' [{:<.1f} s]'.format(testing_time)

                # Save model and early stop
                if max(main_metric_results) == main_metric_results[-1] or \
                        (hasattr(model, 'stage') and model.stage == 1):
                    model.save_model()
                    logging_str += ' *'
                logging.info(logging_str)

                if self.early_stop > 0 and self.eval_termination(main_metric_results):
                    logging.info("Early stop at %d based on dev result." % (epoch + 1))
                    break
        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

        # Find the best dev result across iterations
        best_epoch = main_metric_results.index(max(main_metric_results))
        logging.info(os.linesep + "Best Iter(dev)={:>5}\t dev=({}) [{:<.1f} s] ".format(
            best_epoch + 1, utils.format_metric(dev_results[best_epoch]), self.time[1] - self.time[0]))
        model.load_model()

    def _add_ids(self, batch: dict, out_dict: dict) -> dict:
        """
            extract user ids and item ids from a batch
            and add them into out_dict
        """
        out_dict['user_id'] = batch['user_id'].clone()
        out_dict['item_id'] = batch['item_id'].clone()
        out_dict['rating'] = batch['rating'].clone()
        out_dict['num_pos_items'] = batch['num_pos_items'].clone()
        out_dict['ideal_dcg'] = batch['ideal_dcg'].clone()
        return out_dict

    def fit(self, data: BaseModel.Dataset, epoch=-1) -> float:
        model = data.model
        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        data.actions_before_epoch()  # must sample before multi thread start

        self._adjust_lr(model.optimizer, epoch)

        model.train()
        loss_lst = list()
        dl = DataLoader(data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, desc='Epoch {:<3}'.format(epoch), ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            model.optimizer.zero_grad()
            out_dict = model(batch)
            out_dict = self._add_ids(batch, out_dict)
            loss = model.loss(out_dict, epoch)
            loss.backward()
            if self.clip_value > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), self.clip_value)
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())

        return np.mean(loss_lst).item()

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > 20 and utils.non_increasing(criterion[-self.early_stop:]):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > 20:
            return True
        return False

    def evaluate(self, data: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        predictions, ratings = self.predict(data)
        return self.evaluate_method(predictions, ratings, topks, metrics)

    def predict(self, data: BaseModel.Dataset) -> np.ndarray:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        data.model.eval()
        predictions = list()
        ratings = list()
        dl = DataLoader(data, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=data.collate_batch, pin_memory=self.pin_memory)
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            prediction = data.model(utils.batch_to_gpu(batch, data.model.device))['prediction']
            predictions.extend(prediction.cpu().data.numpy())
            ratings.extend(batch['rating'].cpu().data.numpy())
        return np.array(predictions), np.array(ratings)            # [# of users, # of items], [# of users, # of pos items]

    def print_res(self, data: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(data, self.topk, self.metrics)
        res_str = '(' + utils.format_metric(result_dict) + ')'
        return res_str
