import os
from functools import partial

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

import models.metrics as metrics_module
from data.dataset_loading import PADDED_Y_VALUE
from models.model_utils import get_num_params, log_num_params
from training.early_stop import EarlyStop
from utils.ltr_logging import get_logger
from utils.tensorboard_utils import TensorboardSummaryWriter

logger = get_logger()


def loss_batch(model, loss_func, xb, yb, indices, gradient_clipping_norm, pass_indices_n_qid, qid=None, num_pos=None, num_item=None, ideal_dcg=None, opt=None):
    mask = (yb == PADDED_Y_VALUE)

    if pass_indices_n_qid:
        loss = loss_func(model(xb, mask, indices), yb, qid, indices, num_pos, num_item, ideal_dcg)
    else:
        loss = loss_func(model(xb, mask, indices), yb)

    if opt is not None:
        loss.backward()
        if gradient_clipping_norm:
            clip_grad_norm_(model.parameters(), gradient_clipping_norm)
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def metric_on_batch(metric, model, xb, yb, indices):
    mask = (yb == PADDED_Y_VALUE)
    return metric(model.score(xb, mask, indices), yb)


def metric_on_epoch(metric, model, dl, dev):
    metric_values = torch.mean(
        torch.cat(
            [metric_on_batch(metric, model, xb.to(device=dev), yb.to(device=dev), indices.to(device=dev))
             for xb, yb, indices, _, _, _, _ in dl]
        ), dim=0
    ).cpu().numpy()
    return metric_values


def compute_metrics(metrics, model, dl, dev):
    metric_values_dict = {}
    for metric_name, ats in metrics.items():
        metric_func = getattr(metrics_module, metric_name)
        metric_func_with_ats = partial(metric_func, ats=ats)
        metrics_values = metric_on_epoch(metric_func_with_ats, model, dl, dev)
        metrics_names = ["{metric_name}_{at}".format(metric_name=metric_name, at=at) for at in ats]
        metric_values_dict.update(dict(zip(metrics_names, metrics_values)))

    return metric_values_dict


def epoch_summary(epoch, train_loss, train_metrics, val_metrics):
    summary = "Epoch : {:0>3d} Train loss: {:.6f} \n".format(epoch, train_loss)
    for metric_name, metric_value in train_metrics.items():
        summary += " Train {} {:.6f}".format(metric_name, metric_value)
    summary += "\n"
    for metric_name, metric_value in val_metrics.items():
        summary += " Val {} {:.6f}".format(metric_name, metric_value)

    return summary


def val_metrics_summary(val_metrics):
    summary = "Evaluate on val set \n"
    for metric_name, metric_value in val_metrics.items():
        summary += " Val {} {:.6f}".format(metric_name, metric_value)

    return summary


def get_current_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def fit(epochs, model, loss_func, optimizer, scheduler, train_dl, valid_dl, config, gradient_clipping_norm,
        early_stopping_patience, device, output_dir, tensorboard_output_path, pass_indices_n_qid, pretrained_model_path):
    tensorboard_summary_writer = TensorboardSummaryWriter(tensorboard_output_path)

    num_params = get_num_params(model)
    log_num_params(num_params)

    early_stop = EarlyStop(early_stopping_patience)

    if len(pretrained_model_path) > 0:
        logger.info("load pretrained model model.pkl from {}".format(pretrained_model_path))
        model.load_state_dict(torch.load(pretrained_model_path))
        model.eval()
        with torch.no_grad():
            val_metrics = compute_metrics(config.metrics, model, valid_dl, device)
        logger.info(val_metrics_summary(val_metrics))
        logger.info("Reset the parameters in the last layer.")
        model.reset_last_layer()

    for epoch in range(epochs):
        logger.info("Current learning rate: {}".format(get_current_lr(optimizer)))

        model.train()
        # xb dim: [batch_size, slate_length, embedding_dim]
        # yb dim: [batch_size, slate_length]

        train_losses, train_nums = zip(
            *[loss_batch(model, loss_func, xb.to(device=device), yb.to(device=device), indices.to(device=device),
                         gradient_clipping_norm, pass_indices_n_qid, qid.to(device=device), num_pos.to(device=device), num_item.to(device=device), ideal_dcg.to(device=device), optimizer) for
              xb, yb, indices, qid, num_pos, num_item, ideal_dcg in train_dl])
        train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
        train_metrics = compute_metrics(config.metrics, model, train_dl, device)

        model.eval()
        with torch.no_grad():
            val_metrics = compute_metrics(config.metrics, model, valid_dl, device)

        tensorboard_metrics_dict = {("train", "loss"): train_loss}

        train_metrics_to_tb = {("train", name): value for name, value in train_metrics.items()}
        tensorboard_metrics_dict.update(train_metrics_to_tb)
        val_metrics_to_tb = {("val", name): value for name, value in val_metrics.items()}
        tensorboard_metrics_dict.update(val_metrics_to_tb)
        tensorboard_metrics_dict.update({("train", "lr"): get_current_lr(optimizer)})

        tensorboard_summary_writer.save_to_tensorboard(tensorboard_metrics_dict, epoch)

        logger.info(epoch_summary(epoch, train_loss, train_metrics, val_metrics))

        current_val_metric_value = val_metrics.get(config.val_metric)
        if scheduler:
            if type(scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                args = [val_metrics[config.val_metric]]
                scheduler.step(*args)
            else:
                scheduler.step()

        early_stop.step(current_val_metric_value, epoch)
        if early_stop.stop_training(epoch):
            logger.info(
                "early stopping at epoch {} since {} didn't improve from epoch no {}. Best value {}, current value {}".format(
                    epoch, config.val_metric, early_stop.best_epoch, early_stop.best_value, current_val_metric_value
                ))
            break

    torch.save(model.state_dict(), os.path.join(output_dir, "model.pkl"))
    tensorboard_summary_writer.close_all_writers()

    return {
        "epochs": epoch,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "num_params": num_params
    }
