import torch
from torch_geometric.loader import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from gnn import GNN

from tqdm import tqdm
import argparse
import time
import numpy as np

### importing OGB
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from map_at_k import map_at_k

### using mAP loss
from libauc.sampler import TriSampler
from map_k_loss import meanAveragePrecisionLoss

from libauc.losses import FocalLoss


### the graph dataset with index
class GraphDataset(PygGraphPropPredDataset):
    def __getitem__(self, id_tuple):
        idx, task_id = id_tuple
        item = self.get(self.indices()[idx])
        item.idx = torch.LongTensor([idx])
        item.task_id = torch.LongTensor([task_id])
        return item


cls_criterion = torch.nn.BCEWithLogitsLoss()
reg_criterion = torch.nn.MSELoss()
map_criterion = meanAveragePrecisionLoss(437929, num_labels=128, margin=0.6, gamma=0.9, top_k=300)
focal_criterion = FocalLoss(alpha=0.25, gamma=2.0)

LOSS_TYPE = 'CE' # CE, Focal, mAP (ours)


def set_all_seeds(SEED):
   # REPRODUCIBILITY
   np.random.seed(SEED)
   torch.manual_seed(SEED)
   torch.cuda.manual_seed(SEED)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False

set_all_seeds(2024)


def train(model, device, loader, optimizer, task_type):
    model.train()

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            optimizer.zero_grad()
            pred = model(batch)

            if LOSS_TYPE == 'CE':
                is_labeled = batch.y == batch.y
                loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            elif LOSS_TYPE == 'mAP':
                y_pred = torch.sigmoid(pred)
                y_true = torch.nan_to_num(batch.y)
                loss = map_criterion(y_pred, y_true, batch.idx, batch.task_id)
            elif LOSS_TYPE == 'Focal':
                is_labeled = batch.y == batch.y
                loss = focal_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
            else:
                assert 0, LOSS_TYPE + ' is not supported.'

            loss.backward()
            optimizer.step()
            


def eval(model, device, loader, evaluator):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)

        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                pred = model(batch)

            y_true.append(batch.y.view(pred.shape).detach().cpu())
            y_pred.append(pred.detach().cpu())

    y_true = torch.nan_to_num(torch.cat(y_true, dim = 0)).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()

    #print("y_true:", y_true, y_true.shape)
    #print("y_pred:", y_pred, y_pred.shape)

    #input_dict = {"y_true": y_true, "y_pred": y_pred}
    #return evaluator.eval(input_dict)

    map_val = map_at_k(y_true, y_pred, k=50)
    return {'mAP': map_val}


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='GNN baselines on ogbgmol* data with Pytorch Geometrics')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--gnn', type=str, default='gin-virtual',
                        help='GNN gin, gin-virtual, or gcn, or gcn-virtual (default: gin-virtual)')
    parser.add_argument('--drop_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--num_layer', type=int, default=5,
                        help='number of GNN message passing layers (default: 5)')
    parser.add_argument('--emb_dim', type=int, default=300,
                        help='dimensionality of hidden units in GNNs (default: 300)')
    parser.add_argument('--batch_size', type=int, default=2560,
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers (default: 0)')
    parser.add_argument('--dataset', type=str, default="ogbg-molhiv",
                        help='dataset name (default: ogbg-molhiv)')
    parser.add_argument('--feature', type=str, default="full",
                        help='full feature or simple feature')
    parser.add_argument('--filename', type=str, default="",
                        help='filename to output result (default: )')
    args = parser.parse_args()

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    # build the dataset
    dataset_full = PygGraphPropPredDataset(name = args.dataset)
    dataset_train = GraphDataset(name='ogbg-molpcba')

    split_idx = dataset_full.get_idx_split()
    dataset_full.eval_metric = 'mAP'

    print(len(split_idx["train"]), len(split_idx["valid"]), len(split_idx["test"]))

    # build the sampler
    labels = dataset_train.data.y
    labels = torch.nan_to_num(labels)
    sampler = TriSampler(None, batch_size_per_task=20, labels=labels, sampling_rate=0.5)

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, sampler=sampler, num_workers = args.num_workers)
    valid_loader = DataLoader(dataset_full[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader = DataLoader(dataset_full[split_idx["test"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)

    if args.gnn == 'gin':
        model = GNN(gnn_type = 'gin', num_tasks = dataset_full.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gin-virtual':
        model = GNN(gnn_type = 'gin', num_tasks = dataset_full.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    elif args.gnn == 'gcn':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset_full.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = False).to(device)
    elif args.gnn == 'gcn-virtual':
        model = GNN(gnn_type = 'gcn', num_tasks = dataset_full.num_tasks, num_layer = args.num_layer, emb_dim = args.emb_dim, drop_ratio = args.drop_ratio, virtual_node = True).to(device)
    else:
        raise ValueError('Invalid GNN type')

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    valid_curve = []
    test_curve = []
    train_curve = []

    for epoch in range(1, args.epochs + 1):
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, dataset_full.task_type)

        print('Evaluating...')
        train_perf = eval(model, device, train_loader, evaluator)
        valid_perf = eval(model, device, valid_loader, evaluator)
        test_perf = eval(model, device, test_loader, evaluator)

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})

        train_curve.append(train_perf[dataset_full.eval_metric])
        valid_curve.append(valid_perf[dataset_full.eval_metric])
        test_curve.append(test_perf[dataset_full.eval_metric])

    if 'classification' in dataset_full.task_type:
        best_val_epoch = np.argmax(np.array(valid_curve))
        best_train = max(train_curve)
    else:
        best_val_epoch = np.argmin(np.array(valid_curve))
        best_train = min(train_curve)

    print('Finished training!')
    print('Best validation score: {}'.format(valid_curve[best_val_epoch]))
    print('Test score: {}'.format(test_curve[best_val_epoch]))

    if not args.filename == '':
        torch.save({'Val': valid_curve, 'Test': test_curve, 'Train': train_curve, 'BestVal':best_val_epoch, 'BestTrain': best_train}, args.filename)


if __name__ == "__main__":
    main()
