import argparse

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, SAGEConv
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SAGEConv as SAGEConv_
from torch.nn import Linear
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from logger import Logger


class NGNN_GCNConv(torch.nn.Module):
    def __init__(
        self, in_channels, hidden_channels, out_channels, num_nonl_layers
    ):
        super(NGNN_GCNConv, self).__init__()
        self.num_nonl_layers = (
            num_nonl_layers  # number of nonlinear layers in each conv layer
        )
        self.conv = GraphConv(in_channels, hidden_channels)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x):
        x = self.conv(g, x)

        if self.num_nonl_layers == 2:
            x = F.relu(x)
            x = self.fc(x)

        x = F.relu(x)
        x = self.fc2(x)
        return x



class NGNN_SAGEConv(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_nonl_layers,
        *,
        reduce,
    ):
        super(NGNN_SAGEConv, self).__init__()
        self.num_nonl_layers = (
            num_nonl_layers  # number of nonlinear layers in each conv layer
        )
        self.conv = SAGEConv_(in_channels, hidden_channels, reduce)
        self.fc = Linear(hidden_channels, hidden_channels)
        self.fc2 = Linear(hidden_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        gain = torch.nn.init.calculate_gain("relu")
        torch.nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_uniform_(self.fc2.weight, gain=gain)
        for bias in [self.fc.bias, self.fc2.bias]:
            stdv = 1.0 / math.sqrt(bias.size(0))
            bias.data.uniform_(-stdv, stdv)

    def forward(self, g, x):
        x = self.conv(g, x)

        if self.num_nonl_layers == 2:
            x = F.relu(x)
            x = self.fc(x)

        x = F.relu(x)
        x = self.fc2(x)
        return x






class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,ngnn=False):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()

        num_nonl_layers = (
            1 if num_layers <= 2 else 2
        )  # number of nonlinear layers in each conv layer

        self.convs.append(
            GCNConv(in_channels, hidden_channels, normalize=False))
        for _ in range(num_layers - 2):
            if ngnn == True:
                self.convs.append(NGNN_GCNConv(hidden_channels,hidden_channels,hidden_channels,num_nonl_layers))
            else:
                self.convs.append(GCNConv(hidden_channels, hidden_channels, normalize=False))
        self.convs.append(
            GCNConv(hidden_channels, out_channels, normalize=False))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            print(conv)
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        #print(x.size())
        return x


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout,reduce="mean",ngnn=False):
        super(SAGE, self).__init__()

        self.convs = torch.nn.ModuleList()

        num_nonl_layers = (
            1 if num_layers <= 2 else 2
        )  # number of nonlinear layers in each conv layer

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            if ngnn == True:
                self.convs.append(NGNN_SAGEConv(hidden_channels,hidden_channels,hidden_channels,num_nonl_layers,reduce=reduce))
            else:
                self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):
        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return x


def train(model, data, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(data.x, data.adj_t)[train_idx]
    loss = criterion(out, data.y[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator):
    model.eval()

    y_pred = model(data.x, data.adj_t)

    train_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })
    valid_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })
    test_rocauc = evaluator.eval({
        'y_true': data.y[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })

    return train_rocauc["rocauc"], valid_rocauc["rocauc"],test_rocauc["rocauc"],train_rocauc["acc"], valid_rocauc["acc"],test_rocauc["acc"]


def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', action='store_true')
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--ngnn_sage', action="store_true")
    parser.add_argument('--ngnn_gcn', action="store_true")
    parser.add_argument("--K1", action="store")

    args = parser.parse_args()
    print(args)
    labels = 112

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(
        name='ogbn-proteins', transform=T.ToSparseTensor(attr='edge_attr'))
    data = dataset[0]
    print(data)
    data.x = data.adj_t.mean(dim=1)
    data.x = torch.load("embedding_CC_{}".format(args.K1))
    data.adj_t.set_value_(None)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)


    if args.use_sage:
        model = SAGE(data.num_features, args.hidden_channels, labels,
                     args.num_layers, args.dropout).to(device)
    elif args.ngnn_sage:
        model = SAGE(data.num_features, args.hidden_channels, labels,
                     args.num_layers, args.dropout, ngnn=True).to(device)        
    elif args.ngnn_gcn:
        model = GCN(data.num_features, args.hidden_channels, labels,
                     args.num_layers, args.dropout, ngnn=True).to(device)  
    else:
        model = GCN(data.num_features, args.hidden_channels, labels,
                    args.num_layers, args.dropout).to(device)

        # Pre-compute GCN normalization.
        adj_t = data.adj_t.set_diag()
        deg = adj_t.sum(dim=1).to(torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        adj_t = deg_inv_sqrt.view(-1, 1) * adj_t * deg_inv_sqrt.view(1, -1)
        data.adj_t = adj_t

    data = data.to(device)

    evaluator = Evaluator(name='ogbn-proteins', nlabels=labels)
    logger = Logger(args.runs, args)

    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            loss = train(model, data, train_idx, optimizer)

            if epoch % args.eval_steps == 0:
                result = test(model, data, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc,  test_rocauc, train_acc, valid_acc, test_acc = result


                    if len(result) >  3:
                        print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}% '
                          f'Acc Train: {100 * train_acc:.2f}% '
                          f'Acc Valid: {100 * valid_acc:.2f}% '
                          f'Acc Test: {100 * test_acc:.2f}%')


        logger.print_statistics(run)
    logger.print_statistics()


if __name__ == "__main__":
    main()
