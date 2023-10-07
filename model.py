import torch
import torch.nn as nn

class TransformerLayer(nn.Module):
    def __init__(self, in_dim, layers=1, dropout=.1, heads=8):
        super().__init__()
        self.transformer = nn.Transformer(in_dim, heads, layers, layers, in_dim * 4, dropout=dropout)

    def forward(self, input, mask):
        x = input.permute(1, 0, 2)
        x = self.transformer(x, x, tgt_mask=mask)
        return x.permute(1, 0, 2)


class network(nn.Module):
    def __init__(self, dropout=0.1, edim=32, out_dim=12, hid_dim=64, layers=6, batch_size=32, num_nodes=1):
        super(network, self).__init__()
        self.dstelayer = DecoupledSpatialTemporalEmbeddingLayer(edim=edim, num_nodes=num_nodes, batch_size=batch_size)
        self.translyear = TransformerLayer(in_dim=hid_dim, layers=layers, dropout=dropout)
        self.lin = nn.Linear(hid_dim, out_dim)

    def forward(self, input):
        x = input.transpose(1, 3)
        x, A = self.dstelayer(x)
        x = x.transpose(1, 2)
        mask = A
        x = self.translyear(x, mask)
        x = self.lin(x)
        return x.transpose(1, 2).unsqueeze(-1)
