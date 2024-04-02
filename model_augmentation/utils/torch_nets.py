from torch import nn, Tensor
import torch
import numpy as np

class zero_init_feed_forward_nn(nn.Module): #a simple MLP
    def __init__(self,n_in=6, n_out=5, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh):
        super(zero_init_feed_forward_nn,self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        seq = [nn.Linear(n_in,n_nodes_per_layer),activation()]
        assert n_hidden_layers>0
        for i in range(n_hidden_layers-1):
            seq.append(nn.Linear(n_nodes_per_layer,n_nodes_per_layer))
            seq.append(activation())

        final_layer = nn.Linear(n_nodes_per_layer,n_out)
        seq.append(final_layer)

        self.net = nn.Sequential(*seq)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, val=0)

        nn.init.constant_(final_layer.bias, val=0.0)
        nn.init.constant_(final_layer.weight, val=0.0)

    def forward(self,X):
        return self.net(X) 