from deepSI.utils.torch_nets import feed_forward_nn, simple_res_net
from deepSI.system_data import System_data

from torch import nn, Tensor
import torch

from model_augmentation.utils.torch_nets import zero_init_feed_forward_nn
from model_augmentation.utils.utils import *

class Block(nn.Module):
    '''Basic block implementation with variables that need to be defined for interconnect.'''
    def __init__(self, nz, nw, name=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.nw = nw
        self.nz = nz

        self.name = name
        self.block_ix = None

    def init_block(self, z: Tensor):
        return

    def forward(self, z: Tensor):
        raise NotImplementedError('This function should return w computed from z.')
    
class Static_ANN_Block(Block):
    def __init__(self, net=zero_init_feed_forward_nn, n_nodes_per_layer=64, n_hidden_layers=2, activation=nn.Tanh, *args, **kwargs) -> None:
        super(Static_ANN_Block, self).__init__(*args, **kwargs)

        self.net = net(n_in=self.nz, n_out=self.nw, n_nodes_per_layer=n_nodes_per_layer, n_hidden_layers=n_hidden_layers, activation=activation)

    def forward(self, z: Tensor):
        assert z.size(1) == self.nz
        w = self.net(z.view(-1,self.nz))
        return w.view(-1,self.nw,1)

class Linear_State_Block(Block):
    def __init__(self, A=torch.empty((0,0)), B=torch.empty((0,0)), *args, **kwargs) -> None:
        # Matrices defining the known linear ss model
        self.A = to_tensor(A)
        self.B = to_tensor(B)
        
        self.nx = self.A.size(0) if self.A.numel() else 0
        self.nu = self.B.size(1) if self.B.numel() else 0
        if self.nx == 0:
            self.nx = self.B.size(0) if self.B.numel() else 0

        super().__init__(nw=self.nx, nz=self.nx+self.nu, *args, **kwargs)
    
    def forward(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:,:self.nx,:]
        u = z[:,self.nx:,:]
        w = torch.matmul(self.A, x) + torch.matmul(self.B, u)
        return w
    
class Linear_Output_Block(Block):
    def __init__(self, C=torch.empty((0,0)), D=torch.empty((0,0)), *args, **kwargs) -> None:

        self.C = to_tensor(C)
        self.D = to_tensor(D)

        self.ny = self.C.size(0) if self.C.numel() else 0
        self.nx = self.C.size(1) if self.C.numel() else 0
        self.nu = self.D.size(1) if self.D.numel() else 0
        if self.ny == 0:
            self.ny = self.D.size(0) if self.D.numel() else 0

        super().__init__(nw=self.ny, nz=self.nx+self.nu, *args, **kwargs)

    def forward(self, z: Tensor):
        assert z.size(1) == self.nx + self.nu
        x = z[:,:self.nx,:]
        u = z[:,self.nx:,:]
        w = torch.matmul(self.C, x) + torch.matmul(self.D, u)
        return w


