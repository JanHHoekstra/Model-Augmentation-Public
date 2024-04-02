from deepSI.utils.torch_nets import simple_res_net, feed_forward_nn
from deepSI.fit_systems.encoders import SS_encoder_general_hf

import os
import matplotlib .pyplot as plt
import numpy as np
import torch
import deepSI
from scipy.io import loadmat

from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.mass_spring_damper import *

## ------------- Load data -----------------
dof = 3; nxd = 2*dof

data_file_path = os.path.join(os.getcwd(), "data", "mass_spring_damper")
train_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_train.npz"))
val_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_val.npz"))

## ------------- Add noise -----------------
sigma_n = 15e-5
train_data.y = train_data.y + np.random.normal(0, sigma_n, train_data.y.shape)
val_data.y = val_data.y + np.random.normal(0, sigma_n, val_data.y.shape)

## ------------- Train fit system -----------------
FP_dof = 2
data_file_path = os.path.join(os.getcwd(), "data/mass_spring_damper/msd_{0}dof.mat".format(FP_dof))
mat_contents = loadmat(data_file_path, squeeze_me=False)

nxb = mat_contents['nx'][0,0]; ny = mat_contents['ny'][0,0]; nu = mat_contents['nu'][0,0]
Ts = mat_contents['Ts'][0,0]
nxb = 4

A_bla = mat_contents['Ad']
B_bla = mat_contents['Bd']
C_bla = mat_contents['Cd']
D_bla = mat_contents['Dd']

A_bar_bla, B_bar_bla, C_bar_bla, D_bar_bla = normalize_linear_ss_matrices(A_bla, B_bla, C_bla, D_bla, train_data, state_ix=np.array([0,1,2,3]))

nxa = nxd; nxi = nxb + nxa
interconnect = Interconnect(nxi, nu, ny, debugging=False)

physical_state_model_block = Linear_State_Block(A=A_bar_bla, B=B_bar_bla)
physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)
ANN_state_block = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=64, net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
ANN_output_block = Static_ANN_Block(nz=nxd+nu, nw=ny, n_nodes_per_layer=64, net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)

interconnect.add_block(physical_state_model_block)
interconnect.add_block(physical_output_model_block)
interconnect.add_block(ANN_state_block)
interconnect.add_block(ANN_output_block)

interconnect.connect_block_signals(physical_state_model_block, ["u"], [])
interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxi))
interconnect.connect_signals(physical_state_model_block, "xp", "additive", expansion_matrix(np.array([0,1,2,3]), nxi))

interconnect.connect_block_signals(physical_output_model_block, ["u"], ["y"])
interconnect.connect_signals("x", physical_output_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxi))

interconnect.connect_block_signals(ANN_state_block, ["u"], [])
interconnect.connect_signals("x", ANN_state_block, "concat", selection_matrix(np.array([4,5,6,7,8,9]), nxi))
interconnect.connect_signals(ANN_state_block, "xp", "additive", expansion_matrix(np.array([4,5,6,7,8,9]), nxi))

interconnect.connect_block_signals(ANN_output_block, ["u"], ["y"])
interconnect.connect_signals("x", ANN_output_block, "concat", selection_matrix(np.array([4,5,6,7,8,9]), nxi))

fit_sys = SSE_Interconnect(interconnect=interconnect, na=nxi*2+1, nb=nxi*2+1, e_net_kwargs={"n_nodes_per_layer":64})

nf = 200; epochs = 5000; batch_size = 2000
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf})

# ------------- Save fit system -----------------
model_file_name = "msd_{3}dof_parallel_interconnection_e{0}_nf{1}_batch_size{2}".format(epochs, nf, batch_size, dof)
interconnect_file_path = os.path.join(os.getcwd(), "models", "cdc_paper", model_file_name)
fit_sys.save_system(interconnect_file_path)