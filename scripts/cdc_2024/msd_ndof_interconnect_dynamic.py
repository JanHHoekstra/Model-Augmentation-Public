from deepSI.utils.torch_nets import simple_res_net, feed_forward_nn
from deepSI.fit_systems.encoders import SS_encoder_general_hf

import os
import matplotlib .pyplot as plt
import numpy as np
import torch
import deepSI
from scipy.io import loadmat

from model_augmentation.utils.utils import *
from model_augmentation.utils.torch_nets import identity_init_simple_res_net, zero_init_feed_forward_nn
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.mass_spring_damper import *

## ------------- Load data -----------------
dof = 3

data_file_path = os.path.join(os.getcwd(), "data", "mass_spring_damper")
print(data_file_path)
train_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_train.npz"))
val_data = deepSI.load_system_data(os.path.join(data_file_path, "msd_3dof_multisine_val.npz"))

## ------------- Add noise -----------------
sigma_n = 52e-4 # SNR60:15e-5; SNR30:52e-4; SNR20:15e-3
train_data.y = train_data.y + np.random.normal(0, sigma_n, train_data.y.shape)
val_data.y = val_data.y + np.random.normal(0, sigma_n, val_data.y.shape)

## ------------- Train fit system -----------------
FP_dof = 2
data_file_path = os.path.join(os.getcwd(), "data/mass_spring_damper/msd_{0}dof.mat".format(FP_dof))
# data_file_path = os.path.join(os.getcwd(), "data/mass_spring_damper/msd_{0}dof_non_ideal.mat".format(FP_dof))
mat_contents = loadmat(data_file_path, squeeze_me=False)

nx = mat_contents['nx'][0,0]; ny = mat_contents['ny'][0,0]; nu = mat_contents['nu'][0,0]
Ts = mat_contents['Ts'][0,0]

A_bla = mat_contents['Ad']
B_bla = mat_contents['Bd']
C_bla = mat_contents['Cd']
D_bla = mat_contents['Dd']

A_bar_bla, B_bar_bla, C_bar_bla, D_bar_bla = normalize_linear_ss_matrices(A_bla, B_bla, C_bla, D_bla, train_data, state_ix=np.array([0,1,2,3]))

nxm = 4
nxd = nxm

# ----- Dynamic states
# nxd = 2*dof #uncomment for dynamic aug
# C_bar_bla = np.append(C_bar_bla, [0,0])[np.newaxis] # uncomment for dynamic aug

interconnect = Interconnect(nxd, nu, ny, debugging=False)
physical_state_model_block = Parameterized_MSD_State_Block(nz = 5, nw = 4)
physical_output_model_block = Linear_Output_Block(C=C_bar_bla, D=D_bar_bla)
interconnect.add_block(physical_state_model_block)
interconnect.add_block(physical_output_model_block)

# print(list(physical_state_model_block.parameters()))

# ----- residual -------
# ANN_state_block = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=64, net=zero_init_feed_forward_nn, activation=torch.nn.Tanh)
# # ANN_state_block = Static_ANN_Block(nz=nxd+nu, nw=nxd, n_nodes_per_layer=64, net=zero_init_feed_forward_nn, activation=torch.nn.Identity)
# interconnect.add_block(ANN_state_block)

# interconnect.connect_block_signals(ANN_state_block, ["x", "u"], ["xp"])

# interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxd))
# interconnect.connect_block_signals(physical_state_model_block, ["u"], [])
# interconnect.connect_signals(physical_state_model_block, "xp", "additive", expansion_matrix(np.array([0,1,2,3]), nxd))

# interconnect.connect_block_signals(physical_output_model_block, ["x", "u"], ["y"])

# ----- series -------
# ANN_state_block = Static_ANN_Block(nz=nxd, nw=nxd, n_nodes_per_layer=64, net=identity_init_simple_res_net, activation=torch.nn.Tanh)
# ANN_state_block = Static_ANN_Block(nz=nxm+nxd+nu, nw=nxd, n_nodes_per_layer=64, net=simple_res_net, activation=torch.nn.Tanh)
ANN_state_block = Static_ANN_Block(nz=nxm+nxd+nu, nw=nxd, n_nodes_per_layer=64, net=feed_forward_nn, activation=torch.nn.Tanh)
interconnect.add_block(ANN_state_block)

interconnect.connect_block_signals(ANN_state_block, [physical_state_model_block, "x", "u"], ["xp"])

interconnect.connect_signals("x", physical_state_model_block, "concat", selection_matrix(np.array([0,1,2,3]), nxd))
interconnect.connect_block_signals(physical_state_model_block, ["u"], [])

interconnect.connect_block_signals(physical_output_model_block, ["x", "u"], ["y"])


# ----- deepSI fit -------
fit_sys = SSE_Interconnect(interconnect=interconnect, na=nxd*2+1, nb=nxd*2+1, e_net_kwargs={"n_nodes_per_layer":64})

nf = 200; epochs = 5000; batch_size = 2000
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf}, validation_measure="sim-RMS")

for m in fit_sys.hfn.connected_blocks:
    if isinstance(m, Parameterized_MSD_State_Block):
        print(m.params)
        break
    if isinstance(m, Parameterized_Linear_State_Block):
        # print("Module found")
        print(m.A)
        print(m.A_init)

        print(m.B)
        print(m.B_init)
        break

# ------------- Save fit system -----------------
model_file_name = "msd_{3}dof_static_series_ff_e{0}_nf{1}_batch_size{2}".format(epochs, nf, batch_size, dof)
interconnect_file_path = os.path.join(os.getcwd(), "models", "regularized_ecc", model_file_name)
fit_sys.save_system(interconnect_file_path)