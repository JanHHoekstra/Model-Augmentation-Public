from deepSI.utils.torch_nets import simple_res_net
from deepSI.fit_systems.encoders import SS_encoder_general_hf
import deepSI

import os
import numpy as np
import torch
from scipy.io import loadmat
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from model_augmentation.utils.utils import *
from model_augmentation.fit_systems.interconnect import *
from model_augmentation.fit_systems.blocks import *
from model_augmentation.systems.mass_spring_damper import *

## ------------- Load data -----------------
dof = 3; nxd = 2*dof

data_file_path = "data/mass_spring_damper"
test_data = deepSI.load_system_data(os.path.join(os.getcwd(), data_file_path, "msd_3dof_multisine_test.npz"))

## ------------- Add noise -----------------
sigma_n = 15e-5
test_data.y = test_data.y + np.random.normal(0, sigma_n, test_data.y.shape)

## ------------- Load fit system -----------------
interconnect_mixed_file_path = os.path.join(os.getcwd(), "models\\cdc_paper\\msd_3dof_mixed_interconnection_e5000_nf200_batch_size2000")
interconnect_1 : SSE_Interconnect = deepSI.load_system(interconnect_mixed_file_path)

interconnect_series_file_path = os.path.join(os.getcwd(), "models\\cdc_paper\\msd_3dof_series_interconnection_e5000_nf200_batch_size2000")
interconnect_2 : SSE_Interconnect = deepSI.load_system(interconnect_series_file_path)

interconnect_parallel_file_path = os.path.join(os.getcwd(), "models\\cdc_paper\\msd_3dof_parallel_interconnection_e5000_nf200_batch_size2000")
interconnect_3 : SSE_Interconnect = deepSI.load_system(interconnect_parallel_file_path)

# add loading of parameter matrices from matlab
FP_dof = 2
data_file_path = os.path.join(os.getcwd(), "data\\mass_spring_damper\msd_{0}dof.mat".format(FP_dof))
mat_contents = loadmat(data_file_path, squeeze_me=False)

nx = mat_contents['nx'][0,0]; ny = mat_contents['ny'][0,0]; nu = mat_contents['nu'][0,0]
Ts = mat_contents['Ts'][0,0]

A_bla = mat_contents['Ad']
B_bla = mat_contents['Bd']
C_bla = mat_contents['Cd']
D_bla = mat_contents['Dd']

## ------------- Simulate 2dof -----------------
N = test_data.N_samples

ylog = np.zeros(N)
ulog = np.zeros(N)
x0 = np.zeros((4,1))
for i in range(N):
    u0 = test_data.u[i]
    ulog[i] = u0

    x1 = A_bla @ x0 + B_bla * u0
    ylog[i] = (C_bla @ x0)[0,0]

    x0 = x1.copy()

test_2dof = deepSI.System_data(u=ulog, y=ylog)

## ------------- Plot prediction error -----------------
test_int1 = interconnect_1.apply_experiment(test_data)
test_int2 = interconnect_2.apply_experiment(test_data)
test_int3 = interconnect_3.apply_experiment(test_data)

print(f'NRMS simulation baseline  {test_2dof.NRMS(test_data):.2%}')
print(f'NRMS simulation mixed interconnect  {test_int1.NRMS(test_data):.2%}')
print(f'NRMS simulation series interconnect  {test_int2.NRMS(test_data):.2%}')
print(f'NRMS simulation parallel interconnect  {test_int3.NRMS(test_data):.2%}')

scaling = 6/10; fig1 = plt.figure(figsize=[9.2*scaling, 3*scaling])

for i in range(ny):
    plt.subplot(ny,1,i+1)
    plt.grid()
    plt.plot(test_data.y[:],label="Measured data")
    plt.plot(test_data.y[:] - test_2dof.y[:],label='baseline error')
    plt.plot(test_data.y[:] - test_int1.y[:],label='mixed error')
    plt.plot(test_data.y[:] - test_int2.y[:],label='series error')
    plt.plot(test_data.y[:] - test_int3.y[:],label='parallel error')
    plt.ylabel("y [m]".format(i+1), labelpad=0.0)
    plt.xlim([0,test_data.N_samples])
    plt.tight_layout()
plt.xlabel("k", labelpad=0.0)

fig_file_path = os.path.join(os.getcwd(), "figures\\cdc_paper\\")
plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.svg"), format="svg")
plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.png"), format="png")
plt.savefig(os.path.join(fig_file_path, "fitted_models_pred_error.eps"), format="eps")

plt.show()