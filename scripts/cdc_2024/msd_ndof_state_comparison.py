from deepSI.utils.torch_nets import simple_res_net
from deepSI.fit_systems.encoders import SS_encoder_general_hf
import deepSI

import os
import numpy as np
import torch
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
interconnect_file_path = os.path.join(os.getcwd(), "models\\cdc_paper\\msd_3dof_dyn_NL_interconnection_e5000_nf200_batch_size2000")
interconnect : SSE_Interconnect = deepSI.load_system(interconnect_file_path)

## ------------- Plot state component comparison -----------------
interconnect.hfn.reset_saved_signals()
test_interconnection = interconnect.apply_experiment(test_data)
saved_input_signals = interconnect.hfn.saved_input_signals
saved_output_signals = interconnect.hfn.saved_output_signals

scaling = 6/10
fig1 =  plt.figure(figsize=[9.2*scaling, 8.2*scaling])

for i in range(nxd):
    plt.subplot(nxd,1,i+1)
    plt.plot(saved_output_signals[i,:], label="state")
    plt.plot(saved_input_signals[-nxd+i,:], label="augmentation")
    plt.ylabel("x{0}".format(i+1))
    plt.xlim([0,test_data.N_samples])

    plt.grid()
    plt.xticks([2000, 4000, 6000, 8000, 10000], [])
plt.xticks([2000, 4000, 6000, 8000, 10000], [2000, 4000, 6000, 8000, 10000])
plt.xlabel("k")

state_comp_file_name = "state_component_comparison"
fig_file_path = os.path.join(os.getcwd(), "figures\\cdc_paper\\")
plt.savefig(os.path.join(fig_file_path, "state_component_comparison.svg"), format="svg")
plt.savefig(os.path.join(fig_file_path, "state_component_comparison.png"), format="png")
plt.savefig(os.path.join(fig_file_path, "state_component_comparison.eps"), format="eps")

plt.show()