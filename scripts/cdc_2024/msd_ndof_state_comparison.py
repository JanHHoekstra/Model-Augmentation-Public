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
interconnect_file_path = os.path.join(os.getcwd(), "models\\regularized_ecc\\msd_3dof_dynamic_residual_non_ideal_e500_nf200_batch_size2000")


interconnect : SSE_Interconnect = deepSI.load_system(interconnect_file_path)

## ------------- Plot state component comparison -----------------
print(interconnect.norm.y0)
print(interconnect.norm.ystd)

y0 = interconnect.norm.y0
ystd = interconnect.norm.ystd


interconnect.hfn.reset_saved_signals()
test_interconnection = interconnect.apply_experiment(test_data)
saved_input_signals = interconnect.hfn.saved_input_signals
saved_output_signals = interconnect.hfn.saved_output_signals

print(saved_input_signals.shape)
print(saved_output_signals.shape)

plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times New Roman"
scaling = 6/10
fig1 =  plt.figure(figsize=[9.2*scaling, 8.2*scaling])
plt_range = 400
plt_centre = 200

nxd = 6

for i in range(nxd):
    plt.subplot(nxd,1,i+1)
    plt.plot(saved_output_signals[i, plt_centre - plt_range //2 : plt_centre + plt_range //2 ], label="state")
    plt.plot(saved_input_signals[-nxd+i, plt_centre - plt_range //2 : plt_centre + plt_range //2 ], label="augmentation")

    plt.ylabel(r"$x_{0}$".format(i+1))
    plt.xlim([0,plt_range])
    
    plt.grid()
    plt.xticks([100, 200, 300, 400], [])
# plt.legend()

plt.xticks([100, 200, 300, 400],[100, 200, 300, 400])
plt.xlabel(r"$k$")
# plt.tight_layout()

state_comp_file_name = "state_component_comparison"
fig_file_path = os.path.join(os.getcwd(), "figures\\cdc_paper\\")
plt.savefig(os.path.join(fig_file_path, "state_component_comparison.svg"), format="svg")
plt.savefig(os.path.join(fig_file_path, "state_component_comparison.png"), format="png")
plt.savefig(os.path.join(fig_file_path, "state_component_comparison.eps"), format="eps")

plt.show()