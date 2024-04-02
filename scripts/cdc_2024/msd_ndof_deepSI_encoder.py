import deepSI

import os
import numpy as np
import matplotlib.pyplot as plt

from model_augmentation.utils.utils import determine_yes_no_query_binary_output

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
fit_sys = deepSI.fit_systems.encoders.SS_encoder_general_hf(nx=dof*2, na=dof*4+1, nb=dof*4+1)
    
nf = 200; epochs = 5000; batch_size = 2000
fit_sys.fit(train_sys_data=train_data, val_sys_data=val_data, batch_size=batch_size, epochs=epochs, auto_fit_norm=True, loss_kwargs={'nf':nf})

# ------------- Save fit system -----------------
model_file_name = "msd_{3}dof_ANN_SS_e{0}_nf{1}_batch_size{2}".format(epochs, nf, batch_size, dof)
interconnect_file_path = os.path.join(os.getcwd(), "models", "cdc_paper", model_file_name)
fit_sys.save_system(interconnect_file_path)