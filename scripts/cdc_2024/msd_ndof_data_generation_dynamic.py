import deepSI

import os
import matplotlib.pylab as plt
import numpy as np
from scipy.io import loadmat, savemat

from model_augmentation.systems.mass_spring_damper import Msd_ndof
from model_augmentation.utils.utils import plot_fft_freq, plot_sys_data, determine_yes_no_query_binary_output

## ------------- Define system -----------------
System_dof = 3; ny = 1; nu = 1
dt = 0.02
m = [0.5, 0.4, 0.02]; k = [100, 100, 100]; c = [0.5, 0.5, 0.5]; a = [50, 0, 0]
system = Msd_ndof(n=System_dof, m=m, k=k, c=c, a=a, dt=dt, input_ix=[0], output_ix=[1])

## ------------- Define baseline model -----------------
FP_dof = 2
data_file_path = os.path.join(os.getcwd(), "data\\mass_spring_damper\msd_{0}dof.mat".format(FP_dof))
mat_contents = loadmat(data_file_path, squeeze_me=False)

nx = mat_contents['nx'][0,0]; ny = mat_contents['ny'][0,0]; nu = mat_contents['nu'][0,0]
Ts = mat_contents['Ts'][0,0]

A_bla = mat_contents['Ad']
B_bla = mat_contents['Bd']
C_bla = mat_contents['Cd']
D_bla = mat_contents['Dd']

## ------------- Generate input -----------------
pmax = 4999; amp_scale = 10

periods_train = 2; Ntrain = 10000; Ntrain_total = Ntrain*(periods_train)
periods_val = 1; Nval = 10000; Nval_total = Nval*(periods_val)
periods_test = 1; Ntest = 10000; Ntest_total = Ntest*(periods_test)

utrain = deepSI.deepSI.exp_design.multisine(Ntrain, (periods_train+1), pmax=pmax, n_crest_factor_optim=1)*amp_scale
uval = deepSI.deepSI.exp_design.multisine(Nval, (periods_val+1), pmax=pmax, n_crest_factor_optim=1)*amp_scale
utest = 1.5*deepSI.deepSI.exp_design.multisine(Ntest, (periods_test+1), pmax=pmax, n_crest_factor_optim=1)*amp_scale

Ttrain = np.arange(Ntrain_total)*dt
Tval = np.arange(Nval_total)*dt
Ttest = np.arange(Ntest_total)*dt

RMS_utrain = np.mean((utrain[Ntrain:])**2)**0.5
RMS_uval = np.mean((uval[Nval:])**2)**0.5
RMS_utest = np.mean((uval[Ntest:])**2)**0.5
print(RMS_utrain)
print(RMS_uval)
print(RMS_utest)

## ------------- Simulate system -----------------
train = system.apply_experiment(sys_data=deepSI.System_data(u=utrain), save_state=True); train = train[Ntrain:]; train.y = train.y[:,0]
val = system.apply_experiment(sys_data=deepSI.System_data(u=uval), save_state=True); val = val[Nval:]; val.y = val.y[:,0]
test = system.apply_experiment(sys_data=deepSI.System_data(u=utest), save_state=True); test = test[Ntest:]; test.y = test.y[:,0]

RMS_ytrain = np.mean((train.y)**2)**0.5
RMS_yval = np.mean((val.y)**2)**0.5
RMS_ytest = np.mean((val.y)**2)**0.5
print(RMS_ytrain)
print(RMS_yval)
print(RMS_ytest)

## ------------- Add noise -----------------
sigma_n = 15e-5
noise =  np.random.normal(0, sigma_n, (Ntrain_total,))
Py = np.sum(np.square(train.y + noise))/Ntrain_total
Pn = np.sum(np.square(noise))/Ntrain_total
SNR10 = 10*np.log10(Py/Pn)
print("Py: {0}".format(Py))
print("Pn: {0}".format(Pn))
print("SNR: {0}".format(SNR10))

## ------------- NRMS baseline model -----------------
ylog = np.zeros(Ntrain_total)
ulog = np.zeros(Ntrain_total)
x0 = np.zeros((4,1))
for i in range(Ntrain_total):
    u0 = train.u[i]
    ulog[i] = u0
    x1 = A_bla @ x0 + B_bla * u0
    ylog[i] = (C_bla @ x0)[0,0]
    x0 = x1.copy()

transient = 300
test_2dof = deepSI.System_data(u=ulog[transient:], y=ylog[transient:])
print(f'NRMS simulation ss enc  {test_2dof.NRMS(train[transient:]):.2%}')

Tlog = np.arange(Ntrain_total)*dt
plt.plot(Tlog, train.y[:])
plt.plot(Tlog, train.y[:]-ylog)
plt.show()

## ------------- Save data -----------------
data_file_path = "data\\mass_spring_damper"
# train.save(os.path.join(os.getcwd(), data_file_path, "msd_3dof_multisine_train.npz"))
# val.save(os.path.join(os.getcwd(), data_file_path, "msd_3dof_multisine_val.npz"))
test.save(os.path.join(os.getcwd(), data_file_path, "msd_3dof_multisine_extrapolate.npz"))

# if determine_yes_no_query_binary_output("Should this data be saved? [y/n] "):
#     data_file_name = "msd_{0}dof_multisine_N{1}_amp{2}.npz".format(n, N, 2*amp)
#     data_file_path = os.path.join(os.getcwd(), "data\\mass_spring_damper", data_file_name)
#     data.save(data_file_path)

#     savemat(os.path.join(os.getcwd(), "data\\mass_spring_damper", "msd_u_amp10.mat"), {'u10': data.u})
#     savemat(os.path.join(os.getcwd(), "data\\mass_spring_damper", "msd_y_amp10.mat"), {'y10':data.y})
# else:
#     print("Data was not saved.")