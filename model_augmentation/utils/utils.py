import torch
import numpy as np
import matplotlib.pyplot as plt
from deepSI import system_data

def to_tensor(input):
    if isinstance(input, type(None)):
        return None
    if isinstance(input, torch.Tensor):
        return input
    else:
        try: return torch.Tensor(input)
        except: raise TypeError("Input could not be converted to tensor")

def to_ndarray(input):
    if isinstance(input, torch.Tensor):
        return input.detach().numpy() 
    else:
        try: return np.ndarray(input)
        except: raise TypeError("Input could not be converted to ndarray")

def detect_algebraic_loop(A: np.ndarray):
    n = A.shape[0]
    An = A.copy()

    for ii in range(2, n + 1):
        An = np.dot(An, A)  # Do not re-compute A^n from scratch
        
        if np.trace(An) != 0:
            return True
    
    return False
    
def get_sys_data_dimensions(sys_data: system_data):
    nu = determine_sys_data_signal_dimension(sys_data.u)
    ny = determine_sys_data_signal_dimension(sys_data.y)
    nx = determine_sys_data_signal_dimension(sys_data.x)

    return nu, ny, nx

def determine_sys_data_signal_dimension(u: np.ndarray):
    if u is None: nu = 0
    elif len(u.shape) == 2: nu = u.shape[1]
    else: nu = 1

    return nu

def determine_std_T(x: np.ndarray, nx: int, ix):
    x = x.reshape((-1,nx))
    if nx > 1:
        x = x[:,ix]
    std_x = np.std(x, axis=0)
    Tix = np.diag(std_x)
    Tx = np.linalg.inv(Tix)

    return Tx, Tix

def determine_std_T_sys_data(sys_data: system_data, input_ix=None, output_ix=None, state_ix=None):
    nu, ny, nx = get_sys_data_dimensions(sys_data)
    if not isinstance(input_ix, np.ndarray): input_ix = np.arange(nu)
    if not isinstance(output_ix, np.ndarray): output_ix = np.arange(ny)
    if not isinstance(state_ix, np.ndarray): state_ix = np.arange(nx)

    Tu, Tiu = determine_std_T(sys_data.u, nu, input_ix)
    Ty, Tiy = determine_std_T(sys_data.y, ny, output_ix)

    if nx != 0:
        Tx, Tix = determine_std_T(sys_data.x, nx, state_ix)    
        return Tu, Tiu, Ty, Tiy, Tx, Tix

    return Tu, Tiu, Ty, Tiy

def normalize_linear_ss_matrices(A_bla, B_bla, C_bla, D_bla, sys_data: system_data, input_ix=None, output_ix=None, state_ix=None):
    Tu, Tiu, Ty, Tiy, Tx, Tix = determine_std_T_sys_data(sys_data, input_ix, output_ix, state_ix)

    A_bar_bla = Tx @ A_bla @ Tix
    B_bar_bla = Tx @ B_bla @ Tiu
    C_bar_bla = Ty @ C_bla @ Tix
    D_bar_bla = Ty @ D_bla @ Tiu
    
    return A_bar_bla, B_bar_bla, C_bar_bla, D_bar_bla

def expansion_matrix(ix, n):
    S = np.zeros((n, len(ix)))
    S[ix, np.arange(len(ix)), ] = 1
    return torch.Tensor(S)

def selection_matrix(ix, n):
    E = np.zeros((len(ix), n))
    E[np.arange(len(ix)), ix] = 1
    return torch.Tensor(E)
