import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import time

from train_DNN_search_v2 import *
from autoencoder_synth_blue_temp import *
from z_sbi_functions_temp import *

import warnings
import logging

# Function to allow catching logging warnings
def warning(self, message, *args, **kws):
    if self.isEnabledFor(logging.WARNING):
        self._log(logging.WARNING, message, args, **kws)
        raise Exception(message)

logging.Logger.warning = warning

def get_theta(posterior,observations,limits,mean,std,n_bins):
    samples = posterior.sample((5,), x=observations).cpu().numpy()
    samples = (samples*std + mean).reshape(5,3)

    return samples

def sample(n_spectra,limits,mean_path,std_path,input_filename,output_filename):
    # Getting the predicted theta
    theta_next_round = np.zeros((n_spectra*5,23))
    spectra,_ = load_data_obs(data_file_test)
    spectra = spectra[0:n_spectra,94+512:94+1791]
    spectra = torch.from_numpy(spectra).float()
    z = encoder(spectra).to('cpu')
    valid = np.ones((n_spectra))

    mean = np.load(mean_path).reshape(1,23)
    std = np.load(std_path).reshape(1,23)

    n_bins = np.array([150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,450,450,450,150,300,300])

    for i in range(2):
        limits[:,i] = limits[:,i]*std[0,:] + mean[0,:]

    for i in range(n_spectra):
        try:
            code = z[i:(i+1),:].float().view(1,-1)

            samples = get_theta(posterior,code,limits,mean,std,n_bins)

            theta_next_round[i*5:(i+1)*5,:] = samples
        except Exception as E:
            print(E)
            theta_next_round[i*5:(i+1)*5,:] = theta_next_round[(i-1)*5:(i)*5,:]

    print(theta_next_round)

    # Loading the spectra and copying to new file
    with h5py.File(input_filename,'r') as F:
        F_next = h5py.File(output_filename,'w')

        for key in list(F.keys()):
            F_next.copy(F_next[key],F,key)

    parameters = ['Al', 'Ba', 'C', 'Ca', 'Co', 'Cr', 'Eu', 'Mg', 'Mn', 'N', 'Na', 'Ni', 'O', 'Si', 'Sr', 'Ti', 'Zn', 'logg', 'teff', 'm_h', 'vsini', 'vt', 'vrad']
    for index, parameter in enumerate(parameters):
        F[parameter] = theta_next_round[:,index]
    F['spectra_asymnorm_noiseless'] = np.array(F['spectra_asymnorm_noiseless'])[0:theta_next_round.shape[0]]

def gen_updated_parameters(posterior_path,ae_path,mean_path,std_path,input_filename,output_filename,datafile_obs):
    # Loading the stellar labels from the test
    theta,_ = load_data(input_filename)

    min = np.amin(theta,axis=0)
    max = np.amax(theta,axis=0)

    delta = (max-min)/12
    min = min-delta
    max = max+delta

    limits = []
    for i in range(23):
        limits.append([min[i],max[i]])
    limits = np.array(limits)
    del theta

    # Loading the posterior
    with open(posterior_path,'rb') as handle:
        posterior = pickle.load(handle)

    # Loading the encoder NN
    encoder = get_encoder(ae_path)

    # Finding the number of observed spectra
    spectra,_ = load_data_obs(datafile_obs)
    n_spectra = spectra.shape[0]
    del spectra

    sample(10,limits)
    # sample(int(n_spectra),limits)
