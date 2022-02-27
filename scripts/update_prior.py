import torch
import pickle
import numpy as np
from scipy.signal import savgol_filter
import time

from scripts.train_DNN import *
from scripts.train_ae import *
from scripts.sbi_functions import *

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
    samples = samples.reshape(5,23)

    return samples

def sample(n_spectra,limits,encoder,posterior,mean_path,std_path,input_filename,output_filename,datafile_obs):
    # Getting the predicted theta
    theta_next_round = np.zeros((n_spectra*5,23))
    spectra,_ = load_data_obs(datafile_obs)
    spectra = spectra[0:n_spectra,94:94+1791]
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


    parameters = ['Al','Ba','C','Ca','Co','Cr','Eu','Mg','Mn','N','Na','Ni','O','Si','Sr','Ti','Zn','logg','teff','m_h','vsini','vt','vrad']
    with h5py.File(output_filename,'w') as F_next:
        for index, parameter in enumerate(parameters):
            F_next[parameter] = theta_next_round[:,index]

    # Loading the spectra and copying to new file
#    with h5py.File(input_filename,'r') as F:
#        F_next = h5py.File(output_filename,'w')

#        for key in list(F.keys()):
#            F_next.copy(F[key],F_next,key)

#    parameters = ['Al','Ba','C','Ca','Co','Cr','Eu','Mg','Mn','N','Na','Ni','O','Si','Sr','Ti','Zn','logg','teff','m_h','vsini','vt','vrad']
#    for index, parameter in enumerate(parameters):
#        F_next[parameter][:] = theta_next_round[:,index]
#    with h5py.File(input_filename,'r') as F:
#        F_next['spectra_asymnorm_noiseless'] = np.array(F['spectra_asymnorm_noiseless'])[0:theta_next_round.shape[0]]

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
    with h5py.File(datafile_obs, 'r') as f:
        spectra = np.array(f['spectra'][:,94:94+1791])
    n_spectra = spectra.shape[0]
    del spectra

    #sample(1000,limits,encoder,posterior,mean_path,std_path,input_filename,output_filename,datafile_obs)
    sample(int(n_spectra),limits,encoder,posterior,mean_path,std_path,input_filename,output_filename,datafile_obs)
