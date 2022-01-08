import torch
import numpy as np
import sbi.utils as utils
from sbi.inference.base import infer
import sbi.inference.snle.snle_a as SNLE
from sbi.analysis import pairplot
import pickle

from z_sbi_functions import *
from train_DNN_search_v2 import *
from autoencoder_synth_blue import *

################################################################################
# SBI using SNPE_C
################################################################################
data_file_train = '/home/payeur/scratch/PIGS/sbi/data/data_synth_emulated_noisy.h5'
posterior_name = '/home/payeur/scratch/PIGS/sbi/posteriors/posterior_z_50_5_10_v3.pkl'
path = '/home/payeur/scratch/PIGS/sbi/models/ae_emulated_381.pth'
model = 'nsf'
hidden_features = 50
num_transforms = 5
num_bins = 10
max_epochs = 100
################################################################################

if __name__ == '__main__':
    # Loading the stellar labels from the training data
    theta,x = load_data(data_file_train)
    # x = disturb_spectra(x)

    # Restricting the spectra to the blue arm
    x = x[:,94:94+1791]

    # Creating the prior
    min = np.amin(theta,axis=0)
    max = np.amax(theta,axis=0)

    delta = (max-min)/3
    min = min-delta
    max = max+delta

    prior = create_prior(torch.from_numpy(min),torch.from_numpy(max))

    # Loading the encoder NN
    encoder = get_encoder(path)

    # Getting the simulations data
    theta = torch.from_numpy(theta[:]).float()
    x = torch.from_numpy(x).float()
    z = create_codes(x,encoder).to('cpu')

    # Getting the approximate posterior
    density_estimator = get_density_estimator(model,
        hidden_features,
        num_transforms,
        num_bins)
    posterior = get_posterior(density_estimator,prior,theta,z,max_epochs)

    # Saving the posterior
    with open(posterior_name,'wb') as handle:
        pickle.dump(posterior, handle)
