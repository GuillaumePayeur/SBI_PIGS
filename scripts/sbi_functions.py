import torch
import numpy as np
import sbi.utils as utils
from sbi.inference.base import infer
import sbi.inference.snpe.snpe_c as SNPE
from sbi.analysis import pairplot
from scipy.signal import savgol_filter

from scripts.train_ae import *
################################################################################
# Helper functions for SBI
################################################################################

def get_encoder(path):
    ae = AE(64,120).to('cuda:0')
    ae.load_state_dict(torch.load(path))
    ae.eval()

    def encoder(x):
        spectra_blue = x.view(-1,1,1791).to('cuda:0')
        z = ae.encode(spectra_blue).view(-1,120)
        return z
    return encoder

def create_prior(low,high):
    prior = utils.BoxUniform(
        low=low,
        high=high,
	device='cuda')
    return prior

def create_codes(x,encoder):
    with torch.no_grad():
        print(x.shape)
        z = torch.zeros((x.shape[0],120),device='cuda:0')
        for i in range(x.shape[0]//1000):
            z[i*1000:(i+1)*1000,:] = encoder(x[i*1000:(i+1)*1000,:])
        z[(x.shape[0]//1000)*1000:] = encoder(x[(x.shape[0]//1000)*1000:])
        return z

def get_density_estimator(model,hidden_features,num_transforms,num_bins):
    return utils.get_nn_models.posterior_nn(model=model,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        num_bins=num_bins)

def get_posterior(density_estimator,sbi_agent_path,prior,theta,z,max_epochs):
    if sbi_agent_path:
        with open(sbi_agent_path,'rb') as handle:
            sbi_agent = pickle.load(handle)
    else:
        sbi_agent = SNPE.SNPE_C(
            prior,
    	device='cuda',
            density_estimator=density_estimator)
    sbi_agent.append_simulations(theta,z)
    posterior = sbi_agent.train(
        show_train_summary=True,
        training_batch_size=64,
        learning_rate=5e-4,
        stop_after_epochs=20,
        max_num_epochs=max_epochs)
    posterior = sbi_agent.build_posterior(posterior)
    return sbi_agent, posterior

def disturb_spectra(spectra):
    noise = np.random.uniform(-0.005,0.005,spectra.shape[0])
    noise = np.reshape(np.repeat(noise,spectra.shape[1],axis=0),(spectra.shape[0],spectra.shape[1]))
    return spectra+noise

def continuum_error(spectra):
    b = np.random.uniform(0.9,1.05,spectra.shape[0])
    noise = np.ones(spectra.shape)
    for i in range(spectra.shape[0]):
        for j in range(500):
            noise[i,j] = b[i]+(j/500)*(1-b[i])
    return spectra*noise
