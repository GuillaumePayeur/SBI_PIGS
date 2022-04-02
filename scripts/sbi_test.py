import pickle
import numpy as np
import matplotlib.pyplot as plt
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

def plot_pdf(samples,limits,n_bins,j):

    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 0
    fig, axes = plt.subplots(nrows=23,ncols=23)

    fig.set_size_inches(30,20)
    fig.set_dpi(500)

    # Plotting the diagonal histograms
    for i in range(23):
        axes[i,i].hist(
            samples[:,i],
            bins=60,
            range=(limits[i,0],limits[i,1]),
            density=True,
            )
#        axes[i,i].axvline(true_theta[0,i], color='red')
        axes[i,i].get_yaxis().set_visible(False)

    # Plotting the off diagonal histograms
    for i in range(23):
        for j in range(23):
            if (i != j) & (i < j):
                axes[i,j].axis('off')
                axes[i,j].hist2d(
                    samples[:,j],
                    samples[:,i],
                    bins=[int(n_bins[j]/5),int(n_bins[j]/5)],
                    range=[[limits[j,0],limits[j,1]],[limits[i,0],limits[i,1]]],
                    density=True,
                    )
#                axes[i,j].plot(true_theta[0,j],true_theta[0,i],color='red',marker='.',ms=2)
            elif i == j:
                axes[i,j].set_xlabel(labels[i], fontsize=18)
            else:
                axes[i,j].axis('off')

    plt.subplots_adjust(top=0.99,
                        bottom=0.05,
                        left=0.005,
                        right=0.995,
                        hspace=0.155,
                        wspace=0.165)
    plt.savefig('/home/payeur/scratch/PIGS/SBI_PIGS/results/pdf_plot_{}.h5'.format(j))

def get_mode(posterior,observations,limits,mean,std,n_bins):
    observations = observations.to('cuda:0')
    samples = posterior.sample((30000,), x=observations).cpu().numpy()
    samples = samples*std + mean

    theta_pred = np.zeros((23))
    for m in range(23):
        n,bins = np.histogram(
            a=samples[:,m],
            bins=n_bins[m],
            range=(limits[m,0],limits[m,1]),
            density=True)

        y = savgol_filter(n,29,11)
        theta_pred[m] = (bins[np.argmax(y)] + bins[np.argmax(y)+1])/2

    return theta_pred

def plot_random_posterior(mean_path,std_path):
    theta_pred = np.zeros((23))
    with h5py.File(datafile_test, 'r') as f:
        spectra = np.array(f['spectra'][:,94:94+1791])
        spectra = spectra/np.mean(spectra)

    for j in range(5):
        i = np.random.randint(0,spectra.shape[0])
        spectra = spectra[i,:]
        spectra = torch.from_numpy(spectra).float()
        code = encoder(spectra).to('cpu')

        mean = np.load('mean_path').reshape(1,23)
        std = np.load('std_path').reshape(1,23)

        n_bins = np.array([150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,150,450,450,450,150,300,300])

        for i in range(2):
            limits[:,i] = limits[:,i]*std[0,:] + mean[0,:]

        samples = posterior.sample((300000,), x=code).cpu().numpy()
        samples = samples*std + mean

        # theta_pred = get_mode(posterior,code,limits,mean,std,n_bins)

        plot_pdf(samples,limits,n_bins,j)

def predict(datafile_test,encoder,posterior,mean_path,std_path,n_spectra,limits,results_directory,results_name):
    # Getting the predicted theta
    theta_pred = np.zeros((n_spectra,23))
    with h5py.File(datafile_test, 'r') as f:
        spectra = np.array(f['spectra'][:,94:94+1791])
        mean_obs = np.expand_dims(np.mean(spectra,axis=1),1)
        spectra = spectra/mean_obs
        # spectra = spectra/np.mean(spectra)
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

            theta_pred_ = get_mode(posterior,code,limits,mean,std,n_bins)

            theta_pred[i:(i+1),:] = theta_pred_
        except Exception as E:
            print(E)
            valid[i] = 0

    # Saving the results
    np.save('{}/sols_sbi_{}.npy'.format(results_directory,results_name), theta_pred)
    np.save('{}/valid_sbi_{}.npy'.format(results_directory,results_name), valid)

def generate_predictions(datafile_synth,datafile_test,ae_path,posterior_path,mean_path,std_path,results_directory,results_name):
    logging.Logger.warning = warning

    # Loading the stellar labels from the test
    theta,_ = load_data(datafile_synth)

    min = np.amin(theta,axis=0)
    max = np.amax(theta,axis=0)

    delta = (max-min)/3
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
    with h5py.File(datafile_test, 'r') as f:
        spectra = np.array(f['spectra'][:,94:94+1791])
    n_spectra = spectra.shape[0]
    del spectra

    # Predicting parameters from mode of distributions
    # predict(datafile_test,encoder,posterior,mean_path,std_path,10,limits,results_directory,results_name)
    predict(datafile_test,encoder,posterior,mean_path,std_path,int(n_spectra),limits,results_directory,results_name)
    # Getting posterior for random spectra
    plot_random_posterior()
