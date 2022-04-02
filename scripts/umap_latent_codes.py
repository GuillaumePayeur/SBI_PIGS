import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import h5py
import numpy as np
import os
from tqdm import tqdm
import umap.umap_ as umap
import matplotlib.pyplot as plt

from scripts.train_DNN import *

# Encoder for the blue arm
class Encoder_Blue(nn.Module):
    def __init__(self,filters,latent_dim):
        super().__init__()
        self.filters = filters
        # Blue encoding layers
        self.conv1blue = nn.Conv1d(1,filters,kernel_size=7,stride=4)
        self.conv2blue = nn.Conv1d(filters,2*filters,kernel_size=7,stride=4)
        self.conv3blue = nn.Conv1d(2*filters,4*filters,kernel_size=7,stride=4)
        self.lin1blue = nn.Linear(4*filters*27,latent_dim)

    def forward(self,x):
        x = self.conv1blue(x)
        x = F.relu(x)
        x = self.conv2blue(x)
        x = F.relu(x)
        x = self.conv3blue(x)
        x = F.relu(x.view((-1,1,4*self.filters*27)))
        x = self.lin1blue(x)
        return x

# Decoder for the blue arm
class Decoder_Blue(nn.Module):
    def __init__(self,filters,latent_dim):
        super().__init__()
        self.filters = filters
        # Blue decoding layers
        self.lin2blue = nn.Linear(latent_dim,4*filters*27)
        self.dconv1blue = nn.ConvTranspose1d(4*filters,4*filters,kernel_size=7,stride=4)
        self.dconv2blue = nn.ConvTranspose1d(4*filters,2*filters,kernel_size=7,stride=4)
        self.dconv3blue = nn.ConvTranspose1d(2*filters,filters,kernel_size=7,stride=4)
        self.dconv4blue = nn.ConvTranspose1d(filters,1,kernel_size=1,stride=1)

    def forward(self,z):
        y = self.lin2blue(z).view((-1,4*self.filters,27))
        y = F.relu(self.dconv1blue(y))
        y = F.relu(self.dconv2blue(y))
        y = F.relu(self.dconv3blue(y))
        y = self.dconv4blue(y)
        return y

# Creating the AE
class AE(nn.Module):
    def __init__(self,filters,latent_dim):
        super().__init__()
        self.encoder_blue = Encoder_Blue(filters,latent_dim)
        self.decoder_blue = Decoder_Blue(filters,latent_dim)

    def forward(self,x_blue):
        z_blue = self.encoder_blue(x_blue)
        y_blue = self.decoder_blue(z_blue)
        return y_blue

    def encode(self,x_blue):
        z_blue = self.encoder_blue(x_blue)
        return z_blue

def encode_obs(datafile_obs,latent_dim,ae):
    with h5py.File(datafile_obs, 'r') as f:
        spectra = np.array(f['spectra'][:,94:94+1791])
        mean_obs = np.expand_dims(np.mean(spectra,axis=1),1)
        spectra = spectra/mean_obs
        # spectra = spectra/np.mean(spectra)
    n_spectra = spectra.shape[0]
    print(spectra[0,:])
    codes = np.zeros((n_spectra,latent_dim))

    for i in range(n_spectra//1000):
        spectra_batch = spectra[i*1000:(i+1)*1000,:]
        spectra_tensor = torch.from_numpy(spectra_batch).float().to('cuda:0').view(-1,1,1791)
        codes[i*1000:(i+1)*1000,:] = ae.encode(spectra_tensor).to('cpu').detach().numpy()[:,0,:]
    spectra_batch = spectra[n_spectra//1000*1000:,:]
    spectra_tensor = torch.from_numpy(spectra_batch).float().to('cuda:0').view(-1,1,1791)
    codes[n_spectra//1000*1000:,:] = ae.encode(spectra_tensor).to('cpu').detach().numpy()[:,0,:]
    return codes

def encode_synth(datafile_synth,latent_dim,ae):
    _,spectra = load_data(datafile_synth)
    #print(spectra[0,1791+84:1791+104])
    n_spectra = spectra.shape[0]
    spectra = spectra[:,94:94+1791]
    print(spectra[0,:])
    codes = np.zeros((n_spectra,latent_dim))

    for i in range(n_spectra//1000):
        spectra_batch = spectra[i*1000:(i+1)*1000,:]
        spectra_tensor = torch.from_numpy(spectra_batch).float().to('cuda:0').view(-1,1,1791)
        codes[i*1000:(i+1)*1000,:] = ae.encode(spectra_tensor).to('cpu').detach().numpy()[:,0,:]
    spectra_batch = spectra[n_spectra//1000*1000:,:]
    spectra_tensor = torch.from_numpy(spectra_batch).float().to('cuda:0').view(-1,1,1791)
    codes[n_spectra//1000*1000:,:] = ae.encode(spectra_tensor).to('cpu').detach().numpy()[:,0,:]
    return codes

def make_umap(datafile_synth,datafile_obs,umap_path,ae_path,ae_dim):
    ae = AE(64,ae_dim).to('cuda:0')
    ae.load_state_dict(torch.load(ae_path))
    ae.eval()

    codes_obs = encode_obs(datafile_obs,ae_dim,ae)
    codes_synth = encode_synth(datafile_synth,ae_dim,ae)

    colors = ['red']*codes_synth.shape[0]+['blue']*codes_obs.shape[0]
    codes = np.concatenate((codes_synth,codes_obs))
    embedding = umap.UMAP(densmap=False,n_neighbors=5,min_dist=0.3,metric='euclidean').fit_transform(codes)
    x = embedding[:,0]
    y = embedding[:,1]

    plt.scatter(x,y,s=0.1,alpha=0.5,c=colors)
    plt.title('UMAP of latent codes: Blue=Observed, Red=Synthetic')
    plt.savefig(umap_path)
    plt.clf()
