import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import h5py
import numpy as np
import os
from tqdm import tqdm
################################################################################
# (Denoising) Autoencoder for the blue arm. This version trains on synthetic and observed data simultaneously
################################################################################
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

# Defining weighted l2 loss for spectra only
def l2(x_blue,y_blue):
    return torch.mean((x_blue-y_blue)**2)

# Defining weighted l1 loss for spectra only
def l1(x_blue,y_blue):
    return torch.mean(torch.abs(x_blue-y_blue))

# Defining weighted l2 loss for spectra only
def weighted_l2(x_blue,y_blue,e_blue):
    return torch.mean(((x_blue-y_blue)/e_blue)**2)

# Defining weighted l1 loss for spectra only
def weighted_l1(x_blue,y_blue,e_blue):
    return torch.mean(torch.abs((x_blue-y_blue)/e_blue))

# Function to load spectra, error spectra and wavelength grid into ram
def spectra_data_synth(filename_spectra):
    # Loading data into ram
    with h5py.File(filename_spectra, 'r') as f:
        # spectra, wavelengths
        spectra = np.array(f['spectra_asymnorm_noiseless'])
        spectra_blue = spectra[0:20000,:1980]
        # Cutting the spectra and grid for convenience
        spectra_blue = spectra_blue[:,94:94+1791]

        return spectra_blue

def spectra_data_obs(filename):
    # Loading data into ram
    with h5py.File(filename, 'r') as f:
        # Blue spectra, variances
        spectra_blue = np.array(f['spectra'][:,94:94+1791])
        spectra_blue = spectra_blue/np.mean(spectra_blue)
        e_spectra_blue = np.array(f['error_spectra'][:,94:94+1791])*0+1

        return spectra_blue, e_spectra_blue

# Function to create training batches
def get_batch_synth(spectra_blue_synth,batch_size,i):
    # Preparing spectra batch
    x_blue = torch.from_numpy(spectra_blue_synth[i*batch_size:(i+1)*batch_size]).to('cuda:0').float().view((-1,1,1791))
    return x_blue

# Function to create training batches
def get_batch_obs(spectra_blue_obs,e_spectra_blue,batch_size,i):
    # Preparing spectra batch
    x_blue = torch.from_numpy(spectra_blue_obs[i*batch_size:(i+1)*batch_size]).to('cuda:0').float().view((-1,1,1791))
    e_blue = torch.from_numpy(e_spectra_blue[i*batch_size:(i+1)*batch_size]).to('cuda:0').float().view((-1,1,1791))

    return x_blue,e_blue

# Function to perform a training epoch
def training_epoch(ae,spectra_blue_synth,spectra_blue_obs,e_spectra_blue,n_spectra_synth,n_spectra_obs,batch_size,scheduler,optimizer):
    n_batches_synth = n_spectra_synth//batch_size
    n_batches_obs = n_spectra_obs//batch_size
    n_batches = n_batches_synth + n_batches_obs

    batches_array = np.array(np.arange(n_batches))
    np.random.shuffle(batches_array)

    batches_array_train = batches_array[0:int(batches_array.shape[0]*0.9)]
    batches_array_val = batches_array[int(batches_array.shape[0]*0.9):]

    ae.train()
    SSE = 0
    for batch_index in tqdm(batches_array_train):
        if batch_index < n_batches_synth:
            x_blue = get_batch_synth(spectra_blue_synth,batch_size,batch_index)
            # loss & backprop
            ae.zero_grad()
            y_blue = ae(x_blue)
            loss = l2(x_blue,y_blue)
            SSE += batch_size*loss
            loss.backward()
            optimizer.step()
        else:
            x_blue, e_blue = get_batch_obs(spectra_blue_obs,e_spectra_blue,batch_size,batch_index-n_batches_synth)
            # loss & backprop
            ae.zero_grad()
            y_blue = ae(x_blue)
            loss = weighted_l2(x_blue,y_blue,e_blue)
            SSE += batch_size*loss
            loss.backward()
            optimizer.step()

    scheduler.step()
    MSE = (SSE/(batch_size*(n_batches))).detach().cpu().numpy()
    print('loss train l2', MSE)

    # Validation
    with torch.no_grad():
        ae.eval()
        SSE_l1 = 0
        SSE_l2 = 0
        for batch_index in tqdm(batches_array_val):
            if batch_index < n_batches_synth:
                x_blue = get_batch_synth(spectra_blue_synth,batch_size,batch_index)
                # loss & backprop
                y_blue = ae(x_blue)
                SSE_l1 += batch_size*l1(x_blue,y_blue)
                SSE_l2 += batch_size*l2(x_blue,y_blue)
            else:
                x_blue, e_blue = get_batch_obs(spectra_blue_obs,e_spectra_blue,batch_size,batch_index-n_batches_synth)
                # loss & backprop
                y_blue= ae(x_blue)
                SSE_l1 += batch_size*weighted_l1(x_blue,y_blue,e_blue)
                SSE_l2 += batch_size*weighted_l2(x_blue,y_blue,e_blue)

        MSE_l1 = (SSE_l1/(batch_size*(n_batches))).detach().cpu().numpy()
        MSE_l2 = (SSE_l2/(batch_size*(n_batches))).detach().cpu().numpy()
        print('ME l1: {}'.format(MSE_l1))
        print('ME l2: {}'.format(MSE_l2))

def train_auto_encoder(datafile_synth,datafile_obs,ae_path,config):
    batch_size,epochs,filters,lr,latent_dim = tuple(config)
    lr_enc_blue = lr
    lr_dec_blue = lr

    # Loading data into ram
    spectra_blue_synth = spectra_data_synth(datafile_synth)
    spectra_blue_obs, e_spectra_blue = spectra_data_obs(datafile_obs)
    n_spectra_synth = spectra_blue_synth.shape[0]
    n_spectra_obs = spectra_blue_obs.shape[0]
    n_spectra = n_spectra_synth + n_spectra_obs

    # Setting up the GPU
    GPU = torch.device('cuda:0')

    # Initializing the AE
    ae = AE(filters,latent_dim).to(GPU)
    # summary(ae,[(1,1791)])

    # Setting up optimizer and learning rates
    optimizer = optim.Adam([{'params': ae.encoder_blue.parameters(), 'lr': lr_enc_blue},
                            {'params': ae.decoder_blue.parameters(), 'lr': lr_dec_blue}])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)

    for epoch in range(epochs):
        print('starting epoch {}'.format(epoch+1))
        # Training epoch
        training_epoch(ae,spectra_blue_synth,spectra_blue_obs,e_spectra_blue,n_spectra_synth,n_spectra_obs,batch_size,scheduler,optimizer)
        # # Saving model every 20 epochs
        # if epoch % 20 == 0:
        #   pass
    torch.save(ae.state_dict(), ae_path)
