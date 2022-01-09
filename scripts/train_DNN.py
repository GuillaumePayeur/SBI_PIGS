import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import h5py
import os

# Creating NN
def DNN(sizes):
    # Linear layers
    layers = []
    layers.append(nn.Linear(23,sizes[0]))
    layers.append(nn.LeakyReLU(0.07))
    for i in range(len(sizes)-1):
        layers.append(nn.Linear(sizes[i],sizes[i+1]))
        layers.append(nn.LeakyReLU(0.07))
    layers.append(nn.Linear(sizes[-1],3829))

    model = nn.Sequential(*layers)
    return model

# Defining l2 loss
def l2(y_pred,y_true):
    return torch.sqrt(torch.mean((y_pred-y_true)**2))

# Defining l2 loss
def l1(y_pred,y_true):
    return (torch.mean(torch.abs(y_pred-y_true)))

# Loading data
def load_data(data_file):
    F = h5py.File(data_file,'r')
    names = ['Al','Ba','C','Ca','Co','Cr','Eu','Mg','Mn','N','Na','Ni','O','Si','Sr','Ti','Zn','logg','teff','m_h','vsini','vt','vrad']
    n_spectra = np.array(F['spectra_asymnorm_noiseless']).shape[0]
    # Loading inputs
    inputs = np.zeros((n_spectra,23))
    for i in range(23):
        inputs[:,i] = np.array(F[names[i]])
    # Loading labels
    labels = np.array(F['spectra_asymnorm_noiseless'])
    return inputs,labels

# Function to load observed spectra
def load_data_obs(data_file):
    with h5py.File(data_file,'r') as F:
        spectra = np.ones((F['spectra_red'].shape[0],3829))
        spectra[:,1980+25:1980+25+1791] = np.array(F['spectra_red'])
        spectra[:,94:94+1791] = np.array(F['spectra_blue'])
        e_spectra = 1e10*np.ones((F['e_spectra_red'].shape[0],3829))
        e_spectra[:,1980+25:1980+25+1791] = np.array(F['e_spectra_red'])
        e_spectra[:,94:94+1791] = np.array(F['e_spectra_blue'])

        # Changing errors to help emulator fits
        e_spectra[:,1980+25:1980+25+1791] = 0*e_spectra[:,1980+25:1980+25+1791]+1e10
        e_spectra[:,2471:2476] = 0*e_spectra[:,2471:2476]+1e10
        e_spectra[:,2651:2656] = 0*e_spectra[:,2651:2656]+1e10
        e_spectra[:,3142:3147] = 0*e_spectra[:,3142:3147]+1e10
        e_spectra[np.where(e_spectra > 1.25)] = 1e10
    return spectra, e_spectra

# Function to get a batch of data
def get_batch(inputs,labels,batch_size,n):
    x = torch.from_numpy(inputs[n*batch_size:(n+1)*batch_size]).to('cuda:0').float().view(-1,1,23)
    y = torch.from_numpy(labels[n*batch_size:(n+1)*batch_size]).to('cuda:0').float().view(-1,1,3829)
    return x,y

# Function to execute a training epoch
def train_epoch(NN,inputs,labels,train_frac,batch_size,scheduler,optimizer):
    NN.train()
    loss = 0
    # Passing the data through the NN
    n_train = int(inputs.shape[0]*train_frac)
    for i in range(n_train//batch_size):
        NN.zero_grad()
        x,y_true = get_batch(inputs,labels,batch_size,i)
        y_pred = NN(x)
        batch_loss = l1(y_pred,y_true)
        batch_loss.backward()
        loss += batch_loss*batch_size
        optimizer.step()
    scheduler.step()
    MSE = (loss/(batch_size*(n_train//batch_size))).detach().cpu().numpy()
    return MSE

# Function to execute a validation epoch
def val_epoch(NN,inputs,labels,train_frac,batch_size):
    NN.eval()
    with torch.no_grad():
        loss = 0
        # Passing the data through the NN
        n_train = int(inputs.shape[0]*train_frac)
        n_val = inputs.shape[0] - n_train
        for i in range(n_train//batch_size,(n_train+n_val)//batch_size):
            x,y_true = get_batch(inputs,labels,batch_size,i)
            y_pred = NN(x)
            loss += batch_size*l1(y_pred,y_true)
        MSE = (loss/(batch_size*(n_val//batch_size))).detach().cpu().numpy()
        return MSE

def train_NN(config,name):
    # Loading data into ram
    inputs,labels = load_data(data_file)

    # Initializing the AE
    n,sizes,lr,batch_size = config
    NN = DNN(sizes).to('cuda:0')
    # NN.load_state_dict(torch.load('models\\ae_201.pth'))

    # Saving results to txt file
    f = open(('{}.txt'.format(name)), 'w')

    # Setting up optimizer and learning rates
    optimizer = optim.Adam([{'params': NN.parameters(), 'lr': lr}])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)

    for epoch in range(epochs_max):
        # Training epoch
        loss_train = train_epoch(NN,inputs,labels,0.9,batch_size,scheduler,optimizer)
        loss_val = val_epoch(NN,inputs,labels,0.9,batch_size)
        print(loss_train,loss_val)
        f.write('{}, '.format(loss_train))
        f.write('{}'.format(loss_val))
        f.write('\n')

        if epoch%25 == 0:
            torch.save(NN.state_dict(), 'models\\emulator_{}.pth'.format(epoch+1))

def main():
    for sample in range(num_samples):
        num_layers = np.random.randint(3,7)
        sizes = np.random.randint(0,200,(num_layers))
        sizes[0] += 50
        for i in range(num_layers-1):
            sizes[i+1] += sizes[i]
        lr = np.random.uniform(1e-3,1e-2)
        batch_size = 128*np.random.randint(1,3)

        name = 'n-{}'.format(num_layers)
        for i in range(num_layers):
            name = name + '_{}'.format(sizes[i])
        name = name + '_lr-{}_batchsize-{}'.format(lr,batch_size)
        config = [num_layers,sizes,lr,batch_size]

        print('starting {}'.format(name))
        train_NN(config,name)

if __name__=='__main__':
    # main()
    train_NN([5,[200,268,397,569,739],2.3e-3,256],'a')
