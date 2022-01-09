import numpy as np
import h5py

################################################################################
# functions to create a synthetic training dataset using the emulator starnet
# emulator.
batch_size = 1000
delta = np.array([0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.2,200,0.3,0,0.1,0])
parameters = ['Al', 'Ba', 'C', 'Ca', 'Co', 'Cr', 'Eu', 'Mg', 'Mn', 'N', 'Na', 'Ni', 'O', 'Si', 'Sr', 'Ti', 'Zn', 'logg', 'teff', 'm_h', 'vsini', 'vt', 'vrad']
################################################################################

def gen_emulated_dataset(input_filename,output_filename,emulator_path):
    # Loading synthetic grid to get range of parameters
    with h5py.File(input_filename) as grid:
        n_spectra = np.array(grid['Al']).shape[0]
        labels_grid = np.zeros((n_spectra,23))
        for i, parameter in enumerate(parameters):
            labels_grid[:,i] = np.array(grid[parameter])

    # Loading the emulator model
    emulator = DNN([200,268,397,569,739]).to('cuda:0')
    emulator.load_state_dict(torch.load(emulator_path))
    emulator.eval()

    # Generating the labels
    print('generating {} spectra'.format(n_spectra))
    std = np.load('std.npy')
    delta = delta/std
    labels = np.zeros((n_spectra,23))
    for i in range(n_spectra):
        #print(labels_grid[i,:])
        for j in range(23):
            labels[i,j] = labels_grid[i,j] + np.random.uniform(-delta[j]/2,delta[j]/2,(1))
        #print(labels[i,:])
        #quit()

    # Generating the spectra
    spectra = np.zeros((n_spectra,3829))
    for i in range(int(n_spectra/batch_size)):
        if i&10 == 0:
            print(i*batch_size)
        batch = labels[i*batch_size:(i+1)*batch_size,:]
        batch = torch.from_numpy(batch).float().to('cuda:0')
        spectra[i*batch_size:(i+1)*batch_size,:] = emulator(batch).detach().cpu()

    # Saving the spectra to a file
    with h5py.File(output_filename,'w') as F:
        for i, parameter in enumerate(parameters):
            F.create_dataset(parameter,data=labels[:,i])
        F.create_dataset('spectra_asymnorm_noiseless',data=spectra)

if __name__ == '__main__':
    gen_emulated_dataset(input_filename,output_filename,emulator_path)
