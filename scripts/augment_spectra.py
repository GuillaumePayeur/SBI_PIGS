import numpy as np
import h5py

def add_noise(spectrum,max_noise):
    noise_std = np.random.uniform(0,max_noise)
    noise = np.random.normal(0,noise_std,spectrum.shape)
    spectrum = spectrum + spectrum*noise
    return spectrum, noise_std

def augment_spectra(input_filename,output_filename,config):
    max_noise = config[0]

    # Loading the spectra and copying to new file
    with h5py.File(input_filename,'r') as F_clean:
        F_noisy = h5py.File(output_filename,'w')

        for key in list(F_clean.keys()):
            # F_noisy.create_dataset(key,shape=F_clean[key].shape)
            F_clean.copy(F_clean[key],F_noisy,key)

    # Adding noise and saving the std of the noise
    n_spectra = F_noisy['spectra_asymnorm_noiseless'].shape[0]
    noise_std_array = np.zeros((n_spectra))
    for i in range(n_spectra):
        spectrum = F_noisy['spectra_asymnorm_noiseless'][i]
        noisy_spectrum, noise_std = add_noise(spectrum,max_noise)
        F_noisy['spectra_asymnorm_noiseless'][i] = noisy_spectrum
        noise_std_array[i] = noise_std
    F_noisy.create_dataset('noise_std',data=noise_std_array)
