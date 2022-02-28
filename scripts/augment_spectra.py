import numpy as np
import h5py

def denormalize(n_spectra,spectra_obs,F_augmented):
    n = n_spectra // spectra_obs.shape[0]
    for i,spectrum_obs in enumerate(spectra_obs):
        spectra_synth = np.array(F_augmented['spectra_asymnorm_noiseless'][i*n:(i+1)*n,94:94+1791])
        x = np.arange(0,spectrum_obs.shape[0])/1791
        if i%1000 == 0:
            print(i)
        for j in range(n):
            popt = np.polyfit(x,spectrum_obs,10)
            for k, parameter in enumerate(popt):
                if k>7:
                    popt[k] += np.random.uniform(-parameter/75,parameter/75)
            F_augmented['spectra_asymnorm_noiseless'][i*n+j,94:94+1791] = poly(x,popt)*spectra_synth[j,:]
    return F_augmented

def poly(x,coeffs):
    y = np.zeros(x.shape)
    for i, coeff in enumerate(np.flip(coeffs)):
        y += coeff*x**i
    return y

def add_noise(spectrum,max_noise):
    noise_std = np.random.uniform(0,max_noise)
    noise = np.random.normal(0,noise_std,spectrum.shape)
    spectrum = spectrum + spectrum*noise
    return spectrum, noise_std

def augment_spectra(input_filename,output_filename,datafile_obs,config):
    max_noise = config[0]

    # Loading the spectra and copying to new file
    with h5py.File(input_filename,'r') as F_clean:
        F_augmented = h5py.File(output_filename,'w')

        for key in list(F_clean.keys()):
            F_clean.copy(F_clean[key],F_augmented,key)

    F_obs = h5py.File(datafile_obs,'r')
    spectra_obs = np.array(F_obs['spectra'][:,94:94+1791])
    spectra_obs = spectra_obs/np.mean(spectra_obs)

    # denormalizing
    n_spectra = F_augmented['spectra_asymnorm_noiseless'].shape[0]
    F_augmented = denormalize(n_spectra,spectra_obs,F_augmented)

    # adding noise
    noise_std_array = np.zeros((n_spectra))
    for i in range(n_spectra):
        spectrum = F_augmented['spectra_asymnorm_noiseless'][i]
        noisy_spectrum, noise_std = add_noise(spectrum,max_noise)
        F_augmented['spectra_asymnorm_noiseless'][i] = noisy_spectrum
        noise_std_array[i] = noise_std
    F_augmented.create_dataset('noise_std',data=noise_std_array)
    print(list(F_augmented.keys()))
