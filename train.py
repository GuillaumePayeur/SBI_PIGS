################################################################################
## Training script
# It can
# - Generate a dataset of spectra with continuously varying parameters using an emulator
# - Augment said spectra (add noise)
# - Train an autoencoder on said synthetic spectra and observed spectra
# - Train a density estimator to retrieve stellar parameters from observed spectra

## Datafiles
# Datafile containing the raw synthetic spectra
datafile_synth = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_emulator_train.h5'
# Datafile containing the synthetic spectra generated using the emulator
datafile_synth_emulated = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_synth_emulated.h5'
# Datafile containing the augmented synthetic spectra
datafile_synth_augmented = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_synth_emulated_augmented.h5'
# Datafile containing the observed spectra
datafile_obs = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_obs_combined_all.hdf5'

## Parameters for the creation of emulated spectra

## Parameters for the augmentation of spectra
max_noise_std = 0.04

## Training hyperparameters for the autoencoder
batch_size_ae = 128
epochs_ae = 500
filters_ae = 64
lr_ae = 1e-3
latent_dim_ae = 120

## Training hyperparameters for the density estimator
model = 'nsf'
hidden_features = 50
num_transforms = 5
num_bins = 10
max_epochs = 100

## Models path
emulator_path = '/home/payeur/scratch/PIGS/sbi/models/emulator_v6.pth'
ae_path = '/home/payeur/scratch/PIGS/SBI_PIGS/models/ae_emulated_synth_obs_500.pth'
densityEstimator_path = '/home/payeur/scratch/PIGS/SBI_PIGS/posteriors/posterior_z_50_5_10_v3.pkl'

## Actions to take
create_emulated_dataset = True
augment_spectra = True
train_emulator = False # Not a feature atm
train_autoencoder = True
train_densityEstimator = True
################################################################################
import scripts.train_DNN
import scripts.create_emulated_dataset
import scripts.augment_spectra
import scripts.train_ae
import scripts.sbi_functions
import scripts.sbi_train

# Generating dataset of emulated synthetic spectra
if create_emulated_dataset:
    gen_emulated_dataset(datafile_synth,
                         datafile_synth_emulated,
                         emulator_path)

# Augmenting spectra with noise
if augment_spectra:
    config = [max_noise_std]
    augment_spectra(datafile_synth_emulated,
                    datafile_synth_augmented,
                    config)

# Training autoencoder
if train_autoencoder:
    config = [batch_size_ae,epochs_ae,filters_ae,lr_ae,latent_dim_ae]
    train_auto_encoder(datafile_synth_augmented,
                       datafile_obs,ae_path,
                       config)

# Training density estimator
if train_densityEstimator:
    config = [model,hidden_features,num_transforms,num_bins,max_epochs]
    train_density_estimator(datafile_train,
                            ae_path,
                            posterior_name,
                            config)
