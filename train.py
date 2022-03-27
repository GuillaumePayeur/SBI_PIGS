################################################################################
## Training script
# It can
# - Generate a dataset of spectra with continuously varying parameters using
#   an emulator
# - Augment said spectra (add noise and continuum)
# - Train an autoencoder on said synthetic spectra and observed spectra
# - Train a density estimator to retrieve stellar parameters from observed
#   spectra using SNPE
# - Make a UMAP to visualize the remaining synthetic gap

## Datafiles
# Synthetic datafiles
datafile_synth = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_emulator_train.h5'
datafile_synth_emulated = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_synth_emulated_nonnormed.h5'
datafile_synth_augmented = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_synth_emulated_augmented_nonnormed.h5'
# Synthetic datafiles for multiround
datafile_synth_multiround = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_synth_multiround.h5'
datafile_synth_multiround_emulated = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_synth_multiround_emulated_nonnormed.h5'
datafile_synth_multiround_augmented = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_synth_multiround_emulated_augmented_nonnormed.h5'


# Datafile containing the observed spectra
datafile_obs = '/home/payeur/scratch/PIGS/SBI_PIGS/data/data_obs_combined_all_nonnormed.h5'

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
SNPE_itterations = 2
model = 'nsf'
hidden_features = 50
num_transforms = 10 #5
num_bins = 128 #64
max_epochs = 60
lr_sbi = 5e-4
batch_size_sbi = 128

## Models path
emulator_path = '/home/payeur/scratch/PIGS/sbi/models/emulator_v6.pth'
ae_path = '/home/payeur/scratch/PIGS/SBI_PIGS/models/ae_emulated_synth_obs_nonnormed.pth'
sbi_agent_path = '/home/payeur/scratch/PIGS/SBI_PIGS/models/sbi_agent_test.pkl'
densityEstimator_path = '/home/payeur/scratch/PIGS/SBI_PIGS/models/posterior_3.pkl'

## datafile with normalization parameters for labels
mean_path = '/home/payeur/scratch/PIGS/SBI_PIGS/data/mean.npy'
std_path = '/home/payeur/scratch/PIGS/SBI_PIGS/data/std.npy'

## Actions to take
create_emulated_dataset = True
augment_synth_spectra = True
train_emulator = False # Not a feature atm
train_autoencoder = True
train_densityEstimator = True
umap_synthgap = True
################################################################################
from scripts.train_DNN import *
from scripts.create_emulated_dataset import *
from scripts.augment_spectra import *
from scripts.train_ae import *
from scripts.sbi_functions import *
from scripts.sbi_train import *
from scripts.update_prior import *
from scripts.umap_latent_codes import *

# Generating dataset of emulated synthetic spectra
if create_emulated_dataset:
    print('creating emulated dataset')
    gen_emulated_dataset(datafile_synth,
                         datafile_synth_emulated,
                         emulator_path,
			 std_path)

# Augmenting spectra with noise
if augment_synth_spectra:
    print('augmenting synthetic spectra')
    config = [max_noise_std]
    augment_spectra(datafile_synth_emulated,
                    datafile_synth_augmented,
                    datafile_obs,
                    config)

# Training autoencoder
config_ae = [batch_size_ae,epochs_ae,filters_ae,lr_ae,latent_dim_ae]
if train_autoencoder:
    print('training autoencoder')
    train_auto_encoder(datafile_synth_augmented,
                       datafile_obs,
                       ae_path,
                       config_ae)

# Making UMAP of the synthetic gap
if umap_synthgap:
    umap_path = '/home/payeur/scratch/PIGS/SBI_PIGS/results/UMAP_raw.png'
    make_umap(datafile_synth_augmented,datafile_obs,umap_path,ae_path,latent_dim_ae)

# Training density estimator
if train_densityEstimator:
    print('training density estimator')
    config = [model,hidden_features,num_transforms,num_bins,max_epochs,lr_sbi,batch_size_sbi]
    train_density_estimator(datafile_synth_augmented,
                            ae_path,
                            densityEstimator_path,
                            sbi_agent_path,
                            config,
                            first_round=True)
    for _ in range(SNPE_itterations):
        gen_updated_parameters(densityEstimator_path,
                               ae_path,
                               mean_path,
                               std_path,
                               datafile_synth,
                               datafile_synth_multiround,
                               datafile_obs)
        gen_emulated_dataset(datafile_synth_multiround,
                             datafile_synth_multiround_emulated,
                             emulator_path,
    			             std_path)
        config_augment = [max_noise_std]
        augment_spectra(datafile_synth_multiround_emulated,
                        datafile_synth_multiround_augmented,
                        datafile_obs,
                        config_augment)
        train_auto_encoder(datafile_synth_multiround_augmented,
                           datafile_obs,
                           ae_path,
                           config_ae)
        train_density_estimator(datafile_synth_multiround_augmented,
                                ae_path,
                                densityEstimator_path,
                                sbi_agent_path,
                                config,
                                first_round=True)

# Making UMAP of the synthetic gap
if umap_synthgap:
    umap_path = '/home/payeur/scratch/PIGS/SBI_PIGS/results/UMAP_SNPE.png'
    make_umap(datafile_synth_multiround_augmented,datafile_obs,umap_path,ae_path,latent_dim_ae)
