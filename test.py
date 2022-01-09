################################################################################
## Testing script
# It can
# - Use a trained autoencoder and density estimator to return predictions for
#   stellar parameters of observed spectra

## Datafiles
# datafile of synthetic spectra (for determining the limits of the stellar paramter PDFs)
datafile_synth = '/home/payeur/scratch/PIGS/sbi/data/data_emulator_test.h5'
# datafile of observed spectra
datafile_obs = '/home/payeur/scratch/PIGS/sbi/data/data_obs_combined_all.hdf5'
# datafile with normalization parameters for labels
mean_path = '/home/payeur/scratch/PIGS/SBI_PIGS/data/mean.npy'
std_path = '/home/payeur/scratch/PIGS/SBI_PIGS/data/std.npy'

## Models path
ae_path = '/home/payeur/scratch/PIGS/SBI_PIGS/models/ae_emulated_synth_obs_500.pth'
densityEstimator_path = '/home/payeur/scratch/PIGS/SBI_PIGS/posteriors/posterior_z_50_5_10_v3.pkl'

# name of the results
results_directory = '/home/payeur/scratch/PIGS/SBI_PIGS/results'
results_name = 'v11'

################################################################################
import scripts.train_DNN
import scripts.create_emulated_dataset
import scripts.augment_spectra
import scripts.train_ae
import scripts.sbi_functions
import scripts.sbi_train

generate_predictions(datafile_synth,
                     datafile_obs,ae_path,
                     densityEstimator_path,
                     mean_path
                     std_path
                     results_directory,
                     results_name)
