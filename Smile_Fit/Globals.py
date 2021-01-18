#Global Variables
#Matthew Leung
#December 2020
#######################################################################################

seed_num = 1000 #Numpy seed number; set a fixed random seed for reproducibility

field_size = 0 #Field Size from Zemax; will be assigned to later
M_S = 79.131160/60.115213 #Magnification of the spectrograph

w_data = None #list to store the wavelengths
primary_wavelength_num = 0 #primary wavelength number

root = None #current working directory
dataset_dir = None #directory of dataset
ima_dir = None #directory of IMA files
gia_dir = None #directory of GIA files
centroids_dir = None #directory of csv files storing centroids
smile_results_dir = None #directory to store outputs of smile fits
wavelength_data_file = None #full path of wavelength data file

dataset_dir_name = None #name of the dataset directory relative to cwd
smile_results_dir_name = None #name of the directory to store outputs of smile fits (not full path)
wavelength_filename = None #base filename of the wavelength file, without full path

lambda_y_a_c = None #np.ndarray
