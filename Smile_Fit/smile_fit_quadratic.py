#Smile Fit Quadratic
#Matthew Leung
#December 2020
#######################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
import os
import time
import Globals as Globals
from get_ima_gia_files_info import get_data_file_info, get_wavelength_file_info, get_ima_file_info

np.random.seed(Globals.seed_num) #Set a fixed random seed for reproducibility

"""
Process:
    (NOT IN THIS FILE) Load txt file of simulated data from Zemax into pandas DataFrame
    Determine the wavelengths in the file from wav file from Zemax Wavelength Data
    (NOT IN THIS FILE) From the IMA file, determine the number of micromirrors ON in the DMD (ON along slit), which is n
    for each wavelength:
        (NOT IN THIS FILE) Use k-means clustering algorithm to cluster points into n clusters
        For each cluster, obtain the average/centroid position of the points in the cluster
        Fit a quadratic function to those n centroids, y' = a * x'^2 + c; a and c are constants
    (NOT IN THIS FILE) Find a and c each as a function of wavelength
    
MAJOR CHANGES:
    - The quadratic function y' = a * x'^2 + c is now fitted in a separate function, called smile_fit_quadratic
    - In the original version of the smile_fit.py script, the quadratic function along with a and c
      as functions of wavelength were all fitted in the SAME function, smile_fit. This is not ideal
      because in reality a and c are functions of both wavelength and y_slit. Separating the quadratic
      fitting process allows for better fits for a and c later on.
    - lambda_c_y is now lambda_y_a_c; the column ordering was changed, and I've included the a values
    - Removed some plots
    - slit_y_pos was replaced with y_slit_dmd, which is the slit position on the DMD. Note that
      originally slit_y_pos, denoted as y_slit, was the slit position on the detector.
      y_slit_dmd is the slit position on the DMD, denoted as y_slitDMD. Also, the y in lambda_y_a_c
      is now y_slitDMD. Simply, slit_y_pos = y_slit_dmd * M_S
      
"""

#######################################################################################
#######################################################################################
#GLOBAL VARIABLES
'''
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
'''
#######################################################################################
#######################################################################################
#FIT FUNCTIONS

def smile_fit_fcn(x, a, c):
    return a*x**2 + c

def linear(x, m, b):
    return m*x + b

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def quartic_vertex(x, a, b, h, k):
    return a*(x-h)**4 + b*(x-h)**4 + k

#######################################################################################
#######################################################################################

def smile_fit_quadratic(centroids_file):
    """
    Given some file centroids_file containing all the centroid coordinates,
    returns a np.ndarray containing wavelength values and corresponding values
    of a and c from smile_fit_fcn
    INPUTS:
        ::string:: centroids_file               #path of the centroids file (csv)
    OUTPUTS:
        ::list:: smile_a_fit                    #list containing the fit parameters of a using a_coeff_fit_fcn
        ::list:: smile_c_fit                    #list containing the fit parameters of c using c_coeff_fit_fcn
        ::np.ndarray:: smile_fit_coeff_np       #array where first column is wavelength, second column is a, third column is c

    ###########################################################################
    #Global Variables
    global w_data
    global primary_wavelength_num
    global smile_results_dir
    """
    centroids_data = pd.read_csv(centroids_file, usecols=['X-coord','Y-coord','w'])

    filename_only = os.path.basename(centroids_file)[0:-4] #filename only, without full path or extension
    row = int(filename_only.split("_")[-2]) #row that the DMD micromirros are ON

    smile_fit_coeff = [] #list to store the smile fit coefficients 

    for i in range(0,len(Globals.w_data),1): #for each wavelength
        #centroids_sorted_df is a pd.DataFrame containing the centroids (which are assumed to be already sorted in x) for wavelength number i+1
        centroids_sorted_df = centroids_data.loc[centroids_data['w'] == i+1] # +1 to get the wavelength number
        centroids_sorted = pd.DataFrame.to_numpy(centroids_sorted_df) #convert to np.ndarray
        
        #Fit the centroids to the smile_fit_fcn
        params, params_covariance = scipy.optimize.curve_fit(smile_fit_fcn, centroids_sorted[:,0],centroids_sorted[:,1],p0=[0.2, 0.2])
        smile_fit_coeff += [[Globals.w_data[i],params[0],params[1]]]
        
        #######################################################################
        #Plot fitted function, centroids, points/spots, and predicted spot positions
        if i == 0 or i == Globals.primary_wavelength_num-1 or i == len(Globals.w_data)-1:
            plt.figure(figsize=(10,4), dpi=100)
            #plt.scatter(points[:,0], points[:,1], marker='o', s = 0.05, label='Spot') #points
            plt.scatter(centroids_sorted[:, 0], centroids_sorted[:, 1], c='black', s=50, alpha=0.5, label='Centroid')
            
            axes = plt.gca()
            x_min, x_max = axes.get_xlim() #get the min x and max x of the plot x axis
            #plt.plot(centroids_sorted[:,0],smile_fit_fcn(centroids_sorted[:,0],params[0],params[1]), label='Quadratic Fit')
            plt.plot(np.linspace(x_min,x_max,num=100),smile_fit_fcn(np.linspace(x_min,x_max,num=100),params[0],params[1]), label='Quadratic Fit')
            
            '''
            plt.axvline(x=predicted_spot_pos[0], color='black', linestyle='-', linewidth=0.5, alpha=0.5, label='Predicted spot x position')
            for s in range(1,len(predicted_spot_pos),1):
                plt.axvline(x=predicted_spot_pos[s], color='black', linestyle='-', linewidth=0.5, alpha=0.5) #predicted spot position
            '''
            plt.title("Wavelength: " + ("%.3f" % Globals.w_data[i]) + "µm; Slit Position: " + str(row))
            plt.xlabel('$x_d$ [mm]')
            plt.ylabel('$y_d$ [mm]')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            savefilename = os.path.join(Globals.smile_results_dir, ("%.3f" % Globals.w_data[i]) + "um_plot_" + str(row) + ".png")
            plt.savefig(savefilename, bbox_inches='tight') #https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
            plt.show()
    
    ###########################################################################
    #Convert to Numpy array for use when fitting and plotting later
    #smile_fit_coeff_np is an np.ndarray where first column is wavelength, second column is a, third column is c
    smile_fit_coeff_np = np.array(smile_fit_coeff)
    return smile_fit_coeff_np


#######################################################################################
#######################################################################################
            
def smile_fit_all_files(dataset_dir_name, smile_results_dir_name, wavelength_filename='wavelengths.wav'):
    """
    Iterates through all csv files in centroids directory and calls smile_fit
    to do make a fit for each csv file
    INPUTS:
        ::string:: dataset_dir_name         #name of the dataset directory relative to cwd
        ::string:: smile_results_dir_name   #name of the directory to store outputs of smile fits
        ::string:: wavelength_filename      #base filename of the wavelength file, without full path
    OUTPUTS:
        ::np.ndarray:: lambda_y_a_c         #array of 4 columns; 1st col is wavelength, 2nd col is slit_y_pos, 3rd col is a, 4th col is c
    Directory Structure (that is relevant to this script):
    dataset/
    ├── ima/
    ├── gia/
    ├── centroids/
    ├── smile_results/
    └── wavelengths.wav
    
    ###########################################################################
    #Global Variables
    global w_data
    global primary_wavelength_num
    global field_size
    global M_S

    global wavelength_data_file
    global root
    global dataset_dir
    global ima_dir
    global gia_dir
    global centroids_dir
    global smile_results_dir
    """
    
    Globals.root = os.getcwd() #current working directory
    Globals.dataset_dir = os.path.join(Globals.root, dataset_dir_name) #directory of dataset
    Globals.ima_dir = os.path.join(Globals.dataset_dir, 'ima') #directory of IMA files
    Globals.gia_dir = os.path.join(Globals.dataset_dir, 'gia') #directory of GIA files
    Globals.centroids_dir = os.path.join(Globals.dataset_dir, 'centroids') #directory of csv files storing centroids
    Globals.smile_results_dir = os.path.join(Globals.dataset_dir, smile_results_dir_name) #directory to store outputs of smile fits
    if len(os.listdir(Globals.ima_dir)) == 0 or len(os.listdir(Globals.gia_dir)) == 0 or len(os.listdir(Globals.centroids_dir)) == 0:
        #if directory is empty
        print("ERROR: Empty Directory")
        return False
    if os.path.isdir(Globals.smile_results_dir) == False: #make directory if it doesn't exist
        os.mkdir(Globals.smile_results_dir)
    Globals.wavelength_data_file = os.path.join(Globals.dataset_dir, wavelength_filename) #full path of wavelength data file
    
    Globals.w_data, Globals.primary_wavelength_num = get_wavelength_file_info(Globals.wavelength_data_file)
    
    #get the field_size
    for entry in os.scandir(Globals.gia_dir):
        if entry.is_file() and entry.path.endswith(".txt"):
            Globals.field_size, _ = get_data_file_info(entry.path) #get the field size; note that skiprows is not used
            break
    
    #get the dmd_dim
    for entry in os.scandir(Globals.ima_dir):
        if entry.is_file() and entry.path.endswith(".IMA"):
            dmd_dim, _, _ = get_ima_file_info(entry.path) #get dmd_dim; note that slit_pos and ON_pos not used (irrelevant)
            break
    
    ###########################################################################

    #slit_y_positions = [] #list of predicted y positions of slit [mm]
    lambda_y_a_c = np.empty((0,4)) #Numpy array of lambda, slit position, a, c, slit position
    start_time = time.time()
    for entry in os.scandir(Globals.centroids_dir): #go through directory of centroids files
        if entry.is_file() and entry.path.endswith(".csv"):
            entry_string = os.path.basename(entry)
            entry_string_noext = entry_string[0:-4] #entry basename without file extension
            
            #row that the DMD micromirros are ON
            row = int(entry_string_noext.split("_")[-2]) #row number is last element of the split
            slit_y_pos = -1*(dmd_dim/2 - (row + 1) + 0.5)/(dmd_dim/2) * Globals.field_size/2*Globals.M_S #predicted y position of the slit [mm] (on the detector)
            y_slit_dmd = -1*(dmd_dim/2 - (row + 1) + 0.5)/(dmd_dim/2) * Globals.field_size/2 #predicted y position of the slit [mm] (on the DMD)
            print("Row:", row, "| Slit y position on Detector:", slit_y_pos, "| Slit y position on DMD:", y_slit_dmd)
            
            centroids_file = entry.path #full path of centroid file
            smile_fit_coeff_np = smile_fit_quadratic(centroids_file)
            #slit_y_positions += [slit_y_pos]
            
            #smile_fit_coeff_np_new is a np.ndarray of 4 columns; 1st col is wavelength, 2nd col is slit_y_pos, 3rd col is a, 4th col is c
            #smile_fit_coeff_np_new = np.append(smile_fit_coeff_np[:,0:1], np.ones((len(Globals.w_data),1)) * slit_y_pos, axis=1) #extract 1st col and append 2nd col
            smile_fit_coeff_np_new = np.append(smile_fit_coeff_np[:,0:1], np.ones((len(Globals.w_data),1)) * y_slit_dmd, axis=1) #extract 1st col and append 2nd col
            smile_fit_coeff_np_new = np.append(smile_fit_coeff_np_new, smile_fit_coeff_np[:,1:3], axis=1) #append 3rd and 4th cols
        
            lambda_y_a_c = np.append(lambda_y_a_c, smile_fit_coeff_np_new, axis=0) #append smile_c_vs_wavelength to lambda_y_a_c
    end_time = time.time()
    print('Fitting all files took', end_time-start_time, 'seconds')    
    
    np.savetxt(os.path.join(Globals.smile_results_dir,'lambda_y_a_c.csv'), lambda_y_a_c, delimiter=",") #save Numpy array as csv
    return lambda_y_a_c

"""
if __name__ == "__main__":
    smile_fit_all_files('dataset', 'smile_results')
"""
