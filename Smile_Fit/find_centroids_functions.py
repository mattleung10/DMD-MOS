#Find centroids
#Matthew Leung
#December 2020
#######################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster
import scipy.stats
import os
import Globals as Globals
from get_ima_gia_files_info import get_data_file_info, get_wavelength_file_info, get_ima_file_info

np.random.seed(Globals.seed_num) #Set a fixed random seed for reproducibility with Scikit-Learn

"""
Process:
    Load txt file of simulated data from Zemax into pandas DataFrame
    Determine the wavelengths in the file from wav file from Zemax Wavelength Data
    From the IMA file, determine the number of micromirrors ON in the DMD (ON along slit), which is n
    for each wavelength:
        Use k-means clustering algorithm to cluster points into n clusters
        For each cluster, obtain the average/centroid position of the points in the cluster
        (NOT IN THIS FILE) Fit a quadratic function to those n centroids, y' = a * x'^2 + c; a and c are constants
    (NOT IN THIS FILE) Find a and c each as a function of wavelength
"""

#######################################################################################
#######################################################################################
#GLOBAL VARIABLES

"""
#Not used for this script:
field_size = 0 #Field Size from Zemax; will be assigned to later
M_S = 79.131160/60.115213 #Magnification of the spectrograph
"""
'''
w_data = None #list to store the wavelengths

wavelength_data_file = None
root = None #current working directory
dataset_dir = None #directory of dataset
ima_dir = None #directory of IMA files
gia_dir = None #directory of GIA files
centroids_dir = None #directory of csv files storing centroids
visualizations_dir = None #directory of images of centroid plots
'''
#######################################################################################
#######################################################################################

def plot_Zemax_GIA(data, save=True):
    """
    Plots the Zemax GIA data from a pd.DataFrame. Requires global variable w_data.
    INPUTS:
        ::pd.DataFrame:: data   #the pd.DataFrame used to generate the plot
        ::boolean:: save        #whether or not to save the plot in an image
    
    ###########################################################################
    #Global Variables
    global w_data
    """
    #Plot the data from Zemax Geometric Image Analysis
    #https://stackoverflow.com/questions/28144142/how-can-i-generate-a-colormap-array-from-a-simple-array-in-matplotlib
    #https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
    #https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
    norm = plt.Normalize()
    colors = plt.cm.get_cmap('brg')(norm(Globals.w_data))
    
    #https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend
    plots = []
    plt.figure(figsize=(10,10), dpi=100)
    for i in range(0,len(Globals.w_data),1):
        d = data.loc[data['w'] == i+1] # +1 to get the wavelength number
        plots += [plt.scatter(d['X-coord'], d['Y-coord'], marker='o', s = 0.05, color=colors[i])]
    plt.legend(plots, Globals.w_data, title='Wavelength [µm]', markerscale=20, bbox_to_anchor=(1.05, 1.0), loc='upper left') #Legend location: https://www.delftstack.com/howto/matplotlib/how-to-place-legend-outside-of-the-plot-in-matplotlib/
    plt.title('Zemax Geometric Image Analysis Data')
    plt.xlabel('$x_d$ [mm]')
    plt.ylabel('$y_d$ [mm]')
    plt.axis('scaled')
    plt.grid(linestyle='--')
    if save == True:
        plt.savefig("GIA.png", bbox_inches='tight')
    plt.show()
    return True

def plot_centroids(csv_file, w_data, save_dir='', save=True):
    """
    Plots the centroids contained in a csv file generated with k_means function.
    This is a variant of the plot_Zemax_GIA function
    INPUTS:
        ::string:: csv_file     #path of csv file which stores the pd.DataFrame which is used to generate plot
        ::list:: w_data         #list of wavelength values
        ::string:: save_dir     #path of directory in which the plot image will be saved
        ::boolean:: save        #whether or not to save the plot in an image
    """
    data = pd.read_csv(csv_file)
    csv_filename_only = os.path.basename(csv_file)[0:-4] #filename only, without full path or extension
    
    norm = plt.Normalize()
    colors = plt.cm.get_cmap('brg')(norm(w_data))
    
    #https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend
    plots = []
    plt.figure(figsize=(10,10), dpi=100)
    for i in range(0,len(w_data),1):
        d = data.loc[data['w'] == i+1] # +1 to get the wavelength number
        plots += [plt.scatter(d['X-coord'], d['Y-coord'], marker='o', s = 5, color=colors[i])]
    plt.legend(plots, w_data, title='Wavelength [µm]', markerscale=5, bbox_to_anchor=(1.05, 1.0), loc='upper left') #Legend location: https://www.delftstack.com/howto/matplotlib/how-to-place-legend-outside-of-the-plot-in-matplotlib/
    plt.title('Zemax GIA Data for ' + csv_filename_only)
    plt.xlabel('$x_d$ [mm]')
    plt.ylabel('$y_d$ [mm]')
    plt.axis('scaled')
    plt.xlim(-10,10) #put these after plt.axis('scaled')
    plt.ylim(-15,20)
    plt.grid(linestyle='--')
    if save == True and save_dir != '':
        savepath = os.path.join(save_dir, csv_filename_only + ".png") #full path to save file
        plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    return True

#######################################################################################
#######################################################################################

def k_means(data_file, ima_file, save=True):
    """
    Given a txt file containing the data generated from Zemax GIA, and a
    corresponding IMA file that was used to generate the GIA data, this
    function uses k-means to cluster points in the txt file and returns the
    centroids of the clusters. The centroids of the clusters are placed in a
    pd.DataFrame and this can be saved to a csv file.
    INPUTS:
        ::string:: data_file                #path of txt data file
        ::string:: ima_file                 #path of IMA file
        ::boolean:: save                    #whether or not to save centroids info to a csv file
    OUTPUTS:
        ::pd.DataFrame:: all_centroids_df   #contains the centroids
    
    ###########################################################################
    #Global Variables
    global w_data
    
    global wavelength_data_file
    global root
    global dataset_dir
    global ima_dir
    global gia_dir
    global centroids_dir
    global visualizations_dir
    """
    
    #skiprows used for pd.read_csv; NOT USED: field_size
    _, skiprows = get_data_file_info(data_file)
    #ON_pos used to determine number of clusters; NOT USED: dmd_dim, slit_pos 
    _, _, ON_pos = get_ima_file_info(ima_file, inverted=True)
    
    
    #data is a pandas DataFrame to store the data from the data_file
    #Load the txt file; Notes:
    #   - Data separated by "\t" (tab)
    #   - Skip first 14 rows
    #   - Skip last 9 rows (have to set engine='python' instead of 'c')
    #   - Zemax outputted a txt file that has a UTF-16 LE encoding
    data = pd.read_csv(data_file, sep="\t", header=None, skiprows=skiprows, skipfooter=9, encoding='utf-16-le', engine='python')
    data.columns = ['Ray #','X-field','Y-field','px','py','w','Wgt','X-coord','Y-coord']
    
    all_centroids = np.empty((0,3)) #Numpy array of all centroids               
    
    for i in range(0,len(Globals.w_data),1): #for each wavelength
        d = data.loc[data['w'] == i+1] #part of the DataFrame corresponding to wavelength number i+1
        points = d[['X-coord','Y-coord']].values #Numpy array of points/spots corresponding to wavelength number i+1
        
        #K-means clustering with Scikit-Learn
        kmeans = sklearn.cluster.KMeans(n_clusters=len(ON_pos))
        kmeans.fit(points)
        pred_kmeans = kmeans.predict(points)
        centers = kmeans.cluster_centers_ #centre of the clusters
        
        points = np.append(points, pred_kmeans[:,np.newaxis], axis=1)
        
        #See the results from k-means:
        """
        plt.scatter(points[:,0], points[:,1], c=pred_kmeans, s=0.1, cmap='viridis')
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=0.5)
        plt.title("Wavelength: " + str(Globals.w_data[i]) + "µm")
        plt.xlabel('x position [mm] (on image plane)')
        plt.ylabel('y position [mm] (on image plane)')
        plt.show()
        """
        
        centroids = np.zeros((len(ON_pos),2)) #stores the coordinates of the centroids for current wavelength
        
        for cluster_num in range(0,len(ON_pos),1): #for each cluster
            cluster_points = points[np.where(points[:,2] == cluster_num)] #array of points that fall in the cluster cluster_num
            
            #calculate mean and standard deviation
            mu_x, std_x = scipy.stats.norm.fit(cluster_points[:,0])
            mu_y, std_y = scipy.stats.norm.fit(cluster_points[:,1])
            
            centroids[cluster_num,0] = mu_x
            centroids[cluster_num,1] = mu_y
        
        centroids_sorted = centroids[centroids[:,0].argsort(kind='mergesort')] #sort the centroids array and save it as copy
        centroids_sorted = np.append(centroids_sorted, np.ones((len(ON_pos),1)) * (i+1), axis=1) #append the wavelength number
        all_centroids = np.append(all_centroids, centroids_sorted, axis=0) #append centroids_sorted to all_centroids
    
    all_centroids_df = pd.DataFrame(data=all_centroids, columns=['X-coord','Y-coord','w']) #Pandas DataFrame that stores the centroids' coordinates and wavelength number
    if save == True:
        filename_only = os.path.basename(data_file)[0:-4] #filename only, without full path or extension
        all_centroids_df.to_csv(os.path.join(Globals.centroids_dir, filename_only + "_centroids.csv"))
    return all_centroids_df

def find_centroids(dataset_dir_name, wavelength_filename='wavelengths.wav'):
    """
    Iterates through all GIA files in gia directory and calls k_means to find
    the centroids of each cluster.
    INPUTS:
        ::string:: dataset_dir_name        #name of the dataset directory relative to cwd
        ::string:: wavelength_filename     #base filename of the wavelength file, without full path
    Directory Structure (that is relevant to this script):
    dataset/
    ├── ima/
    ├── gia/
    ├── centroids/
    ├── visualizations/
    └── wavelengths.wav

    ###########################################################################
    #Global Variables
    global w_data
    
    global wavelength_data_file
    global root
    global dataset_dir
    global ima_dir
    global gia_dir
    global centroids_dir
    global visualizations_dir
    """
    Globals.root = os.getcwd() #current working directory
    Globals.dataset_dir = os.path.join(Globals.root, dataset_dir_name) #directory of dataset
    Globals.ima_dir = os.path.join(Globals.dataset_dir, 'ima') #directory of IMA files
    Globals.gia_dir = os.path.join(Globals.dataset_dir, 'gia') #directory of GIA files
    Globals.centroids_dir = os.path.join(Globals.dataset_dir, 'centroids') #directory of csv files storing centroids
    Globals.visualizations_dir = os.path.join(Globals.dataset_dir, 'visualizations') #directory of images of centroid plots
    if os.path.isdir(Globals.centroids_dir) == False: #make directory if it doesn't exist
        os.mkdir(Globals.centroids_dir)
    if os.path.isdir(Globals.visualizations_dir) == False: #make directory if it doesn't exist
        os.mkdir(Globals.visualizations_dir)
    Globals.wavelength_data_file = os.path.join(Globals.dataset_dir, wavelength_filename)
    
    for entry in os.scandir(Globals.gia_dir): #go through directory of GIA files
        if entry.is_file() and entry.path.endswith(".txt"):
            entry_string = os.path.basename(entry)
            entry_string_noext = entry_string[0:-4] #entry basename without file extension
            row = int(entry_string_noext.split("_")[-1]) #row number is last element of the split
            
            data_file = entry.path
            ima_file = os.path.join(Globals.ima_dir, 'dmd_' + str(row) + '_MOS.IMA')
            if os.path.isfile(ima_file) == True: #if the corresponding IMA file also exists
                #primary_wavelength_num is NOT used below
                Globals.w_data, _ = get_wavelength_file_info(Globals.wavelength_data_file)
                df = k_means(data_file, ima_file, save=True)
                
                saved_csv_file_path = os.path.join(Globals.centroids_dir, os.path.basename(data_file)[0:-4] + "_centroids.csv")
                plot_centroids(saved_csv_file_path, Globals.w_data, save_dir=Globals.visualizations_dir, save=True)
            else:
                print("Warning: Cannot find IMA file", ima_file)
    return True

"""
if __name__ == "__main__":
    find_centroids("dataset", wavelength_filename='wavelengths.wav')
"""
