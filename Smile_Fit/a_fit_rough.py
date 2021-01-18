#Rough a_s fit
#Matthew Leung
#December 2020
#######################################################################################

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import scipy.optimize
import os
import time

import Globals as Globals
from find_centroids_functions import find_centroids
from smile_fit_quadratic import smile_fit_all_files
from main import init_dir, check_and_load_lambda_y_a_c, setup

np.random.seed(Globals.seed_num) #Set a fixed random seed for reproducibility

#######################################################################################
#######################################################################################
#FIT FUNCTIONS FOR a_s

def smile_a_fit_fcn(lambda_y, a2, a1, a0, b2, b1, b0):
    l = lambda_y[0]
    y = lambda_y[1]
    return (a2*l**2 + a1*l + a0)*y + (b2*l**2 + b1*l + b0)

#######################################################################################
#######################################################################################

def smile_a_fit(fit_fcn, p0=None):
    """
    Performs a fit using fit_fcn, with parameters determined by
    scipy.optimize.curve_fit.
    INPUTS:
        ::function:: fit_fcn                #the fit function to be used
        ::list:: p0                         #optional parameter; intial values of the parameters for use with scipy.optimize.integrate
    """
    if check_and_load_lambda_y_a_c() == False:
        return None
    lambda_points = Globals.lambda_y_a_c[:,0] #wavelength values
    y_points = Globals.lambda_y_a_c[:,1] #y values
    lambda_y = [lambda_points, y_points] #list of two np.ndarrays
    a_points = Globals.lambda_y_a_c[:,2] #a values

    
    fit_result_dir = os.path.join(Globals.smile_results_dir, "a_fit_" + fit_fcn.__name__) #directory to store results of the fit
    if os.path.isdir(fit_result_dir) == False: #make directory if it doesn't exist
        os.mkdir(fit_result_dir)
    
    #Fit the function
    #https://stackoverflow.com/questions/17934198/curve-fitting-in-scipy-with-3d-data-and-parameters/17934370
    #https://stackoverflow.com/questions/15413217/fitting-3d-points-python
    params, params_covariance = scipy.optimize.curve_fit(fit_fcn, lambda_y, a_points, p0=p0, maxfev=10000) #Note: I set maxfev=10000; feel free to remove this

    a_fit = fit_fcn(lambda_y, *params) #values of c using the fit
    visualize_a_error(a_fit, params, fit_fcn.__name__, fit_result_dir) #save results and see the error
    return params, params_covariance

def visualize_a_error(a_pred, params=None, fit_fcn_name=None, savedir=None, angle=None):
    """
    Given a_pred from a fit function, this function plots the error of the fit
    function. The plot can be saved as a PNG, and the errors and fit parameters
    can be saved in a TXT file.
    With the exception of a_pred, everything is an optional parameter.
    Results will only be saved if params, fit_fcn_name, and savedir are
    provided.
    INPUTS:
        ::np.ndarray:: a_pred                   #array with predicted a values from the fit
        ::np.ndarray or list:: params           #fit parameters
        ::string:: fit_fcn_name                 #name of the fit function
        ::string:: savedir                      #directory to save the results
        ::list:: angle                          #list of 2 elements; elev=angle[0], azim=angle[1]
    """
    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = plt.axes(projection="3d")

    x_points = Globals.lambda_y_a_c[:,0] #lambda (wavelength)
    y_points = Globals.lambda_y_a_c[:,1] #y
    z_points = a_pred - Globals.lambda_y_a_c[:,2] #error in a_s
    
    MSE = np.sum(np.power(z_points,2))/np.size(z_points) #mean squared error
    
    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='jet') #original cmap was 'hsv'
    ax.set_xlabel('$\lambda$ [µm]')
    ax.set_ylabel('$y$ (on DMD) [mm]')
    ax.set_zlabel('$a_s$ error [mm]')
    ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0)) #with scientific notation

    if type(angle) == type([]): #check if angle is a list
        if len(angle) == 2: #check if length of angle is 2
            ax.view_init(angle[0], angle[1]) #elev=angle[0], azim=angle[1]
            plt.savefig(os.path.join(savedir, "a_fit_" + fit_fcn_name + "_" + str(angle[0]) + "_" + str(angle[1]) + ".png"), bbox_inches='tight')
            plt.show()
            return True
    
    if savedir != None and fit_fcn_name != None:
        plt.savefig(os.path.join(savedir, "a_fit_" + fit_fcn_name + ".png"), bbox_inches='tight')
        with open(os.path.join(savedir, "a_fit_" + fit_fcn_name + ".txt"), 'w') as f:
            if type(params) != type(None):
                f.write("params\n")
                for elem in params: #save the fit parameters
                    f.write("%.8f\n" % elem)
            f.write("Maximum error is: " + str(np.max(z_points)) + " mm\n")
            f.write("Minimum error is: " + str(np.min(z_points)) + " mm\n")
            f.write("Maximum absolute error is: " + str(np.max(np.abs(z_points))) + " mm\n")
            f.write("Minimum absolute error is: " + str(np.min(np.abs(z_points))) + " mm\n")
            f.write("Mean squared error is: " + str(MSE) + " mm^2\n")
    
    plt.show()
    if type(params) != type(None):
        print("Fit parameters:", params)
    print("Maximum error is:", np.max(z_points), "mm")
    print("Minimum error is:", np.min(z_points), "mm")
    print("Maximum absolute error is:", np.max(np.abs(z_points)), "mm")
    print("Minimum absolute error is:", np.min(np.abs(z_points)), "mm")
    print("Mean squared error is:", MSE, "mm^2")
    return True

def plot_a_VS_c(include_line=True, save=False):
    """
    Creates a 2D plot of a VS c
    INPUTS:
        ::boolean:: include_line        #whether or not to include a line that joins points of same wavelength
        ::boolean:: save                #whether or not to save the plot
    """
    if check_and_load_lambda_y_a_c() == False:
        return False
    a_points = Globals.lambda_y_a_c[:,2] #a values
    
    lambda_values = np.unique(Globals.lambda_y_a_c[:,0]) #np.ndarray of unique wavelength values
    lambda_values_list = lambda_values.tolist() #list of unique wavelength values
    
    norm = plt.Normalize()
    colors = plt.cm.get_cmap('brg')(norm(lambda_values_list))
    
    plots = []
    plt.figure(figsize=(6,4), dpi=100)
    for i in range(0,len(lambda_values_list),1):
        d = Globals.lambda_y_a_c[np.where(Globals.lambda_y_a_c[:,0]==lambda_values_list[i])] #np.ndarray of all rows where wavelength == l
        plots += [plt.scatter(d[:,3], d[:,2], marker='o', s = 2, color=colors[i])]
        if include_line == True:
            plt.plot(d[:,3], d[:,2], color=colors[i]) #include a line joining points of same wavelength
    plt.legend(plots, lambda_values_list, title='Wavelength [µm]', markerscale=5, bbox_to_anchor=(1.05, 1.0), loc='upper left') #Legend location: https://www.delftstack.com/howto/matplotlib/how-to-place-legend-outside-of-the-plot-in-matplotlib/
    plt.xlabel('$c_s$ [mm]')
    plt.ylabel('$a_s$ [1/mm]')
    plt.ylim(np.min(a_points)-1e-4,np.max(a_points)+1e-4)
    plt.grid(linestyle='--')
    
    if save == True:
        savefilename = os.path.join(Globals.smile_results_dir, "a_VS_c.png")
        plt.savefig(savefilename, bbox_inches='tight')
    plt.show()
    return True

def plot_a_VS_y(include_line=True, save=False):
    """
    Creates a 2D plot of a VS y
    INPUTS:
        ::boolean:: include_line        #whether or not to include a line that joins points of same wavelength
        ::boolean:: save                #whether or not to save the plot
    """
    if check_and_load_lambda_y_a_c() == False:
        return False
    a_points = Globals.lambda_y_a_c[:,2] #a values
    
    lambda_values = np.unique(Globals.lambda_y_a_c[:,0]) #np.ndarray of unique wavelength values
    lambda_values_list = lambda_values.tolist() #list of unique wavelength values
    
    norm = plt.Normalize()
    colors = plt.cm.get_cmap('brg')(norm(lambda_values_list))
    
    plots = []
    plt.figure(figsize=(6,4), dpi=100)
    for i in range(0,len(lambda_values_list),1):
        d = Globals.lambda_y_a_c[np.where(Globals.lambda_y_a_c[:,0]==lambda_values_list[i])] #np.ndarray of all rows where wavelength == l
        plots += [plt.scatter(d[:,1], d[:,2], marker='o', s = 2, color=colors[i])]
        if include_line == True:
            plt.plot(d[:,1], d[:,2], color=colors[i])
    plt.legend(plots, lambda_values_list, title='Wavelength [µm]', markerscale=5, bbox_to_anchor=(1.05, 1.0), loc='upper left') #Legend location: https://www.delftstack.com/howto/matplotlib/how-to-place-legend-outside-of-the-plot-in-matplotlib/
    plt.xlabel('$y$ [mm]')
    plt.ylabel('$a_s$ [1/mm]')
    plt.ylim(np.min(a_points)-1e-4,np.max(a_points)+1e-4)
    plt.grid(linestyle='--')
    
    if save == True:
        savefilename = os.path.join(Globals.smile_results_dir, "a_VS_y.png")
        plt.savefig(savefilename, bbox_inches='tight')
    plt.show()
    return True

def plot_lambda_y_a(save=False):
    """
    Creates a 3D plot of a, lambda, and y
    INPUTS:
        ::boolean:: save                #whether or not to save the plot
    """
    if check_and_load_lambda_y_a_c() == False:
        return False
    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = plt.axes(projection="3d")
    
    x_points = Globals.lambda_y_a_c[:,0] #lambda (wavelength)
    y_points = Globals.lambda_y_a_c[:,1] #y
    z_points = Globals.lambda_y_a_c[:,2] #a
    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='jet')
    ax.set_xlabel('$\lambda$ [µm]')
    ax.set_ylabel('$y$ (on DMD) [mm]')
    ax.set_zlabel('$a_s$ [mm]')
    
    if save == True:
        savefilename = os.path.join(Globals.smile_results_dir, "a_VS_y_and_lambda.png")
        plt.savefig(savefilename, bbox_inches='tight')
    plt.show()
    return True

def plot_lambda_c_a(save=False):
    """
    Creates a 3D plot of a, lambda, and c
    INPUTS:
        ::boolean:: save                #whether or not to save the plot
    """
    if check_and_load_lambda_y_a_c() == False:
        return False
    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = plt.axes(projection="3d")
    
    x_points = Globals.lambda_y_a_c[:,0] #lambda (wavelength)
    y_points = Globals.lambda_y_a_c[:,3] #c
    z_points = Globals.lambda_y_a_c[:,2] #a
    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='jet')
    ax.set_xlabel('$\lambda$ [µm]')
    ax.set_ylabel('$c_s$[mm]')
    ax.set_zlabel('$a_s$ [mm]')
    
    if save == True:
        savefilename = os.path.join(Globals.smile_results_dir, "a_VS_c_and_lambda.png")
        plt.savefig(savefilename, bbox_inches='tight')
    plt.show()
    return True

#######################################################################################
#######################################################################################

def main_fcn():
    Globals.dataset_dir_name = 'dataset'
    Globals.smile_results_dir_name = 'smile_results'
    Globals.wavelength_filename = 'wavelengths.wav'
    
    #initialize directories
    if init_dir(Globals.dataset_dir_name, Globals.smile_results_dir_name, Globals.wavelength_filename) == False:
        return False #if missing directories
    
    #IMPORTANT:
    #If you want to run k-means and find the centroids, set do_find_centroids=True below
    #If you want to fit the quadratic functions y_d = a_s*x_d^2 + c_s, set do_quadratic_smile_fit=True below
    setup(do_find_centroids=False, do_quadratic_smile_fit=False)
    
    #plot_a_VS_c(save=True)
    #plot_a_VS_y(save=True)
    #plot_lambda_y_a(save=True)
    #plot_lambda_c_a(save=True)
    
    smile_a_fit(smile_a_fit_fcn)
    return True

if __name__ == "__main__":
    main_fcn()
