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

np.random.seed(Globals.seed_num) #Set a fixed random seed for reproducibility

#######################################################################################
#######################################################################################

def init_dir(dataset_dir_name, smile_results_dir_name, wavelength_filename='wavelengths.wav'):
    """
    Initializes the names of the directories. For use if e.g. we don't want to
    call find_centroids or smile_fit_all_files again but we still need to
    denote the directory names for the global variables.
    INPUTS:
        ::string:: dataset_dir_name         #name of the dataset directory relative to cwd
        ::string:: smile_results_dir_name   #name of the directory to store outputs of smile fits
        ::string:: wavelength_filename      #base filename of the wavelength file, without full path
    """
    Globals.root = os.getcwd() #current working directory
    Globals.dataset_dir = os.path.join(Globals.root, dataset_dir_name) #directory of dataset
    Globals.ima_dir = os.path.join(Globals.dataset_dir, 'ima') #directory of IMA files
    Globals.gia_dir = os.path.join(Globals.dataset_dir, 'gia') #directory of GIA files
    Globals.centroids_dir = os.path.join(Globals.dataset_dir, 'centroids') #directory of csv files storing centroids
    Globals.visualizations_dir = os.path.join(Globals.dataset_dir, 'visualizations') #directory of images of centroid plots
    Globals.smile_results_dir = os.path.join(Globals.dataset_dir, smile_results_dir_name) #directory to store outputs of smile fits
    if os.path.isdir(Globals.ima_dir) == False or os.path.isdir(Globals.gia_dir) == False:
        #if the ima or gia directory does not exist
        print("ERROR: ima directory and/or gia directory does not exist")
        return False
    if len(os.listdir(Globals.ima_dir)) == 0 or len(os.listdir(Globals.gia_dir)) == 0:
        #if directory is empty
        print("ERROR: Empty ima directory and/or gia directory")
        return False
    Globals.wavelength_data_file = os.path.join(Globals.dataset_dir, wavelength_filename)
    return True

def check_and_load_lambda_y_a_c():
    """
    This function checks if the smile_results_dir exists and is non-empty.
    Then loads lambda_y_a_c.csv into a global variable as a np.ndarray, if it
    is not already loaded
    """
    if os.path.isdir(Globals.smile_results_dir) == False:
        #if smile_results_dir does not exist
        print("ERROR: Results directory", Globals.smile_results_dir, "does not exist")
        return False
    if len(os.listdir(Globals.smile_results_dir)) == 0:
        #if smile_results_dir is empty
        print("ERROR: Results directory", Globals.smile_results_dir, "is empty")
        return False
    if type(Globals.lambda_y_a_c) == type(None):
        lambda_y_a_c_path = os.path.join(Globals.smile_results_dir, 'lambda_y_a_c.csv')
        if os.path.isfile(lambda_y_a_c_path) == False:
            print("ERROR: lambda_y_a_c file does not exist")
            return False
        Globals.lambda_y_a_c = np.loadtxt(lambda_y_a_c_path, delimiter=',')
    return True

#######################################################################################
#######################################################################################
#FIT FUNCTIONS FOR c_s

def smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam):
    #Original fit function
    l = lambda_y[0]
    y = lambda_y[1]
    return f_cam * np.tan(np.arcsin(G*l - np.sin(theta_0 + np.arctan(-y/f_coll))) - theta_0) #NOTE THE NEGATIVE y
    #return f_cam * (np.arcsin(G*l - np.sin(theta_0 + -y/f_coll)) - theta_0) #small angle approx

def tan_angle_cubic(lambda_y, c0, c1, c2, c3):
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    tan_ang = smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) / f_cam
    return f_cam*(c3*tan_ang**3 + c2*tan_ang**2 + c1*tan_ang + c0)

def tan_angle_cubic_one(lambda_y, c3):
    #THIS ONE IS TERRIBLE
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    tan_ang = smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) / f_cam
    return f_cam*(c3*tan_ang**3)

def tan_angle_cubic_with_factor(lambda_y, c0, c1, c2, c3, a1):
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    tan_ang = np.tan(a1*(np.arcsin(G*l - np.sin(theta_0 + np.arctan(-y/f_coll)))) - theta_0)
    return f_cam*(c3*tan_ang**3 + c2*tan_ang**2 + c1*tan_ang + c0)

def tan_factor(lambda_y, a1, a2):
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return f_cam * np.tan(a1*(np.arcsin(G*l - np.sin(theta_0 + a2*np.arctan(-y/f_coll)))) - theta_0)

def tan_angle_add(lambda_y, a, b):
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + a*(l-0.55)**3 + b*y**3

def tan_angle_add_corr_1(lambda_y, a3, a2, a1, a0, b3, b2, b1, b0):
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + (a3*y**3 + a2*y**2 + a1*y + a0)*(l-0.55)**3 + (b3*y**3 + b2*y**2 + b1*y + b0)*(l-0.55)

def tan_angle_add_corr_2(lambda_y, a3, a2, a1, a0, b3, b2, b1, b0, c1, c0):
    #GOOD
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + (a3*y**3 + a2*y**2 + a1*y + a0)*(l-0.55)**3 + (c1*y+c0)*(l-0.55)**2 + (b3*y**3 + b2*y**2 + b1*y + b0)*(l-0.55)

def tan_angle_add_corr_3(lambda_y, a3, a2, a1, a0, b3, b2, b1, b0, c1, c0, d1, d0):
    #TOO EXTRA
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + (d1*y+d0)*(l-0.55)**4 + (a3*y**3 + a2*y**2 + a1*y + a0)*(l-0.55)**3 + (c1*y+c0)*(l-0.55)**2 + (b3*y**3 + b2*y**2 + b1*y + b0)*(l-0.55)

def tan_angle_add_corr_4(lambda_y, a3, a2, a1, a0, b3, b2, b1, b0, c3, c2, c1, c0):
    #TOO EXTRA
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + (a3*y**3 + a2*y**2 + a1*y + a0)*(l-0.55)**3 + (c3*y**3 + c2*y**2 + c1*y+c0)*(l-0.55)**2 + (b3*y**3 + b2*y**2 + b1*y + b0)*(l-0.55)

def tan_angle_add_corr_5(lambda_y, a3, a2, a1, a0, b3, b2, b1, b0, c1, c0, d3, d2, d1, d0):
    #TOO EXTRA
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + (d3*y**3 + d2*y**2 + d1*y+d0)*(l-0.55)**4 + (a3*y**3 + a2*y**2 + a1*y + a0)*(l-0.55)**3 + (c1*y+c0)*(l-0.55)**2 + (b3*y**3 + b2*y**2 + b1*y + b0)*(l-0.55)

def angle_add_corr(lambda_y, a3, a2, a1, a0, b3, b2, b1, b0, c1, c0):
    #similar to tan_angle_add_corr_2 but we remove the tan
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return f_cam * (np.arcsin(G*l - np.sin(theta_0 + np.arctan(-y/f_coll))) - theta_0) + \
    (a3*y**3 + a2*y**2 + a1*y + a0)*(l-0.55)**3 + (c1*y+c0)*(l-0.55)**2 + (b3*y**3 + b2*y**2 + b1*y + b0)*(l-0.55)
    
######################################################################

def tan_angle_add_corr_roots(lambda_y, a3, a2, a1, a0, b1, b0, c1, c0):
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + (a3*y**3 + a2*y**2 + a1*y + a0)*(l-0.55)*(l-(b1*y + b0))*(l-(c1*y+c0))

def tan_angle_add_corr_roots12(lambda_y, a3, a2, a1, a0, b3, b2, b1, b0, c3, c2, c1, c0):
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + (a3*y**3 + a2*y**2 + a1*y + a0)*(l-0.55)*(l-(b3*y**3+b2*y**2+b1*y + b0))*(l-(c3*y**3+c2*y**2+c1*y+c0))

def tan_angle_add_corr_roots_quad(lambda_y, a3, a2, a1, a0, b3, b2, b1, b0, c3, c2, c1, c0):
    #GOOD
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + (l-0.55)*((a3*y**3 + a2*y**2 + a1*y + a0)*l**2 + (b3*y**3+b2*y**2+b1*y + b0)*l + (c3*y**3+c2*y**2+c1*y+c0))

def tan_angle_add_corr_roots_quad_2(lambda_y, a2, a1, a0, b2, b1, b0, c2, c1, c0):
    #quadratic function of y instead of cubic function of y (as in tan_angle_add_corr_roots_quad)
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return smile_c_fit_fcn_og(lambda_y, G, theta_0, f_coll, f_cam) + (l-0.55)*((a2*y**2 + a1*y + a0)*l**2 + (b2*y**2 + b1*y + b0)*l + (c2*y**2 + c1*y + c0))

def angle_add_corr_roots_quad(lambda_y, a2, a1, a0, b2, b1, b0, c2, c1, c0):
    #quadratic function of y instead of cubic function of y (as in tan_angle_add_corr_roots_quad)
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return f_cam * (np.arcsin(G*l - np.sin(theta_0 + np.arctan(-y/f_coll))) - theta_0) + \
    (l-0.55)*((a2*y**2 + a1*y + a0)*l**2 + (b2*y**2 + b1*y + b0)*l + (c2*y**2 + c1*y + c0))

def angle_add_corr_roots_quad_2(lambda_y, a2, a1, a0, b2, b1, b0, c2, c1, c0, d1, d0):
    #VERY GOOD
    #similar to angle_add_corr_roots_quad but with d1*(l-0.55)**2 * (l-0.55+d2)*(l-0.55-d2) term added
    f_cam = 79.131160
    f_coll = 60.115213
    theta_0 = 12.870 * np.pi/180
    G = 0.810
    l = lambda_y[0]
    y = lambda_y[1]
    return f_cam * (np.arcsin(G*l - np.sin(theta_0 + np.arctan(-y/f_coll))) - theta_0) + \
    (l-0.55)*((a2*y**2 + a1*y + a0)*l**2 + (b2*y**2 + b1*y + b0)*l + (c2*y**2 + c1*y + c0)) + d1*(l-0.55)**2 * (l-0.55+d0)*(l-0.55-d0)


#######################################################################################
#######################################################################################
#Functions for fitting c_s

def smile_c_fit(fit_fcn, p0=None):
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
    c_points = Globals.lambda_y_a_c[:,3] #c values

    
    fit_result_dir = os.path.join(Globals.smile_results_dir, fit_fcn.__name__) #directory to store results of the fit
    if os.path.isdir(fit_result_dir) == False: #make directory if it doesn't exist
        os.mkdir(fit_result_dir)
    
    #Fit the function
    #https://stackoverflow.com/questions/17934198/curve-fitting-in-scipy-with-3d-data-and-parameters/17934370
    #https://stackoverflow.com/questions/15413217/fitting-3d-points-python
    params, params_covariance = scipy.optimize.curve_fit(fit_fcn, lambda_y, c_points, p0=p0, maxfev=10000) #Note: I set maxfev=10000; feel free to remove this

    c_fit = fit_fcn(lambda_y, *params) #values of c using the fit
    visualize_c_error(c_fit, params, fit_fcn.__name__, fit_result_dir) #save results and see the error
    return params, params_covariance

def smile_c_fit_baseline():
    """
    Plots the baseline fit function
    """
    if check_and_load_lambda_y_a_c() == False:
        return False
    lambda_points = Globals.lambda_y_a_c[:,0] #wavelength values
    y_points = Globals.lambda_y_a_c[:,1] #y values
    lambda_y = [lambda_points, y_points] #list of two np.ndarrays
    c_points = Globals.lambda_y_a_c[:,3] #c values
    
    fit_result_dir = os.path.join(Globals.smile_results_dir, "baseline") #directory to store results of the fit
    if os.path.isdir(fit_result_dir) == False: #make directory if it doesn't exist
        os.mkdir(fit_result_dir)
    
    c_pred = smile_c_fit_fcn_og(lambda_y, G=0.810, theta_0=12.87*np.pi/180, f_coll = 60.115213, f_cam = 79.131160)
    visualize_c_error(c_pred, fit_fcn_name="baseline", savedir=fit_result_dir)
    
    #see other angles of plot
    visualize_c_error(c_pred, fit_fcn_name="baseline", savedir=fit_result_dir, angle=[0,0])
    visualize_c_error(c_pred, fit_fcn_name="baseline", savedir=fit_result_dir, angle=[0,-90])
    return True

def visualize_c_error(c_pred, params=None, fit_fcn_name=None, savedir=None, angle=None):
    """
    Given c_pred from a fit function, this function plots the error of the fit
    function. The plot can be saved as a PNG, and the errors and fit parameters
    can be saved in a TXT file.
    With the exception of c_pred, everything is an optional parameter.
    Results will only be saved if params, fit_fcn_name, and savedir are
    provided.
    INPUTS:
        ::np.ndarray:: c_pred                   #array with predicted c values from the fit
        ::np.ndarray or list:: params           #fit parameters
        ::string:: fit_fcn_name                 #name of the fit function
        ::string:: savedir                      #directory to save the results
        ::list:: angle                          #list of 2 elements; elev=angle[0], azim=angle[1]
    """
    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = plt.axes(projection="3d")

    x_points = Globals.lambda_y_a_c[:,0] #lambda (wavelength)
    y_points = Globals.lambda_y_a_c[:,1] #y
    z_points = c_pred - Globals.lambda_y_a_c[:,3] #error in c_s
    
    MSE = np.sum(np.power(z_points,2))/np.size(z_points) #mean squared error
    
    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='jet') #original cmap was 'hsv'
    ax.set_xlabel('$\lambda$ [µm]')
    ax.set_ylabel('$y$ (on DMD) [mm]')
    ax.set_zlabel('$c_s$ error [mm]')
    
    if type(angle) == type([]): #check if angle is a list
        if len(angle) == 2: #check if length of angle is 2
            ax.view_init(angle[0], angle[1]) #elev=angle[0], azim=angle[1]
            plt.savefig(os.path.join(savedir, fit_fcn_name + "_" + str(angle[0]) + "_" + str(angle[1]) + ".png"), bbox_inches='tight')
            plt.show()
            return True
    
    if savedir != None and fit_fcn_name != None:
        plt.savefig(os.path.join(savedir, fit_fcn_name + ".png"), bbox_inches='tight')
        with open(os.path.join(savedir, fit_fcn_name + ".txt"), 'w') as f:
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

def visualize_c_fit(fit_fcn, params, save=False, angle=None):
    """
    Given some fit function fit_fcn and its parameters params, creates a plot
    of the fit function (a surface), with the actual c_s points superimposed
    on the plot.
    INPUTS:
        ::function:: fit_fcn_name           #fit function
        ::np.ndarray or list:: params       #fit parameters
        ::boolean:: save                    #whether or not to save the plot
        ::list:: angle                          #list of 2 elements; elev=angle[0], azim=angle[1]
    """
    if save == True:
        fit_result_dir = os.path.join(Globals.smile_results_dir, fit_fcn.__name__) #directory to store results of the fit
        if os.path.isdir(fit_result_dir) == False: #make directory if it doesn't exist
            os.mkdir(fit_result_dir)
    
    fig = plt.figure(figsize=(7,6), dpi=100)
    ax = fig.gca(projection='3d')

    x_points = Globals.lambda_y_a_c[:,0] #lambda (wavelength)
    y_points = Globals.lambda_y_a_c[:,1] #y 
    z_points = Globals.lambda_y_a_c[:,3] #actual c_s       
        
    #Make data
    X = np.arange(np.min(x_points), np.max(x_points), 0.005)
    Y = np.arange(np.min(y_points), np.max(y_points), 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = fit_fcn([X,Y], *params)
    
    surf = ax.plot_surface(X, Y, Z, cmap="coolwarm", linewidth=0) #plot the surface
    ax.scatter3D(x_points, y_points, z_points, c=z_points) #plot the actual points
    ax.set_xlabel('$\lambda$ [µm]')
    ax.set_ylabel('$y$ (on DMD) [mm]')
    ax.set_zlabel('$c_s$ [mm]')
    
    # Add a color bar which maps values to colors.
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.ax.set_ylabel('Fitted $c_s$ value [mm]')
    
    if type(angle) == type([]): #check if angle is a list
        if len(angle) == 2: #check if length of angle is 2
            ax.view_init(angle[0], angle[1]) #elev=angle[0], azim=angle[1]
            if save == True:
                plt.savefig(os.path.join(fit_result_dir, fit_fcn.__name__ + "_" + str(angle[0]) + "_" + str(angle[1]) + "_surface.png"), bbox_inches='tight')
    else:
        if save == True:
            plt.savefig(os.path.join(fit_result_dir, fit_fcn.__name__ + "_surface.png"), bbox_inches='tight')
    
    plt.show()
    return True

#######################################################################################
#######################################################################################
#Rough Experimentation (can ignore this part)

def quartic_special(x, a, b, c):
    #BAD
    return a*((x-0.55)**2 - b**2)*((x-0.55)**2 - c**2)

def quartic_special_2(x, a, b, c):
    #BAD
    return a*(x**2 - b)**2 + c

def quartic(x, a, b, c, d, e):
    return a*x**4 + b*x**3 + c*x**2 + d*x + e

def quartic_special_3(x, a, b):
    return a*(x-0.55)**2 * (x-0.55+b) * (x-0.55-b)

def advanced_angle_add_corr_roots_quad():
    """
    For angle_add_corr_roots_quad; idea: take average value of error for each
    wavelength, and then fit a quartic curve for that (function of wavelength),
    and subtract this from the error.
    I.e., the fit is (current fit) - (additonal_term), where additional_term is
    the quartic function of wavelength.
    """
    if check_and_load_lambda_y_a_c() == False:
        return False
    lambda_y_points = Globals.lambda_y_a_c[:,0:2] #np.ndarray with 2 columns; 1st is wavelength, 2nd is y
    lambda_points = Globals.lambda_y_a_c[:,0] #wavelength values
    y_points = Globals.lambda_y_a_c[:,1] #y values
    lambda_y = [lambda_points, y_points] #list of two np.ndarrays
    c_points = Globals.lambda_y_a_c[:,3] #c values
    
    fit_fcn = angle_add_corr_roots_quad
    params, params_covariance = scipy.optimize.curve_fit(fit_fcn, lambda_y, c_points)
    c_fit = fit_fcn(lambda_y, *params) #values of c using the fit
    
    error = c_fit - c_points
    error = error[:,np.newaxis] #https://stackoverflow.com/questions/33480985/convert-numpy-vector-to-2d-array-matrix/33481152#33481152
    lambda_y_error = np.append(lambda_y_points, error, axis=1) #np.ndarray where the columns are lambda, y, and error
    
    lambda_values = np.unique(lambda_y_error[:,0]) #np.ndarray of unique wavelength values
    lambda_values_list = lambda_values.tolist() #list of unique wavelength values
    
    lambda_error = np.empty((0,2))
    for l in lambda_values_list: #for each unique wavelength in lambda_values_lis
        a = lambda_y_error[np.where(lambda_y_error[:,0]==l)] #np.ndarray of all rows where wavelength == l
        average_error = np.sum(a[:,2])/(a.shape[0]) #average error of all points for given wavelength l
        lambda_error_new_row = np.array([l, average_error])
        lambda_error_new_row = lambda_error_new_row[:,np.newaxis].T #1st column is wavelength, 2nd is average error
        lambda_error = np.append(lambda_error, lambda_error_new_row, axis=0) #append new row to the np.ndarray lambda_error

    #np.savetxt('lambda_error.csv', lambda_error, delimiter=",")
    par, _ = scipy.optimize.curve_fit(quartic_special_3, lambda_error[:,0], lambda_error[:,1], maxfev=10000)
    
    plt.figure(figsize=(6,4), dpi=100)
    plt.scatter(lambda_error[:,0], lambda_error[:,1], marker='o') #points
    
    axes = plt.gca()
    x_min, x_max = axes.get_xlim() #get the min x and max x of the plot x axis
    plt.plot(np.linspace(x_min,x_max,num=100),quartic_special_3(np.linspace(x_min,x_max,num=100),*par))
    print(par)
    plt.xlabel('$\lambda$ [mm]')
    plt.ylabel('Error [mm]')
    plt.show()
    return par

def good_fit():
    """
    For angle_add_corr_roots_quad. This function gives the fit obtained by
    using the process in advanced_angle_add_corr_roots_quad and shows the error
    """
    if check_and_load_lambda_y_a_c() == False:
        return False
    lambda_points = Globals.lambda_y_a_c[:,0] #wavelength values
    y_points = Globals.lambda_y_a_c[:,1] #y values
    lambda_y = [lambda_points, y_points] #list of two np.ndarrays
    c_points = Globals.lambda_y_a_c[:,3] #c values
    
    fit_fcn = angle_add_corr_roots_quad
    params, params_covariance = scipy.optimize.curve_fit(fit_fcn, lambda_y, c_points)
    c_fit = fit_fcn(lambda_y, *params) #values of c using the fit
    
    #calculate the additonal_term; note that -2.22599564e+02 and 1.35843923e-01 were calculated from advanced_angle_add_corr_roots_quad
    additional_term = quartic_special_3(lambda_points, -2.22599564e+02, 1.35843923e-01)
    c_fit_add = c_fit - additional_term #this is the final fit
    visualize_c_error(c_fit_add)
    return True

def plot_h(save=False):
    """
    Plots the h(\lambda, y) function (see progress report).
    Using the baseline fit, we can represent the associated error \Delta c_s
    as (\lambda - \lambda_c)*h(\lambda, y). This function below plots h.
    INPUTS:
        ::boolean:: save        #if True then saves an image of the plot of h
    """
    if check_and_load_lambda_y_a_c() == False:
        return False
    lambda_points = Globals.lambda_y_a_c[:,0] #wavelength values
    y_points = Globals.lambda_y_a_c[:,1] #y values
    lambda_y = [lambda_points, y_points] #list of two np.ndarrays
    c_points = Globals.lambda_y_a_c[:,3] #c values
    
    c_pred = smile_c_fit_fcn_og(lambda_y, G=0.810, theta_0=12.87*np.pi/180, f_coll = 60.115213, f_cam = 79.131160)
    c_error = c_pred-c_points #error in c
    h = np.divide(c_error,(lambda_points-0.55)) #divide the error by (\lambda - \lambda_c) to obtain h
    
    fig = plt.figure(figsize=(6,6), dpi=100)
    ax = plt.axes(projection="3d")

    #If you want to plot the plane z=0 on the same plot
    #xx, yy = np.meshgrid(np.linspace(0.4, 0.7,2),np.linspace(-5,5,2))
    #z = (xx+yy)*0
    # Add an axes
    #ax = fig.add_subplot(111,projection='3d')
    # plot the surface
    #ax.plot_surface(xx, yy, z, alpha=0.5)
    
    x_points = Globals.lambda_y_a_c[:,0] #lambda (wavelength)
    y_points = Globals.lambda_y_a_c[:,1] #y
    z_points = h #error in c_s
    ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='jet') #original cmap was 'hsv'
    ax.set_xlabel('$\lambda$ [µm]')
    ax.set_ylabel('$y$ (on DMD) [mm]')
    ax.set_zlabel('$h(\lambda,y)$ [mm]')
    
    if save == True: #save the plot as an image
        fit_result_dir = os.path.join(Globals.smile_results_dir, "h") #directory to store result
        if os.path.isdir(fit_result_dir) == False: #make directory if it doesn't exist
            os.mkdir(fit_result_dir)
        plt.savefig(os.path.join(fit_result_dir, "h.png"), bbox_inches='tight')
    plt.show()
    return True

#######################################################################################
#######################################################################################


def setup(do_find_centroids, do_quadratic_smile_fit):
    """
    INPUTS:
        ::boolean:: do_find_centroids           #if True then will run k-means and find the centroids
        ::boolean:: do_quadratic_smile_fit      #if True then will fit the quadratic functions y_d = a_s*x_d^2 + c_s
    """
    if do_find_centroids == True:
        find_centroids(Globals.dataset_dir_name, wavelength_filename=Globals.wavelength_filename)
    if do_quadratic_smile_fit == True:
        Globals.lambda_y_a_c = smile_fit_all_files(Globals.dataset_dir_name, Globals.smile_results_dir_name, Globals.wavelength_filename)
    return True

def main():
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
    
    ###########################################################################
    #Fits:
    
    #smile_c_fit_baseline()
    #smile_c_fit(smile_c_fit_fcn_og, [0.810, 12.870, 60.115213, 79.131160])
    #smile_c_fit(tan_factor)
    #smile_c_fit(tan_angle_cubic)
    #smile_c_fit(tan_angle_add)
    #smile_c_fit(tan_angle_cubic_one)
    
    #smile_c_fit(tan_angle_add_corr_1)
    #smile_c_fit(tan_angle_add_corr_2)
    #smile_c_fit(angle_add_corr)

    #smile_c_fit(tan_angle_add_corr_roots_quad_2)
    #smile_c_fit(angle_add_corr_roots_quad)
    params, _ = smile_c_fit(angle_add_corr_roots_quad_2) #BEST FIT
    visualize_c_fit(angle_add_corr_roots_quad_2, params, save=True)
    visualize_c_fit(angle_add_corr_roots_quad_2, params, save=True, angle=[0,0])
    visualize_c_fit(angle_add_corr_roots_quad_2, params, save=True, angle=[0,-90])
    ###########################################################################
    #Just some experimentation: (can ignore this)
    #advanced_angle_add_corr_roots_quad()
    #good_fit()
    #plot_h(True)
    ###########################################################################
    return True

if __name__ == "__main__":
    main()
