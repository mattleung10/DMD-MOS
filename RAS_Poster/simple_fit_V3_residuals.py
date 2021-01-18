#Simple Fit
#Matthew Leung
#August 2020
#######################################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.cluster
import scipy.stats
import scipy.optimize
import PIL
import os
import time

np.random.seed(1000) #Set a fixed random seed for reproducibility with Scikit-Learn

"""
Process:
    Load txt file of simulated data from Zemax into pandas DataFrame
    Determine the wavelengths in the file from wav file from Zemax Wavelength Data
    From the IMA file, determine the number of micromirrors ON in the DMD (ON along slit), which is n
    for each wavelength:
        Use k-means clustering algorithm to cluster points into n clusters
        For each cluster, obtain the average/centroid position of the points in the cluster
        Fit a quadratic function to those n centroids, y' = a * x'^2 + c; a and c are constants
    Find a and c each as a function of wavelength
"""

#######################################################################################
#######################################################################################
#GLOBAL VARIABLES

data_file = 'data_1000k.txt'
wavelength_data_file = 'wavelengths.wav'
ima_file = 'dmd_499_500_MOS.IMA'

field_size = 0 #Field Size from Zemax; will be assigned to later
M_S = 79.131160/60.115213 #Magnification of the spectrograph

data = None #pandas DataFrame to store the data from the data_file
w_data = None #list to store the wavelengths

#######################################################################################
#######################################################################################

def get_data_file_info(data_file):
    linenum = 0
    with open(data_file, encoding='utf-16-le') as f: #Careful: Zemax outputted txt file is UTF-16 LE
        while True:
            line = f.readline()
            #print(linenum, line)
            if "Field Width" in line and linenum != 2: #get the line with "Field Width"
                temp = line.split(' ')
                break
            linenum += 1
        for elem in temp:
            try:
                field_size = float(elem)
                break
            except ValueError:
                pass
        while True:
            line = f.readline()
            if "Ray" in line:
                break
            linenum += 1

    skiprows = linenum + 2
    return field_size, skiprows

def get_wavelength_file_info(wavelength_data_file):
    w_data_raw = []
    w_data = [] #stores the wavelength values in the Zemax file
    with open(wavelength_data_file, encoding='utf-16-le') as f:
        for line in f:
            w_data_raw += [line.split(' ')]        
    for i in range(1,len(w_data_raw),1):
        w_data += [float(w_data_raw[i][0])] #w_data_raw[i][1] is the weight, we don't need that
    
    temp = ''
    for c in w_data_raw[0][0]:
        if c.isdigit():
            temp += c
    primary_wavelength_num = int(temp) #Primary wavelength
    return w_data, primary_wavelength_num

def get_ima_file_info(ima_file, inverted=False):
    slit_pos = -1
    ON_pos = [] #stores the x-position of the micromirrors that are ON
    with open(ima_file) as f:
        dmd_dim = int(f.readline()) #max dimension of DMD
        while True:
            line = f.readline()
            slit_pos += 1
            if '1' in line:
                for i in range(0,len(line),1):
                    if line[i] == '1':
                        ON_pos += [i]
                break

    predicted_spot_pos = []
    for i in range(0,len(ON_pos),1):
        #predicted_spot_pos += [(ON_pos[i]/(dmd_dim/2) * field_size/2 * M_S ) - (field_size/2 * M_S)]

        predicted_spot_pos += [-1*(dmd_dim/2 - (ON_pos[i] + 1) + 0.5)/(dmd_dim/2) * field_size/2*M_S]

    if inverted == True: #if the image is inverted
        #in-place reverse the list
        for i in range(0,int(len(predicted_spot_pos)/2),1):
            temp = predicted_spot_pos[i]
            predicted_spot_pos[i] = predicted_spot_pos[len(predicted_spot_pos)-1-i]
            predicted_spot_pos[len(predicted_spot_pos)-1-i] = temp
        #multiply every element in list by -1
        for i in range(0,len(predicted_spot_pos),1):
            predicted_spot_pos[i] = -1*predicted_spot_pos[i]
    
    return dmd_dim, slit_pos, ON_pos, predicted_spot_pos


#######################################################################################
#######################################################################################

def plot_Zemax_GIA(save=True):
    #Plot the data from Zemax Geometric Image Analysis
    #https://stackoverflow.com/questions/28144142/how-can-i-generate-a-colormap-array-from-a-simple-array-in-matplotlib
    #https://stackoverflow.com/questions/6063876/matplotlib-colorbar-for-scatter
    #https://stackoverflow.com/questions/12236566/setting-different-color-for-each-series-in-scatter-plot-on-matplotlib
    norm = plt.Normalize()
    colors = plt.cm.get_cmap('brg')(norm(w_data))
    
    #https://stackoverflow.com/questions/17411940/matplotlib-scatter-plot-legend
    plots = []
    plt.figure(figsize=(10,10), dpi=100)
    for i in range(0,len(w_data),1):
        d = data.loc[data['w'] == i+1] # +1 to get the wavelength number
        plots += [plt.scatter(d['X-coord'], d['Y-coord'], marker='o', s = 0.05, color=colors[i])]
    plt.legend(plots, w_data, title='Wavelength [µm]', markerscale=20, bbox_to_anchor=(1.05, 1.0), loc='upper left') #Legend location: https://www.delftstack.com/howto/matplotlib/how-to-place-legend-outside-of-the-plot-in-matplotlib/
    plt.title('Zemax Geometric Image Analysis Data')
    plt.xlabel('$x_d$ [mm]')
    plt.ylabel('$y_d$ [mm]')
    plt.axis('scaled')
    plt.grid(linestyle='--')
    if save == True:
        plt.savefig("GIA.png", bbox_inches='tight')
    plt.show()
    return True

#######################################################################################
#######################################################################################

def smile_fit_fcn(x, a, c):
    return a*x**2 + c

def cubic_const0(x, a, c):
    #Fit function for the centroid deviation/difference (keystone distortion)
    #This function f(x) must satisfy:
    #   f(0) = 0
    #   f(x) = -f(-x)
    #For a cubic, this is only possible for f(x) = a*x^3 + c*x
    return a*x**3 + c*x

def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

def cubic(x, a, b, c, d):
    return a*x**3 + b*x**2 + c*x + d

def quartic_vertex(x, a, b, h, k):
    return a*(x-h)**4 + b*(x-h)**4 + k

def xd_fit(x_pred, wavelength, keystone_a_fit, keystone_c_fit):
    a_k = 0
    for i in range(0,len(keystone_a_fit),1):
        a_k += keystone_a_fit[i] * wavelength**(len(keystone_a_fit)-i-1)
    c_k = 0
    for i in range(0,len(keystone_c_fit),1):
        c_k += keystone_c_fit[i] * wavelength**(len(keystone_c_fit)-i-1)
    return cubic_const0(x_pred, a_k, c_k+1) #remember +1

def yd_fit(x_pred, wavelength, smile_a_fit, smile_c_fit, keystone_a_fit, keystone_c_fit):
    #takes in the PREDICTED spot pos x_pred
    xd = xd_fit(x_pred, wavelength, keystone_a_fit, keystone_c_fit)
    a_s = 0
    for i in range(0,len(smile_a_fit),1):
        a_s += smile_a_fit[i] * wavelength**(len(smile_a_fit)-i-1)
    c_s = 0
    for i in range(0,len(smile_c_fit),1):
        c_s += smile_c_fit[i] * wavelength**(len(smile_c_fit)-i-1)
    return smile_fit_fcn(xd, a_s, c_s)


def main():
    #Global Variables
    global data_file
    global wavelength_data_file
    global ima_file
    global field_size
    global M_S
    global data
    global w_data
    
    field_size, skiprows = get_data_file_info(data_file)
    w_data, primary_wavelength_num = get_wavelength_file_info(wavelength_data_file)
    dmd_dim, slit_pos, ON_pos, predicted_spot_pos = get_ima_file_info(ima_file, inverted=True)
    
    #Load the txt file; Notes:
    #   - Data separated by "\t" (tab)
    #   - Skip first 14 rows
    #   - Skip last 9 rows (have to set engine='python' instead of 'c')
    #   - Zemax outputted a txt file that has a UTF-16 LE encoding
    data = pd.read_csv(data_file, sep="\t", header=None, skiprows=skiprows, skipfooter=9, encoding='utf-16-le', engine='python')
    data.columns = ['Ray #','X-field','Y-field','px','py','w','Wgt','X-coord','Y-coord']

    #Plot the data from Zemax Geometric Image Analysis
    start_time = time.time()
    plot_Zemax_GIA(save=True)
    end_time = time.time()
    print("Plotting the Zemax Geometric Image Analysis data took", str(end_time - start_time), "seconds.")

    smile_fit_coeff = [] #list to store the smile fit coefficients
    diff_fit_coeff = [] #list to store the (keystone) centroid difference fit coefficients
    predicted_spot_pos_np = np.array(predicted_spot_pos).T #Numpy array of predicted spot positions, for plotting
    
    all_centroids = np.empty((0,3)) #Numpy array of all centroids  
    
    start_time = time.time()
    for i in range(0,len(w_data),1): #for each wavelength
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
        plt.title("Wavelength: " + str(w_data[i]) + "µm")
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
        
        #Fit the centroids to the smile_fit_fcn
        params, params_covariance = scipy.optimize.curve_fit(smile_fit_fcn, centroids_sorted[:,0],centroids_sorted[:,1],p0=[0.2, 0.2])
        smile_fit_coeff += [[w_data[i],params[0],params[1]]]
        
        #######################################################################
        #Plot fitted function, centroids, points/spots, and predicted spot positions
        if i == 0 or i == primary_wavelength_num-1 or i == len(w_data)-1:
            plt.figure(figsize=(10,4), dpi=100)
            plt.scatter(points[:,0], points[:,1], marker='o', s = 0.05, label='Spot') #points
            plt.scatter(centroids_sorted[:, 0], centroids_sorted[:, 1], c='black', s=50, alpha=0.5, label='Centroid')
            
            axes = plt.gca()
            x_min, x_max = axes.get_xlim() #get the min x and max x of the plot x axis
            #plt.plot(centroids_sorted[:,0],smile_fit_fcn(centroids_sorted[:,0],params[0],params[1]), label='Quadratic Fit')
            plt.plot(np.linspace(x_min,x_max,num=100),smile_fit_fcn(np.linspace(x_min,x_max,num=100),params[0],params[1]), label='Quadratic Fit')
            
            plt.axvline(x=predicted_spot_pos[0], color='black', linestyle='-', linewidth=0.5, alpha=0.5, label='Predicted spot x position')
            for s in range(1,len(predicted_spot_pos),1):
                plt.axvline(x=predicted_spot_pos[s], color='black', linestyle='-', linewidth=0.5, alpha=0.5) #predicted spot position
            plt.title("Wavelength: " + ("%.3f" % w_data[i]) + "µm")
            plt.xlabel('$x_d$ [mm]')
            plt.ylabel('$y_d$ [mm]')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.savefig(("%.3f" % w_data[i]) + "um_plot.png", bbox_inches='tight') #https://stackoverflow.com/questions/10101700/moving-matplotlib-legend-outside-of-the-axis-makes-it-cutoff-by-the-figure-box
            plt.show()
        #######################################################################
        
        #Centroid x direction deviation
        diff_params, _ = scipy.optimize.curve_fit(cubic_const0, predicted_spot_pos_np, centroids_sorted[:,0]-predicted_spot_pos_np,p0=[0.2, 0.2])
        diff_fit_coeff += [[w_data[i],diff_params[0],diff_params[1]]]
        
        plt.figure(figsize=(8,6), dpi=100)
        plt.scatter(predicted_spot_pos_np, centroids_sorted[:,0]-predicted_spot_pos_np)
        plt.plot(predicted_spot_pos_np, cubic_const0(predicted_spot_pos_np, diff_params[0], diff_params[1]))
        plt.title("Wavelength: " + ("%.3f" % w_data[i]) + "µm\nCentroid x position deviation")
        plt.xlabel('$M_S x$ [mm]')
        plt.ylabel('$x_d - M_S x$ [mm]')
        plt.grid()
        plt.tight_layout()
        plt.savefig(str(i) + '_centroid_pos.png')
        plt.close('all')
    end_time = time.time()
    print("K-means clustering and centroid curve fitting (smile) took", str(end_time - start_time), "seconds.")
    
    all_centroids_df = pd.DataFrame(data=all_centroids, columns=['X-coord','Y-coord','w']) #Pandas DataFrame that stores the centroids' coordinates and wavelength number
    all_centroids_df.to_csv(data_file[0:-4] + "_centroids.csv")
    
    ###########################################################################
    #Convert to Numpy array for use when plotting later
    smile_fit_coeff_np = np.array(smile_fit_coeff)
    diff_fit_coeff_np = np.array(diff_fit_coeff)
    
    #Fit coefficients for the relationship between distortion fit coefficient and wavelength
    smile_a_fit = []
    smile_c_fit = []
    keystone_a_fit = []
    keystone_c_fit = []
    
    #Plot of smile fit a coefficient VS wavelength
    smile_a_fit, _ = scipy.optimize.curve_fit(cubic, smile_fit_coeff_np[:,0],smile_fit_coeff_np[:,1],p0=[0.2, 0.2, 0.2, 0.2])
    plt.plot(smile_fit_coeff_np[:,0], cubic(smile_fit_coeff_np[:,0],smile_a_fit[0],smile_a_fit[1],smile_a_fit[2],smile_a_fit[3]),label='Cubic Fit')
    plt.scatter(smile_fit_coeff_np[:,0], smile_fit_coeff_np[:,1])
    plt.title("Smile fit $a_s$ coefficient [in $y_d = a_s x_d^2 + c_s$]")
    plt.xlabel("$\lambda$ [µm]")
    plt.ylabel("$a_s$")
    plt.axis('tight')
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.savefig("smile_a.png", bbox_inches='tight')
    plt.show()
    
    #Plot of smile fit c coefficient VS wavelength
    smile_c_fit, _ = scipy.optimize.curve_fit(cubic, smile_fit_coeff_np[:,0],smile_fit_coeff_np[:,2],p0=[0.2, 0.2, 0.2, 0.2])
    plt.plot(smile_fit_coeff_np[:,0], cubic(smile_fit_coeff_np[:,0],smile_c_fit[0],smile_c_fit[1],smile_c_fit[2],smile_c_fit[3]),label='Cubic Fit')
    plt.scatter(smile_fit_coeff_np[:,0], smile_fit_coeff_np[:,2])
    plt.title("Smile fit $c_s$ coefficient [in $y_d = a_s x_d^2 + c_s$]")
    plt.xlabel("$\lambda$ [µm]")
    plt.ylabel("$c_s $")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.savefig("smile_c.png", bbox_inches='tight')
    plt.show()
    
    #Plot of keystone fit a coefficient
    keystone_a_fit, _ = scipy.optimize.curve_fit(quadratic, diff_fit_coeff_np[:,0], diff_fit_coeff_np[:,1], p0=[0.2, 0.2, 0.2])
    plt.plot(diff_fit_coeff_np[:,0], quadratic(diff_fit_coeff_np[:,0],keystone_a_fit[0],keystone_a_fit[1],keystone_a_fit[2]),label='Quadratic Fit')
    #https://stackoverflow.com/questions/37062866/applying-bounds-to-specific-variable-during-curve-fit-scipy-leads-to-an-error
    #keystone_a_fit, _ = scipy.optimize.curve_fit(quartic_vertex, diff_fit_coeff_np[:,0], diff_fit_coeff_np[:,1], p0=[0.2, 10, 0.2, np.amin(diff_fit_coeff_np[:,1])], bounds=((0,0,-np.inf,-np.inf),(np.inf,np.inf,np.inf,np.amin(diff_fit_coeff_np[:,1])*1.005)))
    #plt.plot(diff_fit_coeff_np[:,0], quartic_vertex(diff_fit_coeff_np[:,0],keystone_a_fit[0],keystone_a_fit[1],keystone_a_fit[2],keystone_a_fit[3]),label='Quartic Fit')
    plt.scatter(diff_fit_coeff_np[:,0], diff_fit_coeff_np[:,1])
    plt.title('Keystone fit $a_k$ coefficient [in $x_d - (M_S x) = a_k (M_S x)^3 + c_k (M_S x)$]')
    plt.xlabel("$\lambda$ [µm]")
    plt.ylabel("$a_k$")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.savefig("keystone_a.png", bbox_inches='tight')
    plt.show()
    
    #Plot of keystone fit c coefficient
    keystone_c_fit, _ = scipy.optimize.curve_fit(cubic, diff_fit_coeff_np[:,0],diff_fit_coeff_np[:,2],p0=[0.2, 0.2, 0.2, 0.2])
    plt.plot(diff_fit_coeff_np[:,0], cubic(diff_fit_coeff_np[:,0],keystone_c_fit[0],keystone_c_fit[1],keystone_c_fit[2],keystone_c_fit[3]),label='Cubic Fit')
    plt.scatter(diff_fit_coeff_np[:,0], diff_fit_coeff_np[:,2])
    plt.title('Keystone fit $c_k$ coefficient [in $x_d - (M_S x) = a_k (M_S x)^3 + c_k (M_S x)$]')
    plt.xlabel("$\lambda$ [µm]")
    plt.ylabel("$c_k$")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    plt.grid()
    plt.savefig("keystone_c.png", bbox_inches='tight')
    plt.show()
    
    ###########################################################################
    #Residuals
    max_deviation = 0 #maximum deviation of spot position between actual and fit
    wavelength_of_interest = 13
    for wavelength_of_interest in range(1,14,1):
        centroids_df = pd.read_csv(data_file[0:-4] + "_centroids.csv")
        centroids_df_w = centroids_df.loc[centroids_df['w'] == wavelength_of_interest] #part of the DataFrame corresponding to wavelength_of_interest
        centroids_of_interest = centroids_df_w[['X-coord','Y-coord']].values
        
        xd = xd_fit(predicted_spot_pos_np, w_data[wavelength_of_interest-1], keystone_a_fit, keystone_c_fit)
        plt.scatter(predicted_spot_pos_np, xd-centroids_of_interest[:,0])
        plt.title('Wavelength: ' + ("%.3f" % w_data[wavelength_of_interest-1]) + 'µm\nResiduals of $x_d$ fit')
        plt.xlabel('$M_S x$ [mm]')
        plt.ylabel('Deviation [mm]')
        plt.show()
        
        yd = yd_fit(predicted_spot_pos_np, w_data[wavelength_of_interest-1], smile_a_fit, smile_c_fit, keystone_a_fit, keystone_c_fit)
        plt.scatter(predicted_spot_pos_np, yd-centroids_of_interest[:,1])
        plt.title('Wavelength: ' + ("%.3f" % w_data[wavelength_of_interest-1]) + 'µm\nResiduals of $y_d$ fit')
        plt.xlabel('$M_S x$ [mm]')
        plt.ylabel('Deviation [mm]')
        plt.show()
        
        plt.scatter(centroids_of_interest[:,0], centroids_of_interest[:,1], label='Actual')
        plt.scatter(xd, yd, label='Fit')
        plt.title('Wavelength: ' + ("%.3f" % w_data[wavelength_of_interest-1]) + 'µm\nComparison of spot positions')
        plt.xlabel('x direction [mm]')
        plt.ylabel('y direction [mm]')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.savefig("Comparison" + ("%.3f" % w_data[wavelength_of_interest-1]) + "um.png", bbox_inches='tight')
        plt.show()
        
        plt.plot(xd, np.sqrt((yd-centroids_of_interest[:,1])**2 + (xd-centroids_of_interest[:,0])**2))
        plt.title('Wavelength: ' + ("%.3f" % w_data[wavelength_of_interest-1]) + 'µm\nResiduals of fit')
        plt.xlabel('$M_S x$ [mm]')
        plt.ylabel('Absolute Deviation [mm]')
        plt.savefig("Residuals" + ("%.3f" % w_data[wavelength_of_interest-1]) + "um.png", bbox_inches='tight')
        plt.show()
        
        max_deviation = max(max_deviation, np.amax(np.sqrt((yd-centroids_of_interest[:,1])**2 + (xd-centroids_of_interest[:,0])**2)))
    
    print("The maximum deviation is", max_deviation, " mm")
    
    ###########################################################################
    #Save centroid x direction deviation plots as gif
    images = []
    for i in range(0,len(w_data),1): #add plots to images list
        im = PIL.Image.open(str(i) + '_centroid_pos.png')
        images += [im]
    images_to_save = images
    for i in range(len(images)-1,-1,-1): #add plots in reverse order
        images_to_save += [images[i]]
    images_to_save[0].save('Centroid_x_deviation.gif',save_all=True, append_images=images[1:], optimize=False, duration=350, loop=0)
    for i in range(0,len(w_data),1):
        os.remove(str(i) + '_centroid_pos.png') #remove images
    
    with open('fit_coeff.txt', 'w') as f:
        f.write("Smile a\n")
        for elem in smile_a_fit:
            f.write("%.8f\n" % elem)
        f.write("Smile c\n")
        for elem in smile_c_fit:
            f.write("%.8f\n" % elem)
        f.write("Keystone a\n")
        for elem in keystone_a_fit:
            f.write("%.8f\n" % elem)
        f.write("Keystone c\n")
        for elem in keystone_c_fit:
            f.write("%.8f\n" % elem)

if __name__ == "__main__":
    main()
