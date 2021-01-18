#DMD-MOS rough simulation - Convolution
#Matthew Leung
#July 2020
#######################################################################################

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.integrate

#######################################################################################
#######################################################################################
#GLOBAL VARIABLES

#DMD
p_M = 13.7e-6 #pitch of micromirror in m
N_Mx = 750 #number of micromirrors in x diretion
N_My = 1000 #number of micromirrors in y direction

#Detector
p_D = 9.00e-6 #pitch of a pixel
N_Dx = 4000 #number of pixels in x direction (spatial)
N_Dy = 4000 #number of pixels in y direction (spectral)

#General
f_coll = 60.115213e-3 #focal length of collimator in m

M_S = p_D * 2 / p_M #magnification of spectrograph
f_cam = M_S * f_coll #focal length of camera optics in m

lambda_min = 0.4 #minimum wavelength in µm
lambda_max = 0.7 #maximum wavelength in µm
lambda_cen = 0.55 #central wavelength in µm

#Grating
G = 810 #groove density in lines per mm
alpha = 12.87 #incidence angle wrt grating normal, in degrees
m = 1 #diffraction order

###############################################################################

EE50_p = 1.2 #EE50 in units of pixels
EE50 = EE50_p * p_D * 1e6 #EE50 in units of µm
sigma = EE50 * np.sqrt(-1 / (2 * np.log(0.5))) #sigma in Line Spread Function (LSF) in µm

###############################################################################

#DMD array
#The DMD is represented by a numpy array, where each element corresponds to a 
#micromirror. Elements with a value 1 correspond to a micromirror in the ON 
#configuration (micromirror tilted towards the spectral channel), whereas
#elements with a value 0 correspond to a micromirror in the OFF configuration
dmd = np.zeros((N_My, N_Mx)) #shape: (height, width)

#Detector array
#The CCD detector is represented by a numpy array, where each element 
#corresponds to a pixel on the detector. Each element in the array takes on a
#float that represents the wavelength of light that hits that pixel.
ccd = np.zeros((N_Dy, N_Dx))
ccd_no_dispersion = np.zeros((N_Dy, N_Dx)) #CCD detector without dispersion

#Output
#list to store info (wavelength positions on CCD) to write to an output file
output_info = []

#######################################################################################
#######################################################################################

def plotCCD(ccd):
    """
    Function to show the CCD detector
    """
    fig = plt.figure(figsize=(7,6), dpi=100)
    plt.title('CCD')
    colormap = plt.imshow(ccd,aspect='auto',cmap='nipy_spectral')
    fig.axes[0].set_xlabel('Pixel Number (x-direction)')
    fig.axes[0].set_ylabel('Pixel Number (y-direction)')
    #Colorbar:
    cbar = plt.colorbar(colormap, orientation='vertical')
    cbar.set_label('Wavelength [µm]')
    plt.savefig("CCD_plot.png")
    plt.show()

def plotDMD(dmd):
    """
    Function to show the DMD
    """
    plt.figure(figsize =(6,4.5), dpi=100)
    plt.imshow(dmd, cmap='gray')
    plt.title('DMD')
    plt.xlabel('Micromirror Number (x-direction)')
    plt.ylabel('Micromirror Number (y-direction)')
    plt.savefig("DMD_plot.png")
    plt.show()

def plotCCD_no_dispersion(ccd_no_dispersion):
    """
    Function to show the CCD detector, without dispersion at the centre wavelength
    """
    r = get_boundary()
    plt.figure(figsize =(6,6), dpi=100)
    ax = plt.subplot(111)
    rect1 = patches.Rectangle((r[0][1], r[0][0]), r[1][1]-r[0][1], r[1][0]-r[0][0], linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect1)
    plt.imshow(ccd_no_dispersion, cmap='gray')
    plt.title('CCD (No Dispersion) at ' + str(lambda_cen) + " µm")
    plt.xlabel('Pixel Number (x-direction)')
    plt.ylabel('Pixel Number (y-direction)')
    plt.savefig("CCD_no_dispersion_plot.png")
    plt.show() 

def getDiffractionAngle(wavelength, m, G, alpha):
    """
    INPUTS:
        ::float:: wavelength           #wavelength in µm
        ::int:: m                      #diffraction order
        ::float:: G                    #groove density in lines/mm
        ::float:: alpha                #incidence angle in degrees
    OUTPUT:
        ::float::                      #diffraction angle in degrees
    """
    return np.arcsin((G*1e3)*m*(wavelength*1e-6) - np.sin(alpha * np.pi/180)) * 180/np.pi

def getBlazeAngle(lambda_cen, m, G):
    """
    Returns the blaze angle given some wavelength lambda_cen [um],
    diffraction order m, and groove density G [lines/mm]
    """
    return np.arcsin((G*1e3)*m*(lambda_cen*1e-6) / 2) * 180/np.pi

#######################################################################################
#######################################################################################

def turnMicromirrorsON(p1, p2):
    """
    Sets some rectangle of micromirrors in DMD to be in the ON configuration,
    i.e. makes some elements in the dmd array to be 1.
    This "rectangle of micromirrors" is defined by p1 and p2, which are the
    micromirror numbers / array indices for the top left and bottom right 
    micromirrors in this rectangle respectively.
    INPUTS:
        ::list of length 2:: p1    #array indices for top left micromirror ([y,x])
        ::list of length 2:: p2    #array indices for bottom right micromirror ([y,x])
    """
    
    #check that the indices do not go out of bounds
    if p1[0] < 0 or p1[1] < 0 or p2[0] >= dmd.shape[0] or p2[1] >= dmd.shape[1]:
        return False
    
    #Make some micromirrors in the DMD be in the ON configuration (set elements to 1)
    dmd[p1[0]:p2[0]+1, p1[1]:p2[1]+1] = 1
    
    #simulate where rays will land on CCD
    r = propagate(p1, p2)
    
    #write this to an output file later on
    global output_info
    output_info += ["For the micromirror(s) from (x,y) = (" + str(p1[0]) + "," + str(p1[1]) + ") to (x,y) = (" + str(p2[0]) + "," + str(p2[1]) + "):"]
    output_info += ["Centre wavelength " + str(lambda_cen) + " µm is at pixel number (x,y) = (" + str(r[2]) + "," + str(r[3]) + ")"]
    output_info += ["Minimum wavelength " + str(lambda_min) + " µm is at pixel number x = " + str(r[0])]
    output_info += ["Maximum wavelength " + str(lambda_max) + " µm is at pixel number x = " + str(r[1])]
    output_info += ["\n"]
    return True

def propagate(p1, p2):
    """
    For the rectangle of micromirrors defined by p1 and p2, simulate where rays will land on the CCD
    """
    
    min_x_pos = ccd.shape[0] #minimum pixel number in x-direction for this spectrum
    max_x_pos = 0 #maximum pixel number in x-direction for this spectrum
    d = {'x': [], 'y': []} #stores the positions of CCD pixels with centre wavelength
    
    for row in range(p1[0],p2[0]+1,1): #iterate over DMD rows/height of interest
        for col in range(p1[1],p2[1]+1,1): #iterate over DMD columns/width of interset

            ###############################################################
            #for y direction (spatial)
            #dmd_img_y_pos is position of micromirror on DMD in y-direction wrt horizontal centerline in DMD
            dmd_img_y_pos = (row - N_My/2 + 0.5) * p_M #add 0.5 so that we are measuring from centre of micromirror
            ccd_img_y_pos = dmd_img_y_pos * M_S #Position of corresponding pixel on detector in y-direction
            y_pos = int(round(ccd_img_y_pos/p_D + N_Dy/2 - 0.5)) #y-direction pixel number on detector; minus 0.5 to get the correct array index
            ###############################################################
            
            dmd_img_height = (col - N_Mx/2 + 0.5) * p_M #height of micromirror wrt optical axis
            
            u = (0 - dmd_img_height * (1/f_coll)) * 180/np.pi #ray angle [degrees] wrt optical axis after passing through collimator (refraction equation)
            ind_angle = alpha + u #actual incidence angle wrt to grating normal [degrees]
            
            diff_angle_cen = getDiffractionAngle(lambda_cen, m, G, alpha) #angle between grating and optical axis after dispersion [degrees]
            
            ###############################################################
            #for ccd_no_dispersion
            x_pos = grating_to_CCD(lambda_cen, ind_angle, diff_angle_cen)
            if x_pos < N_Dx and x_pos >= 0:
                ccd_no_dispersion[y_pos,x_pos] = 1
                d['x'] += [x_pos]
                d['y'] += [y_pos]
            ###############################################################

            #get the pixels where lamba_max and lambda_min are
            max_x_pos = max(max_x_pos, grating_to_CCD(lambda_max, ind_angle, diff_angle_cen))
            min_x_pos = min(min_x_pos, grating_to_CCD(lambda_min, ind_angle, diff_angle_cen))
            
            ###############################################################
            mult_factor = 1e6 #arbitrary factor
            #wavelength_iter is just a variable so that you can iterate over the wavelengths; mult_factor was selected arbitrarily
            for wavelength_iter in range(int(lambda_min*mult_factor), int(lambda_max*mult_factor), int(0.0003*mult_factor)):
                wavelength = wavelength_iter / mult_factor #actual wavelength

                x_pos = grating_to_CCD(wavelength, ind_angle, diff_angle_cen)
                
                if x_pos < N_Dx and x_pos >= 0: #ensure we do not index array out of range
                    #Match wavelength to CCD pixel; match to 3 pixels in y-direction for each pixel, for easier viewing,
                    #and to eliminate random black (missing) lines in y-direction due to rounding with int()
                    ccd[y_pos,x_pos] = wavelength
    
    cen_x_pos = int(round((d['x'][0] + d['x'][len(d['x'])-1]) / 2)) #pixel number in x-direction with centre wavelength
    cen_y_pos = int(round((d['y'][0] + d['y'][len(d['y'])-1]) / 2)) #pixel number in y-direction with centre wavelength
    return [min_x_pos, max_x_pos, cen_x_pos, cen_y_pos]

def grating_to_CCD(wavelength, ind_angle, diff_angle_cen):
    """
    This function traces a dispersed ray with some wavelength from grating to CCD.
    Returns corresponding CCD pixel number.
    INPUTS:
        ::float:: wavelength           #wavelength in µm
        ::float:: ind_angle            #actual incidence angle wrt to grating normal [degrees]
        ::float:: diff_angle_cen       #angle between grating and optical axis after dispersion for centre wavelength [degrees]
    OUTPUT:
        ::int:: x_pos                  #x-direction index for the ccd numpy array
    """
    b = getDiffractionAngle(wavelength, m, G, ind_angle) #diffraction angle in degrees
    angle_rel_oa = b - diff_angle_cen #ray angle wrt optical axis in degrees
    
    #Paraxial ray trace from grating to detector:
    h = 0 + (angle_rel_oa * np.pi/180) * f_cam #ray height wrt optical axis at camera optics (transfer equation)
    u_after_cam = (angle_rel_oa * np.pi/180) - h * (1/f_cam) #ray angle [radians] wrt optical axis after passing through camera optics (refraction equation)
    h_after_cam = h + u_after_cam * f_cam #ray height wrt optical axis at detector (transfer equation)
    
    x_pos = int(round(h_after_cam/p_D + N_Dx/2 - 0.5)) #x-direction pixel number on detector
    return x_pos

def get_boundary():
    """
    Returns the a pair of coordinates representing the boundary of the CCD 
    without dispersion at the centre wavelength
    """
    corner = [[0,0], [dmd.shape[0]-1, dmd.shape[1]-1]]
    out = []
    for coor in corner:
        ###############################################################
        #for y direction (spatial)
        #dmd_img_y_pos is position of micromirror on DMD in y-direction wrt horizontal centerline in DMD
        dmd_img_y_pos = (coor[0] - N_My/2 + 0.5) * p_M #add 0.5 so that we are measuring from centre of micromirror
        ccd_img_y_pos = dmd_img_y_pos * M_S #Position of corresponding pixel on detector in y-direction
        y_pos = int(round(ccd_img_y_pos/p_D + N_Dy/2 - 0.5)) #y-direction pixel number on detector; minus 0.5 to get the correct array index
        ###############################################################
        
        dmd_img_height = (coor[1] - N_Mx/2 + 0.5) * p_M #height of micromirror wrt optical axis
        u = (0 - dmd_img_height * (1/f_coll)) * 180/np.pi #ray angle [degrees] wrt optical axis after passing through collimator (refraction equation)
        ind_angle = alpha + u #actual incidence angle wrt to grating normal [degrees]
        
        diff_angle_cen = getDiffractionAngle(lambda_cen, m, G, alpha) #angle between grating and optical axis after dispersion [degrees]
        
        ###############################################################
        #for ccd_no_dispersion
        x_pos = grating_to_CCD(lambda_cen, ind_angle, diff_angle_cen)
        out += [[y_pos, x_pos]]
    return out

#######################################################################################
#######################################################################################
    
def getLSFConvolved(p1, p2, wavelength):
    """
    Plots the an "intensity" curve of a dispersed "beam" with some wavelength
    when it hits the CCD. The curve is plotted about the centroid of the "beam"
    """
    
    #check that the indices do not go out of bounds
    if p1[0] < 0 or p1[1] < 0 or p2[0] >= dmd.shape[0] or p2[1] >= dmd.shape[1]:
        return False
    
    #centre position of group of mirrors wrt optical axis [µm]
    y_cen = (((p2[0] - N_My/2 + 0.5) * p_M) + ((p1[0] - N_My/2 + 0.5) * p_M)) / 2.0
    x_cen = (((p2[1] - N_Mx/2 + 0.5) * p_M) + ((p1[1] - N_Mx/2 + 0.5) * p_M)) / 2.0
    
    DMD_width = (p2[1] - p1[1] + 1) * p_M #width of tophat function on DMD
    CCD_width = DMD_width * M_S #width of tophat function on CCD
    
    ###############################################################
    #for y direction (spatial)
    #dmd_img_y_pos is position of micromirror on DMD in y-direction wrt horizontal centerline in DMD
    dmd_img_y_pos = y_cen
    ccd_img_y_pos = dmd_img_y_pos * M_S #Position of corresponding pixel on detector in y-direction
    y_pos = int(round(ccd_img_y_pos/p_D + N_Dy/2 - 0.5)) #y-direction pixel number on detector; minus 0.5 to get the correct array index
    ###############################################################

    dmd_img_height = x_cen
    
    u = (0 - dmd_img_height * (1/f_coll)) * 180/np.pi #ray angle [degrees] wrt optical axis after passing through collimator (refraction equation)
    ind_angle = alpha + u #actual incidence angle wrt to grating normal [degrees]
    
    diff_angle_cen = getDiffractionAngle(lambda_cen, m, G, alpha) #angle between grating and optical axis after dispersion [degrees]
    
    b = getDiffractionAngle(wavelength, m, G, ind_angle) #diffraction angle in degrees
    angle_rel_oa = b - diff_angle_cen #ray angle wrt optical axis in degrees
    #Paraxial ray trace from grating to detector:
    h = 0 + (angle_rel_oa * np.pi/180) * f_cam #ray height wrt optical axis at camera optics (transfer equation)
    u_after_cam = (angle_rel_oa * np.pi/180) - h * (1/f_cam) #ray angle [radians] wrt optical axis after passing through camera optics (refraction equation)
    h_after_cam = h + u_after_cam * f_cam #ray height wrt optical axis at detector (transfer equation)
    
    x_pos = int(round(h_after_cam/p_D + N_Dx/2 - 0.5)) #x-direction pixel number on detector
    x_smallest = x_pos * p_D #distance from CCD left edge to current pixel left edge [µm]
    x_abs_dist = h_after_cam + (N_Dx/2)*p_D #distance from CCD left edge to spot [µm]
    x_diff = x_abs_dist - x_smallest #distance from spot to current pixel left edge [µm]
    
    ###############################################################
    
    CCD_width_microns = CCD_width * 1e6
    ds = CCD_width_microns/1000 #step
    start = -1 * CCD_width_microns/2 - 4*p_D*1e6 #starting point to plot
    stop = CCD_width_microns/2 + 4*p_D*1e6 #ending point to plot
    x_arr = np.arange(start, stop, ds)
    y_arr = np.zeros(x_arr.shape[0]) #for convolved function
    
    #plot Gaussian LSF
    y1 = gaussian(x_arr)
    plt.plot(x_arr, y1,color='b')
    plt.grid()
    plt.title('Gaussian LSF with σ = ' + str(round(sigma,3)) + ' µm')
    plt.xlabel('Distance away from centroid (x-direction) on CCD [µm]')
    plt.ylabel('Intensity [Normalized units]')
    plt.savefig("01_LSF.png")
    plt.show()
    
    #plot rect/tophat
    y2 = rect_np_array(x_arr, CCD_width_microns)
    plt.plot(x_arr,y2,color='b')
    plt.grid()
    plt.title("DMD input for micromirror(s) \n(x,y) = (" + str(p1[0]) + "," + str(p1[1]) + ") to (x,y) = (" + str(p2[0]) + "," + str(p2[1]) + ") inclusive")
    plt.xlabel('Distance away from centroid (x-direction) on CCD [µm]')
    plt.ylabel('Input [Arbitrary units]')
    plt.savefig("02_tophat.png")
    plt.show()
    
    #plot convolved function
    for i in range(0, y_arr.shape[0], 1):
        y_arr[i] = convolve(x_arr[i], rect, gaussian, CCD_width_microns) #convolution
    plt.plot(x_arr, y_arr)
    plt.grid()
    plt.title("CCD at " + str(round(wavelength,3)) + " µm;\n Centroid on pixel (x,y) = (" + str(x_pos) + "," + str(y_pos) + ")")
    plt.xlabel('Distance away from centroid (x-direction) on CCD [µm]')
    plt.ylabel('Intensity [Arbitrary units]')
    iterator = np.arange(x_diff*1e6, stop, p_D*1e6)
    for val in iterator:
        plt.axvline(val, color='tab:orange') #plot orange vertical lines to represent pixel borders
    iterator = np.arange(x_diff*1e6 - p_D*1e6, start, -1*p_D*1e6)
    for val in iterator:
        plt.axvline(val, color='tab:orange') #plot orange vertical lines to represent pixel borders
    plt.plot(x_arr, y_arr, color='b')
    plt.savefig("03_spot_plot.png")
    plt.show()

    return [x_pos, y_pos]

def rect_np_array(x, w):
    return np.where(abs(x) <= w/2, 1, 0)

def rect(x, w):
    if np.abs(x) < w/2:
        return 1
    else:
        return 0

def gaussian(x):
    return np.exp(-1 * x**2 / (2 * sigma**2))

def integrand(t, x, f, g, w):
    return f(t, w) * g(x - t)

def convolve(x, f, g, w):
    #https://docs.scipy.org/doc/scipy/reference/tutorial/integrate.html
    return scipy.integrate.quad(integrand, -np.inf, np.inf, args=(x, f, g, w))[0]

#######################################################################################
#######################################################################################
#MAIN PROGRAM

def main():
    temp_factor = 0.20 #some random scaling factor for testing purposes
    test_wavelength = 0.60 #arbitrary test wavelength in µm
    test_micromirrors = [[int(N_My * temp_factor),int(N_Mx/2)],[int(N_My * temp_factor + 10),int(N_Mx/2 + 10)]]
    turnMicromirrorsON(test_micromirrors[0],test_micromirrors[1])
    getLSFConvolved(test_micromirrors[0],test_micromirrors[1],test_wavelength)

    plotDMD(dmd) #show a plot of the DMD
    plotCCD_no_dispersion(ccd_no_dispersion) #show a plot of the CCD without dispersion
    plotCCD(ccd) #show a plot of the CCD
    
    #write the wavelength positions to a file
    with open('CCD_wavelength_positions.txt', 'w') as f:
        for line in output_info:
            f.write("%s\n" % line)
            print(line)
    return True

if __name__ == "__main__":
    main()

