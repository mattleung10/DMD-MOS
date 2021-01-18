#Matthew Leung
#This script generates a IMA files which each represents a DMD
#This script also allows you to visualize the IMA files
#IMA file for use in Zemax OpticStudio

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

#Note that the x and y direction are in terms of those on the image plane
#in the Zemax file

n = 1000 #x-direction height of DMD (number of micromirrors)
y_width = 750 #y-direction width of DMD

def gen_ima(row):
    global n
    global y_width
    output = []
    #output is a n x n array with each entry being '0'
    for i in range(0,n,1): #iterate over rows
        output += [[]]
        for j in range(0,n,1): #iterate over columns
            output[i] += ["0"]
    
    on_slits = [row] #row of slits that are ON
    #on_slits = [int(n/4)-1, int(n/4)]
    
    for s in on_slits:
        for i in range(0, int(n/2), 50):
            output[s][i] = "1"
            output[s][n-1-i] = "1"
    
    output_write = [] #output array which will be written to IMA file
    for i in range(0,n,1):
        output_write += [""]
        for j in range(0,n,1):
            output_write[i] += output[i][j]
    
    with open('dmd_' + str(row) + '_MOS.IMA', 'w') as f:
        f.write(str(n) + "\n")
        for line in output_write:
            f.write("%s\n" % line)

def visualize_ima(ima_file):
    ima_list = []
    with open(ima_file) as f:
        dmd_dim = int(f.readline()) #max dimension of DMD
        for i in range(0,dmd_dim,1):
            line = f.readline()[0:-1]
            ima_list += [[int(char) for char in line]]
    
    ima_numpy = np.array(ima_list)
    #ima_numpy_crop = ima_numpy[125:876,:]
    plt.figure(figsize=(11.25,15), dpi=120)
    ax = plt.subplot(111)
    rect1 = patches.Rectangle((-1,124), 1001, 751, linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect1)
    plt.imshow(ima_numpy, cmap='gray')
    plt.ylabel('Micromirror Number (y direction)')
    plt.xlabel('Micromirror Number (x direction)')
    plt.savefig(ima_file[0:-4] + "_visualized.png", bbox_inches='tight')
    plt.show()

def main():
    # 124 < row < 624
    for row in range(125, int(n/2), 50):
        gen_ima(row)
        gen_ima(n-1-row)
        visualize_ima('dmd_' + str(row) + '_MOS.IMA')
        visualize_ima('dmd_' + str(n-1-row) + '_MOS.IMA')
    return True

if __name__ == "__main__":
    main()

