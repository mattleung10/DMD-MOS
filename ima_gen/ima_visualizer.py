import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ima_file = 'dmd_130_MOS.IMA'
ima_list = []

with open(ima_file) as f:
    dmd_dim = int(f.readline()) #max dimension of DMD
    for i in range(0,dmd_dim,1):
        line = f.readline()[0:-1]
        ima_list += [[int(char) for char in line]]

ima_numpy = np.array(ima_list)
ima_numpy_crop = ima_numpy[125:876,:]
plt.figure(figsize=(11.25,15), dpi=120)
#ax = plt.subplot(111)
#rect1 = patches.Rectangle((-1,125), 1001, 750, linewidth=1,edgecolor='r',facecolor='none')
#ax.add_patch(rect1)
plt.imshow(ima_numpy_crop, cmap='gray')
plt.ylabel('Micromirror Number (y direction)')
plt.xlabel('Micromirror Number (x direction)')
plt.savefig("DMD_visualized.png", bbox_inches='tight')
plt.show()
