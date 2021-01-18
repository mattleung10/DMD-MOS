#Matthew Leung
#October 2020
#######################################################################################

def get_data_file_info(data_file):
    """
    Obtains the field size and the number of rows to skip (when calling
    pd.DataFrame later on) for the GIA txt data file.
    INPUTS:
        ::string:: data_file    #path of GIA txt data file
    OUTPUTS:
        ::float:: field_size    #field size
        ::int:: skiprows        #number of rows to skip when calling pd.DataFrame
    """
    linenum = 0
    with open(data_file, encoding='utf-16-le') as f: #Careful: Zemax outputted txt file is UTF-16 LE
        while True:
            line = f.readline()
            #print(linenum, line)
            if "Field Width" in line and linenum != 2: #get the line with "Field Width"
                temp = line.split(' ') #temp is a list containing the different words (separated by ' ') in line
                break
            linenum += 1
        for elem in temp: #iterate through temp to find the field size
            try:
                field_size = float(elem)
                break #field size has been found
            except ValueError: #if elem is not a number
                pass
        while True: #keep on reading a new line until "Ray" is in the line, then break
            line = f.readline()
            if "Ray" in line:
                break
            linenum += 1

    skiprows = linenum + 2
    return field_size, skiprows

def get_wavelength_file_info(wavelength_data_file):
    """
    Obtains info from the wavelength data file.
    INPUTS:
        ::string:: wavelength_data_file     #path of the wavelength data file
    OUTPUTS:
        ::list:: w_data                     #list of floats representing wavelength values
        ::int:: primary_wavelength_num      #primary wavelength number in wavelength data file
    """
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
    """
    This function provides basic info of the IMA file including the dimension
    of the DMD, the slit position, and a list of the positions of the ON
    micromirrors.
    INPUTS:
        ::string:: ima_file             #path of IMA file
        ::boolean:: inverted            #whether or not the image was inverted (system dependent)
    OUTPUTS:
        ::int:: dmd_dim                 #max dimension of DMD (largest length along one edge)
        ::int:: slit_pos                #position of slit (in units of DMD micromirrors)
        ::list:: ON_pos                 #list of ints of ON positions of micromirrors (in units of DMD micromirrors)
    """    
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

    return dmd_dim, slit_pos, ON_pos

def get_predicted_spot_pos(field_size, M_S, dmd_dim, ON_pos, inverted=False):
    """
    Given the field size, magnification of the spectrograph, and the parameters
    obtained by the get_ima_file_info function, this function returns a list
    of predicted spot positions
    INPUTS:
        ::float:: field_size            #field size
        ::float:: M_S                   #magnification of spectrograph
        ::int:: dmd_dim                 #max dimension of DMD (largest length along one edge)
        ::int:: slit_pos                #position of slit (in units of DMD micromirrors)
        ::list:: ON_pos                 #list of ints of ON positions of micromirrors (in units of DMD micromirrors)
    OUTPUTS:
        ::list:: predicted_spot_pos     #list of floats of predicted spot positions (same units as field_size)
    """
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
    
    return predicted_spot_pos
