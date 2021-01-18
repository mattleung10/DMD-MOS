# DMD-MOS

## Overview

A multi-object spectrograph (MOS) is an instrument that can acquire the spectra of hundreds of astrophysical objects simultaneously, saving much on observing time. In the recent years, the [digital micromirror device (DMD)](https://en.wikipedia.org/wiki/Digital_micromirror_device) has shown potential in becoming the central component of the MOS, being used as a programmable slit mask. A DMD-based multi-object spectrograph is being developed at the [Dunlap Institute for Astronomy and Astrophysics](http://www.dunlap.utoronto.ca/) at the University of Toronto. I am currently working on a wavelength calibration fit to correct hyperspectral imaging distortion in this DMD-MOS. For some wavelength λ, a point on the DMD given by (x,y) can be mapped to a point on the image plane, (x<sub>d</sub>,y<sub>d</sub>), and the goal is to find this mapping. For more information about the fit, see the ```Reports``` folder.

In September 2020, I submitted a [poster](https://ras.ac.uk/poster-contest/matthew-c-h-leung) for the Royal Astronomical Society (RAS) Early Career Poster Competition which shows some of my earlier work in this project, for the case of y=0. 

![](https://github.com/mattleung10/DMD-MOS/blob/main/RAS_Poster/Poster_Images/DMD-MOS_Optical_Design.jpg)

## Brief Summary of Steps
1) Generate IMA files for use as masks in Geometric Image Analysis (GIA) in Zemax OpticStudio, on the plane of the DMD surface; each "opening" in the mask represents a micromirror on the DMD
2) Launch 1,000 to 1,000,000 rays in the DMD-MOS system using GIA; determine where the rays land on the detector
3) Use k-means to cluster points together corresponding to rays from the same micromirror
4) For each λ, use the equation y<sub>d</sub> = a<sub>s</sub>x<sub>d</sub><sup>2</sup>+c<sub>s</sub> to fit smile distortion, where (x<sub>d</sub>,y<sub>d</sub>) are centroids obtained from k-means
5) Determine a<sub>s</sub> and c<sub>s</sub> as functions of x, y, and λ
6) Fit keystone distortion

Smile fit for minimum wavelength λ=0.400µm:![](https://github.com/mattleung10/DMD-MOS/blob/main/RAS_Poster/Outputs/0.400um_plot.png)

Fit for c<sub>s</sub> as a function of y and λ (c<sub>s</sub> has no dependency on x):![](https://github.com/mattleung10/DMD-MOS/blob/main/Smile_Fit/dataset/smile_results/angle_add_corr_roots_quad_2/angle_add_corr_roots_quad_2_surface.png)

## Key Milestones and Current Progress
- December 2020 - Fit determined for the c<sub>s</sub> coefficient in smile distortion fit (case of x=0) with a maximum fit error of 17.6µm on the detector, which is less than 2× the pitch of a pixel (fit error is shown in the figure below)
- August 2020 - Fit determined for the case of y=0 with a maximum fit error of around 22µm on the detector

![](https://github.com/mattleung10/DMD-MOS/blob/main/Smile_Fit/dataset/smile_results/angle_add_corr_roots_quad_2/angle_add_corr_roots_quad_2.png)

## Repository Folders
-	```Smile_Fit``` contains the code for the smile fit
-	```RAS_Poster``` contains the code used to generate the results in the poster I submitted for the RAS Early Career Poster Competition
-	```Paraxial_Sim_Convolution``` contains code used in a preliminary simulation of the DMD-MOS system using paraxial approximation and by convolving the input signal
-	```ima_gen``` contains the code used to generate IMA files for GIA in Zemax-OpticStudio
- ```Reports``` contains some PDF progress reports of the project
-	```MATLAB``` contains the MATLAB standalone application used with ZOS-API
