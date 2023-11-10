"""
Creator: Nick Russo
"""
import re
import numpy as np
import pandas as pd

import matplotlib.pyplot as pp
from glob import glob
from scipy.optimize import curve_fit
from scipy import signal
from copy import deepcopy

from multiprocessing import Pool
from itertools import product

"""
Yet Another Rixs Analysis Software
YARAS 

The Work Flow should be as follows:

1. read in the dark spectrum, as an image. 
2. Filter out Cosmic rays from the dark image

3. For each line in the 'Andor' Folder:
    a. read in the 2D data file
    b. substract the dark spectrum
    c. Filter cosmic rays using any of the provided methods.
    d. sum the spectrum along the short axis
    e. return the line.

4. return the complete set of lines excitation energies as from the top level summary file.

5. Calibrate the spectra using known emission lines and the elastic peak (hopefully)
"""


def read_summary(file):
    
    """
    Read in a summary data spectrum
    and save it as a pd array. Usually in the top level folder above the "Andor" folder
    
    """
    data = pd.read_csv(file,sep="\t", engine="python",skiprows=12)
    return data

def read_rixs_1D(file):
    """ Read 1D data file. Is easier to quickly analyze then the 2D dataset, 
    with the admission it is harder to get a good cosmic ray discrimination
    """
    
    data=pd.read_csv(file,engine="python",sep='\t', skiprows= 9)
    return data

def read_rixs_2D(filename):

    """
    Read in a single CCD map
    return the CCD image as numpy array unfiltered, and the unfiltered RIXS spectrum line.
       
    """
    #data = pd.read_csv(file,sep="\t", engine="python",skiprows=11)
    
    x_px = 0
    y_px = 0 
    
    with open(filename,"r") as file:
        
        data=file.readlines(0)[:4]
        x_px=int(data[1].strip("\n").strip("X Size:"))
        y_px=int(data[2].strip("\n").strip("Y Size:"))
    
    ccd_data= np.loadtxt(filename,
                     skiprows=10,
                     delimiter="\t")
    
    new_shape=(y_px,x_px)
    print(new_shape)
    
    ccd_image = ccd_data[:,2].reshape(new_shape)
    spectrum = np.sum(ccd_image,axis=0)

    return ccd_image, spectrum 



def line(x,a=1,b=0):

    return a*x + b



def assemble_rixs_line(ccd_image,
                dark_image,
                rejection='pych',
                dark_scale=1,**kwargs):

    """
    Assemble the Line data profile from:
    1. the dark image
    2. the CCD image
    3. the rejection scheme of choice: Default is 'pych'
    Other's not yet implemented.

    The dark image should be rescaled to have
     the same integration time as the image 
    """
    # Subtract the Background
    adjusted_image= ccd_image-dark_image/dark_scale

    # Ensure there are no negative counts:
    adjusted_image[adjusted_image < 0 ] =0

    # Do the rejection Scheme
    new_image, cosmic_rays= pych_rejection(adjusted_image,**kwargs)

    #Compute the Emission spectrum
    rixs_line = np.sum(new_image,axis=0)

    cosmic_line=np.sum(cosmic_rays)

    return rixs_line, cosmic_line


def assemble_rixs_map(folder, 
                dark_image,
                rejection='pych',
                dark_scale=1,**kwargs):

    """ Assemble the map from a bunch of RIXS lines. 
    This is highly parellelizable"""

    image_list= glob(folder+"*2D.txt")

    length=len(image_list)

    print(length)


    star_list= product(image_list,[dark_image*dark_scale]) # has to be in [] to avoid making a product over the numpy array
    print(list(star_list))
    #output_array= np.zeros((2048,length))
    output=[]
    if __name__ == '__main__':
        with Pool(4) as p:
            output.append(p.starmap(assemble_rixs_line,star_list))
            print(output)
    
    return np.array(output)

def assemble_rixs_map_loop(folder,
                dark_image,
                rejection='pych',
                dark_scale=1,**kwargs):

    """ Assemble the map from a bunch of RIXS lines. 
    This is highly parellelizable"""

    image_list= glob(folder+"*2D.txt")

    length=len(image_list)

    print(length)
    output=[]

    for i, file in enumerate(image_list):
        print("File No: ", i , file)
        data,line=read_rixs_2D(file)
        output.append(assemble_rixs_line(data,dark_image,dark_scale=dark_scale,**kwargs))

    return np.array(output)


def quick_rixs_map(XASFILE, 
                      Folder,
                      ax=None,
                      levels=100,
                      offset= 00,
                      estart=0,
                      slope=1,
                      skip=14,
                      **kwargs):
    
    """ Assemble a RIXS Map from a collection of Files from Beamline 8
    This assumes that:

    Cosmic rays have been subtracted and 
    Dark background has been subtracted.
    Very fast plotting of multiple emission lines

    1. Grab the spectra files
    2. Assemble the Files into a data array
    3. Collect Ranges of Energies
    4. Renormalize Emission Yields by IO
    Collect Max Value from Map
    
    """
    
    FileList=glob(Folder+"*1D.txt")
    #print(FileList)
    L=len(FileList)
    print(L)
    
    num_rows=np.shape(np.array(pd.read_csv(FileList[1],delimiter="\t", skiprows=9))[:,1])[0]
    
    data=np.zeros((L,2048))
    
    for i,File in enumerate(FileList):
        x=np.array(pd.read_csv(File,delimiter="\t", skiprows=9))[:,1]
        data[i,:]=x[:]
    
    
    XASdata=pd.read_csv(XASFILE,delimiter="\t",skiprows=skip)
    for i in range(len(XASdata.iloc[:,3])):
        data[i,:]=data[i,:]/XASdata.iloc[i,3]
    print("Renormalized Yield by IO")               
    Energies=np.array(XASdata.iloc[:,2])
    
    pixel=np.linspace(1,2048,2048)
    
    emission_energies= slope*(pixel-offset)+estart
    x_axis=emission_energies
    y_axis=Energies

    pp.contourf(x_axis,y_axis,data,levels,**kwargs)

    pp.colorbar()
    
    pp.ylabel("Excitation Energy (eV)")
    pp.xlabel("Emission Energy (eV)")
    pp.tight_layout()

    return Energies,pixel,data;



# Rejection Schemes for Cosmic Rays
def pych_rejection(ccd_image, 
                step=2, 
                thresh=3,
                sigma_clip=3,
                verbose= False,
                dispersion_axis=0,
                ):
    """ 

    ccd_image - the original data set from the CCD camera. 
    Should be background (dark) subtracted

    thresh- ratio for rejection

    step- horizontal bin size. default is 2, for 1024 bins
    4 is OK for 64*4 pixels

    Reject Cosmic Rays using the approach from the paper:
    Pych,Wojtek, Pub. Astr. Pacific, 116:114-153, Feb. 2004,
    'A Fast Algorithm for Cosmic-Ray Removal from Single Images'

    Steps
    1. Select small Size frames which cover the frame, with overlap
    2. compute the standard deviation
    3. apply sigma clipping once (i.e. clip off counts
     in the distribution that are outside [med-a*sigma,med+a*sigma] 
     for some value of a?)
    4. HISTogram it (with bins = max count)
    5. find the mode (peak)
    6. For counts > mode, look for gaps (i.e. regions of zeros), 
    7. for the first gap     wider than a threshold,say 3 sigma
    8. if such a gap exists, flag pixels with counts above the gap.
    
    the CCD detectors give rectangular images, with the x-ray dispersion along one axis. 
    Thus it will be a good idea to box along the perpendicular axis.
    """
    
    im_shape=ccd_image.shape

    
    bins= int(im_shape[1]/step) #usually 2048/2 = 1024 bins
    new_image= np.zeros(im_shape)
    cosmic_ray_map=np.zeros(im_shape)

    for i in range(bins):
        
        box=ccd_image[:,(i*step):step*(i+1)] # this can be parallelized better.

        sigma_box= np.std(box)
        median_box = np.median(box)
        box[ box > median_box + sigma_box * sigma_clip]  = median_box
        # One step of Sigma_Clipping, very rough rejection of 3sigma + events
        sigma_box= np.std(box)
        median_box = np.median(box)


        #hist= np.histogram(box, bins=200, 
         #   range=(0,median_box + sigma_box*sigma_clip ))

        box_hist=np.histogram(box.flatten(),
            bins=100,range=(0,np.max(box.flatten()))); 
        print(box_hist)

        box_hist_nonzero=np.nonzero(box_hist[0])[0]
        print("non zero indices:", box_hist_nonzero)
        #Find the bin edges with non zero counts:
        gaps=np.diff(box_hist[1][box_hist_nonzero])
        
        print("gaps at:", gaps)
        gap_idx=np.where(gaps>thresh*sigma_box)+1
        print("gap_}idx at:", gap_idx)
        
        threshold_point=box_hist[1][box_hist_nonzero][gap_idx]
        

        newbox=deepcopy(box)
    
        if len(threshold_point)> 0: 
            print(i)
            print("box thresholds: ", threshold_point)

            cosmic_mask=np.where(box>np.max(threshold_point))
            #print("Mask for Box: " ,cosmic_mask)
            newbox[cosmic_mask]=median_box
            cosmic_ray_map[:,step*i:step+step*i][cosmic_mask] =1

        #Assemble New image from box frame.
        new_image[:,step*i:step+step*i]=newbox

        #For Debugging:
        if verbose==True:
            print("bin: ",i)
            print("nonzero bins in hist",box_hist_nonzero)
            print("nonzero bin labels in hist",box_hist[1][box_hist_nonzero])
            
            print("median and std. ",median_box, sigma_box)
            print(box[box > median_box + thresh*sigma_box] )
            print("gaps in spectrum",gaps)
            print("which gaps are bigger than sigma*3",gaps>thresh*sigma_box)
            print(gap_idx[0])
            print("box thresholds: ", threshold_point)
            if len(threshold_point)> 0: 
                print("Mask for Box: " ,cosmic_mask)

    return new_image,cosmic_ray_map



def diff_reject(ccd_image1,ccd_image2):
    
    """
    Take the difference between two arrays.
    
    Compute per pixel:
        
        (n1 - n2)/SQRT(n1n2)
    
    """
    
    image_diff=  ccd_image1-ccd_image2
    
    ratio= image_diff / (np.sqrt(ccd_image1) * np.sqrt(ccd_image2) )
    
    
    final_image = np.zeros(np.shape(ccd_image1))
    
    filtered_pixels = np.where(ratio<1)
    
    
    return final_image



def laplacian_reject(ccd_image):
    
    """
    First the image needs to be subsampled to 4 times its size by interpolation
    then it must be convolved with a Laplacian 
    
    then negative values are to be removed.
    

    Implementation Paper
    
    """
    
    
    LAPLACIAN=-.25*np.array([[0,1,0],
                             [1,-4,1],
                             [0,1,0]])
    
    dim=np.shape(ccd_image)
    print(dim)
    
    new_dim = (dim[0]*2,dim[1]*2)
    
    image_interp=np.zeros(new_dim)
    
    new_image=np.zeros(new_dim)
    
    image_interp[::2,::2]=ccd_image
    
    #do the convolution, filter out the negative points
    x = signal.convolve2d(image_interp,LAPLACIAN) 
    x[x<0]=0
    
    
    pass
