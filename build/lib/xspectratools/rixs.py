import numpy as np
import pandas as pd

import matplotlib.pyplot as pp
from glob import glob
from scipy.optimize import curve_fit
from scipy import signal


def read_summary(file):
    
    """
    Read in a summary data spectrum
    and save it as a pd array
    
    """
    data = pd.read_csv(file,sep="\t", engine="python",skiprows=11)
    return data

def read_rixs(filename):

    """
    Read in a single CCD map
    return it, unfiltered, and the unfiltered RIXS spectrum.
       
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
    
    image = ccd_data[:,2].reshape(new_shape)
    spectrum = np.sum(image,axis=1)

    return image, spectrum 

# 'C:\\Users\\rando\\Documents\\Work\\BlueBronze\\BL4_XAS_RIXS\\CalibrationAttempt1\\CCDScan8558'
def calibrate_ccd(path):
    """
    
    
    Ideal: Function that returns emission energy as a 
    function of location for a given spectrometer angle: 
    Emission Energy = f(Theta,Pix)
    Theta = ~ spectrometer energy
    
    Heres the Idea:
        
    For a given 'spectrometer energy' there is going to be a few energies for 
    excitation.
    
    
    The Spectrometer energy and file name are in the main file.
    We need to take that data, read in the CCD frame, and then integrate along the width
    to get a function that is "roughly" a spectrum for a given image,
    and use the elastic peak placement to get thee energy calibration
    for a given spectrometer energy and frame.
    """
    
    print("Using this path:  ",path)
    summary_file=glob(path+"*.txt")[0]
    
    print("Frame- Excitation File: ", summary_file)
        
    try:
        data_summary=read_summary(summary_file)
    
    except: 
        print("No")
        pass
    finally:
        print("OK")
        pass 
    print("Obtained File List: ",data_summary.Filename )
    filelist=list(data_summary.Filename)
    # Calibration=glob(path)
    
   
    calibration_data= np.zeros((len(data_summary.Filename), 5 )) 
   
    calibration_data[:,:3] = np.array(data_summary.iloc[:,:3])
   
   
   
    for i, file in enumerate(filelist):
     
        
        try:
            rixs , spectrum = read_rixs(file) 
        except:
            print("Bad File Format/ Nothing to read here.")
            continue
        finally:
            print(f"Read File: {file} number {i}")
        
        pp.plot(spectrum, label= f"{i} / Exc = {data_summary.iloc[i,1]}")
        
        elastic_idx = np.argmax(spectrum)
        
        calibration_data[i,3] = elastic_idx
        
        print(f"Frame {i} with Ex= {calibration_data[i,1]} and SE={calibration_data[i,2]}")
    
    print("AcquistionComplete")
    
    
    print("Begining Energy Calibration")
    
    ex_spec_diff = calibration_data[:,1]-calibration_data[:,2]
    
    params,cov= curve_fit(line,
                          ex_spec_diff,
                          calibration_data[:,3],
                          p0=(300,500))
    
    
    return calibration_data, params, cov


def line(x,a=1,b=0):

    return a*x + b




def single_reject(ccd_image, box=16, thresh=3,
                  dispersion_axis=0,
                  interp_radii=(4,16)
                  ):
    """ 
    Reject Cosmic Rays using the approach from the paper:
    Pych,Wojtek, Pub. Astr. Pacific, 116:114-153, Feb. 2004,
    'A Fast Algorithm for Cosmic-Ray Removal from Single Images'
    """
    
    im_shape=ccd_image.shape
    
    num_y_bins = 2*im_shape[0]/(box) - 2
    num_x_bins = 2*im_shape[1]/(box) - 2

    print(num_y_bins,num_x_bins)
    bins=np.linspace(0,500,501)
    step= int(box/2)

    for i in range(num_y_bins):
        for j in range(num_x_bins):
            
            newbox=ccd_image[step*i:step*(2+i),(j*step):step*(j+2)]
            histogram= np.histogram(newbox.flatten(),bins=bins)
    pass



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
    x = signal.convolve2d(image_interpp,LAPLACIAN) 
    x[x<0]=0
    
    
    pass













