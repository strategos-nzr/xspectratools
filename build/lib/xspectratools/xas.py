# Spectra Analysis tools for XAS
import numpy as np
import pandas as pd
import scipy.optimize as spo
import scipy.signal as ss
from glob import glob
import matplotlib.pyplot as pp
from scipy.stats import bootstrap

import adaptive as ad


def resample_xas():

    """
    To DO: put in 'coarse' XAS spectrum and energy spacings
    return adaptively resampled energy spacing file for new data
    to be acquired with
    
    """
    pass 

def batch_average(data_list):

    num_spectra=len(data_list)

    print("Num Spectra:",num_spectra)

    data_shape=np.shape(data_list[0])
    data_ave= np.zeros(data_shape)

    for data in data_list:
        data_ave+=data
    
    data/=num_spectra
    
    return data

def batch_std():
    """ Compute the error in the measurement from a collection of seperate spectra
    Somewhat analogous to increasing integration time.

    Sum of non correlated random variables is itself a random variable.
    For a count distribution sqrt(n) is the error,
    but for this we want not the standard deviation (the mean error), but instead the error in the mean
    
    """
    pass

def renormalize(data):
    """ 
    Ideally want to make XAS spectrum 'pretty' in the following sense:
    the pre-edge region is distinquishable from the edge jump feature
    the post edge 'flatline' is also visible
    
    This is done by:
    1. rescaling the data by the Incident flux (IO)
    2. shifting the data in the dataset slightly such that the lowest value in the set is 0
    Optionally:
    3. rescaling again by the integrand/max spectrum value to ensure that the highest data point is 1

    It is not possible, except in transmission mode, to get % absorbance for TEY and TFY signals.

    """
    data_shape=np.shape(data)
    rescaled_data=np.zeros(data_shape)
    rescaled_data[:,0]=data.iloc[:,0] #Monochromator Energies
    rescaled_data[:,1]=data.iloc[:,1]/data.iloc[:,1] #I0/I0
    rescaled_data[:,2]=data.iloc[:,2]/data.iloc[:,1] #Rescaled TEY
    rescaled_data[:,3]=data.iloc[:,3]/data.iloc[:,1] #Rescaled TEY 
    
    header=['Monochromator Energy','Io\/Io','rn TEY','rn TFY']
    return pd.DataFrame(rescaled_data,columns=header) 

def reffle():
    """
    Rearrange 2 Motor Scan files to make data plottable on a contour or 3d plot.
    """
    pass

def transmission_diff(data1,data2):

    pass