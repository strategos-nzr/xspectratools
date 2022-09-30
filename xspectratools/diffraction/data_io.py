import numpy as np
import matplotlib.pyplot as pp

#https://xrayutilities.sourceforge.io/simulations.html
# Maybe Commit to this?


def GrabDataFromRange(DATA,Theta1,Theta2):
    """ 
    Splice an arbitrary powder or 1-D pattern and return the data from a specific range.
    Useful for fitting. 
    """

    pass


def GrabData(Filename,delimiter=" ",**kwargs):
    """ Use Numpy loadtxt to read the csv file,
    returns numpy array for quick plotting"""

    return np.loadtxt(Filename,delimiter=delimiter,skiprows=1,**kwargs)


def ImportCif(FileName,**kwargs):
    """ Read a Cif file and get the atomic data"""
    pass


def QuickXRDplot(data,scale=1,offset=0,Normalize = False,**kwargs):
    """
    Quickly Plot Numpy array of XRD data
    """
    N=1
    if Normalize == True:
        N= np.trapz(data[:,1])

    pp.plot(data[:,0],offset+scale*data[:,1]/N,**kwargs)
   
          
