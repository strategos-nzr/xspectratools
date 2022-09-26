import numpy as np
import pandas as pd
import matplotlib.pyplot as pp

#https://xrayutilities.sourceforge.io/simulations.html
# Maybe Commit to this?
"""
Cell Params

a,b,c,alpha,beta,gamma

"""
#Dummy Example
CellParams=[5,5,5,90,90,90]

# Defining Fit Functions for Peaks:

def GrabDataFromRange(DATA,Theta1,Theta2):
    pass


def GrabData(Filename,delimiter=" ",**kwargs):
    return sp.array(pd.read_csv(Filename,delimiter=delimiter,**kwargs))[:,:]


def ImportCif(FileName,**kwargs):
    """ Read a Cif file and get the atomic data"""
    pass


def QuickXRDplot(dataname,scale=1,offset=0,Normalize = False,sign=1,**kwargs):
    if Normalize == True:
        pp.plot(dataname[:,0],offset+sign*scale*dataname[:,1]/sp.trapz(dataname[:,1]),**kwargs)
    else:
        pp.plot(dataname[:,0],offset+scale*sign*dataname[:,1],**kwargs)  
          
