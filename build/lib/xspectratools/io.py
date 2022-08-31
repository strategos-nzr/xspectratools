import numpy as np
import pandas as pd
from glob import glob




def read_bl7(File,col=[0,2,3,4,5,6,7]):
    skip=14
        
    df = pd.read_csv(File, delimiter = "\t",
        skiprows= skip,**kwargs)
    
    return df.iloc[:,col]
    

def read_bl8(File,col=[1,8,9,10,11]):
 
    skip=15
    df = pd.read_csv(File, delimiter = "\t", 
        skiprows= skip,**kwargs)
    
    return df.iloc[:,col]

def read_bl4():

    pass


beamlines_dict = {
    "BL8": read_bl8,
    "BL7": read_bl7,
    "BL4": read_bl4,
    
     }

def read_xas(FILENAME, format = "BL8"):
    """
    read XAS data file from certain beamlines. 
    Specifiying the format for the given beamline, default is 
    beamline 8.0.1
    Beamlines = {7.3.1, 8.0.1.1 }

    Each beamline is slightly different, but the idea 
    is to build a dataframe with the following information:
     
    Frame Keys :
    0. Time of Day
    1. Beamline Energy 
    2. I0
    3. TEY UHV XAS
    4. TFY UHV XAS
    5. Beam Current
    """

    func= beamlines_dict.get(format)

    return func(FILENAME)

def batch_read(NameStem,format="BL7",**kwargs):

    """
    Read all XAS spectrum in an entire Folder. 
    No processing is done here, it just returns a list of pandas arrays.
    
    NameStem is the path/common file name.
    
    kwargs are those for the pandas dataframe in pandas.read_csv
    
    """
    FileList= glob(NameStem+"*.txt")
    DataList=[]
    
    for File in FileList:
        DataList.append(read_xas(File,format,**kwargs))
        
    N = len(FileList)
    
    print(str(N)," Spectra Acquired")
    
    return DataList





