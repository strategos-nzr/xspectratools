import numpy as np
import pandas as pd
from glob import glob




def read_bl7():

    pass

def read_bl8():

    pass

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
#[MAIN DATA, ]

