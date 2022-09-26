# -*- coding: utf-8 -*-
"""
Created on Thu Jul  7 16:48:49 2022

@author: Nick
"""
from glob import glob
import numpy as np
import matplotlib.pyplot as pp
from rixs import read_rixs


ddx= 4
ddy= 4

th=3
flist=glob('C:\\Users\\rando\\Documents\\Work\\BlueBronze\\BL4_XAS_RIXS\\CCDScan8533\\Andor CCD\\*2D.txt')

for i,file in enumerate(flist):
    
    rixs,sp= read_rixs(file)
    
    histo=np.histogram(rixs.flatten(),bins=np.linspace(0,1000,1001))
    
    pp.bar(x=histo[1][:-1],height=histo[0])
    
    stdev=np.std(rixs.flatten())
    
    print(histo[0].argmax(), histo[0][histo[0].argmax()])
    
    histo[0]
    
    print( np.diff(np.nonzero( histo[0] ) )> th*stdev )  
    
    c_ray_idx=np.where(np.diff(np.nonzero( histo[0] ) )> th*stdev)[1]+1
    
    print(c_ray_idx)
    if len(c_ray_idx) < 1:
        print("no bad pixels")
        continue
    
    
    bad_counts= np.nonzero(histo[0])[0][c_ray_idx]
    
    flagged_pixels=  np.where(rixs >bad_counts.min() )
    
    num_bad_px= np.shape(flagged_pixels)[1]
    
    for i in range(num_bad_px):
                
        rixs[flagged_pixels[i,0] ,flagged_pixels[i,1] ] = 0
        
pp.ylim(0,10)



# 'C:\\Users\\rando\\Documents\\Work\\BlueBronze\\BL4_XAS_RIXS\\CalibrationAttempt1\\CCDScan8558'
def calibrate_ccd(path):
    """   
    DEPRECATED
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

