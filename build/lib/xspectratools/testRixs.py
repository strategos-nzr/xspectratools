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
                
        rixs[flagged_pixels[i,0] ,flagged_pixels[i,1] ] = 
        
pp.ylim(0,10)


