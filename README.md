# X Spectra Tools

A Set of tools for Dealing with Processing X-ray Spectra from XAs/RIXS Beamlines at the ALS. Includes some postprocessing methods to make life easier for analysis in other tools. 

### Currently covered Beamlines
- 7.3.1 
- 8.0.1

### Sub modules

- io  
    - read
    - write
    - assemble
    - splice
    - batchread
- xas
    - preedge
    - findE0
    - plot
    - diff 
    - reffle
- rixs
    - calibrate
    - plot
    - cosmic_filter
    - 
### Usage
Basically, a wrapper for `pandas` dataframe to read the data from beamline output formats. By default, data not of interest to the current experiment is discarded from the dataframe, but is possible to flag the inclusion for it. 

Different beamlines have different instrumentation. As part of an ongoing effort to standardize collection; this will likely be broken in the future.
This is expected, and we will try expedite such workflows as fast as possible. 

For XAS, background substraction and normalization between the edge and pre-edge is important.
For RIXS, it is the background subtraction and cosmic ray removal that are the key steps for data prep, although this is often done at the beamline.
