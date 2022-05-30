# X Spectra Tools

A Set of tools for Dealing with Processing X-ray Spectra from different Beamlines at the ALS and different acquisition modes. Includes some postprocessing methods to make life easier for analysis in other tools. 

### Currently covered Beamlines

- 7.3.1 
- 8.0.1.1

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
- rixs
    - calibrate
    - plot
    - cosmic_filter
- ARPES
    - Igor Conversion
    - 
- uncertainty 
    - noise

### Usage
 Essentially, everything here utilized `pandas` dataframes in order to read the data from various beamline output formats. By default, data not of interest to the current experiment is discarded from the dataframe, but is possible to flag the inclusion. 

 