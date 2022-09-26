# X Spectra Tools

A Set of tools for Dealing with Processing X-ray Spectra from different Beamlines at the ALS and different acquisition modes. Includes some postprocessing methods to make life easier for analysis in other tools. 

### Currently covered Beamlines
- 4.0.3
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
    - reffle
- rixs
    - calibrate
    - plot
    - cosmic_filter
- ARPES
    - Igor Conversion
    - data plotting
- uncertainty 
    - noise
- diffraction
    - data_io
    - diffractanal

### Usage
 Essentially, everything here utilizes either a  `pandas` dataframe in order to read the data from various beamline output formats. By default, data not of interest to the current experiment is discarded from the dataframe, but is possible to flag the inclusion for it. 

 Different beamlines have different instrumentation and thus require different analysis workflows. This is expected, and for the ones I am using, i am trying to expedite such workflows as fast as possible. For XAS, background substraction and normalization between the edge and pre-edge is important. In the case of EXAFS, this is required for a large range for good k-space resolution.  For RIXS, it is the background subtraction and cosmic ray removal that are the key steps for data prep, although this is often done at the beamline.

 For ARPES, since Igor Pro:tm: is the standard way of analyzing output waves from the spectrometer, I would like to get away from that and instead make the data more digestable to the average (read: low budget) experimenter. Sure, why not have your institution pay for Igor pro so you dont have to? But, on the other hand, why not make data analysis more accessable and streamline standardization with open source methods?