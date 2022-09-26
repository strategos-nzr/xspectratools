import numpy as np
from scipy.optimize import curve_fit

from numpy import sin, cos, pi


#https://xrayutilities.sourceforge.io/simulations.html
# Maybe Commit to this?
"""
Cell Params

a,b,c,alpha,beta,gamma

"""
#Dummy Example
CellParams=[5,5,5,90,90,90]

# Defining Fit Functions for Peaks:

def Lorentzian(x,x0=0,C=1,w=1):
    """
    C is a scaling parameter
    w is the half width at half max
    https://en.wikipedia.org/wiki/Cauchy_distribution
    """
    
    return C*np.power(np.pi*w*(1+ np.power(2*(x-x0)/w,2)),-1)


def Gaussian(x,x0=0,C=1,w=1):
    """
    C is a scaling parameter 
    w is the standard deviation
    FWHM is 2 SQRT(2 Ln(2))w 
    """
    return C/(w* np.sqrt(2*np.pi)) * np.exp(-np.power(x-x0,2)/(2*w**2))

def PseudoVoigt(x,x0,wL=1,wG=1,C=1):
    FG= 2.355*wG
    FL= 2*wL
    
    FWHM= np.power( FG**5
                   + 2.69269*FL*FG**4 
                   + 2.42843*FL**2 * FG**3 
                   + 4.47163*FL**3 * FG**2
                   + 0.07842*FL**4 * FG
                   + FL**5
                   ,1/5)
        
    eta = 1.36603*(FL/FWHM) - 0.47719 *(FL/FWHM)**2 +0.11116*(FL/FWHM)**3
    
    return C*(Lorentzian(x,x0,w=wL)*eta+(1-eta)*Gaussian(x,x0,w=wG))

def PseudoVoigtFWHM(params,cov):
    """
    Returns the associated FWHM obtained from the PseudoFit
    https://en.wikipedia.org/wiki/Full_width_at_half_maximum
    """
    theta = params[0]
    wL = params[1]
    wG = params[2]
    C = params[3]
    theta_error = cov[0,0]
    WL_error = cov[1,1]
    wG_error = cov[2,2]
    height_error = cov[3,3]
    
    
    FG= 2.355*wG
    FL= 2*wL
    
    FWHM= np.power( FG**5
                   + 2.69269*FL*FG**4 
                   + 2.42843*FL**2 * FG**3 
                   + 4.47163*FL**3 * FG**2
                   + 0.07842*FL**4 * FG
                   + FL**5
                   ,1/5)
    
    FG_error =    (2.355*wG_error)**2 *(5*FG**4
                   + 4*2.69269*FL*FG**3 
                   + 3*2.42843*FL**2 * FG**2 
                   + 2*4.47163*FL**3 * FG
                   + 0.07842*FL**4 )**2
    
    FL_error =     (2*WL_error)**2 *( 2.69269*FG**4 
                   + 2*2.42843*FL * FG**3 
                   + 3*4.47163*FL**2 * FG**2
                   + 4*0.07842*FL**3 * FG
                   + 5*FL**4)**2
    
    denom=       np.power(FG**5
                   + 2.69269*FL*FG**4 
                   + 2.42843*FL**2 * FG**3 
                   + 4.47163*FL**3 * FG**2
                   + 0.07842*FL**4 * FG
                   + FL**5
                   ,4/5)
    
    FWHM_error = np.sqrt((FG_error+FL_error))
    
    eta = 1.36603*(FL/FWHM) - 0.47719 *(FL/FWHM)**2 +0.11116*(FL/FWHM)**3
    
    return [FWHM,FG,FL,eta,theta,FWHM_error,FG_error,FL_error, theta_error]



def DSEQ(FWHM,theta,SigmaFWHM=0,SigmaTheta=0):
    """ Return the debye scherrer estimate of the nanoparticle size
    FWHM - Voight Fullwidth half max
    Theta-peak angle in degrees
    
    assumes K =1 for the Bruker D8 and Horizon D2
    with a copper Ka wavelength of 1.402 Angstrom
    """
    
    BETA=FWHM*3.1415/180
    Debye_ScherrerEst= 1.402/(BETA*np.cos(theta*3.14/180))
    DS_error = Debye_ScherrerEst*np.sqrt((SigmaFWHM/FWHM)**2 + (np.tan(theta*3.14/180)*SigmaTheta*3.14/180)**2)
    return Debye_ScherrerEst,DS_error

def FitRange(DATA,THETA,p0=[11,.017,.05,.015],getrange=False):
    
    """ Fit A PseudoVoigt Function to a peak in a range
    
    base value for 0 assumes a normalized data set, which is done by force here
    may rescale the data later to fit """
    theta1=THETA[0]
    theta2=THETA[1]
    
    datax=DATA[np.logical_and(theta1<DATA[:,0],DATA[:,0] <theta2 ),0]
    
    datay=DATA[np.logical_and(theta1<DATA[:,0],DATA[:,0] <theta2 ),1]/ np.trapz(DATA[:,1])

    """ Need to find optimal Fit Parameters
    """
    ThetaMaxIndex=np.where(datay==np.max(datay))[0][0]
    
    FWHM_LeftIndex=np.where(datay>np.max(datay)/2)[0][0]  
    FWHM_RightIndex=np.where(datay>np.max(datay)/2)[0][-1]  
    FWHM_Guess= datax[FWHM_RightIndex] -datax[FWHM_LeftIndex]
    
    CGuess= np.power(1/(3.14*FWHM_Guess/2)+1/(2.5066*FWHM_Guess/2.355),-1 )
    pGuess= [datax[ThetaMaxIndex],FWHM_Guess/2,FWHM_Guess/2.355, CGuess]
    try:
        params,cov=curve_fit(PseudoVoigt,datax,datay,p0=pGuess)
    except:
        print("I am Error")
    if getrange==True:
        return datax,datay,params,cov
    else:
        return params, cov

def QuickDebye(DATA,THETA,p0=None):

    """ Combines all the previous functions to get a size estimate in angstroms
    
    1. Find the Fit Parameters
    2. Make some Error Estimates
    3. find the debye scherrer sizes
    
    """
    
    params,cov =FitRange(DATA,THETA,p0=p0)
    

    FWHM,FG,FL,eta,theta,FWHM_error,FG_error,FL_error, theta_error = PseudoVoigtFWHM(params,cov)


    Debye_ScherrerEst,DS_error = DSEQ(FWHM,theta,SigmaFWHM=FWHM_error,SigmaTheta=theta_error)
    
    return [Debye_ScherrerEst, DS_error]



    
# Defining Models for Diffraction analysis
    
    
# Get the structure Factor from the Unit Cell Positions    
def StructureFactor(h,k,l,UnitCellPositions,FormFactors,**CellParams):
    """ Unit Cell positions is a List of Positions of Atoms in the Unit Cell
    
    Form Factors are the "scattering amplitudes" for Individual Atoms in the Unit Cell
    
    **Cell Params is currently Unused
    """
    
    Shkl=0
    for i in range(len(UnitCellPos)):
        Shkl+=FormFactors[i]*np.exp(1j*2*np.pi *np.dot(UnitCellPositions[i],(h,k,l)))
    return Shkl



def BraggAngle(h,k,l,Wvlgth=1.41,**CellParams):
    """ Returns the Bragg angle Theta, not 2Theta, of peak at [hkl]
    
    Default wavelength is for a Copper Alpha Source, 1.41 Angstrom
    
    Unit Cell Params are given by a,b,c, alpha,beta,gamma
    
    """
    RecipricalLatticeVector = np.sqrt( (h/a)**2 + (k/b)**2 + (l/c)**2 )
    
    return np.arcsin(.5*Wvlgth*RecipricalLatticeVector)*180/np.pi




def DiscreteIntensityModel(CellPositions,FormsF,n=10):
    """
    A Basic Discrete Intensity model for determining the peak locations from Unit Cell positions
    
    CellPositions is a list of atomic coordinates in the unit cell. (3xN)
    
    FormsF are form factors, which depend on atom ~Z, must be same length as CellPositions
    (3xN)
    """
    
    hmax=n
    kmax=n
    lmax=n
    
    Angles=np.zeros(hmax*kmax*lmax)
    
    Intensities=np.zeros(hmax*kmax*lmax)
    m=0
    for h in range(n):
        for k in range(n):
            for l in range(n):
                Angles[m] = 2*BraggAngle(h,k,l)
                Intensities[m]=np.absolute(StructureFactore(h,k,l,CellPositions,FormsF))
                m+=1
                
    indices=np.argsort(Angles)
    results= [Angles[indices],Intensities[indices]]
    print(len(Angles),len(Intensities))       
    
    return  results


#The all important continuous model
def LorentzFitModel(Theta,FitCoefficients,UnitCellPositions,FormFactors,hmax=8,kmax=8,lmax=8):
    """ A more Complicated Fitting Procedure
    
    I(theta) = |Sum_{hkl} S_{hkl} L_{hkl}(x,x0,C,W)|^2
    
    
    1. Get Bragg Angles
    
    2. Get Structure Factors, Lorentzian Funcs
    
    3. Add together, Square
    
    FitCoefficients are coefficients of the Lorentzian for each hkl peak. Need hmax*kmax*lmax of them
    
    """
    
    Shkl=np.zeros(hmax*kmax*lmax)
    
    Natoms= len(UnitCellPositions)
    
    
    FuncList= 0
    #LorentzFitParams= np.ones((hmax*kmax*lmax,3))
    
    StructureFunction= np.zeros(len(Theta))
    i=0
    for h in range(hmax):
        for k in range(kmax):
            for l in range(lmax):
                
                Shkl = StructureFactore(h,k,l,UnitCellPositions,FormFactors,**CellParams)
                
                Theta_hkl=BraggAngle(h,k,l,Wvlgth=1.41,**CellParams)
                    
                LFunc= Lorentzian(Theta,Theta_hkl,C,W)
                
                
                i=+1
                
                
    FiMax=np.ones(Natoms)
    FWHM=np.ones(Natoms)
    
    for i in range(len(UnitCellPos)):
        Shkl+=FormFactors[i]*np.exp(1j*2*np.pi *np.dot(UnitCellPositions[i],(h,k,l)))
    
    
    return Shkl

    

def ContinuousIntensityModel():
    
    pass



def InterplanarSpacing(h,k,l,a=5,b=5,c=5,alpha=90,beta=90,gamma=90):
    """ Calculates Planar 
    Spacing with angles in degree, default values are meaningless"""
    
    
    part1= (h*sin(pi*alpha/180)/a)**2 + (k*sin(pi*beta/180)/b)**2 +(l*sin(pi*gamma/180)/c)**2 
    
    part2 = (2*k*l/(b*c))*(cos(beta)*cos(gamma)-cos(alpha))
    
    part3 = (2*h*l/(a*c))*(cos(alpha)*cos(gamma)-cos(beta))
    part4 = (2*k*h/(b*a))*(cos(beta)*cos(alpha)-cos(alpha))

    denom= 1 -cos(alpha)**2 -cos(beta)**2 -cos(gamma)**2 +2*cos(alpha)*cos(beta)*cos(gamma)
    
    return np.power((part1+part2+part3+part4)/denom,-1/2)