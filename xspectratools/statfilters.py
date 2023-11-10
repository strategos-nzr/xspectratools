# Nick Russo
"""
Statistical Filtering and Analysis of Image or Spectral Data

Contains:

Max Ent:  10.1103/PhysRevB.84.235111
Cambridge Algorithm:   10.1214/lnms/1215460511

CCD Cosmic Ray Removal Algorithms
Pych:
Laplace Corrections
"""

import numpy as np
import numexpr as ne



def maximum_entropy(Image,
                     Broadening= 0.08,
                     Model = "Gaussian" ):
    """ Employs Cambridge Algorithm to unbroaden / sharpen image

    """


return sharpened_image