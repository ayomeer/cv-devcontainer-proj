import matplotlib.pyplot as plt
import numpy as np
from lib.cppmodule import HomographyReconstruction as HomRec


def getNorms(polyPts):
    polyEdges = np.array([polyPts[1]-polyPts[0], 
                          polyPts[2]-polyPts[1],
                          polyPts[3]-polyPts[2],
                          polyPts[0]-polyPts[3],])
    
    rot = np.array([[ 0,  1], # different rot matrix in cv-coords!
                    [-1, 0]])
    
    polyNormals = ((rot@polyEdges.T).T).astype(int)
    return polyNormals

# --- prepare test data -------------------------------------------------
pt = np.array([25, 25])
polyPts = np.array([[20, 10],[20, 40],[30,30],[30,20]]) # cv-coords
polyNrm = getNorms(polyPts)

img = np.zeros((50,50,3))

# --- test cppmodule ----------------------------------------------------
obj = HomRec(img)

# test __host__ function
print(obj.pointInPoly(pt, polyPts, polyNrm))

# Test homography
# H_ = np.zeros((3,3))
# imOut = np.array(obj.pointwiseTransform(H_, ), copy=False)

dummy = 0
