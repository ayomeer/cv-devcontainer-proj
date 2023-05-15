import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
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

def pointInPoly(p, polyPts):
    polyNrm = getNorms(polyPts)

    # -- kernel -----------------------------------------
    pVect = p - polyPts

    inside = True
    for i, n in enumerate(polyNrm):
        if pVect[i].dot(n) < 0:      
            inside = False
    
    return inside
        

if __name__ == '__main__':

    # Create example image w/ poly
    img = np.zeros((50,50,3))

    polyPts = np.array([[20, 10],[20, 40],[30,30],[30,20]]).astype(int) # cv-coords
    cv.polylines(img, [polyPts.reshape(-1,1,2)], True, (1,0,0))
    
    # Show poly and choose point
    plt.imshow(img)
    pt = np.array(plt.ginput(1)[0]).T.astype(int) # ginput returns list of tuples -> take first (and only)
    img[tuple(np.round(pt[::-1]).astype(int))] = (0,1,0) # write chosen point into image
    
    # perform check
    # print(pointInPoly(pt, polyPts))
    
    polyNrm = getNorms(polyPts).astype(int)
    print(HomRec.py_pointInPoly(pt, polyPts, polyNrm))
    
    # show poly with chosen point
    plt.imshow(img)
    plt.show()
