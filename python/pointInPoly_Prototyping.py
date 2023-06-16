import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import lib.cppmodule as cpp

def switchCoords(points):
    return np.array([p[::-1] for p in points])

def getNorms(polyPts):
    polyEdges = np.array([polyPts[1]-polyPts[0], 
                          polyPts[2]-polyPts[1],
                          polyPts[3]-polyPts[2],
                          polyPts[0]-polyPts[3],])
    
    rot = np.array([[0, -1], # different rot matrix in cv-coords!
                    [1, 0]])
    
    polyNormals = (rot@polyEdges.T).T.astype(int)
    return polyNormals

def _pointInPoly(p, polyPts, polyNrm):

    pVect = p - polyPts

    inside = True
    for i, n in enumerate(polyNrm):
        dotP = pVect[i].dot(n)
        print(dotP)
        if dotP < 0:      
            inside = False
    
    return inside
        

if __name__ == '__main__':

    # Create example image w/ poly

    # xonar box example
    # img = np.zeros((1960,4032,3))
    # polyPts = np.array([[ 755,  972],
    #                     [1637,  981],
    #                     [1273, 2958],
    #                     [ 542, 2506]])
    # polyPts_cv = switchCoords(polyPts)

    # small example
    img = np.zeros((50,50,3))
    polyPts = np.array([[10, 10],
                        [30, 15],
                        [25, 40],
                        [5, 35]])
    polyPts_cv = switchCoords(polyPts)
    cv.polylines(img, [polyPts_cv.reshape(-1,1,2)], True, (1,0,0))
    
    # Show poly and choose point
    plt.imshow(img)
    pt_cv = np.array(plt.ginput(1)[0]).T.astype(np.int64) # ginput returns list of tuples -> take first (and only)
    pt = pt_cv[::-1]

    print(pt)
    img[tuple(np.round(pt).astype(np.int64))] = (0,1,0) # write chosen point into image
    
    # perform check
    # print(pointInPoly(pt, polyPts))
    
    polyNrm = getNorms(polyPts).astype(np.int64)
    # print(cpp.pointInPoly(pt, polyPts.flatten(), polyNrm.flatten()))
    print(_pointInPoly(pt, polyPts, polyNrm))
    
    # show poly with chosen point
    plt.imshow(img)
    plt.show()
