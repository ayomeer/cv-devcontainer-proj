import matplotlib.pyplot as plt
import numpy as np
from lib.cppmodule import HomographyReconstruction as HomRec

# --- prepare test data -------------------------------------------------
pt = np.array([50, 50])
polyPts = np.array([[20, 10],[20, 40],[30,30],[30,20]]) # cv-coords
img = plt.imread('/app/_img/xonar_perspective.jpg')

# --- test cppmodule ----------------------------------------------------
obj = HomRec(img)

# test __host__ function
print(obj.py_pointInPoly(pt, polyPts))

# Test homography
# H_ = np.zeros((3,3))
# imOut = np.array(obj.pointwiseTransform(H_, ), copy=False)

dummy = 0
