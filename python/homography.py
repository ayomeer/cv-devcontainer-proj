
import numpy as np
import matplotlib.pyplot as plt
import lib.cppmodule as cpp # path from vs code root --> cd to /app/python
import cv2
import time
from concurrent.futures import ThreadPoolExecutor

# --- Constants ----------------------------------------------------------- #
FIGURE_SIZE = (12, 9)

# --- Functions ----------------------------------------------------------- #

def plotPointsOnImage(img: np.ndarray, points: np.ndarray):
    plt.figure()
    plt.imshow(img)
    for point in points:
        plt.plot(point[1], point[0], 'o')
    plt.title("image with points at specified coordinates")
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 0")
    plt.show(block=False)


# solve system of equations to find H_d_u
def homographyFrom4PointCorrespondences(x_d: np.ndarray, x_u: np.ndarray) -> np.ndarray:

    A = np.zeros((8,8))
    y = np.zeros(8)

    for n in range(4): # for each reference point generate two lines of the EqSys
        xu = x_u[n][0]; yu = x_u[n][1]
        xd = x_d[n][0]; yd = x_d[n][1]
                       
        x_idx = 2*n
        y_idx = 2*n + 1
                       
        A[x_idx] = np.array([xu, yu, 1, 0 , 0 , 0, -xd*xu, -xd*yu])
        A[y_idx] = np.array([0 , 0 , 0, xu, yu, 1, -yd*xu, -yd*yu])
        
        y[x_idx] = xd
        y[y_idx] = yd
        
    h_coefs = np.linalg.solve(A, y)
    H_d_u = np.append(h_coefs, [1]).reshape((3,3))
    return H_d_u


def hom2inhom(xhom):
    return xhom[0:2]/xhom[2]


def inhom2hom(x):
    xhom = np.ones(3)
    xhom[0:2] = x
    return xhom


def pointwiseUndistort(H_d_u, img_d, M, N):
    img_u = np.empty((M,N,3), np.uint8)
    for m in range(M):
        for n in range(N):
            # Change to hom. coords, do transform, go back to inhom coords
            xu = np.array([m, n])
            xu_hom = inhom2hom(xu)
            
            xd_hom = H_d_u@xu_hom # hom. transform
            
            xd = hom2inhom(xd_hom)
            xd = np.round(xd).astype(int) # get integer coords
            
            # Use transformed coords to get pixel value
            img_u[m][n] = img_d[xd[0], xd[1], :] # last dimensions: rgb channels  
    return img_u

def multiproc_undistortPixel(idx): # all passed by ref since threads in same process -> same memory
    # Change to hom. coords, do transform, go back to inhom coords
   
    print("current index: ", idx)
    return idx
"""
    m = idx[0]
    n = idx[1]
    
    xu = np.array([m, n])
    xu_hom = inhom2hom(xu)
    
    xd_hom = H@xu_hom # hom. transform
    
    xd = hom2inhom(xd_hom)
    xd = np.round(xd).astype(int) # get integer coords
    return img_d[xd[0], xd[1], :]"""

def main():
    # Reading image
    imgName = '/app/_img/chessboard_perspective.jpg'
    img_d = plt.imread(imgName)

    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(img_d)
    plt.show(block=False)

    # Define shape undistorted image
    M = N = 800
 
    # Pre-defined reference points in distorted and undistorted image
    x_d = np.array([[48, 385], [188, 927], [424, 55], [665, 721]])
    x_u = np.array([[0, 0], [0, N-1], [M-1, 0], [M-1, N-1]])

    # Show reference point in original, distorted image
    plotPointsOnImage(img_d, x_d)

    # Compute correspondence matrix
    H_d_u = homographyFrom4PointCorrespondences(x_d, x_u)

    # Apply correspondence matrix to each point of img_u
    img_u_shape = (M, N)

    t1 = time.perf_counter()
    # img_u = pointwiseUndistort(H_d_u, img_d, M, N)
    # img_u = np.array(cpp.pointwiseUndistort(img_d, H_d_u, img_u_shape))
    
    # multithreading test
    # prepare array containing coordinates to iterate over
    idx = np.indices((M,N))
    idx = np.stack((idx[0], idx[1]), axis=-1).reshape(M*N,2)

    with ThreadPoolExecutor(max_workers=4) as ex:
        img_u = np.array(ex.map(multiproc_undistortPixel, idx))
    


    t2 = time.perf_counter()
    print("tUndistort ", t2-t1)
    
    # Show result
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(img_u)
    plt.title("Undistorted Image")
    plt.show()

    dummy = 1

if __name__ == "__main__":
    main()




    
