
import numpy as np
import matplotlib.pyplot as plt
import lib.cppmodule as cpp # path from vs code root --> cd to /app/python
import cv2 as cv
import time

# --- CONSTANTS ------------------------------------------------------------------------ #
FIGURE_SIZE = (12, 9)

# AR Wireframe
points_b = np.array(
        [
            # qube wireframe
            [0, 0, 0], # 0: A[0,0]
            [1, 0, 0], # 1: A[1,0]
            [1, 0, 1], # 2: F[1,0]
            [0, 0, 1], # 3: F[0,0]
            [0, 1, 0], # 4: A[0,1]
            [1, 1, 0], # 5: A[1,1]
            [1, 1, 1], # 6: F[1,1]
            [0, 1, 1], # 7: F[0,1]
            # axes representation
            [0.3, 0, 0],
            [0, 0.3, 0],
            [0, 0, 0.3]
        ]).T

# Individual scaling of the cube's axes
res = 2 # px/mm
points_b = (res*np.diag((165, 240, 60)) @ points_b) # xonar box
# points_b = (res*np.diag((30, 30, 30)) @ points_b)

# Lines on the cuboid as sequence of tuples containing the indices of the starting point and the endpoint

"""
edges = [
         [4, 5], [5, 6], [6, 7], [7, 4],    # Edges right plane
         [0, 4], [1, 5], [2, 6], [3, 7],    # Edges connecting left w/ right
         [0, 1], [1, 2], [2, 3], [3, 0],    # Edges left plane
         [0, 8], [0, 9], [0, 10],           # Edges indicating the coordinate frame
        ]
"""
edges = [
         [0, 1], [0, 4], [4, 5], [1, 5],    # A
         [3, 7], [3, 2], [7, 6], [2, 6],    # F
         [0, 3], [4, 7], [5, 6], [1, 2],    # corder edges
         [0, 8], [0, 9], [0, 10],           # Edges indicating the coordinate frame
        ]

# Line colors can be given one of the strings 'r' for red, 'g' for green, 'b' for blue
edgecolors = [
               '1.0','1.0','1.0','1.0', # front face
               '0.5','0.5','0.5','0.5', # sides
               '0.2','0.2','0.2','0.2', # back face
                'r','g','b',            # coord frame
             ]


# --- HELPER FUNCTIONS ------------------------------------------------------------------ #

def plotPointsOnImage(img: np.ndarray, points: np.ndarray):
    plt.figure()
    plt.imshow(img)
    for i, point in enumerate(points):
        plt.plot(point[1], point[0], 'o', label='p{}'.format(i))
    plt.title("image with points at specified coordinates")
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 0")
    plt.legend()
    plt.show()

def plotPointsOnImageCV(img: np.ndarray, points: np.ndarray):
    plt.figure()
    plt.imshow(img)
    for i, point in enumerate(points):
        plt.plot(point[0], point[1], 'o', label='p{}'.format(i))
    plt.title("image with points at specified coordinates")
    plt.xlabel("coordinate 0")
    plt.ylabel("coordinate 1")
    plt.legend()
    plt.show(block=False)

# Plot box wireframe edges
def plot_edges(ax, image_points: np.ndarray, edges: list, edgcolors: list = [], title: str="view"):
    #plt.ylim(3000, 0)
    #plt.xlim(0, 4000)
    for edgeId, edge in enumerate(edges):
        pt1 = image_points[:, edge[0]]
        pt2 = image_points[:, edge[1]]
        x1 = pt1[0]
        x2 = pt2[0]
        y1 = pt1[1]
        y2 = pt2[1]
        # in the plot y points upwards so i use -y to let positive y point downwards in the plot
        colorString = edgecolors[edgeId]
        ax.plot([y1, y2], [x1, x2], colorString)
    plt.title(title)
    plt.xlabel('y - axis')
    plt.ylabel('x - axis')
    return

# Plot RANSAC matches
def plotMatches(trainImage, trainPoints, queryImage, queryPoints, mask, matches):

    imgMatches = cv.drawMatches(
        img1=queryImage,
        keypoints1=queryPoints,
        img2=trainImage,
        keypoints2=trainPoints,
        matches1to2=matches,
        outImg=None,
        matchesThickness=1,
        matchColor = None,  # (100,255,100), # draw matches in green color
        singlePointColor = None,
        matchesMask = mask.ravel().tolist(),  # draw only inliers
        flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,  # cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    plt.imshow(imgMatches)
    plt.title('Matches refined with homography-RANSAC')
    plt.show()

# Solve sytem of eqs for docal lengths and pose
def recoverRigidBodyMotionAndFocalLengths(H_c_b):
    """
    Input
      H_c_b is a homography from a planar object in b-coordinates to an image c-coordinates with the ORIGIN on the IMAGE CENTER!
    Output
      Rotation matrix R_c_b
      Translation vector t_c_cb
      focal lenghts fx and fy
    """
    H = H_c_b
    A = np.array([  [H[0,0]**2,     H[1,0]**2,     H[2,0]**2    ],
                    [H[0,1]**2,     H[1,1]**2,     H[2,1]**2    ],
                    [H[0,0]*H[0,1], H[1,0]*H[1,1], H[2,0]*H[2,1]]])
    
    y = np.array([[1], [1], [0]])

    # Solve sytem for focal lengths
    x = np.linalg.inv(A) @ y                # x = [a**2, b**2, c**2].T
    LambdaInv = np.diag(np.sqrt(x.ravel())) # LabdaInv = diag(lambda/fx lambda/fx lambda)**2
    E_ = LambdaInv @ H_c_b

    # Rearrange E_ into R_c_b and t_c_cb
    rx = E_[:, [0]]
    ry = E_[:, [1]]
    rz = np.cross(rx.ravel(), ry.ravel()).reshape((3,1))
    R_c_b = np.hstack((rx, ry, rz))
    t_c_cb = E_[:,[2]]
    
    # Get focal lengths from LambdaInv
    fx = LambdaInv[2, 2] / LambdaInv[0, 0]
    fy = LambdaInv[2, 2] / LambdaInv[1, 1]
    return R_c_b, t_c_cb, fx, fy

def hom2inhom(xhom):
    return xhom[0:2]/xhom[2]

def inhom2hom(x):
    xhom = np.ones(3)
    xhom[0:2] = x
    return xhom

def homographyTransform(xu, H_d_u):
    xu_hom = inhom2hom(xu)
    xd_hom = H_d_u@xu_hom # hom. transform
    xd = hom2inhom(xd_hom).astype(int)
    return xd

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
            if (xd[0]<img_d.shape[0] and xd[1]<img_d.shape[1]):
                img_u[m][n] = img_d[xd[0], xd[1], :] # last dimensions: rgb channels  
    return img_u

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

# Switch (x,y) <-> (y,x)
def switchCoords(points):
    return np.array([p[::-1] for p in points])


def getAbc2(H):
    A = np.array([  [H[0,0]**2,     H[1,0]**2,     H[2,0]**2    ],
                    [H[0,1]**2,     H[1,1]**2,     H[2,1]**2    ],
                    [H[0,0]*H[0,1], H[1,0]*H[1,1], H[2,0]*H[2,1]]])
    
    y = np.array([[1], [1], [0]])

    # Solve sytem for focal lengths
    x = np.linalg.inv(A) @ y
    x = x.reshape(3,)  
    return np.array([x[0], x[1], x[2]])

def genAbcNeighborhoods(xd, xu, N):

    R = N//2 # neighborhood radius    
    abc2 = np.empty((4,N,N,3)) # points, x-dim window, y-dim window, abc
    
    for k in range(4):
        for i in range(N):
            for j in range(N):
                deltaX = np.zeros((4,2))
                windowX = np.array([i-R,j-R])
                deltaX[k] += windowX
                
                xd_ = xd + deltaX 

                H = homographyFrom4PointCorrespondences(xd_, xu)
                abc2[k,i,j,:] = getAbc2(H)
                  
    return abc2
    
def homographyCorrection(x_d, x_u, N):
    focalParams = genAbcNeighborhoods(x_d, x_u, N)

    masks = np.empty((4, N, N), dtype=bool)
    for i, _ in enumerate(focalParams):
        masks[i] = (focalParams[i,:,:,0] > 0) & (focalParams[i,:,:,1] > 0) & (focalParams[i,:,:,2] > 0)
    
    # Create data for testing how close (a, b) ~ (fx, fy) are
    ratio = np.zeros((4, N, N))
    for i, _ in enumerate(focalParams):
        ratio[i] = np.abs(focalParams[i,:,:,0] / focalParams[i,:,:,1]) # |a**2-b**2| neighborhood around each point
        
    ratio_masked = ratio[masks] # subset array where mask == True
    
    # Create array of indexes to track original idx after masking
    masked_origIdx = np.arange(ratio.size)[masks.flatten()] # Idx values of original array where mask == True
    masked_minIdx = np.argmin(np.abs(ratio_masked-1)) # find min idx within subset array
    
    minIdx = masked_origIdx[masked_minIdx]
    
    correctedPoint = np.array(np.unravel_index(minIdx, ratio.shape))
    correctedPoint[-2:] = correctedPoint[-2:] - N//2 # change to centered neighborhood coords 
    print("correctedPoint: ", correctedPoint)
    
    x_d[correctedPoint[0]] += correctedPoint[-2:]
    
    return x_d

# === MAIN ==================================================================================== #
if __name__ == "__main__":   
   
    ## -- Read images ----------------------------------------------------------------
    trainImage = plt.imread('/app/_img/xonar_template.jpg')
    queryImage = plt.imread('/app/_img/xonar_perspective.jpg')
    
    ## -- Feature Detection: SIFT ------------------------------------------------------
    sift = cv.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.001,
        edgeThreshold=20,
        sigma=1.5,
        nOctaveLayers=4
    )
    trainPoints, trainDescriptors = sift.detectAndCompute(trainImage, None)
    queryPoints, queryDescriptors = sift.detectAndCompute(queryImage, None)

    ## -- Feature Matching: BFMatcher ------------------------------------------------- 
    matcher = cv.BFMatcher_create(cv.NORM_L1, crossCheck=True)
    matches = matcher.match(queryDescriptors, trainDescriptors)
    print('{} matches found'.format(len(matches)))

    """
    # Show matches, starting with the most reliable
    sortedMatches = sorted(matches, key = lambda x:x.distance)
    pltImage = cv.drawMatches(trainImage, trainPoints, queryImage, queryPoints, sortedMatches[:400], trainImage, flags=2)
    plt.imshow(pltImage)
    plt.title('Brute force matching result')
    plt.show()
    """
    
    xd = np.float32([queryPoints[m.queryIdx].pt for m in matches]).reshape(-1,2)
    xu = np.float32([trainPoints[m.trainIdx].pt for m in matches]).reshape(-1,2)
    

    ## -- Fit homography using top face reference and RANSAC ---------------------------
    H_c_b, mask = cv.findHomography(xu, xd, method=cv.RANSAC, ransacReprojThreshold=4.0)

    # plotMatches(trainImage, trainPoints, queryImage, queryPoints, mask, matches) 

    ## -- Estimate Pose from homography ------------------------------------------------
    # Homography found using RANSAC is not guaranteed to be valid for pose estimation.
    # To correct the homography, we switch to a 4-point-correspondence model, so that
    # there are fewer variables to tweak.
    
    # -- Find 4 correspondence points in query image --
    M_t, N_t = trainImage.shape[0], trainImage.shape[1]
    rect_u_ = np.array([[0, 0], [0, N_t-1], [M_t-1, N_t-1], [M_t-1, 0]]) 
    rect_u = switchCoords(rect_u_) # switch to openCV coord system, since homographyy H_c_b was found by openCV function cv.findHomography
    rect_d = np.empty((4,2), dtype=int)

    # Transform each corner of the reference image back into the query image using homography
    for i in range(rect_d.shape[0]):
        rect_d[i] = homographyTransform(rect_u[i], H_c_b).astype(int)
    
    # Check transform result 
    # plotPointsOnImageCV(queryImage, rect_d)
    # plt.show()
    
    # homography from 4 point correspondence
    M_u, N_u = res*165, res*240 
    x_d = np.array([p[::-1] for p in rect_d]) # change to x-down-coordinates
    x_u = np.array([[0, 0], [0, N_u-1], [M_u-1, N_u-1], [M_u-1, 0]])
    
    M_d, N_d = queryImage.shape[0], queryImage.shape[1] 
    x_d_center = np.array([M_d/2, N_d/2])
    cx_d = x_d-x_d_center

    # Tweak correspondence points in query Image that are valid for pose estimation
    cx_d_corrected = homographyCorrection(cx_d, x_u, 10)
    
    # Recompute homography from tweaked 4 point correspondence
    cH_d_u = homographyFrom4PointCorrespondences(cx_d_corrected, x_u)
    
    # Finally actually do the pose and focal length estimation 
    # (sign flip to match negative z component as a result of numpy based coords)
    R_c_b, t_c_cb, fx, fy = recoverRigidBodyMotionAndFocalLengths(cH_d_u)
  

    # Create camera intrinsics matrix
    # Note: inclusion of cx, cy brings back coordinate offset
    cx = M_d / 2
    cy = N_d / 2
    K_c = np.array([[fx, 0, cx], 
                    [0, fy, cy], 
                    [0, 0, 1]])

 
    # Transform wireframe coordinates into camera coordinates
    points_c_query = R_c_b @ points_b + t_c_cb

    # Project onto camera sensor
    x_query_hom = (K_c @ points_c_query)
    x_query = x_query_hom[:2, :] / x_query_hom[2, :]

    # View the image and the wireframe overlay
    plt.imshow(queryImage)
    plt.title("Undestorted Image")
    ax = plt.gca()
    plot_edges(ax, x_query, edges, title="AR Wireframe")
    plt.show(block=False)

    ## -- Alter wireframe pose ----------------------------------------------------
    # rx_ = np.array([1,0,0]).reshape((3,1))
    # ry_ = np.array([0,1,0]).reshape((3,1))
    # rz_ = np.array([0,0,1]).reshape((3,1))
    # R_c_b_ = np.hstack((rx_, ry_, rz_))

    # t_ = np.array([0, 0, 983]).reshape(3,1)
    # t_c_cb = t_

    phi = np.deg2rad(-30)
    R_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                    [np.sin(phi),  np.cos(phi), 0],
                    [0,            0,           1]])

    
    # Transform wireframe coordinates into camera coordinates   
    points_c_output = t_c_cb + R_c_b @ R_z @ points_b # wireframe points in outputImage

    # Project onto camera sensor and switch to inhom. coords
    x_output_hom = (K_c @ points_c_output)
    x_output = x_output_hom[:2, :] / x_output_hom[2, :]    

    # Plot altered wireframe
    fig, ax = plt.subplots()
    outputImage = np.zeros((queryImage.shape))
    plt.imshow(outputImage)
    ax = plt.gca()
    plot_edges(ax, x_output, edges, title="Altered Wireframe")
    plt.show(block=False)

    polyPts_cv = switchCoords(x_output.T[[0,4,5,1],:].astype(np.int32))

    t_start = time.perf_counter()
    cv.fillPoly(outputImage, [polyPts_cv], (255,0,0))
    t_stop = time.perf_counter()

    print("fillPoly time: ", t_stop-t_start, "seconds")

    plt.imshow(outputImage)
    plt.show(block=False)
    ## -- Compute new homographies queryImage -> outputImage

    H_A = homographyFrom4PointCorrespondences(x_query.T, x_output.T) # wireframe points are hstacked, whereas 4pointcorr takes vstacked points

    dummy = 1

    
    # cpp.pointwiseUndistort(queryImage, H_A, trainImage.shape)







    
