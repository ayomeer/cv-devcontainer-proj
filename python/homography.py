
import numpy as np
import matplotlib.pyplot as plt
import lib.cppmodule as cpp # path from vs code root --> cd to /app/python
import cv2 as cv
import time

# --- Constants ----------------------------------------------------------- #
FIGURE_SIZE = (12, 9)

# AR Wireframe
points_b = np.array(
        [
            # box wireframe
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            # axes representation
            [0.3, 0, 0],
            [0, 0.3, 0],
            [0, 0, 0.3]
        ]).T

# Individual scaling of the cube's axis
res = 2 # px/mm
points_b = (res*np.diag((165, 240, 60)) @ points_b)

# Lines on the cuboid as sequence of tuples containing the indices of the starting point and the endpoint
edges = [
         [4, 5], [5, 6], [6, 7], [7, 4],  # Lines of back plane
         [0, 4], [1, 5], [2, 6], [3, 7], # Lines connecting front with back-plane
         [0, 1], [1, 2], [2, 3], [3, 0],  # Lines of front plane
         [0, 8], [0, 9], [0, 10],  # Lines indicating the coordinate frame
        ]

# Line colors can be given one of the strings 'r' for red, 'g' for green, 'b' for blue
edgecolors = [
               '0.8','0.8','0.8','0.8',
               '0.5','0.5','0.5','0.5',
               'k','k','k','k',
                'r','g','b'
             ]


# --- Helper Functions ---------------------------------------------------- #

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

def coordTransform(xu, H_d_u):
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

def switchCoords(points):
    return np.array([p[::-1] for p in points])

# --- Main -----------------------------------------------------------------#
def main():
    ## -- Read images --
    trainImage = plt.imread('/app/_img/xonar_template.jpg')
    queryImage = plt.imread('/app/_img/xonar_perspective.jpg')

    ## Select corners to check homography fit later
    # plt.imshow(trainImage) # don't adjust window size!
    # corners = np.round(plt.ginput(n=4, mouse_add=None, timeout=0)).astype(int)
    # plt.close()

    # hardcoded top corners xonar box top (in cv-coordinates!)
    corners = np.array([[1207,  400],
                        [2940,  401],
                        [2934, 1640],
                        [1204, 1637]])

    # TODO: Fit rectangle to selection!



    ## -- Feature Detection: SIFT --
    sift = cv.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.001,
        edgeThreshold=20,
        sigma=1.5,
        nOctaveLayers=4
    )
    trainPoints, trainDescriptors = sift.detectAndCompute(trainImage, None)
    queryPoints, queryDescriptors = sift.detectAndCompute(queryImage, None)

    # Feature Matching: Brute Force Matcher 
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
    
    # Fit the homography model: RANSAC --
    xd = np.float32([queryPoints[m.queryIdx].pt for m in matches]).reshape(-1,2)
    xu = np.float32([trainPoints[m.trainIdx].pt for m in matches]).reshape(-1,2)
    
    H_c_b, mask = cv.findHomography(xu, xd, method=cv.RANSAC, ransacReprojThreshold=4.0)

    # plotMatches(trainImage, trainPoints, queryImage, queryPoints, mask, matches) 

    ## Transform x_u-corners back to queryImage
    """ cv.perspectiveTransform
    rect_u = np.float32([ corners ]).reshape(-1,1,2)
    rect_d = cv.perspectiveTransform(rect_u, H_c_b)
    img2 = cv.polylines(queryImage, [np.int32(rect_d)], True, 255 , 3, cv.LINE_AA)
    """
    M_t, N_t = trainImage.shape[0], trainImage.shape[1]
    rect_u_ = np.array([[0, 0], [0, N_t-1], [M_t-1, N_t-1], [M_t-1, 0]]) # cv coords
    rect_u = switchCoords(rect_u_)
    rect_d = np.empty((4,2)).astype(int)

    for i in range(rect_d.shape[0]):
        rect_d[i] = coordTransform(rect_u[i], H_c_b).astype(int)
    
    plotPointsOnImageCV(queryImage, rect_d)
    plt.show()
    
    ## Second Homography using 4-point correspondence to square points
    M_u, N_u = res*165, res*240 
    x_d_ = np.array([[760, 977], [215, 2117], [583, 2845], [1352, 1674]])
    # x_d = np.array([p[::-1] for p in rect_d]) # change to x-down-coordinates
    x_d = x_d_
    x_u = np.array([[0, 0], [0, N_u-1], [M_u-1, N_u-1], [M_u-1, 0]])
    
    M_d, N_d = queryImage.shape[0], queryImage.shape[1] 
    x_d_center = np.array([M_d/2, N_d/2])

    cH_d_u = homographyFrom4PointCorrespondences(x_d-x_d_center, x_u)
    
    ## -- Pose and Focal Length Estimation --
    R_c_b, t_c_cb, fx, fy = recoverRigidBodyMotionAndFocalLengths(cH_d_u)

    
    M_d, N_d = queryImage.shape[0], queryImage.shape[1]
    cx = M_d / 2
    cy = N_d / 2
    K_c = np.array([[fx, 0, cx], 
                    [0, fy, cy], 
                    [0, 0, 1]])

    # Transform wireframe coordinates into camera coordinates
    
    points_c = R_c_b @ points_b + t_c_cb

    # Project onto camera sensor
    image_points_homogeneous = (K_c @ points_c)
    image_points = image_points_homogeneous[:2, :] / image_points_homogeneous[2, :]

    # View the image and the wireframe overlay
    plt.imshow(queryImage)
    plt.title("Undestorted Image")
    ax = plt.gca()
    plot_edges(ax, image_points, edges, title="Perspective view")
    plt.show()



    
    
    
    # cpp.pointwiseUndistort(queryImage, cH_c_b, trainImage.shape)

if __name__ == "__main__":
    main()




    
