
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
            [0.1, 0, 0],
            [0.1, 0, 0.1],
            [0, 0, 0.1],
            [0, 0.1, 0],
            [0.1, 0.1, 0],
            [0.1, 0.1, 0.1],
            [0, 0.1, 0.1],
            # axes representation
            [0.1, 0, 0],
            [0, 0.1, 0],
            [0, 0, 0.1]
        ]).T

# Individual scaling of the cube's axis
points_b = (5*np.diag((1.5, 1.5, 1.5)) @ points_b)

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

# plot box wireframe edges
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

def plotMatches(queryImage, queryPoints, mask, trainImage, trainPoints, matches):
    matchesMask = mask.ravel().tolist()

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
        matchesMask = matchesMask,  # draw only inliers
        flags = cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,  # cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    plt.imshow(imgMatches)
    plt.title('Matches refined with homography-RANSAC')
    plt.show()

def recoverRigidBodyMotionAndFocalLengths(H_c_b):
    # Input
    #   H_c_b is a homography from a planar object in b-coordinates to an image c-coordinates with the ORIGIN on the IMAGE CENTER!
    # Output
    #   Rotation matrix R_c_b
    #   Translation vector t_c_cb
    #   focal lenghts fx and fy
    H = H_c_b
    A = np.array([  [H[0,0]**2,     H[1,0]**2,     H[2,0]**2    ],
                    [H[0,1]**2,     H[1,1]**2,     H[2,1]**2    ],
                    [H[0,0]*H[0,1], H[1,0]*H[1,1], H[2,0]*H[2,1]]])
    
    y = np.array([[1], [1], [0]])

    # Solve sytem for focal lengths
    x = np.linalg.inv(A) @ y                # x = [a**2, b**2, c**2].T
    LambdaInv = np.diag(np.sqrt(x.ravel())) # LabdaInv = lambda/fx etc
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


# --- Main -----------------------------------------------------------------#
def main():
    #%% Reading images
    queryImage = plt.imread('/app/_img/perspective.jpg')
    trainImage = plt.imread('/app/_img/cropTop.jpg')

    #%% Feature Detection: SIFT
    sift = cv.SIFT_create(
        nfeatures=3000,
        contrastThreshold=0.001,
        edgeThreshold=20,
        sigma=1.5,
        nOctaveLayers=4
    )
    # Run SIFT detection on both images
    trainPoints = sift.detect(trainImage, None)
    queryPoints = sift.detect(queryImage, None)

    # Compute descriptors
    _, trainDescriptors = sift.compute(trainImage, trainPoints)
    _, queryDescriptors = sift.compute(queryImage, queryPoints)

    #%% Feature Matching: Brute Force Matcher (cv.bfmatcher)
    matcher = cv.BFMatcher_create(cv.NORM_L1, crossCheck=True)
    matches = matcher.match(queryDescriptors, trainDescriptors)
    print('{} matches found'.format(len(matches)))

    # Show matches, starting with the most reliable
    sortedMatches = sorted(matches, key = lambda x:x.distance)
    pltImage = cv.drawMatches(queryImage, queryPoints, trainImage, trainPoints, sortedMatches[:400], queryImage, flags=2)
    plt.imshow(pltImage)
    plt.title('Brute force matching result')
    plt.show()

    #%% Fit the homography model: RANSAC 
    dstPtsCoords = np.float32([queryPoints[m.queryIdx].pt for m in matches]).reshape(-1,2)
    srcPtsCoords = np.float32([trainPoints[m.trainIdx].pt for m in matches]).reshape(-1,2)
    
    # Apply square pixel assumption to remove 2 degrees of freedom in pose estimation later on
    M = queryImage.shape[0]
    N = queryImage.shape[1]
    srcPts_center = np.array([M/2, N/2])

    H, mask = cv.findHomography(
        srcPoints=srcPtsCoords,#-srcPts_center, # find homography for centered camera coords
        dstPoints=dstPtsCoords, 
        method=cv.RANSAC, 
        ransacReprojThreshold=4.0)

    # show correspondences
    # plotMatches(queryImage, queryPoints, mask, trainImage, trainPoints, matches) 



    #%% Pose and Focal Length Estimation
    """
    
    R_c_b, t_c_cb, fx, fy = recoverRigidBodyMotionAndFocalLengths(-H)
    cx = srcPts_center[0]
    cy = srcPts_center[1]
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
    plt.show()
    """

    #%% Compute box wireframe for homography input boundaries




    #%% Compute Homographies: custom cpp-cuda module
    
    img_u_shape = (600, 800, 3)

    t1 = time.perf_counter()
    img_u = np.array(cpp.pointwiseUndistort(queryImage, H, img_u_shape), copy=False)
    t2 = time.perf_counter()
    tUndistort = t2-t1
    print("tUndistort ", tUndistort)
    
    # Show result
    plt.figure(figsize=FIGURE_SIZE)
    plt.imshow(img_u)
    plt.title("Undistorted Image")
    plt.show()
    

if __name__ == "__main__":
    main()




    
