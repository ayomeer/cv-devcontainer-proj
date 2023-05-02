
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Points on the object expressed in the b-frame
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
points_b = (res*np.diag((94, 131, -60)) @ points_b)

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

def homographyFrom4PointCorrespondences(x_d: np.ndarray, x_u: np.ndarray) -> np.ndarray:
    # Matrix D und Vektor y
    A = np.zeros((8, 8))
    y = np.zeros((8,))
    for n in range(4):
        A[2*n, :] = [x_u[n, 0], x_u[n, 1], 1, 0, 0, 0, -x_u[n, 0]*x_d[n, 0], -x_u[n, 1]*x_d[n, 0]]
        A[2*n+1, :] = [0, 0, 0, x_u[n, 0], x_u[n, 1], 1, -x_u[n, 0]*x_d[n, 1], -x_u[n, 1]*x_d[n, 1]]
        y[2*n] = x_d[n, 0]
        y[2*n+1] = x_d[n, 1]

    # Compute coeffizientenvector theta = [a b c ...  h]
    theta = np.linalg.solve(A, y)

    # Compute the homography that maps points from the undistorted image to the distorted image
    H_d_u = np.ones((3, 3))
    H_d_u[0, :] = theta[0:3]
    H_d_u[1, :] = theta[3:6]
    H_d_u[2, 0:2] = theta[6:8]
    return H_d_u

def PointwiseUndistort(img_d, xdFromXu, rect_shape):
    img_u = np.zeros(rect_shape, np.uint8)
    M, N, _ = rect_shape
    Ms = img_d.shape[0]
    Ns = img_d.shape[1]
    for m in range(M):
        for n in range(N):
            x_u = [m, n]  # Point in undistorted image
            x_d = xdFromXu(x_u)  # Point in distorted image
            x_d = np.round(x_d).astype(np.int32)  # round takes the "nearest neighbour"
            if not (x_d[0] < 0 or Ms <= x_d[0] or x_d[1] < 0 or Ns <= x_d[1]):  # Skip in case the point lies outside the image border
                img_u[x_u[0], x_u[1], :] = img_d[x_d[0], x_d[1], :]  # read point from distorted image
    return img_u

def inhomogeneousHomographicTransform(H_d_u, x_u):
    xhom_u = np.ones((3,))
    xhom_u[0:2] = x_u
    xhom_d = H_d_u @ (xhom_u)
    x_d = xhom_d[0:2]/xhom_d[2]
    return x_d

"""
def recoverRigidBodyMotion(H_c_b, K_c): # Factorize
    V = np.linalg.inv(K_c) @ H_c_b
    rx = V[:, 0] / np.linalg.norm(V[:, 0])
    ry = V[:, 1] / np.linalg.norm(V[:, 0])
    t_c_cb = V[:, [2]] / np.linalg.norm(V[:, 0])
    rz = np.cross(rx, ry)
    R_c_b = np.hstack((rx, ry, rz))
    return R_c_b, t_c_cb
"""
def recoverRigidBodyMotionAndFocalLengths(H_c_b):
    """
    # Input
    #   H_c_b --> homography from a planar object in b-coordinates to an image c-coordinates 
    #             with the ORIGIN at IMAGE CENTER!
    # Output
    #   Rotation matrix R_c_b
    #   Translation vector t_c_cb
    #   focal lenghts fx and fy
    """
    
    # Setup System of Eqs for determining focal lengths of the camera
    H = H_c_b
    A = np.array([[H[0,0]**2,     H[1,0]**2,     H[2,0]**2],
                   [H[0,1]**2,     H[1,1]**2,     H[2,1]**2],
                   [H[0,0]*H[0,1], H[1,0]*H[1,1], H[2,0]*H[2,1]]])   
    y = np.array([[1], [1], [0]])

    # Solve sytem for focal lengths
    diags = np.linalg.inv(A) @ y
    LambdaInv = np.diag(np.sqrt(diags.ravel()))
    B = LambdaInv @ H_c_b
    rx = B[:, [0]]
    ry = B[:, [1]]
    rz = np.cross(rx.ravel(), ry.ravel()).reshape((3,1))
    R_c_b = np.hstack((rx, ry, rz))
    t_c_cb = B[:,[2]]
    fx = LambdaInv[2, 2] / LambdaInv[0, 0]
    fy = LambdaInv[2, 2] / LambdaInv[1, 1]
    return R_c_b, t_c_cb, fx, fy


if __name__ == "__main__":

    FIGURE_SIZE = (12, 9)
    # Bilder einlesen
    imgName = '/app/_img/perspective_ar.jpg'
    img_d = plt.imread(imgName)
    
    # 4-point correspondences white box 
    m, n = res*94, res*131 # px/mm * mm
    x_d = np.array([[164, 958], [491, 2206], [1295, 1570], [723, 240]])
    x_u = np.array([[0, 0], [0, n-1], [m-1, n-1], [m-1, 0]])

    # plotPointsOnImage(img_d, x_d)

    # Estimate the homography from the body planar surface to the image coordinates
    # with the origin in the center
    M = img_d.shape[0]
    N = img_d.shape[1]
    x_d_center = np.array((M/2, N/2))
    x_u_center = np.mean(x_u, axis=0)
    cH_c_b = homographyFrom4PointCorrespondences((x_d-x_d_center), x_u)

    # Determine the pose and the focal lengths {b} -> {c}
    R_c_b, t_c_cb, fx, fy = recoverRigidBodyMotionAndFocalLengths(-cH_c_b)
    cx = x_d_center[0]
    cy = x_d_center[1]
    K_c = np.array([[fx, 0, cx], 
                    [0, fy, cy], 
                    [0, 0, 1]])

    # Transform wireframe coordinates into camera coordinates
    points_c = R_c_b @ points_b + t_c_cb

    # Project onto camera sensor
    image_points_homogeneous = (K_c @ points_c)
    image_points = image_points_homogeneous[:2, :] / image_points_homogeneous[2, :]

    # View the image and the wireframe overlay
    plt.imshow(img_d)
    plt.title("Undestorted Image")
    ax = plt.gca()
    plot_edges(ax, image_points, edges, title="Perspective view")
    plt.show()
    plt.show()

