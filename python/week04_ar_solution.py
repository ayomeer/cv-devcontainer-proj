
import numpy as np
import matplotlib.pyplot as plt
import cv2


# Virtual object points expressed in the b-frame and in mm
edge_length_mm = 35
points_b = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [0, 1, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ])
points_b =  edge_length_mm * points_b.T


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
        ax.plot([x1, x2], [y1, y2], colorString)
    plt.title(title)
    plt.xlabel('x - axis')
    plt.ylabel('y - axis')
    return


def plotPointsOnImage(img: np.ndarray, points: np.ndarray):
    plt.figure()
    plt.imshow(img)
    for point in points:
        plt.plot(point[0], point[1], 'o')
    plt.title("image with points at specified coordinates")
    plt.xlabel("coordinate 1")
    plt.ylabel("coordinate 0")
    plt.show()


def homographyFrom4PointCorrespondences(x_c: np.ndarray, x_b: np.ndarray) -> np.ndarray:
    # Matrix D und Vektor y
    A = np.zeros((8, 8))
    y = np.zeros((8,))
    for n in range(4):
        A[2*n, :] = [x_b[n, 0], x_b[n, 1], 1, 0, 0, 0, -x_b[n, 0]*x_c[n, 0], -x_b[n, 1]*x_c[n, 0]]
        A[2*n+1, :] = [0, 0, 0, x_b[n, 0], x_b[n, 1], 1, -x_b[n, 0]*x_c[n, 1], -x_b[n, 1]*x_c[n, 1]]
        y[2*n] = x_c[n, 0]
        y[2*n+1] = x_c[n, 1]
    # Compute coefficient vector theta = [a b c ...  h]
    theta = np.linalg.solve(A, y)
    # Compute the homography that maps points from the undistorted image to the distorted image
    H_d_u = np.ones((3, 3))
    H_d_u[0, :] = theta[0:3]
    H_d_u[1, :] = theta[3:6]
    H_d_u[2, 0:2] = theta[6:8]
    return H_d_u


def rigidBodyMotion(H_c_b, f):
    K_c = np.array([[f, 0, 0], [0, f, 0], [0, 0, 1]])
    V = np.linalg.inv(K_c) @ H_c_b
    rx = V[:, 0] / np.linalg.norm(V[:, 0])
    rz = np.cross(rx, V[:, 1])
    rz = rz / np.linalg.norm(rz)
    ry = np.cross(rz, rx)
    R_c_b = np.hstack((rx, ry, rz)).reshape((3,3)).T
    t_c_cb = V[:, [2]] / np.linalg.norm(V[:, 0])
    return R_c_b, t_c_cb


def focalLength(H_c_b):
    h11 = H_c_b[0, 0]
    h12 = H_c_b[0, 1]
    h21 = H_c_b[1, 0]
    h22 = H_c_b[1, 1]
    h31 = H_c_b[2, 0]
    h32 = H_c_b[2, 1]
    fsquare = - (h11 * h12 + h21 * h22) / (h31 * h32)
    return np.sqrt(fsquare)


if __name__ == "__main__":

    # Read image
    imgName = '/app/_img/chessboard_perspective.jpg'
    img_d = plt.imread(imgName)

    # Define point correspondences

    # Image coordinates are defined as [horizontal, vertical] with [0, 0 ] at the left top corner
    x_d = np.array([[48, 385], [188, 927], [424, 55], [665, 721]])
    x_c = np.array([p[::-1] for p in x_d])
    # x_c = np.array([[246, 178], [810, 230], [817, 513], [36, 350]])
    # Keyboard coordinates in units of mm are defined as [horizontal vertical] with [0,0] at
    # the left bottom corner
    x_b = np.array([[0, 165], [305, 165], [305, 0], [0, 0], ])

    plotPointsOnImage(img_d, x_c)

    # Estimate the homography from the body planar survace to the image coordinates with the origin in the image center
    M = img_d.shape[0]
    N = img_d.shape[1]
    cx = M/2
    cy = N/2
    x_c_center = np.array((cx, cy))
    cH_c_b = homographyFrom4PointCorrespondences(x_c - x_c_center, x_b)

    # Determine focal length and pose
    f = focalLength(cH_c_b)
    R_c_b, t_c_cb = rigidBodyMotion(cH_c_b, f)

    # Transform wireframe coordinates into camera coordinates and project onto the camera sensor
    points_c = R_c_b @ points_b + t_c_cb
    K_c = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    image_points_homogeneous = (K_c @ points_c)
    image_points = image_points_homogeneous[:2, :] / image_points_homogeneous[2, :]

    # View the image and the wireframe overlay
    plt.imshow(img_d)
    plt.title("Undestorted Image")
    ax = plt.gca()
    plot_edges(ax, image_points, edges, title="Perspective view")
    plt.show()
    plt.show()

