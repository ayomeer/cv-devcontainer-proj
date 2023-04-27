import matplotlib.pyplot as plt
import numpy as np


ROTATE_OBJECT_ARROUND_ITS_Z_AXIS_BY_DEGREE = +0


# Change the corner point coordinate system from b to c
def change_coordinate_system(pose_c_b, points_b):
    points_b_homogeneous = np.vstack((points_b, np.ones((1, points_b.shape[1]))))
    points_c_homogeneous = (pose_c_b @ points_b_homogeneous)
    points_c = points_c_homogeneous[:3, :]
    return points_c


def plot_edges(
    ax, image_points: np.ndarray, 
    edges: list, 
    edgcolors: list = [], 
    title: str="view"
):
    plt.ylim(3000, 0) # px resolution of camera
    plt.xlim(0, 4000)
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


# Points on the object expressed in the b-frame
points_b = np.array(
        [
            # box wireframe
            [0, 0, 0],
            [0.3, 0, 0],
            [0.3, 0, 0.15],
            [0, 0, 0.15],
            [0, 0.1, 0],
            [0.3, 0.1, 0],
            [0.3, 0.1, 0.15],
            [0, 0.1, 0.15],
            # coordinate frame vis
            [0.05, 0, 0],
            [0, 0.05, 0],
            [0, 0, 0.05]
        ]).transpose()


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


# Columns of rotmat_b_c are the axis of the b-frame expressed in terms of the c-frame
rotmat_b_c = np.array([[0, 0, -1],
                       [1, 0, 0],
                       [0, 1, 0]])


# Rotation Matrix arround z-Axis
phi = np.deg2rad(ROTATE_OBJECT_ARROUND_ITS_Z_AXIS_BY_DEGREE)
R_z = np.array([[np.cos(phi), -np.sin(phi), 0],
                [np.sin(phi), np.cos(phi), 0],
                [0, 0, 1]])

# Change of Basis Matrix E_b_c
t_b_c = np.array([0, 0.3, 1]).reshape((3, 1)) # (dx, dy, dz) in c coords
E_b_c = np.block([[rotmat_b_c, t_b_c],
                  [np.zeros((1,3)), 1]])

# Camera Matrix, homogeneous
fx = fy = 3330
mx = 1500; my = 2000
K = np.array([[fx, 0,  mx,  0],
              [0,  fy, my,  0],
              [0,  0,  1,   0],
              [0,  0,  0,   1]])


if __name__ == "__main__":
      
    # augment image points by ones to go into projective space
    points_b_hom = np.vstack((points_b, np.ones((1, points_b.shape[1]))))
    
    # Change of Basis B -> C (Body coords --> Camera Coords)
    points_c_hom = E_b_c@points_b_hom 

    # Camera Projection
    points_image_hom = K@points_c_hom
    points_image = points_image_hom[0:2, :] / points_image_hom[2, :]

    # plot camera perspective
    fig, ax = plt.subplots()
    plot_edges(ax, points_image, edges, title="Perspective view")
    plt.show()
    
    
    
    
