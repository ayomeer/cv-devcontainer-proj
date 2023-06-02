import matplotlib.pyplot as plt
import numpy as np

RIGHT_BUTTON = 3


# Points on the object expressed in the b-frame
points_b = np.array(
        [
            [0, 0, 0],
            [0.3, 0, 0],
            [0.3, 0, 0.15],
            [0, 0, 0.15],
            [0, 0.1, 0],
            [0.3, 0.1, 0],
            [0.3, 0.1, 0.15],
            [0, 0.1, 0.15],
            [0.4, 0, 0],
            [0, 0.4, 0],
            [0, 0, 0.4]
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



def plot_edges(image_points: np.ndarray, edges: list, edgcolors: list = [], title: str="view"):
#    plt.axis('equal')  # Set aspect ratio to one
    plt.xlim(-1,1)
    plt.ylim(1,-1)
    # plt.gca().invert_yaxis()  # Invert y-axis as is typical in image processing


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


def on_motion(event):
    x = event.xdata
    y = event.ydata

    if True:#(event.button == RIGHT_BUTTON):
        p = np.array([x,-y,0])
        angle = np.linalg.norm(p)
        axis = p / angle
        R = axang2rotmat(axis, angle)
        points_g =  R @ points_b
        image_points = points_g[:2, :]  # Orthographic projecion along the z-axis means to skip the z-axis
        plt.cla()
        plot_edges(image_points, edges, title="Orthographic view")
        plt.show()


def axang2rotmat(rotationAxis :np.ndarray , rotationAngle :float) -> np.ndarray:
    u = rotationAxis
    alpha = rotationAngle

    U = np.matmul(u.reshape(3, 1), u.reshape(1, 3))
    Ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])

    R = U + (np.eye(3)-U)*np.cos(alpha) + Ux*np.sin(alpha)
    Rs = np.eye(3) + np.sin(alpha) * Ux + (1-np.cos(alpha)) * Ux @ Ux
    return R


if __name__ == "__main__":
    # To test the plot_image function, the 3d points are plotted by ignoring their z-component. This is an orthographic projection
    rotationAxis = np.array([1,1,1]) / np.sqrt(3)
    rotationAngle = -np.pi / 10
    rotmat = axang2rotmat(rotationAxis, rotationAngle)
    points_g =  rotmat @ points_b
    image_points = points_g[:2, :]  # Orthographic projecion along the z-axis means to skip the z-axis

    fig, ax = plt.subplots()
    plot_edges(image_points, edges, title="Orthographic view")
    a=fig.canvas.mpl_connect('motion_notify_event', on_motion)
    plt.show()