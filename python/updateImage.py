import matplotlib.pyplot as plt
import numpy as np


plt.ion()

img = np.zeros((10,10,3))
red = np.array([1, 0, 0])

fig, ax = plt.subplots()
im = ax.imshow(img)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        img[i,j] = red
        im.set_data(img)
        fig.canvas.draw()
        fig.canvas.flush_events()

# im.set_data(img)
# plt.show()

plt.pause(2)