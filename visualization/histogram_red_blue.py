import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate random 2D data for two sets
data1 = np.random.normal(loc=[2, 3], scale=[1, 1], size=(100, 2))
data2 = np.random.normal(loc=[4, 5], scale=[1, 1], size=(100, 2))

# Create 3D histogram
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

hist1, xedges, yedges = np.histogram2d(data1[:,0], data1[:,1], bins=4)
hist2, _, _ = np.histogram2d(data2[:,0], data2[:,1], bins=(xedges, yedges))

# Plot the histogram
xpos, ypos = np.meshgrid(xedges[:-1] + 0.25, yedges[:-1] + 0.25, indexing="ij")
xpos = xpos.ravel()
ypos = ypos.ravel()
zpos = 0

dx = dy = 0.5 * np.ones_like(zpos)
dz1 = hist1.ravel()
dz2 = hist2.ravel()

ax.bar3d(xpos, ypos, zpos, dx, dy, dz1, color='b', zsort='average')
ax.bar3d(xpos, ypos, zpos, dx, dy, dz2, color='r', zsort='average')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Frequency')
ax.set_title('3D Histogram')

plt.show()

