import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

URL = 'https://storage.googleapis.com/picrystal-bucket/hiring/445b7773-3431-4c34-a762-ce8986670aa3_main_hiring_updated.csv'

# Generate random 2D data for two sets
data1 = np.random.normal(loc=[2, 3], scale=[1, 1], size=(100, 2))
data2 = np.random.normal(loc=[4, 5], scale=[1, 1], size=(100, 2))

print(data1.shape)
print(data1)

# Create 3D histogram
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

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
ax.set_xlabel('Race')
ax.set_ylabel('Gender')
ax.set_zlabel('Number of points')
ax.set_title('Hiring model')

plt.show()

