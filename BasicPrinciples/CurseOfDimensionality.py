import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import  mplot3d
from mpl_toolkits.mplot3d import Axes3D

# This code is for demonstrating curse of Dimensionality
# which is depicted by visualizing the gamma function
# for L1- manhattan distance and L2 - euclidean distance
# in high dimensionality space
points = []
points2 = []
for n in range(100,1001):                   # n instances
    print(n)
    for d in range(1,101):                  # d dimensions
        track = []
        track2 = []
        data = np.random.randint(low=1,high=100 ,size=(n, d))  # generate n d-dimensional points with feature value 0-100
        for x in range(n):
            for y in range(x,n):
                if x != y:
                    try:
                        dist = np.linalg.norm(data[x] - data[y])  #euclidean distance
                        dist2 = np.linalg.norm(data[x]-data[y],ord=1)  # first-order distance
                        track.append(dist)
                        track2.append(dist2)
                    except:
                        print("ignoring this pair due to divide by 0 error")
        min_val = min(track)
        max_val = max(track)
        min_val2 = min(track2)
        max_val2 = max(track2)
        gamma_func = math.log((max_val - min_val)/min_val)
        gamma_func2 = math.log((max_val2 - min_val2)/min_val2)
        points.append([n,d,gamma_func])
        points2.append([n,d,gamma_func2])

points.pop(0)
points2.pop(0)
# Generate the Plots for euclidean distance
data = np.array(points)
instances = data[:,0]
dimensions = data[:,1]
gamma = data[:,2]
plot = plt.axes(projection='3d')
plot.set_title("euclidean distance")
plot.scatter3D(instances,dimensions,gamma)

# Generate the plot for L1 distance
data2 = np.array(points2)
instances2 = data2[:,0]
dimensions2 = data2[:,1]
gamma2 = data2[:,2]

plot2 = plt.axes(projection='3d')
plot2.set_title("L1 distance")
plot2.scatter3D(instances2,dimensions2,gamma2)

# %matplotlib notebook for jupyter plot to see in ipython
