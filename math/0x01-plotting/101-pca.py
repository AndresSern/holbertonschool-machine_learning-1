#!/usr/bin/env python3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()
X = iris.data 
y = iris.target

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')
X_reduced = PCA(n_components=3).fit_transform(iris.data)

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
X_reduced  = np.matmul(norm_data, Vh[:3].T)

ax.scatter(X_reduced[:, 0], X_reduced[:, 1], X_reduced[:,2], c=y,
           cmap='plasma', edgecolor='face', s=100)
ax.set_xlim((-3,4))
ax.set_ylim((-1.5,1))
ax.set_zlim((-0.8,0.6))
ax.set_title("PCA of Iris Dataset",fontsize=18)
ax.set_xlabel("U1")
ax.set_ylabel("U2")
ax.set_zlabel("U3")

plt.show()