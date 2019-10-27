from sklearn.datasets import make_circles
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import cholesky

#rand 10; noise 0.25 0.7

data_l_ratio = 0.25
n_sample = 400
rand = 10
n_train = int(n_sample*2/3)
plt.title('Data generation')
fig = plt.figure(1)
x0, y0 = make_circles(n_samples=n_sample, factor=0.1, noise=0.25, random_state=rand)
x0[:, 1] = x0[:, 1]-1.5

class0 = x0[np.argwhere(y0 == 0), :].reshape((-1, 2))
class1 = x0[np.argwhere(y0 == 1), :].reshape((-1, 2))
class2 = x0[np.argwhere(y0 == 2), :].reshape((-1, 2))

x1, y1 = make_circles(n_samples=n_sample, factor=0.1, noise=0.25, random_state=rand)
x1[:, 1] = x1[:, 1]+1.5
y1 = np.where(y1 == 1, 2, y1)

class0 = np.vstack((class0, x1[np.argwhere(y1 == 0), :].reshape((-1, 2))))
class1 = np.vstack((class1, x1[np.argwhere(y1 == 1), :].reshape((-1, 2))))
class2 = np.vstack((class2, x1[np.argwhere(y1 == 2), :].reshape((-1, 2))))

center = [[-2, 0], [2, 0]]
std = 0.65
x2, y2 = make_blobs(n_samples=n_sample, centers=center, n_features=2, cluster_std=std, random_state=rand)
y2 = np.where(y2 == 0, 2, y2)

class0 = np.vstack((class0, x2[np.argwhere(y2 == 0), :].reshape((-1, 2))))
class1 = np.vstack((class1, x2[np.argwhere(y2 == 1), :].reshape((-1, 2))))
class2 = np.vstack((class2, x2[np.argwhere(y2 == 2), :].reshape((-1, 2))))

data_train = np.vstack((x0[:n_train, :], x1[:n_train, :], x2[:n_train, :])).astype(np.float32)
label_train = np.hstack((y0[:n_train], y1[:n_train], y2[:n_train]))

data_val = np.vstack((x0[n_train:, :], x1[n_train:, :], x2[n_train:, :])).astype(np.float32)
label_val = np.hstack((y0[n_train:], y1[n_train:], y2[n_train:]))

data_conf = np.vstack((x0[:int(n_train * data_l_ratio), :], x1[:int(n_train * data_l_ratio), :], x2[:int(n_train * data_l_ratio), :])).astype(np.float32)
label_conf = np.hstack((y0[:int(n_train * data_l_ratio)], y1[:int(n_train * data_l_ratio)], y2[:int(n_train * data_l_ratio)]))

data_non_conf = np.vstack((x0[int(n_train * data_l_ratio):n_train, :], x1[int(n_train * data_l_ratio):n_train, :], x2[int(n_train * data_l_ratio):n_train, :])).astype(np.float32)
label_non_conf = np.hstack((y0[int(n_train * data_l_ratio):n_train], y1[int(n_train * data_l_ratio):n_train], y2[int(n_train * data_l_ratio):n_train]))
#
# mu = np.array([0, 0])
# Sigma = np.array([[4, 0], [0, 4]])
# R = cholesky(Sigma)
# s = np.float32(np.dot(np.random.randn(n_sample, 2), R) + mu)
#
# data_conf = s
# label_conf = np.zeros(n_sample, dtype=np.int64)
#
plt.scatter(class1[:, 0], class1[:, 1], marker='x', s=10, c='red')
plt.scatter(class2[:, 0], class2[:, 1], marker='^', s=10, c='green')
plt.scatter(class0[:, 0], class0[:, 1], marker='.', s=10, c='blue')

# plt.scatter(data_non_conf[:, 0], data_non_conf[:, 1], marker='.', c='blue')
# plt.scatter(data_conf[:, 0], data_conf[:, 1], marker='.', c='red')
# plt.scatter(class2[:, 0], class2[:, 1], marker='.', c='green')

np.save('data_train.npy', data_train)
np.save('data_val.npy', data_val)
np.save('data_conf.npy', data_conf)
np.save('data_non_conf.npy', data_non_conf)
np.save('label_train.npy', label_train)
np.save('label_val.npy', label_val)
np.save('label_conf.npy', label_conf)
np.save('label_non_conf.npy', label_non_conf)

np.save('class0.npy', class0)
np.save('class1.npy', class1)
np.save('class2.npy', class2)

plt.show()


