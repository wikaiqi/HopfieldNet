import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import cv2
from Hopfield import HopfieldNN

# set network parameters
N_nodes    = 900
Q_patterns = 40
T_steps    = 12
N_tests    = 1
ini_olp    = 0.18
T          = 0.02
retrieval_pattern = 0

# load mnist test example
Xdata = np.loadtxt('mnist/mnistexample.txt')
i     = 1

this_data     = Xdata[i]
img           = this_data.reshape(28, 28)
resized_image = cv2.resize(img, (30, 30))
resized_image[np.where(resized_image < 0.5)] = -1
resized_image[np.where(resized_image > 0.5)] = 1
test_img = resized_image.reshape(N_nodes)

# Hopfield neural network
a        = HopfieldNN(N_nodes, Q_patterns, test_img, T, T_steps, N_tests, ini_olp, retrieval_pattern)
ev_state = a.run_Ntest()

    
# Plot results
def plot_data(data, size, start, nx, ny, picture_size=2):
    
    size_x = nx*picture_size
    size_y = ny*picture_size
    fig, axes = plt.subplots(ny, nx, figsize=(size_x, size_y))
    for i in range(ny):
        for j in range(nx):
            pid   = i*nx + j
            datai = data[:,pid].reshape(size, size)
            
            
            axes[i][j].set_title('step: {label}'.format(label=pid), color="red")
            axes[i][j].imshow(datai, cmap='gray', interpolation='nearest')
            
            axes[i][j].set_xticks([])
            axes[i][j].set_yticks([])

nx = 4
ny = 3
picture_size=3

plot_data(ev_state, 30, 0, nx, ny, picture_size)
plt.show()
