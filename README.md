# Little-Hopfield neural network (Python Version)

A [Little-Hopfield network](https://en.wikipedia.org/wiki/Hopfield_network) is a form of recurrent artificial neural network popularized by American physicist John Hopfield in 1982, but described earlier by Little in 1974. A Hopfield network can be used as an associative memory. 

## Isomorphism between the Hopfield and Ising models

Physicists have analyzed the Hopfield model in such exquisite detail because it is isomorphic to the Ising model of magnetism (at temperature zero). The total magnetic field $h_i$ sensed by the atom i in an ensemble of particles is the sum of the fields induced by each atom and the external field.

## Hebbian learning rule

A synapse between two neurons is strengthened when the neurons on either side of the synapse have highly correlated outputs. In essence, when an input neuron fires, if it frequently leads to the firing of the output neuron, the synapse is strengthened. 

## Run Hopfield neural network
```
from Hopfield import HopfieldNN
import numpy as np
import matplotlib.pyplot as plt
import cv2

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

HopNN  = HopfieldNN(N_nodes, Q_patterns, test_img, T, T_steps, N_tests, ini_olp, retrieval_pattern)
states = HopNN.run_Ntest()
```

## Example:MNIST handwritting
![MNIST handwrittinh](https://github.com/wikaiqi/HopfieldNet/blob/master/hardwrittingexample.png)
