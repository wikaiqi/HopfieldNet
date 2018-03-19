# Little-Hopfield neural network (Python Version)

A [Little-Hopfield network](https://en.wikipedia.org/wiki/Hopfield_network) is a form of recurrent artificial neural network popularized by American physicist John Hopfield in 1982, but described earlier by Little in 1974. A Hopfield network can be used as an associative memory. 

## Isomorphism between the Hopfield and Ising models

Physicists have analyzed the Hopfield model in such exquisite detail because it is isomorphic to the Ising model of magnetism (at temperature zero). The total magnetic field $h_i$ sensed by the atom i in an ensemble of particles is the sum of the fields induced by each atom and the external field.

## Hebbian learning rule

A synapse between two neurons is strengthened when the neurons on either side of the synapse have highly correlated outputs. In essence, when an input neuron fires, if it frequently leads to the firing of the output neuron, the synapse is strengthened. 

## Run Hopfield neural network
```
HopNN  = HopfieldNN(N_nodes, Q_patterns, test_img, T, T_steps, N_tests, ini_olp, retrieval_pattern)
states = HopNN.run_Ntest()
```
* N_nodes: number of neurons
* Q_patterns: number of patterns store in memory
* test_img: test retrieval image
* T: temperature
* T_steps: number of time steps for each test
* N_tests: number of tests
* ini_olp: intial overlap
* retrieval_pattern: the tetrieval pattern

## Example:MNIST handwritting
![MNIST handwrittinh](https://github.com/wikaiqi/HopfieldNet/blob/master/hardwrittingexample.png)
