# Gaussian Process Regression
This is a program that performs Gaussian process regression (also known as ['kriging'](https://en.wikipedia.org/wiki/Kriging)) on data for interpolation and prediction.

## Background
This type of regression is premised on the assumption that the values at each point are [normally distributed](https://en.wikipedia.org/wiki/Normal_distribution) and therefore that a collection of points form a multivariate normal (or Gaussian) distribution. If we assume that the values of any two points are correlated in some proportion to their distance apart, e.g. that the closer together two points are, the more similar their values are likely to be, we can define a function known as a kernel. A kernel, or covariance function, defines the covariance of the multivariate distributions from pairs/sets of points as a function of their distance apart. Essentially, the kernel defines how we expect the underlying function of a dataset to act.

If we take a collection of many points over an interval and apply our kernel's covariance to their multivariate Gaussian, the result is a distribution of 'plausible' values sets. That is, the probability of a particular set of values for our point collection is defined by how well it obeys our assumption about covariance. In the absence of data, this basically generates random functions of the class defined by our kernel. When we have training data, however, the distribution can be updated using Bayesian inference (as detailed in Chapter 2 of [this resource](http://www.gaussianprocess.org/gpml/chapters/RW.pdf)) to prioritise those functions which agree with the training data. The mean of this distribution represents the best value estimate at each point in our collection, given the training data and assuming the covariance defined by the kernel.

## Program
<img src="https://github.com/funatparties/gaussian-process/blob/master/images/example_plot.png" width="600">

##### An example plot showing the distribution mean and confidence interval using a [squared exponential](https://www.cs.toronto.edu/~duvenaud/cookbook/) kernel.

This program supplies an interface for generating, modifying, and visualising Gaussian process models. The `Manager` class in `manager.py` and the `GaussianProcess` class in `gaussianprocess.py` are the primary user interfaces. The user can use the methods of `GaussianProcess` to set kernel parameters and training data as desired, whilst the methods of `Manager` connect the `GaussianProcess` object to a `Plotter` object to allow conveniently generating graphs of the model (as seen above).
