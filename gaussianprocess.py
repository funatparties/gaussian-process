#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: Josh"""

from kernel import Kernel, SquaredExponential
import numpy as np
from scipy.spatial.distance import cdist

class GaussianProcess():
    """Class for representing gaussian processes. Stores covariance kernel, 
    training data, and hyperparameters. Has methods for sampling prior and 
    posterior distributions, as well as generating mean and covariance.
        
    The process models data as coming from an underlying function with covariance
    over space defined by a kernel function plus added Gaussian white noise.
    """
    def __init__(self, kernel=SquaredExponential(), training_X=None, training_y=None,
                 noise_var=1, rng_seed = None):
        """
        Parameters
        ----------
        kernel : Kernel
            The covariance kernel (with set hyperparameters). The default is
            the squared exponential kernel.
        noise_var : float, optional
            The variance of the added Gaussian white noise, must be non-negative.
            The default is 1.
        training_X : array-like, optional
            Training data used for generating posterior distribution.    
            An n x d array of n training inputs in d dimensions. The default is None.
            Missing training data means that only a prior distribution can be accessed.
        training_y : array-like, optional
            An n x 1 array of n scalar training outputs. The default is None.
        rng_seed : int, optional
            Seed for the rng used when sampling distributions. The default of
            None means a random seed will be pulled from the OS.
            (as per numpy.random.default_rng documentation)

        Raises
        ------
        TypeError
            Raises TypeError if incorrect kernel object or incomplete training data
            supplied.
        ValueError
            Raises ValueError if negative value for noise_var or mismatched 
            arrays for training data supplied.

        Returns
        -------
        None.

        """
        
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be Kernel object.")
        self._kernel = kernel
        
        if not noise_var >= 0:
            raise ValueError("noise_var must be non-negative.")
        self._noise_var = noise_var
        
        #initialise interal rng
        self.rng = np.random.default_rng(rng_seed)
        
        #determine if process lacks training or if sufficient training supplied
        if (training_X is None) and (training_y is None):
            self._has_training = False
        elif (training_X is None) or (training_y is None):
            raise TypeError("Must supply both X and y training data.")
        elif training_X.shape[0] != training_y.shape[0]:
            raise ValueError("Must have equal number of rows in training data.")
        elif len(training_X) == 0 and len(training_y) == 0:
            raise ValueError("Training data must not be empty")
            self._has_training = False
        else:
            self._has_training = True
            self._training_X = training_X
            self._training_y = training_y
            self.num_points, self.dims = self._training_X.shape[:2]
            self._training_cov() #precalculate training covariance
        return
    
    @property
    def has_training(self):
        """
        Returns
        -------
        bool
            Flag storing whether the object has training data..

        """
        
        return self._has_training
    
    @property
    def training_X(self):
        """
        Returns
        -------
        array
            The n x d array of n points of training data inputs in d dimensions.

        """
        
        return self._training_X
    
    @property
    def training_y(self):
        """
        Returns
        -------
        array
            The n x 1 array of training data outputs.

        """
        
        return self._training_y
    
    @property
    def noise_var(self):
        """
        Returns
        -------
        float
            The variance of the Gaussian white noise in the model.

        """
        
        return self._noise_var
    
    @noise_var.setter
    def noise_var(self, new_val):
        """Sets the variance of the Gaussian white noise in the model.

        Parameters
        ----------
        new_val : float
            The new variance of the noise.  Must be non-negative.

        Raises
        ------
        ValueError
            Raises ValueError if new_val is negative.

        Returns
        -------
        None.

        """
        
        if new_val >= 0:
            self._noise_var = new_val
            if self._has_training:
                self._training_cov() #update training covariance
        else:
            raise ValueError("Noise_var must be non-negative")
        return
    
    @property
    def kernel(self):
        """
        Returns
        -------
        Kernel
            The kernel object used for the covariance function in the model.

        """
        
        return self._kernel
    
    @kernel.setter
    def kernel(self, new_kernel):
        """Replaces the kernel object in the model.

        Parameters
        ----------
        new_kernel : Kernel
            The new kernel object to use in the model.

        Raises
        ------
        TypeError
            Raises TypeError if new_kernel is not a Kernel object.

        Returns
        -------
        None.

        """
        
        if not isinstance(new_kernel, Kernel):
            raise TypeError("kernel must be Kernel object.")
        self._kernel = new_kernel
        if self._has_training:
            self._training_cov() #update training covariance
        return
    
    @property
    def kernel_config(self):
        """
        Returns
        -------
        dict
            The attribute dictionary of the kernel object used in the model.

        """
        
        return self._kernel.config
    
    @kernel_config.setter
    def kernel_config(self, new_config):
        """Sets the configuration of the kernel object used in the model.

        Parameters
        ----------
        new_config : dict
            The new settings to use for the kernel. Need not specify all
            attributes.
            
        Raises
        ------
        ValueError
            Raises ValueError if a supplied config value does not meet the 
            constraints  of the kernel object.

        Returns
        -------
        None.

        """
        
        self._kernel.config = new_config
        if self._has_training:
            self._training_cov() #update training covariance
        return
    
    def update_training(self, training_X, training_y):
        """Updates training data to supplied values. Supply None or empty arrays
        to remove training training data.

        Parameters
        ----------
        training_X : array-like
            Training data used for generating posterior distribution.    
            An n x d array of n training inputs in d dimensions.
            Missing training data means that only a prior distribution can be accessed.
        training_y : array-like
            An n x 1 array of n scalar training outputs.

        Raises
        ------
        TypeError
            Raises TypeError if incomplete training data supplied.
        ValueError
            Raises ValueError if mismatched training data supplied.

        Returns
        -------
        None.

        """
        
        if (training_X is None) and (training_y is None):
            self._has_training = False
            self._training_X = None
            self._training_y = None
        elif (training_X is None) or (training_y is None):
            raise TypeError("Must supply both X and y training data")
        elif training_X.shape[0] != training_y.shape[0]:
            raise ValueError("Must have equal number of rows in training data.")
        elif len(training_X) == 0 and len(training_y) == 0:
            self._has_training = False
            self._training_X = None
            self._training_y = None
        else:
            self._has_training = True
            self._training_X = training_X
            self._training_y = training_y
            self.num_points, self.dims = self._training_X.shape[:2]
            self._training_cov() #update training covariance
        return
    
    def cov_matrix(self, X, Y):
        """Returns the covariance matrix between all pairs of points in X and Y
        according to the process' kernel.

        Parameters
        ----------
        X : array-like
            n x d array of n points in d dimensions.
        Y : array-like
            m x d array of m points in d dimensions.
            
        Raises
        ------
        ValueError
            Raises ValueError if X and Y have mismatched dimensions.

        Returns
        -------
        array
            n x m array of all pairwise covariances between rows of X and Y.

        """
        return cdist(X,Y,metric=self._kernel.apply)
    
    def _training_cov(self):
        """Private method for precalculating the inverse of the training covariance
        with noise. This value is used for several methods, takes non-trivial
        time to calculate, and does not change unless training or kernel changes
        so it is stored for efficiency.

        Returns
        -------
        None.

        """
        
        self._y_cov = self.cov_matrix(self._training_X,self._training_X)
        self._noise_cov = self._noise_var * np.identity(self.num_points)
        self._inv_y_noise_cov = np.linalg.inv(self._y_cov + self._noise_cov)
        return
    
    def predictive_mean(self, X):
        """Calculates the mean vector of the posterior distribution over the 
        function space for the supplied points. That is, the expected value of 
        the underlying function at each test input value, given the training data.
        If there is no training data, the mean is equal to zero everywhere
        (the mean of the prior).

        Parameters
        ----------
        X : array-like
            An n x d array of n test inputs in d dimensions. These inputs are
            the points to interpolate over.

        Returns
        -------
        array
            An n x 1 array of the calculated mean values at each point in X.

        """
        if not self._has_training:
            #prior mean
            return np.zeros(X.shape[0])
        cross_cov = self.cov_matrix(X, self._training_X)
        return cross_cov @ self._inv_y_noise_cov @ self._training_y
    
    def predictive_cov(self, X):
        """Calculates the covariance matrix for the posterior distribution over
        the function space for the supplied points. That is, the covariance matrix
        between all pairs of test inputs, given the training data. If there is 
        no training data, the covariance matrix is simply that defined by the
        kernel (the covariance of the prior).

        Parameters
        ----------
        X : array-like
            An n x d array of n test inputs in d dimensions. These inputs are
            the points to interpolate over.

        Returns
        -------
        array
            An n x n array of the covariances between each pair of points in X.

        """
        
        if not self._has_training:
            #prior cov
            return self.cov_matrix(X,X)
        cross_cov = self.cov_matrix(self._training_X,X)
        b = cross_cov.T @ self._inv_y_noise_cov @ cross_cov
        return self.cov_matrix(X,X) - b
    
    def predictive_sigma(self, X):
        """Calculates the standard deviation for the posterior distribution
        over the function space for the supplied points. That is, the standard
        deviation for the underlying function at each test input, given the
        training data. If there is no training data, the standard deviation is
        simply equal to the standard deviation of the noise.
        
        Parameters
        ----------
        X : array-like
            An n x d array of n test inputs in d dimensions. These inputs are
            the points to interpolate over.

        Returns
        -------
        array
            An n x 1 array of the calculated standard deviation values at each
            point in X.

        """
        
        return np.sqrt(np.diag(self.predictive_cov(X)))
    
    
    def sample_posterior(self, X, n):
        """Sample the posterior distribution over the function space for the
        supplied points. That is, generate 'feasible' functions according to
        the covariance matrix, given the training data. If there is no training
        data, the distribution is equal to the prior distribution (mean zero and 
        covariance defined by the kernel).
        
        
        Parameters
        ----------
        X : array-like
            An m x d array of m test inputs in d dimensions. These inputs are
            the points to interpolate over.
        n : int
            Size of sample (number of examples to take).

        Returns
        -------
        array
            An n x m array of the sampled values where each row is a generated
            function.

        """
        
        mean = self.predictive_mean(X)
        cov = self.predictive_cov(X)
        return self.rng.multivariate_normal(mean.flatten(),cov,n)
    
    def sample_prior(self, X, n):
        """Sample the prior distribution over the function space. The prior
        distribution is essentially 'random' functions with mean zero which
        follow the covariance function defined by the kernel.

        Parameters
        ----------
        X : array-like
            An m x d array of m test inputs in d dimensions. These inputs are
            the points to interpolate over.
        n : int
            Size of sample (number of examples to take).

        Returns
        -------
        array
            An n x m array of the sampled values where each row is a generated
            function.

        """
        cov = self.cov_matrix(X,X)
        return self.rng.multivariate_normal(np.zeros(X.shape[0]),cov,n)
    
