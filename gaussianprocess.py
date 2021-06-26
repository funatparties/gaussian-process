#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: Josh"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import sqeuclidean,cdist,euclidean

class Kernel(ABC):
    @abstractmethod
    def apply(self, p, q):
        """Returns covariance between points p and q.

        Parameters
        ----------
        p : array-like
            Vector representing point p
        q : array-like
            Vector representing point q

        Returns
        -------
        float
            Covariance according to kernel.

        """
        return 0.0
    
    
class Linear(Kernel):
    def __init__(self):
        return
    
    def apply(self, p, q):
        return np.dot(p.T,q)
    
class SquaredExponential(Kernel):
    def __init__(self, l=1):
        self.l = l
        return
    
    def apply(self, p, q):
        c = -1/(2*self.l*self.l)
        return np.exp(c*sqeuclidean(p,q))
    
class Periodic(Kernel):
    def __init__(self,l=1,P=1):
        self.l = l
        self.P = P
    
    def apply(self, p, q):
        c = -2/(2*self.l*self.l)
        d = np.pi*euclidean(p,q)/self.P
        s = np.sin(d)
        return np.exp(c*s*s)
        
class Polynomial(Kernel):
    def __init__(self, d=2, c=0):
        self.d = d
        self.c = c
        
    def apply(self, p, q):
        return (np.dot(p,q)+self.c)**self.d

class GaussianProcess():
    """Class for representing gaussian processes. Stores covariance kernel, 
    training data, and hyperparameters. Has methods for sampling prior and 
    posterior distributions, as well as generating mean and covariance.
        
    The process models data as coming from an underlying function with added
    Gaussian white noise.
    """
    def __init__(self, kernel, noise_var=1, training_X=None, training_y=None, rng_seed = None):
        """

        Parameters
        ----------
        kernel : Kernel
            The covariance kernel (with set hyperparameters)
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
            DESCRIPTION.
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        if not isinstance(kernel, Kernel):
            raise TypeError("kernel must be Kernel object.")
        self.kernel = kernel
        
        if not noise_var >= 0:
            raise ValueError("noise_var must be non-negative.")
        self._noise_var = noise_var
        
        self.rng = np.random.default_rng(rng_seed)
        
        if (training_X is None) and (training_y is None):
            self._has_training = False
        elif (training_X is None) or (training_y is None):
            raise TypeError("Must supply both X and y training data.")
        else:
            if len(training_X) == 0 or len(training_y) == 0:
                raise ValueError("Training data must not be empty")
            self._has_training = True
            self.training_X = training_X
            self.training_y = training_y
            self.num_points, self.dims = self.training_X.shape[:2]
            self._training_cov() #precalculate training covariance
        return
    
    @property
    def has_training(self):
        return self._has_training
    
    @property
    def noise_var(self):
        return self._noise_var
    
    @noise_var.setter
    def noise_var(self, new_val):
        if new_val >= 0:
            self._noise_var = new_val
            if self._has_training:
                self._training_cov() #update training covariance
        else:
            raise ValueError("Noise_var must be non-negative")
        return
    
    def update_training(self, training_X, training_y):
        if (training_X is None) and (training_y is None):
            self._has_training = False
            self.training_X = None
            self.training_y = None
        elif (training_X is None) or (training_y is None):
            raise TypeError("Must supply both X and y training data")
        else:
            if len(training_X) == 0 or len(training_y) == 0:
                raise ValueError("Training data must not be empty")
            self._has_training = True
            self.training_X = training_X
            self.training_y = training_y
            self.num_points, self.dims = self.training_X.shape[:2]
            self._training_cov() #update training covariance
        return
    
    #TODO:kernel config setters and getters
    
    def cov_matrix(self, X, Y):
        """

        Parameters
        ----------
        X : array-like
            DESCRIPTION.
        Y : array-like
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return cdist(X,Y,metric=self.kernel.apply)
    
    def _training_cov(self):
        self._y_cov = self.cov_matrix(self.training_X,self.training_X)
        self._noise_cov = self._noise_var * np.identity(self.num_points)
        self._inv_y_noise_cov = np.linalg.inv(self._y_cov + self._noise_cov)
        return
    
    def sample_prior(self, X, n):
        """

        Parameters
        ----------
        X : array-like
            DESCRIPTION.
        n : int
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        cov = self.cov_matrix(X,X)
        return self.rng.multivariate_normal(np.zeros(X.shape[0]),cov,n)
    
    def predictive_mean(self, X):
        """

        Parameters
        ----------
        X : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if not self._has_training:
            #prior mean
            return np.zeros(X.shape[0])
        cross_cov = self.cov_matrix(X,self.training_X)
        return cross_cov @ self._inv_y_noise_cov @ self.training_y
    
    def predictive_cov(self, X):
        if not self._has_training:
            #prior cov
            return self.cov_matrix(X,X)
        cross_cov = self.cov_matrix(self.training_X,X)
        b = cross_cov.T @ self._inv_y_noise_cov @ cross_cov
        return self.cov_matrix(X,X) - b
    
    def predictive_sigma(self, X):
        return np.sqrt(np.diag(self.predictive_cov(X)))
    
    def sample_posterior(self, X, n):
        mean = self.predictive_mean(X)
        cov = self.predictive_cov(X)
        return self.rng.multivariate_normal(mean.flatten(),cov,n)
    