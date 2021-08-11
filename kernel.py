#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: JoshM"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import sqeuclidean, euclidean

class Kernel(ABC):
    """Abstract base class for covariance kernel objects. All kernel classes
    should sublass this class and override the apply method.
    """
    @abstractmethod
    def apply(self, p, q):
        """Returns covariance between points p and q according the kernel function.

        Parameters
        ----------
        p : array-like
            Vector representing point p
        q : array-like
            Vector representing point q

        Returns
        -------
        float
            Covariance according to kernel function.

        """
        return 0.0
    
    @property
    def name(self):
        """Returns the preferred string name of the kernel.
        
        Defaults to the class name.

        Returns
        -------
        str
            Name of the kernel.

        """
        return self.__class__.__name__
    
    @property
    def config(self):
        """Returns the hyperparameters/attributes of the kernel.

        Returns
        -------
        dict
            The attribute dict of the object.

        """
        
        return self.__dict__
    
    @config.setter
    def config(self, new_config):
        """Sets the hyperparameters/attributes of the kernel. Can be subject to
        constraints in subclasses and potentially raise a ValueError.

        Parameters
        ----------
        new_config : dict
            The dict containing new values to assign for attributes. Need not
            specify all attributes if only some are being changed.

        Returns
        -------
        None.

        """
        #TODO: better condition checking, maybe stored dict of lambdas
        self.__dict__.update(new_config)
        return
    
    
class Linear(Kernel):
    """The linear kernel is equivalent to Bayesian linear regression and 
    applies a covariance function k(p,q) = p^T @ q.
    """
    def __init__(self):
        return
    
    def apply(self, p, q):
        return np.dot(p.T,q)
    
class SquaredExponential(Kernel):
    """The Squared Exponential kernel, also known as the Gaussian kernel or
    Radial Basis Function (RBF) kernel, applies the covariance function
    k(p,q) = exp(-(||p - q||^2)/(2*l^2)). This kernel produces smooth models
    which can approximate any continuous target function.
    """
    def __init__(self, l=1.0):
        """
        Parameters
        ----------
        l : float, optional
            The length scale of the covariance function. It determines the range
            over which points influence the variance of their neighbours. Higher 
            values mean distant points have more influence. Must be positive. 
            The default is 1.0.

        Raises
        ------
        ValueError
            Raises ValueError if l is non-positive.

        Returns
        -------
        None.

        """
        
        if l <= 0:
            raise ValueError("l must be positive.")
        self.l = l
        return
    
    @property
    def name(self):
        return "Squared Exponential"
    
    @property
    def config(self):
        return self.__dict__
    
    @config.setter
    def config(self, new_config):
        if 'l' in new_config.keys():
            if new_config['l'] <= 0:
                raise ValueError("l must be positive.")
        self.__dict__.update(new_config)
        return
    
    def apply(self, p, q):
        c = -1/(2*self.l*self.l)
        return np.exp(c*sqeuclidean(p,q))
    
class Periodic(Kernel):
    """The periodic kernel produces periodic models good for fitting cyclical data.
    It applies a covariance function k(p,q) = exp(-(2*sin^2(pi*||p-q||/P))/l^2).
    """
    def __init__(self,l=1.0,P=1.0):
        """
        Parameters
        ----------
        l : float, optional
            The length scale of the covariance function. It determines the range
            over which points influence the variance of their neighbours. Higher 
            values mean distant points have more influence. Must be positive.
            The default is 1.0.
        P : float, optional
            The period of repetition in the function. Must be positive.
            The default is 1.0.

        Raises
        ------
        ValueError
            Raises ValueError if l or p is non-positive.

        Returns
        -------
        None.

        """
        
        if l <= 0:
            raise ValueError("l must be positive.")
        if P <= 0:
            raise ValueError("P must be positive.")
        self.l = l
        self.P = P
        
    @property
    def config(self):
        return self.__dict__
    
    @config.setter
    def config(self, new_config):
        if 'l' in new_config.keys():
            if new_config['l'] <= 0:
                raise ValueError("l must be positive.")
        if 'P' in new_config.keys():
            if new_config['P'] <= 0:
                raise ValueError("P must be positive.")
        self.__dict__.update(new_config)
        return
    
    
    def apply(self, p, q):
        c = -2/(self.l*self.l)
        d = np.pi*euclidean(p,q)/self.P
        s = np.sin(d)
        return np.exp(c*s*s)
        
class Polynomial(Kernel):
    """The polynomial kernel is effectively Bayesian polynomial regression of
    arbitrary degree. It applies the covariance function k(p,q) = (p^T @ q + c)^d.
    """
    def __init__(self, d=2, c=0.0):
        """
        Parameters
        ----------
        d : int, optional
            The degree of the polynomial model. Must be a positive integer.
            The default is 2.
        c : float, optional
            Determines the coefficient of lower order terms in the polynomial
            as can be calculated with the binomial theorem. Zero corresponds to 
            a homogenous polynomial (no lower order terms).
            The default is 0.0.

        Raises
        ------
        ValueError
            Raises ValueError if c is negative or d is not a positive integer.

        Returns
        -------
        None.

        """
        
        if d <= 0 or d != int(d):
            raise ValueError("d must be a positive integer.")
        if c < 0:
            raise ValueError("c must be non-negative.")
        self.d = int(d)
        self.c = c
        
    @property
    def config(self):
        return self.__dict__
    
    @config.setter
    def config(self, new_config):
        if 'd' in new_config.keys():
            int_d = int(new_config['d'])
            if new_config['d'] <= 0 or new_config['d'] != int_d:
                raise ValueError("d must be a positive integer.")
            new_config['d'] = int_d
        if 'c' in new_config.keys():
            if new_config['c'] < 0:
                raise ValueError("c must be non-negative.")
        self.__dict__.update(new_config)
        return
        
    def apply(self, p, q):
        return (np.dot(p.T,q)+self.c)**self.d
    
class LocalPeriodic(Periodic):
    """The locally periodic kernel is similar to the periodic kernel except the
    periodicity can change over time. Equivalent to multiplying the squared
    exponential kernel by the periodic kernel.
    """
    @property
    def name(self):
        return "Locally Periodic"
    
    def apply(self, p, q):
        c = -2/(self.l*self.l)
        d = np.pi*euclidean(p,q)/self.P
        s = np.sin(d)
        a = -1/(2*self.l*self.l)
        return np.exp(c*s*s)*np.exp(a*sqeuclidean(p,q))