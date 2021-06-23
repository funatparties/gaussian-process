#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: Josh"""

from abc import ABC, abstractmethod
import numpy as np
from scipy.spatial.distance import sqeuclidean,cdist,euclidean
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

class Kernel(ABC):
    @abstractmethod
    def apply(self, p, q):
        pass
    
class Linear(Kernel):
    def __init__(self):
        return
    
    def apply(self, p, q):
        return np.dot(p.T,q)
    
class SquaredExponential(Kernel):
    #TODO: doc
    def __init__(self, l=1):
        #TODO: doc
        self.l = l
        return
    
    def apply(self, p, q):
        #TODO: doc
        c = -1/(2*self.l*self.l)
        return np.exp(c*sqeuclidean(p,q))
    
#TODO:periodic
#TODO:polynomial
#TODO:gaussian-periodic/products

class GaussianProcess():
    #TODO
    def __init__():
        pass
    
    def cov_matrix():
        pass
    #TODO
    
    def predictive_mean():
        pass
    #TODO
    
    def predictive_cov():
        pass
    #TODO