# -*- coding: utf-8 -*-
"""@author: JoshM"""

import gaussianprocess as gp
from plotting import Plotter
import numpy as np

class Manager():
    #holds training data, updates process object and dynamically creates plotters
    def __init__(self, proc=None, test_X=None):
        self._plotter = None
        if proc is not None:
            if not isinstance(proc, gp.GaussianProcess):
                raise TypeError("process must be GaussianProcess object.")
        self._proc = proc
        self._test_X = test_X
        return
    
    @property
    def proc(self):
        return self._proc
    
    @proc.setter
    def proc(self, new_process):
        if not isinstance(new_process, gp.GaussianProcess):
            TypeError("process must be GaussianProcess object.")
        self._proc = new_process
        return
    
    @property
    def plotter(self):
        return self._plotter
    
    @plotter.deleter
    def plotter(self):
        self._plotter = None
        return
    
    @property
    def test_X(self):
        return self._test_X
    
    @test_X.setter
    def test_X(self, new_X):
        if new_X is not None:
            if len(new_X) == 0:
                self._test_X = None
        self._test_X = new_X
        return
    
    def create_plotter(self):
        if self._test_X is None:
            raise TypeError("test_X must not be None.")
        X = self._test_X
        mean = self.proc.predictive_mean(X).flatten()
        sigma = self.proc.predictive_sigma(X).flatten()
        if self.proc.has_training:
            train_x = self.proc.training_X.flatten()
            train_y = self.proc.training_y.flatten()
        else:
            train_x, train_y = None,None
        self._plotter = Plotter(X.flatten(),mean,sigma,train_x,train_y)
        
    def generate_samples(self, n):
        #if no plotter, create plotter
        if not self._plotter:
            self.create_plotter()
        #potential bug if proc settings have changed and create_plotter not called
        self._plotter.add_samples(self.proc.sample_posterior(self._test_X,n))
        return
    
    def plot(self):
        #if not plotter, create plotter
        if not self._plotter:
            self.create_plotter()
        self._plotter.plot()
        return
    
    #TODO: read write files
    #TODO: update kernel function
    #TODO: kernel config function
    
def test():
    def read_input(filename):
        with open(filename, 'r') as f:
            ar = np.array([float(line.strip()) for line in f])
        ar = ar.reshape(ar.shape[0],1)
        return ar

    test_inputs = read_input('test_inputs.txt')
    train_inputs = read_input('train_inputs.txt')
    train_outputs = read_input('train_outputs.txt')
    
    sqk = gp.SquaredExponential()
    p = gp.GaussianProcess(sqk,train_inputs,train_outputs,1,0)
    return Manager(p,test_inputs)