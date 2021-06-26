# -*- coding: utf-8 -*-
"""@author: JoshM"""

import gaussianprocess as gp
from plotting import Plotter
import numpy as np

class Manager():
    #holds training data, updates process object and dynamically creates plotters
    def __init__(self, proc=None, x_range=None):
        self._plotter = None
        if proc is not None:
            if not isinstance(proc, gp.GaussianProcess):
                raise TypeError("process must be GaussianProcess object.")
        self._proc = proc
        self.x_range = x_range
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
    
    def update_plotter(self):
        if self.x_range is None:
            raise TypeError("x_range must not be None.")
        self._plotter = self._create_plotter(self.x_range)
    
    def _create_plotter(self, X):
        mean = self.proc.predictive_mean(X).flatten()
        sigma = self.proc.predictive_sigma(X).flatten()
        if self.proc.has_training:
            train_x = self.proc.training_X.flatten()
            train_y = self.proc.training_y.flatten()
        else:
            train_x, train_y = None,None
        return Plotter(X.flatten(),mean,sigma,train_x,train_y)
        
    def generate_samples(self, n):
        #if no plotter, create plotter
        #TODO: gen samples, add to plotter
        pass
    
    def plot(self):
        if not self._plotter:
            self.update_plotter()
        self._plotter.plot()
        return
    
    #TODO: read write files
    
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
    p = gp.GaussianProcess(sqk,1,train_inputs,train_outputs,0)
    return Manager(p,test_inputs)