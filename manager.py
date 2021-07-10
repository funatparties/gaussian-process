#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: JoshM"""

import kernel as k
from gaussianprocess import GaussianProcess
from plotting import Plotter
import numpy as np

class Manager():
    """A top level class that provides an interface for accessing, updating, and
    plotting gaussian processes.
    """
    def __init__(self, proc, domain=None):
        """
        Parameters
        ----------
        proc : GaussianProcess
            The model object that is to be managed and plotted.
        domain : array-like, optional
            The X domain over which the interpolation/prediction is performed
            by the gaussian process model. This must be supplied before plotting
            is possible. The default is None.

        Raises
        ------
        TypeError
            Raises TypeError if proc is not a GaussianProcess object.

        Returns
        -------
        None.

        """
        
        self._plotter = None
        if not isinstance(proc, GaussianProcess):
            raise TypeError("proc must be GaussianProcess object.")
        self._proc = proc
        self._test_X = domain
        return
    
    @property
    def proc(self):
        """
        Returns
        -------
        GaussianProcess
            The gaussian process model object being managed.

        """
        
        return self._proc
    
    @proc.setter
    def proc(self, new_process):
        """Replaces the gaussian process object being managed.

        Parameters
        ----------
        new_process : GaussianProcess
            The new process object to be managed.
            
        Raises
        ------
        TypeError
            Raises TypeError if new_process is not a GaussianProcess object.

        Returns
        -------
        None.

        """
        
        if not isinstance(new_process, GaussianProcess):
            TypeError("proc must be GaussianProcess object.")
        self._proc = new_process
        return
    
    @property
    def plotter(self):
        """
        Returns
        -------
        Plotter
            The plotter object being used for plotting data from the model.

        """
        
        return self._plotter
    
    @plotter.deleter
    def plotter(self):
        """Removes the stored plotter object.

        Returns
        -------
        None.

        """
        
        self._plotter = None
        return
    
    @property
    def domain(self):
        """
        Returns
        -------
        array
            The domain over which interpolation/prediction is performed by the
            model.

        """
        
        return self._test_X
    
    @domain.setter
    def domain(self, new_X):
        """Sets the domain over which interpolation/prediction is performed by
        the model.

        Parameters
        ----------
        new_X : array-like
            An n x 1 array of the points at which predictions will be made by
            the model.

        Returns
        -------
        None.

        """
        
        if new_X is not None:
            if len(new_X) == 0:
                self._test_X = None
        self._test_X = new_X
        return
    
    def create_plotter(self):
        """Calculates predictions from the model and creates an internal 
        Plotter object with the the generated values. This plotter is used for
        creating plots and should be recreated if the model is modified.

        Raises
        ------
        TypeError
            Raises TypeError if the process object or test domain data are missing.

        Returns
        -------
        None.

        """
        
        if self._proc is None:
            raise TypeError("Gaussian process required.")
        if self._test_X is None:
            raise TypeError("Test data required.")
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
        """Samples the distribution of the model and adds the results to the
        plotter object for visualisation. plot() will need to be called to see
        them.

        Parameters
        ----------
        n : int
            The sample size (number of examples to draw).

        Returns
        -------
        None.

        """
        
        #if no plotter, create plotter
        if not self._plotter:
            self.create_plotter()
        #potential bug if proc settings have changed and create_plotter not called
        self._plotter.add_samples(self.proc.sample_posterior(self._test_X,n))
        return
    
    def plot(self):
        """Uses the plotter object to show a plot of the stored data. If no
        plotter object exists, one is generated.

        Returns
        -------
        None.

        """
        
        #if not plotter, create plotter
        if not self._plotter:
            self.create_plotter()
        self._plotter.plot()
        return
    

def write_array_txt(filename, array):
    """Saves an array as a txt file using the numpy filewriter

    Parameters
    ----------
    array : array
        The numpy array to be saved.
    filename : str
        The path of the file to save the array. Existing files will be 
        overwritten.

    Returns
    -------
    None.

    """
    
    np.savetxt(filename,array,fmt='%.6f')
    return


def read_array_txt(filename):
    """Reads a txt file and returns the stored array. If the array is 1D, converts
    to column vector format.

    Parameters
    ----------
    filename : str
        The path of the file to be read.

    Returns
    -------
    array
        The array generated by the file data.

    """
    
    x = np.genfromtxt(filename)
    if len(x.shape) == 1:
        x = x.reshape((*x.shape,1))
    return x
    
def test():
    
    test_inputs = read_array_txt('data/test_inputs.txt')
    train_inputs = read_array_txt('data/train_inputs.txt')
    train_outputs = read_array_txt('data/train_outputs.txt')
    
    sqk = k.SquaredExponential()
    p = GaussianProcess(sqk,train_inputs,train_outputs,1,0)
    m = Manager(p,test_inputs)
    m.plot()
    return m