#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""@author: Josh"""

import matplotlib.pyplot as plt

class Plotter():
    """Class for plotting data from gaussian processes and storing calculated
    values for quick replotting. When the model changes, a new Plotter should
    be generated.
    """
    colours = 'bgmcykr' #for plotting samples
    num_colours = len(colours)
    config = None #TODO: support storing plot settings
    def __init__(self, x, mean, sigma, training_X=None, training_y=None,
                 samples = None):
        """
        Parameters
        ----------
        x : array-like
            A flattened array of test inputs indicating the x domain over which
            the (non-training) data will be plotted.
        mean : array-like
            A flattened array of the distribution mean vector.
        sigma : array-like
            A flattened array of the distribution standard deviation.
        training_X : array-like, optional
            A flattened array of X values of any training data. 
            The default None indicates no training data.
        training_y : array-like, optional
            A flattened array of y values of any training data. 
            The default None indicates no training data.
        samples : array, optional
            An n x m array of n sampled functions from the distribution over the
            m test inputs. The default None indicates no samples.

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
        
        self.x = x
        self.mean = mean
        self.sigma = sigma
        if (training_X is None) and (training_y is None):
            self._has_training = False
        elif (training_X is None) or (training_y is None):
            raise TypeError("Must supply both X and y training data.")
        elif len(training_X) != len(training_y):
            raise ValueError("Must have equal number of rows in training data.")
        elif len(training_X) == 0 and len(training_y) == 0:
            self._has_training = False
            self.training_X = None
            self.training_y = None
        else:
            self.train_x = training_X
            self.train_y = training_y
            self._has_training = True
        self.samples = samples
        return
    
    def add_samples(self, samples):
        """Adds samples to the plotter. No changes to mean or training data needed.

        Parameters
        ----------
        samples : array
            An n x m array of n sampled functions from the distribution over m
            test inputs.

        Returns
        -------
        None.

        """
        
        self.samples = samples
        return
    
    def plot_samples(self):
        """Plots any samples on the axes.

        Returns
        -------
        None.

        """
        if self.samples is not None:
            for i,y in enumerate(self.samples):
                self.ax.plot(self.x, y, self.colours[i%self.num_colours]+'-')
        return
    
    def prepare_figure(self):
        """Sets up the figure and axis settings for plotting.

        Returns
        -------
        None.

        """
        
        self.fig = plt.figure(figsize=(6,4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Input")
        self.ax.set_ylabel("Output")
        return
    
    def plot_CI(self):
        """Plots the 95% confidence interval (approx. equal to 2 standard 
        deviations from the mean) on the axes.

        Returns
        -------
        None.

        """
        
        self.ax.plot(self.x,self.mean,'r-')
        plt.fill_between(self.x,self.mean+(2*self.sigma),
                         self.mean-(2*self.sigma),alpha=0.5)
        return
    
    def plot_training(self):
        """Plots any training data on the axes.

        Returns
        -------
        None.

        """
        
        if self._has_training:
            self.ax.plot(self.train_x, self.train_y, 'k+')
        return
    
    def plot(self):
        """Calls the plotting functions to plot all supplied information such
        as training, samples, mean, and confidence interval, and shows the plot.

        Returns
        -------
        None.

        """
        #TODO: add interactivity and more modularity around which elements to display
        self.prepare_figure()
        self.plot_CI()
        self.plot_training()
        self.plot_samples()
        plt.show()
        return