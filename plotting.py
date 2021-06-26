# -*- coding: utf-8 -*-
"""@author: Josh"""

import matplotlib.pyplot as plt

class Plotter():
    colours = 'bgmcykr'
    config = None #TODO: store plot settings
    def __init__(self, x, mean, sigma, training_X=None, training_y=None,
                 samples = None):
        self.x = x #flattened
        self.mean = mean
        self.sigma = sigma
        if (training_X is None) and (training_y is None):
            self._has_training = False
        elif (training_X is None) or (training_y is None):
            raise TypeError("Must supply both X and y training data.")
        else:
            self.train_x = training_X
            self.train_y = training_y
            self._has_training = True
        self.samples = samples
        return
    
    def add_samples(self, samples):
        self.samples = samples
        return
    
    def plot_samples(self):
        if self.samples is not None:
            for i,y in enumerate(self.samples):
                self.ax.plot(self.x, y, self.colours[i]+'-')
        return
    
    def prepare_figure(self):
        self.fig = plt.figure(figsize=(6,4))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlabel("Input")
        self.ax.set_ylabel("Output")
        return
    
    def plot_CI(self):
        self.ax.plot(self.x,self.mean,'r-')
        plt.fill_between(self.x,self.mean+(2*self.sigma),
                         self.mean-(2*self.sigma),alpha=0.5)
        return
    
    def plot_training(self):
        if self._has_training:
            self.ax.plot(self.train_x, self.train_y, 'k+')
        return
    
    def plot(self):
        self.prepare_figure()
        self.plot_CI()
        self.plot_training()
        self.plot_samples()
        plt.show()
        return