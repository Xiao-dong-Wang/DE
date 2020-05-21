# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 19:27:27 2020

@author: xdwang
"""

import numpy as np
import random
from util import better

class DE:
    def __init__(self, funct, NP, max_iter, lb, ub, F, CR):
        self.funct = funct
        self.NP = NP
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.F = F
        self.CR = CR
        self.dim = self.lb.shape[0]
        self.outdim = len(self.funct)
        
        self.best_vio = np.inf
        self.best_y = np.zeros((self.outdim))
        self.best_y[0] = np.inf
        self.best_x = np.zeros((self.dim))
        self.init_dataset()
        
    
    # initialize dataset
    def init_dataset(self):
        x = np.zeros((self.dim, self.NP))
        y = np.zeros((self.outdim, self.NP))
        for i in range(self.dim):
            x[i, :] = np.random.uniform(self.lb[i], self.ub[i], self.NP)
        self.x = x
        for j in range(self.outdim):
            y[j,:] = self.funct[j](x)
        self.y = y
        self.dataset = {}
        self.dataset['x'] = self.x
        self.dataset['y'] = self.y
    
    # obtain the current best objective value
    def get_best_y(self, x, y):
        for i in range(y.shape[1]):
            vio = np.maximum(y[1:,i],0).sum()
            if vio < self.best_vio and self.best_vio > 0:
                self.best_vio = vio
                self.best_y = np.copy(y[:,i])
                self.best_x = np.copy(x[:,i])
            elif vio <= 0 and self.best_vio <=0 and y[0,i] < self.best_y:
                self.best_vio = vio
                self.best_y = np.copy(y[:,i])
                self.best_x = np.copy(x[:,i])
    
    # mutation
    def mutator(self):
        self.get_best_y(self.x, self.y)
        self.mutation_x = np.zeros((self.dim, self.NP))
        for i in range(self.NP):
            r1 = r2 = 0
            while r1 == i or r2 == i or r1 == r2:
                r1 = random.randint(0, self.NP-1)
                r2 = random.randint(0, self.NP-1)
            tmp_x = self.best_x + self.F * (self.x[:,r1] - self.x[:,r2])
            # check boundary
            for j in range(self.dim):
                tmp_x[j] = np.maximum(self.lb[j], np.minimum(self.ub[j], tmp_x[j]))
            self.mutation_x[:,i] = tmp_x
    
    # crossover
    def crossover(self):
        self.cr_x = np.zeros((self.dim, self.NP))
        for i in range(self.NP):
            for j in range(self.dim):
                rand_j = random.randint(0, self.dim-1)
                tmp = random.random()
                if tmp <= self.CR or rand_j == j:
                    self.cr_x[j,i] = self.mutation_x[j,i]
                else:
                    self.cr_x[j,i] = self.x[j,i]
    
    # select
    def selector(self):
        self.select_x = np.zeros((self.dim, self.NP))
        self.select_y = np.zeros((self.outdim, self.NP))
        next_y = np.zeros((self.outdim, self.NP))
        for i in range(self.outdim):
            next_y[i,:] = self.funct[i](self.cr_x)
        for j in range(self.NP):
            if better(next_y[:,j], self.y[:,j]):
                self.select_y[:,j] = next_y[:,j]
                self.select_x[:,j] = self.cr_x[:,j]
            else:
                self.select_y[:,j] = self.y[:,j]
                self.select_x[:,j] = self.x[:,j]
        self.x = self.select_x
        self.y = self.select_y
    
    def optimize(self):
        for i in range(self.max_iter):
            print('**********************************************')
            print('Generation ', i)
            self.mutator()
            self.crossover()
            self.selector()
            print('The best x in this generation ', self.best_x)
            print('The best y in this generation ', self.best_y)
            self.dataset['x'] = np.concatenate((self.dataset['x'].T, self.x.T)).T
            self.dataset['y'] = np.concatenate((self.dataset['y'].T, self.y.T)).T
        self.get_best_y(self.dataset['x'], self.dataset['y'])
        return self.best_x, self.best_y
            
        