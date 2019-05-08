
import numpy as np
import sys


class DataSet:
    '''
    Base class for data objects.
    '''
    
    def __init__(self, paras={}):
        self.X_tr = None
        self.X_te = None
        self.y_tr = None
        self.y_te = None
        self.n_tr = None
        self.n_te = None
        self.paras = paras
    
    def init_tr(self, X, y, C=None):
        self.X_tr = X
        self.y_tr = y
        self.n_tr = X.shape[0]
        if C is not None:
            self.C_tr = C
        
    def init_te(self, X, y, C=None):
        self.X_te = X
        self.y_te = y
        self.n_te = X.shape[0]
        if C is not None:
            self.C_te = C
