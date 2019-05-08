
import math
import pprint
import numpy as np
import os


def makedir_safe(dirname):
    '''
    A simple utility for making new directories
    after checking that they do not exist.
    '''
    if not os.path.exists(dirname):
        os.makedirs(dirname)


# Callback function for the step size.
def alpha_fixed(t, val):
    return val
def make_step(u):
    def mystep(t, model=None, data=None, newdir=None):
        return alpha_fixed(t=t, val=u)
    return mystep


def vlnorm(meanlog, sdlog):
    '''
    Variance of the log-Normal distribution.
    '''
    return (math.exp(sdlog**2) - 1) * math.exp((2*meanlog + sdlog**2))

def mlnorm(meanlog, sdlog):
    '''
    Mean of log-Normal distribution.
    '''
    return math.exp((meanlog + sdlog**2/2))


def vnorm(shift, scale):
    '''
    Variance of the Normal distribution.
    '''
    return scale**2

def mnorm(shift, scale):
    '''
    Mean of Normal distribution.
    '''
    return shift


BETA_GEMAN_ABS = 0.3851295
BETA_GEMAN_QUAD = 0.3443205

def chi_geman_abs(x, beta):
    return np.abs(x) / (1+np.abs(x)) - beta

def chi_geman_quad(x, beta):
    return x**2 / (1+x**2) - beta

def scale_chi_compute(x, thres, iters, chi_fn, beta):
    '''
    A general-purpose routine for scale estimation using
    M-estimators that are built with one of any number of
    "chi" functions.
    '''
    
    s_new = np.std(x) # initialize to sd, check for degeneracy.
    if s_new <= 0:
        return 0.001

    diff = 1.0 # note that thres << 1.

    for t in range(iters):
        s_old = s_new
        val = x / s_old
        s_new = s_old*np.sqrt(1+np.mean(chi_fn(x=val, beta=beta))/beta)
        diff = abs(s_new-s_old)
        if diff <= thres:
            break

    return s_new


def psi_gud(u):
    '''
    Gudermannian function.
    '''
    return 2 * np.arctan(np.exp(u)) - np.pi/2


def est_gud(x, s, thres=1e-03, iters=50):
    '''
    M-estimate of location using Gudermannian function.
    '''
    
    new_theta = np.mean(x)
    old_theta = None
    diff = 1.0

    # Solve the psi-condition.
    for t in range(iters):
        old_theta = new_theta
        new_theta = old_theta + s * np.mean(psi_gud((x-old_theta)/s))
        diff = abs(old_theta-new_theta)
        if diff <= thres:
            break

    return new_theta
