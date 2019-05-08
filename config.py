
import os
import numpy as np

import helpers as hlp

# Number of times to record performance.
TORECORD_TIMES = 100


# Universal settings.

s_min = 0.001 # Minimum value for s scale, to deal with underflow.
mest_thres = 1e-03 # termination condition for M-estimator sub-routine; min difference.
mest_iters = 50 # termination condition for M-estimator sub-routine; max iterations.
mest_delta = 0.01 # confidence parameter for M-estimator sub-routine


# Task-specific parameter fetcher.

def task_paras(task, subtask=None):

    paras = {}

    if task == "POC":

        # Data-related.
        paras["n"] = 500 # sample size
        paras["d"] = 2 # number of parameters
        paras["init_range"] = 5.0 # controls support of random init.
        paras["num_trials"] = 250

        # Algorithm-related.
        paras["t_max"] = 45 # termination condition; max number of iters.
        paras["thres"] = -1.0 # termination condition; update threshold.
        paras["alphaval"] = 0.1 # pre-fixed step size value.

        # Clerical matters.
        paras["mth_names"] = ["erm", "oracle", "rgd"]
        paras["mth_colours"] = ["gray", "black", "darkgreen"]
        paras["mth_linestyles"] = ["dashed", "dashdot", "dotted"]
        paras["fontsize"] = "xx-large" # Font size setup.
        #matplotlib.rcParams['pdf.fonttype'] = 42 # for Type-1 fonts if needed.
        
        if subtask == "lognormal":
            
            noise_sdlog = 1.75
            noise_meanlog = 0.0
            paras["noise_var"] = hlp.vlnorm(meanlog=noise_meanlog,
                                            sdlog=noise_sdlog)
            paras["noise_mean"] = hlp.mlnorm(meanlog=noise_meanlog,
                                             sdlog=noise_sdlog)
            noise_fn = lambda u : np.random.lognormal(
                mean=noise_meanlog,
                sigma=noise_sdlog,
                size=u
            )-paras["noise_mean"]
            paras["noise_fn"] = noise_fn
            
        elif subtask == "normal":
            
            noise_sd = 20.0
            noise_mean = 0.0
            paras["noise_var"] = hlp.vnorm(shift=noise_mean,
                                           scale=noise_sd)
            paras["noise_mean"] = hlp.mnorm(shift=noise_mean,
                                            scale=noise_sd)
            noise_fn = lambda u : np.random.normal(
                loc=noise_mean,
                scale=noise_sd,
                size=u
            )-paras["noise_mean"]
            paras["noise_fn"] = noise_fn
        
        else:
            raise ValueError("Please pass a valid subtask name.")


    return paras


