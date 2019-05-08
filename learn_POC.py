
import sys
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import algorithms as al
import models as mod
import classes
import helpers as hlp
import config


## Experiment setup.

task = "POC"

s = sys.stdin.read()
splitstr = s.split(" ")

if (splitstr[0] == ""):
    raise ValueError("Invalid input; please provide a task name.")

subtask = splitstr[0]

p = config.task_paras(task=task, subtask=subtask) # holds everything we need.

w_star = np.ones(p["d"]).reshape((p["d"],1)) # vector specifying true model
w_init = w_star + np.random.uniform(low=-p["init_range"],
                                    high=p["init_range"],
                                    size=p["d"]).reshape((p["d"],1))
# NOTE: Initial weights are randomly generated in advance, 
# so that per-trial randomness here is only due to the data.

num_mths = len(p["mth_names"])


## Running the algorithms.

# Sub-task specific data setup.
data = classes.DataSet() # Initialize data object; re-populate at each trial.
cov_X = np.eye(p["d"]) # covariance matrix of the inputs.

# Prepare storage for performance metrics.
perf_shape = (p["num_trials"], p["t_max"], num_mths)
riskvals = np.zeros(perf_shape, dtype=np.float32)
loss_tr = np.zeros(perf_shape, dtype=np.float32)
truedist = np.zeros(perf_shape, dtype=np.float32)

# Loop over trials.
for tri in range(p["num_trials"]):
    
    # Generate new data (with *centred* noise).
    X = np.random.normal(loc=0.0, scale=1.0, size=(p["n"],p["d"]))
    epsilon = p["noise_fn"]((p["n"],1))
    y = np.dot(X, w_star) + epsilon
    data.init_tr(X=X, y=y)
    
    # Initialize models.
    mod_oracle = mod.Oracle_LinearL2(w_star=w_star,
                                     A=cov_X,
                                     b=math.sqrt(p["noise_var"]))
    mod_learner = mod.LinearL2(data=data)
    risk_star = mod_oracle.risk(w=w_star) # optimal risk value.
    
    # Initialize all algorithms.
    al_gd = al.Algo_OracleGD(w_init=w_init,
                             step=hlp.make_step(u=p["alphaval"]),
                             t_max=p["t_max"],
                             thres=p["thres"])
    al_erm = al.Algo_GD(w_init=w_init,
                        step=hlp.make_step(u=p["alphaval"]),
                        t_max=p["t_max"],
                        thres=p["thres"])
    al_gud = al.Algo_RGD_Mest(w_init=w_init,
                              step=hlp.make_step(u=p["alphaval"]),
                              t_max=p["t_max"],
                              thres=p["thres"],
                              delta=config.mest_delta,
                              mest_thres=config.mest_thres,
                              mest_iters=config.mest_iters)
    
    # Run all algorithms and save their performance.
    
    ## ERM-GD.
    mthidx = 0
    idx = 0
    for mystep in al_erm:
        al_erm.update(model=mod_learner, data=data)
        # Record performance
        loss_tr[tri,idx,mthidx] = np.mean(mod_learner.l_tr(w=al_erm.w, data=data))-np.mean(mod_learner.l_tr(w=w_star, data=data))
        riskvals[tri,idx,mthidx] = mod_oracle.risk(w=al_erm.w)-risk_star
        truedist[tri,idx,mthidx] = mod_oracle.dist(w=al_erm.w)-0
        idx += 1
    
    ## GD Oracle.
    mthidx = 1
    idx = 0
    for mystep in al_gd:
        al_gd.update(model=mod_oracle, data=data)
        # Record performance
        loss_tr[tri,idx,mthidx] = np.mean(mod_learner.l_tr(w=al_gd.w, data=data))-np.mean(mod_learner.l_tr(w=w_star, data=data))
        riskvals[tri,idx,mthidx] = mod_oracle.risk(w=al_gd.w)-risk_star
        truedist[tri,idx,mthidx] = mod_oracle.dist(w=al_gd.w)-0
        idx += 1
        
    ## RGD using M-estimator based on function of Gudermann.
    mthidx = 2
    idx = 0
    for mystep in al_gud:
        al_gud.update(model=mod_learner, data=data)
        # Record performance
        loss_tr[tri,idx,mthidx] = np.mean(mod_learner.l_tr(w=al_gud.w, data=data))-np.mean(mod_learner.l_tr(w=w_star, data=data))
        riskvals[tri,idx,mthidx] = mod_oracle.risk(w=al_gud.w)-risk_star
        truedist[tri,idx,mthidx] = mod_oracle.dist(w=al_gud.w)-0
        idx += 1
        

# Finally, take statistics of the performance metrics over all trials.
ave_loss_tr = np.mean(loss_tr, axis=0)
ave_riskvals = np.mean(riskvals, axis=0)
ave_truedist = np.mean(truedist, axis=0)
sd_loss_tr = np.std(loss_tr, axis=0)
sd_riskvals = np.std(riskvals, axis=0)
sd_truedist = np.std(truedist, axis=0)


## Visualization of performance.

tvals = np.arange(p["t_max"])+1 # better to start from the first update.

# Average over trials.
myfig = plt.figure(figsize=(15,5))

ax_loss_tr = myfig.add_subplot(1,3,1)
plt.axhline(y=0.0, linestyle="-", color="black")
for m in range(num_mths):
    vals = ave_loss_tr[:,m]
    err = sd_loss_tr[:,m]
    ax_loss_tr.plot(tvals, vals,
                    linestyle=p["mth_linestyles"][m],
                    color=p["mth_colours"][m],
                    label=p["mth_names"][m])
    ax_loss_tr.tick_params(labelsize=p["fontsize"])
    ax_loss_tr.legend(loc=1,ncol=1, fontsize=p["fontsize"])
    plt.title("Excess empirical risk", size=p["fontsize"])
    

ax_riskvals = myfig.add_subplot(1,3,2)
for m in range(num_mths):
    vals = ave_riskvals[:,m]
    err = sd_riskvals[:,m]
    err_log = err / vals
    ax_riskvals.semilogy(tvals, vals,
                         linestyle=p["mth_linestyles"][m],
                         color=p["mth_colours"][m],
                         label=p["mth_names"][m])
    ax_riskvals.tick_params(labelsize=p["fontsize"])
    plt.title("Excess risk", size=p["fontsize"])

ax_variation = myfig.add_subplot(1,3,3)
plt.axhline(y=0.0, linestyle="-", color="black")
for m in range(num_mths):
    vals = sd_riskvals[:,m]
    ax_variation.plot(tvals, vals,
                      linestyle=p["mth_linestyles"][m],
                      color=p["mth_colours"][m],
                      label=p["mth_names"][m])
    ax_variation.tick_params(labelsize=p["fontsize"])
    plt.title("Variation of risk over samples", size=p["fontsize"])

hlp.makedir_safe("img")
plt.savefig(fname=("img/"+task+"_"+subtask+".pdf"), bbox_inches="tight")



