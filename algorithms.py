
import math
import numpy as np

import config
import helpers as hlp


class Algo_LineSearch:
    '''
    Basic archetype of an iterator for implementing a line
    search algorithm.
    
    Note that we assume w_init is a nparray matrix with the
    shape (d,1), where d represents the number of total
    parameters to be determined.
    '''

    def __init__(self, w_init, step,
                 t_max, thres, mntm=None,
                 verbose=False, store=False,
                 lamreg=None, budget=None):

        # Attributes passed from user.
        self.w = w_init
        self.step = step
        self.t_max = t_max
        self.thres = thres
        self.mntm = mntm
        self.verbose = verbose
        self.store = store
        self.lamreg = lamreg
        self.budget = budget

        # Attributes determined internally.
        if self.w is None:
            self.nparas = None
            self.w_old = None
        else:
            self.nparas = self.w.size
            self.w_old = np.copy(self.w)
            
        self.t = None
        self.diff = np.inf
        self.stepcost = 0 # for per-step costs.
        self.cumcost = 0 # for cumulative costs.
        self.torecord = False
        
        if self.budget is not None:
            self.epoch_cost = 0
            self.epoch_cost_limit = self.budget // config.TORECORD_TIMES

        # Keep record of all updates (optional).
        if self.store:
            self.wstore = np.zeros((self.w.size,t_max+1), dtype=np.float32)
            self.wstore[:,0] = self.w.flatten()
        else:
            self.wstore = None
        
        
    def __iter__(self):

        self.t = 0

        if self.verbose:
            print("(via __iter__)")
            self.print_state()
            
        return self
    

    def __next__(self):
        '''
        Check the stopping condition(s).
        '''
        
        if self.t >= self.t_max:
            raise StopIteration

        if self.diff <= self.thres:
            raise StopIteration

        if self.budget is not None:
            if self.cumcost >= self.budget:
                raise StopIteration

        if self.verbose:
            print("(via __next__)")
            self.print_state()
            
            
    def update(self, model, data):
        '''
        Carry out the main parameter update.
        '''
        
        # Parameter update.
        newdir = self.newdir(model=model, data=data)
        stepsize = self.step(t=self.t, model=model, data=data, newdir=newdir)
        if self.mntm is not None:
            self.w = self.w + stepsize * (self.mntm*np.transpose(newdir) + (1-self.mntm)*(self.w-self.w_old))
        else:
            self.w = self.w + stepsize * np.transpose(newdir)
        
        # Update the monitor attributes.
        self.monitor_update(model=model, data=data)
        
        # Run cost updates.
        self.cost_update(model=model, data=data)

        # Keep record of all updates (optional).
        if self.store:
            self.wstore[:,self.t] = self.w.flatten()


    def newdir(self, model, data):
        '''
        By default returns None. This will be
        implemented by sub-classes that inherit
        this class.
        '''
        raise NotImplementedError
    

    def monitor_update(self, model, data):
        '''
        By default returns None. This will be
        implemented by sub-classes that inherit
        this class.
        '''
        raise NotImplementedError

    
    def cost_update(self, model, data):
        '''
        By default returns None. This will be
        implemented by sub-classes that inherit
        this class.
        '''
        raise NotImplementedError

    
    def print_state(self):
        print("t =", self.t, "( max =", self.t_max, ")")
        print("diff =", self.diff, "( thres =", self.thres, ")")
        if self.verbose:
            print("w = ", self.w)
        print("------------")


class Algo_GD(Algo_LineSearch):
    '''
    Iterator which implements a line-search steepest descent method,
    where the direction of steepest descent is measured using the
    Euclidean norm (this is the "usual" gradient descent). Here the
    gradient is a sample mean estimate of the true risk gradient.
    That is, this is ERM-GD.
    '''

    def __init__(self, w_init, step,
                 t_max, thres,
                 store=False, lamreg=None,
                 budget=None):

        super(Algo_GD, self).__init__(w_init=w_init,
                                      step=step,
                                      t_max=t_max,
                                      thres=thres,
                                      store=store,
                                      lamreg=lamreg,
                                      budget=budget)

    def newdir(self, model, data):
        '''
        Determine the direction of the update.
        '''
        return (-1) * np.mean(model.g_tr(w=self.w,
                                         data=data,
                                         lamreg=self.lamreg),
                              axis=0, keepdims=True)

    def monitor_update(self, model, data):
        '''
        Update the counters and convergence
        monitors used by the algorithm. This is
        executed once every step.

        For GD, increment counter and check the
        differences at each step.
        '''
        self.t += 1
        self.diff = np.linalg.norm((self.w-self.w_old))
        self.w_old = np.copy(self.w)
        
        
    def cost_update(self, model, data):
        '''
        Update the amount of computational resources
        used by the routine.

        Cost here is number of gradient vectors
        computed. GD computes one vector for each
        element in the sample.
        '''

        # Compute costs for this step.
        self.stepcost = data.n_tr

        # If costs have accumulated enough, record performance.
        if self.budget is not None:
            self.epoch_cost += self.stepcost
            if self.epoch_cost >= self.epoch_cost_limit:
                self.torecord = True
                self.epoch_cost = 0
                
        # Update cumulative costs.
        self.cumcost += self.stepcost


class Algo_OracleGD(Algo_LineSearch):
    '''
    Iterator which implements a line-search steepest descent method,
    under the special setting of an "oracle" procedure, in which we
    have access to the true risk gradient, and do not need to use
    any data to approximate it.
    '''

    def __init__(self, w_init, step, t_max, thres):

        super(Algo_OracleGD, self).__init__(w_init=w_init,
                                            step=step,
                                            t_max=t_max,
                                            thres=thres)

    def newdir(self, model, data):
        '''
        Determine the direction of the update.
        '''
        return (-1) * model.g_imp(w=self.w)


    def monitor_update(self, model, data):
        '''
        Update the counters and convergence
        monitors used by the algorithm. This is
        executed once every step.

        For GD, increment counter and check the
        differences at each step.
        '''
        self.t += 1
        self.diff = np.linalg.norm((self.w-self.w_old))
        self.w_old = np.copy(self.w)

        
    def cost_update(self, model, data):
        '''
        Update the amount of computational resources
        used by the routine.
        
        Cost here is number of gradient vectors
        computed. GD computes one vector for each
        element in the sample.
        '''
        self.stepcost = 1
        self.cumcost += self.stepcost


class Algo_RGD_Mest(Algo_LineSearch):
    '''
    Iterator which implements our first "robust gradient descent"
    proposal, using the Catoni M-estimator of the gradient's
    mean. This is almost a standard line-search steepest descent
    procedure, but with the empirical mean replaced by a more
    sophisticated estimator of the gradient vector mean.
    '''
    
    def __init__(self, w_init, step, t_max, thres,
                 delta, mest_thres, mest_iters,
                 mntm=None,
                 batchsize=None, replace=False,
                 store=False, lamreg=None, budget=None):
        
        super(Algo_RGD_Mest, self).__init__(w_init=w_init,
                                            step=step,
                                            t_max=t_max,
                                            thres=thres,
                                            mntm=mntm,
                                            store=store,
                                            lamreg=lamreg,
                                            budget=budget)
        self.delta = delta
        self.mest_thres = mest_thres
        self.mest_iters = mest_iters
        self.batchsize = batchsize
        self.replace = replace
        
        
    def newdir(self, model, data):
        '''
        Determine the direction of the update.
        '''
        
        # Prepare the matrix of gradient vectors.
        # If mini-batch size is specified, use it.
        if self.batchsize is not None:
            n = self.batchsize
            shufidx = np.random.choice(data.n_tr,
                                       size=self.batchsize,
                                       replace=self.replace)
            Gmtx = model.g_tr(w=self.w, n_idx=shufidx,
                              data=data, lamreg=self.lamreg)
        else:
            n = data.n_tr
            Gmtx = model.g_tr(w=self.w, data=data, lamreg=self.lamreg)
        
        d = Gmtx.shape[1]
        
        # Initialize to the empirical mean.
        g_est = np.mean(Gmtx, axis=0, keepdims=True)
        
        idx_robustify = np.random.choice(d, size=d, replace=False)
        
        # Compute core quantities, in dimension-wise fashion.
        for j in range(d):
            
            # For selected subset, robustify.
            if j in idx_robustify:
                
                s_factor = math.sqrt(n/math.log(1/self.delta))
                s_disp = hlp.scale_chi_compute(x=(Gmtx[:,j]-np.mean(Gmtx[:,j])),
                                               thres=self.mest_thres,
                                               iters=self.mest_iters,
                                               chi_fn=hlp.chi_geman_quad,
                                               beta=hlp.BETA_GEMAN_QUAD)
                s_est = max(s_disp*s_factor, config.s_min)
                g_est[0,j] = hlp.est_gud(x=Gmtx[:,j], s=s_est,
                                         thres=self.mest_thres,
                                         iters=self.mest_iters)
                
            # For the rest, leave as-is using summation.
            else:
                continue

        # Once we've covered all the dims, return
        # the upgraded gradient.
        return (-1) * g_est


    def monitor_update(self, model, data):
        '''
        Update the counters and convergence
        monitors used by the algorithm. This is
        executed once every step.
        '''
        self.t += 1
        self.diff = np.linalg.norm((self.w-self.w_old))
        self.w_old = np.copy(self.w)


    def cost_update(self, model, data):
        '''
        Update the amount of computational resources
        used by the routine.

        Cost here is number of gradient vectors
        computed. GD computes one vector for each
        element in the sample.
        '''

        # Compute costs for this step.
        if self.batchsize is not None:
            self.stepcost = self.batchsize
        else:
            self.stepcost = data.n_tr

        # If costs have accumulated enough, record performance.
        if self.budget is not None:
            self.epoch_cost += self.stepcost
            if self.epoch_cost >= self.epoch_cost_limit:
                self.torecord = True
                self.epoch_cost = 0
                
        # Update cumulative costs.
        self.cumcost += self.stepcost
