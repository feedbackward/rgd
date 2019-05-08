
import numpy as np

import helpers as hlp


class Model:
    '''
    Base class for model objects.
    '''

    def __init__(self, name=None):
        self.name = name

    def l_imp(self, w=None, X=None, y=None, lamreg=None):
        raise NotImplementedError
    
    def l_tr(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.l_imp(w=w, X=data.X_tr,
                              y=data.y_tr,
                              lamreg=lamreg)
        else:
            return self.l_imp(w=w, X=data.X_tr[n_idx,:],
                              y=data.y_tr[n_idx,:],
                              lamreg=lamreg)
    
    def l_te(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.l_imp(w=w, X=data.X_te,
                              y=data.y_te,
                              lamreg=lamreg)
        else:
            return self.l_imp(w=w, X=data.X_te[n_idx,:],
                              y=data.y_te[n_idx,:],
                              lamreg=lamreg)

    def g_imp(self, w=None, X=None, y=None, lamreg=None):
        raise NotImplementedError
    
    def g_tr(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.g_imp(w=w, X=data.X_tr,
                              y=data.y_tr,
                              lamreg=lamreg)
        else:
            return self.g_imp(w=w, X=data.X_tr[n_idx,:],
                              y=data.y_tr[n_idx,:],
                              lamreg=lamreg)
    
    def g_te(self, w, data, n_idx=None, lamreg=None):
        if n_idx is None:
            return self.g_imp(w=w, X=data.X_te,
                              y=data.y_te,
                              lamreg=lamreg)
        else:
            return self.g_imp(w=w, X=data.X_te[n_idx,:],
                              y=data.y_te[n_idx,:],
                              lamreg=lamreg)


class LinReg(Model):
    '''
    General-purpose linear regression model.
    No losses are implemented, just a predict()
    method.
    '''
    
    def __init__(self, data=None, name=None):
        super(LinReg, self).__init__(name=name)

        # If given data, collect model information.
        if data is not None:
            self.numfeats = data.X_tr.shape[1]
            

    def predict(self, w, X):
        '''
        Predict real-valued response.
        w is a (d x 1) array of weights.
        X is a (k x d) matrix of k observations.
        Returns array of shape (k x 1) of predicted values.
        '''
        return X.dot(w)


class LinearL2(LinReg):
    '''
    An orthodox linear regression model
    using the l2 error; typically this
    will be used for the classical least
    squares regression.
    '''
    
    def __init__(self, data=None):
        super(LinearL2,self).__init__(data=data)

        # Dimension of linear regression model.
        self.d = self.numfeats
        
    
    def l_imp(self, w, X, y, lamreg=None):
        '''
        Implementation of l2-loss under linear model
        for regression.

        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeats) matrix of k observations.
        y is a (k x 1) matrix of labels in {-1,1}.
        lamreg is a regularization parameter (l2 penalty).

        Output:
        A vector of length k with losses evaluated at k points.
        '''
        if lamreg is None:
            return (y-self.predict(w=w,X=X))**2/2
        else:
            penalty = lamreg * np.linalg.norm(w)**2
            return (y-self.predict(w=w,X=X))**2/2 + penalty
    
    
    def g_imp(self, w, X, y, lamreg=None):
        '''
        Implementation of the gradient of the l2-loss
        under a linear regression model.

        Input:
        w is a (d x 1) matrix of weights.
        X is a (k x numfeats) matrix of k observations.
        y is a (k x 1) matrix of labels in {-1,1}.

        Output:
        A (k x numfeats) matrix of gradients evaluated
        at k points.
        '''
        if lamreg is None:
            return (y-self.predict(w=w,X=X))*(-1)*X
        else:
            penalty = lamreg*2*w.T
            return (y-self.predict(w=w,X=X))*(-1)*X + penalty

    
class Oracle_LinearL2:
    '''
    A somewhat special model; this is
    the RISK of the l2 loss under a 
    linear regression model, assuming
    oracle knowledge of the underlying
    distribution.

    The assumed form of the risk is:
      R(w) = (w-w_star)^{T}A(w-w_star)+b^2.

    If the loss l(w) = (y-<w,x>)^2, then
    we have that the risk R(w)=El(w) takes
    the desired form, with A=Exx^{T}, and
    with b^2 = Eepsilon^2, where epsilon is
    the zero-mean additive noise used in
    the model y = <w_star,x> + epsilon.
    '''

    def __init__(self, w_star, A, b):
        self.w_star = w_star
        self.A = A
        self.d = self.A.shape[0]
        self.b = b

    def risk(self, w):
        '''
        Compute the risk function.
        '''
        diff = w-self.w_star
        quad = np.dot(np.transpose(diff),
                      np.dot(self.A,diff).reshape(diff.shape))
        return np.float32(quad)/2 + self.b**2

    def dist(self, w):
        '''
        Compute the distance from the
        true underlying parameter.
        '''
        return np.linalg.norm(w-self.w_star)

    def g_imp(self, w):
        '''
        Compute the gradient of the
        risk function with respect to
        parameter w, assuming of course
        that we can invert the order of
        integration and differentiation.

        Returns a (1 x d) vector, to be
        transposed again in the line
        search algorithm.
        '''
        diff = w-self.w_star
        return np.transpose(np.dot(self.A, diff))
