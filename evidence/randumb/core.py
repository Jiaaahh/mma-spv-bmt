"""RanDumb-SPV: RBF features and pooled-residual OAS, exact dual solve.
Adapted mechanism; no deep learning, no test-time fitting.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.kernel_approximation import RBFSampler

def embedder(x_ideal, dim, multiplier, seed):
    gamma=float(multiplier/(x_ideal.shape[1]*np.var(x_ideal,dtype=np.float64)))
    if not np.isfinite(gamma): raise ValueError('invalid training variance')
    obj=RBFSampler(gamma=gamma,n_components=dim,random_state=seed)
    obj.fit(np.asarray(x_ideal,dtype=np.float64))
    return obj

def transform(obj,x,batch=96):
    out=np.empty((len(x),obj.n_components),dtype=np.float64)
    for i in range(0,len(x),batch):
        out[i:i+batch]=obj.transform(np.asarray(x[i:i+batch],dtype=np.float64))
    return out

class ResidualOAS:
    def __init__(self,z,y,repeat=1):
        self.classes=np.unique(y)
        self.means=np.stack([z[y==c].mean(0) for c in self.classes])
        # Repeat changes OAS n, not empirical covariance or class means.
        self.e=z-self.means[np.searchsorted(self.classes,y)]
        self.n,self.p=self.e.shape
        self.g=self.e@self.e.T/self.n
        mu=float(np.trace(self.g)/self.p)
        alpha=float(np.sum(self.g*self.g)/(self.p*self.p))
        den=(self.n*repeat+1)*(alpha-mu*mu/self.p)
        self.s=1.0 if den==0 else min((alpha+mu*mu)/den,1.0)
        self.mu=mu
        self.em=self.e@self.means.T

    def solve(self,rho):
        a=(1-rho)*(1-self.s)
        b=(1-rho)*self.s*self.mu+rho
        if b<=0: raise ValueError('non-positive regularizer')
        if a==0:
            w=self.means.T/b
        else:
            system=np.array(self.g,order='F',copy=True)
            system.flat[::self.n+1]+=b/a
            factor=cho_factor(system,lower=True,overwrite_a=True,check_finite=False)
            solution=cho_solve(factor,self.em,check_finite=False)
            w=(self.means.T-self.e.T@solution/self.n)/b
        bias=-0.5*np.sum(self.means.T*w,axis=0)
        assert np.isfinite(w).all() and np.isfinite(bias).all()
        return w,bias

def predict(z,w,b): return np.argmax(z@w+b,axis=1)
