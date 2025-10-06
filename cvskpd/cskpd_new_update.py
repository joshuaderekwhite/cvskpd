import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from statsmodels.genmod.families.links import *
from scipy.linalg import norm, qr
from sklearn.linear_model import Lasso, Ridge, LinearRegression
import itertools
from numpy import ravel as vec
from collections import namedtuple
from joblib import Parallel,delayed
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score, confusion_matrix, check_scoring
import random
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
def auc(y_true, y_pred):
    y_pred = np.array(y_pred) >= 0.5
    y_true = np.array(y_true) >= 0.5
    return roc_auc_score(y_true, y_pred)
scorers = {
    'MSE': mean_squared_error,
    'AUC': auc
}

def inv_vec(x, cols):
    return x.reshape(cols, -1).T
def orthonormalize(x):
    return qr(x, mode='economic')[0]
def b_next_iter(y,X,A,lamb=0):
    R = A.shape[1]
    lamb = lamb*np.prod(A.shape)
    Xa = [vec((x.T @ A).T) for x in X]
    return inv_vec(Ridge(fit_intercept = False, alpha=lamb).fit(Xa,y).coef_, R) # Ridge regression to stabilize B
def a_next_iter(y,X,B,lama=0):
    R = B.shape[1]
    lama = lama*np.prod(B.shape)
    Xb = [vec((x @ B).T) for x in X]
    return orthonormalize(inv_vec(Lasso(fit_intercept = False, alpha=lama).fit(Xb,y).coef_, R))
def zeta_next_iter(y,Z,lamz=0):
    # print(f"y: {y.shape}, Z: {Z.shape}")
    return Ridge(fit_intercept = False, alpha=lamz).fit(Z, y).coef_
def score_AB(A,B,zeta=None,score=None):
    stability = np.var(B)/(norm(B, 2)**2)
    sparsity = np.where(vec(A) == 0)[0].shape[0]/vec(A).shape[0]
    norms = [
        np.linalg.norm(A, 'fro'),
        np.linalg.norm(B, 'fro'),
        np.linalg.norm(zeta) if zeta is not None else 0
    ]
    if score is not None:
        dif = np.array([
            (np.linalg.norm(score['A'], 'fro') - norms[0])/norms[0],
            (np.linalg.norm(score['B'], 'fro') - norms[1])/norms[1],
            (np.linalg.norm(score['zeta']) - norms[2])/norms[2] if zeta is not None else 0
        ])
        history = score['history']
    else: 
        dif = np.array(norms)
        history = []
    history.append(dif)
    return {'stability': stability, 'sparsity': sparsity, 'dif': dif, 'zeta': zeta, 'A': A, 'B': B, 'history': history}
def Rearrange(X,p,d,inverse=False):
    p, d = np.array(p), np.array(d)
    slices = []
    Rx = []
    if inverse:
        if len(p) == 2:
            return Rearrange(X,[p[0],d[0]], [p[1],d[1]])
        else:
            return np.array([Rearrange(np.reshape(r, (p[1]*p[2],d[1]*d[2])), [p[1],d[1]], [p[2],d[2]]) for r in Rearrange(X, [p[0],d[0]], [p[1]*p[2],d[1]*d[2]])])
    else:
        assert X.ndim > 1, "The size of the tensor must be of dimensions of 2 or more."
        assert X.ndim == len(p) == len(d), f"The dimension size of X {X.ndim}, and the lengths of p ({len(p)}) and d ({len(d)}) must all be equal"
        assert (X.shape == p*d).all(), f"The dimensions of X ({X.shape}), must be equal to the product of each element of p*d ({p*d})"
        for dim in range(X.ndim):
            slices.append([])
            for i in range(p[dim]):
                slices[dim].append(slice(d[dim]*i, d[dim]*(i+1)))
        Rx = [X[s].reshape(-1,1) for s in list(itertools.product(*slices))]
        return np.concatenate(Rx, axis=1).T
def Rearrange_new(X,p,inverse=False):
    p, D = np.array(p), np.array(X.shape)
    d = D/p
    print(f"p: {p}, D: {D}, d:{d}")
    slices = []
    Rx = []
    if inverse:
        if len(p) == 2:
            return Rearrange_new(X,[p[0],d[0]])
        else:
            return np.array([Rearrange_new(np.reshape(r, (p[1]*p[2],d[1]*d[2])), [p[1],d[1]], [p[2],d[2]]) for r in Rearrange_new(X, [p[0],d[0]])])
    else:
        assert X.ndim > 1, "The size of the tensor must be of dimensions of 2 or more."
        assert X.ndim == len(p) == len(d), f"The dimension size of X {X.ndim}, and the lengths of p ({len(p)}) and d ({len(d)}) must all be equal"
        assert (X.shape == p*d).all(), f"The dimensions of X ({X.shape}), must be equal to the product of each element of p*d ({p*d})"
        for dim in range(X.ndim):
            slices.append([])
            for i in range(int(p[dim])):
                slices[dim].append(slice(int(d[dim])*i, int(d[dim])*(i+1)))
        Rx = [X[s].reshape(-1,1) for s in list(itertools.product(*slices))]
        return np.concatenate(Rx, axis=1).T
def pack_parameters(parameters_dict, static={}, max_vals=None, seed=None):
    values = [item if type(item)==list else [item] for item in parameters_dict.values()]
    combinations = list(itertools.product(*values))
    if max_vals is not None:
        if seed is not None:
            random.seed(seed)
        combinations = random.sample(combinations, max_vals)
    return [{**static, **dict(zip(parameters_dict.keys(), c))} for c in combinations]
def best_fit(X, y, Z=None, parameters_dict={}, max_vals=None, seed=None, tol=1e-3, n_cores=1):
        values = [item if type(item)==list else [item] for item in parameters_dict.values()]
        combinations = list(itertools.product(*values))
        if max_vals is not None:
            if seed is not None:
                random.seed(seed)
            combinations = random.sample(combinations, max_vals)
        parameters = [{**dict(zip(parameters_dict.keys(), c))} for c in combinations]
        # print(parameters)
        models = [CSKPD(**parameters[i]) for i in range(len(parameters))]

        if n_cores > 1:
            vals = Parallel(n_jobs= n_cores, verbose=False)(delayed(models[i].fit)(X,y,Z=Z) for i in range(len(parameters)))
        else:
            vals = [models[i].fit(X,y,Z=Z) for i in range(len(parameters))]

        opt = np.argmin([v.score['MBIC'] for v in vals])
        return vals[opt], vals
def custom_cross_validate(X, y, Z=None, scorers=scorers, k_folds=5, **kwargs):
    K = kwargs.get('K', None)
    if K is not None:
        assert type(K) == convolution, "K must be a convolution object."
        X = np.array([K.convolve(x) for x in X]) 
    valid_params = {'Z', 'tol', 'static', 'max_vals'}
    fit_params = {k: v for k, v in kwargs.items() if k in valid_params}
    pred_params = fit_params.copy()

    # Initialize a KFold object
    if k_folds > 1:
        kf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=0).split(X, y)
    else:
        kf = [(np.array(range(len(y))), np.array(range(len(y))))]
    
    # Initialize lists to store scores
    models = []
    scores = {key: [] for key in scorers}
    
    for train_index, test_index in kf:
        # Split the data
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if Z is not None:
            fit_params['Z'], pred_params['Z'] = Z[train_index], Z[test_index]

        # Clone the estimator to ensure it's a fresh model each time
        model = clone(model)
        model.fit(X_train, y_train, **fit_params)

        # Predict on the test set
        y_pred = model.predict(X_test, **pred_params)

        # Calculate and store each score
        for scorer_name, scorer in scorers.items():
            score = scorer(y_test, y_pred)
            scores[scorer_name].append(score)
        
        results = {f'test_{key}': np.array(value) for key, value in scores.items()}
        models.append({'model': model, 'results': results, 'y_hat': y_pred, 'y_true': y_test})
    
    return models

def relu(x):
    return np.maximum(0, x)

# Samplers for test and train data sets
def modified_k_fold(Y, folds=5, shuffle=True):
    kf = KFold(folds,shuffle=shuffle)
    sample = np.array(range(len(Y)))
    test_k = [idx[1] for idx in list(kf.split(sample))]
    train_k = [np.setdiff1d(sample, k) for k in test_k]
    return test_k, train_k
def pick_fold(data, test_or_train, fold_number):
    if type(data) == list:
        return [data[k] for k in test_or_train[fold_number]]
    else:
        return np.array([data[k] for k in test_or_train[fold_number]])

# Scorers
def mse(y_true, y_pred):
    return np.average((y_true - y_pred)**2)
def calc_mbic(S,R,p):
    def mbic(y_true, y_pred):
        # print(R,p)
        Cn, n = [np.log(np.log(R*np.prod(p))), len(y_true)]
        # return np.log(mse(y_true, y_pred))
        return np.log(mse(y_true, y_pred)) + (Cn * np.log(n)/n * S) + (2*S*np.log(0.1*np.prod(p)/4-1))
    return mbic

# Results Output
def print_results(folds, X, Y, test_k):
    aucs = []
    accs = []
    for i, f in enumerate(folds):
        X_pick = np.array([X[k] for k in test_k[i]])
        y_hat = f.predict(X_pick)
        aucs.append(auc(y_hat, Y[test_k[i]]))
        accs.append(np.average(np.round(y_hat,0) == Y[test_k[i]]))
    auc_score = {}
    acc_score = {}
    for i, f in enumerate(folds):
        for j, e in enumerate(f.models):
            try:
                y_h = e.predict(X_pick)
                auc_score[(i,j)] = auc(y_h, Y[test_k[i]])
                # y_hats[(i,j)] = list(y_h)
                acc_score[(i,j)] = np.average(np.round(y_h,0) == Y[test_k[i]])
            except:
                pass
    for k in range(len(folds[0].models)):
        score_auc = np.array([s for (i,j), s in auc_score.items() if j == k])
        score_acc = np.array([s for (i,j), s in acc_score.items() if j == k])
        print(f"Model {k}: AUC {np.round(np.average(score_auc),4)} ({np.round(np.std(score_auc),4)}), Accuracy {np.round(np.average(score_acc),4)} ({np.round(np.std(score_acc),4)}), Best AUC/Accuracy {np.max(score_auc)}/{np.max(score_acc)}")
    print(f"Ensemble Model: AUC {np.round(np.average(aucs),4)} ({np.round(np.std(aucs),4)}), Accuracy {np.round(np.average(accs),4)} ({np.round(np.std(accs),4)}), Best AUC/Accuracy {np.max(aucs)}/{np.max(accs)}")
    return aucs, accs

class CSKPD(RegressorMixin, BaseEstimator):
    def __init__(self,p=None,lama=0.0,lamb=0.0,lamz=0.0,R=None,g=Identity(),K=None,L=None,n_cores = -1,max_iter = 20,print_iter = 5,cuda=False,seed=None):
        self.p = p
        # self.d = d # d is now dependent on p
        self.lama = lama
        self.lamb = lamb
        self.lamz = lamz
        self.R = R
        self.g = g
        self.K = K
        self.L = L
        self.n_cores = n_cores
        self.max_iter = max_iter
        self.print_iter = print_iter
        self.cuda = cuda
        self.seed = seed

        # Defined after init
        self.S, self.Cn, self.C = None, None, None

    def fit(self, X, y, **kwargs):
        # Initializations and adjustments for X, y
        Z = kwargs.get('Z', None)
        if self.L is not None:
            assert type(self.L) == convolution, "L must be a convolution object."
            X = np.array([self.L.convolve(x) for x in X])
        if self.K is not None:
            assert type(self.K) == convolution, "K must be a convolution object."
            X = np.array([self.K.convolve(x) for x in X])
        if self.g is not None:
            g = self.g
            y = self.g(y)
        tol = kwargs.get('tol', 1e-3)
        if self.p is None:
            p_list = []
            for n in X[0].shape:
                pairs = []
                recs = []
                for i in range(1, int(n**0.5) + 1):
                    if n % i == 0:
                        pairs.append(i)
                        recs.append(n//i)
                p_list.append(pairs + recs[::-1])
            self.p = set(itertools.product(*p_list))
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        assert X is not None and y is not None and self.p is not None, "X, y, and p must be provided"
        
        self.d = [X[0].shape[i] // self.p[i] for i in range(len(X[0].shape))]
        Rx = [Rearrange(x,self.p,self.d) for x in X]

        # Initialize A and z
        y_iterations = 1 if y.ndim == 1 else y.shape[1]
        A = np.linalg.svd(sum([y[i] * Rx[i] for i in range(len(Rx))]))[0][:, :1]
        z = np.zeros(len(y))

        score = None
        zeta = None
        for i in range(self.max_iter+1):
            B = b_next_iter(y-z,Rx,A,self.lamb)
            A = a_next_iter(y-z,Rx,B,self.lama)
            # Changed to R=1
            if Z is not None:
                C = Rearrange(vec(A).reshape(-1,1) @ vec(B).reshape(1,-1), self.p, self.d, True)
                # print(f"C: {self.C.shape}, g(y): {len(g(y))}, X: r{X.shape}")
                zeta = zeta_next_iter(y-np.array([np.vdot(x,C) for x in X]),Z,self.lamz)
                z = np.array(Z @ zeta)
            score = score_AB(A,B,zeta,score)
            # print(f"dif: {score['dif']}")
            if np.all(np.abs(score['dif']) < tol):
                # print(f"Completed after {i} iterations.")
                break

        C = Rearrange(vec(A).reshape(-1,1) @ vec(B).reshape(1,-1), self.p, self.d, True)
        if self.R is not None:
            u, s, v = np.linalg.svd(C)
            C = u[:,(self.R-1):self.R] @ np.diag(s[(self.R-1):self.R]) @ v[(self.R-1):self.R, :]
        self.A = A
        self.B = B
        self.zeta = zeta
        # g intentionally left off since y is generalized here
        MSE = mse(y, np.sum(X * C) + np.array(Z @ self.zeta) if Z is not None else np.sum(X * C)) # MSE is for the generalized form
        S, Cn, n = [(1-score['sparsity']) * np.prod(self.d), np.log(np.log(np.prod(self.p))), len(y)]
        # MBIC = np.log(MSE)
        MBIC = np.log(MSE) + (Cn * np.log(n)/n * S) + (2*S*np.log(0.05*np.prod(C.shape)/4-1))
        print_detail = f"Iter: {i}, Instability: {np.round(score['stability']*100,1)}%, Sparsity: {np.round(score['sparsity']*100,1)}%, gMSE: {np.round(MSE,2)}, MBIC: {np.round(MBIC,2)}, dif: {score['dif']}, zeta: {zeta}"

        self.C = C
        self.score = {"gMSE":MSE, "A":A, "B":B, "C":C, "zeta": zeta, "print_detail":print_detail, "MBIC":MBIC, "S":S, "Cn":Cn, "score":score}
        return self

    def predict(self, X, **kwargs):
        Z = kwargs.get('Z', None)
        if self.L is not None:
            X = np.array([self.L.convolve(x) for x in X])
        if self.K is not None:
            X = np.array([self.K.convolve(x) for x in X])
        z = np.array(Z @ self.zeta) if Z is not None else np.zeros(X.shape[0])
        res = [self.g.inverse(np.sum(x*self.C)+z[i]) for i,x in enumerate(X)]
        return res
    
class EnsembleCSKPD(RegressorMixin, BaseEstimator):
    def __init__(self,**kwargs):
        self.kwargs = kwargs
        self.models = []
        self.seed = kwargs.get('seed', np.random.randint(1000,size=1)[0])

    def fit(self, X, y, **kwargs):
        # Formatted as array of dictionaries
        # Change X_meta to new_C and then predict with the non-sparse C of the meta_model
        C_new = []
        Z = kwargs.get('Z', None)
        model_parameters = kwargs.get('model_parameters', {})
        common_parameters = kwargs.get('common_parameters', {})
        predictions = []
        for params in model_parameters:
            model, _ = best_fit(X, y, Z=Z, parameters_dict={**params, **common_parameters}, **self.kwargs)
            self.models.append(model)
            predictions.append(model.predict(X, Z=Z))
        # print(predictions)
        y_meta = np.array(predictions).T
        self.meta_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.4,
            min_samples_leaf=8,
            random_state=self.seed
        )
        self.meta_model.fit(y_meta, y)
        return self
    
    def predict(self, X, **kwargs):
        Z = kwargs.get('Z', None)
        predictions = []
        for model in self.models:
            predictions.append(model.predict(X, Z=Z))
        y_meta = np.array(predictions).T
        return self.meta_model.predict(y_meta)
    
class convolution:
    def __init__(self, convolve_function, convolve_params, deconvolve_function=None):
        self.convolve_function = lambda x: convolve_function(x, **convolve_params)
        self.deconvolve_function = deconvolve_function

    def convolve(self, X):
        convolution = self.convolve_function(X)
        assert type(convolution) == tuple, "The convolution function must be returned as a tuple."
        # order of declaration must also be the same in the deconvolve function
        if len(convolution) > 1:
            self.deconvolve_params = convolution[1:]
        return convolution[0]

    def deconvolve(self, X):
        return self.deconvolve_function(X, *self.deconvolve_params)