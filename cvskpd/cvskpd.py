import matplotlib.pyplot as plt
import nibabel as nb
import numpy as np
import os
import seaborn as sns
import pandas as pd
import xmltodict
from PIL import Image
import re
import statsmodels.api as sm
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from SKPD import skpdRegressor
from SKPD import *
from sklearn.impute import SimpleImputer
# Change the tag to enable or disable warnings
import warnings, sys, io

# Cyclic-voxel SKPD
class SKPDRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, impute=None, pd_list=None, run_type='standard', fn=None, R_list=[1,2,3], lmbda_set=[0.4,0.6,1], n_cores=4, max_iter=100, print_iter=30):
        self.impute = impute
        self.pd_list = pd_list
        self.run_type = run_type
        self.fn = fn
        self.R_list = R_list
        self.lmbda_set = lmbda_set
        self.n_cores = n_cores
        self.max_iter = max_iter
        self.print_iter = print_iter
    def extract_features(self, data):
        """
        Extract the features from the data.

        Parameters
        ----------
        data : pandas.DataFrame or pandas.Series
            The data to extract the features from.

        Returns
        -------
        X : pandas.DataFrame or pandas.Series
            The extracted features.
        Z : pandas.DataFrame or pandas.Series
            The extracted covariates, if any.
        impute : str or None
            The imputation strategy, if any.
        """
        if hasattr(data, 'X'):
            X = data['X']
            Z = data.loc[:, data.columns != 'X']
        else:
            X = data
            Z = None
        if hasattr(data, 'impute'):
            impute = data['impute']
            Z = data.loc[:, data.columns != 'X']
        else:
            impute = None
        return X, Z, impute

    def fit(self, X, y):
        self.X_, self.Z_, _ = self.extract_features(X)
        if self.impute is not None:
            imp = SimpleImputer(
                missing_values=np.nan, 
                strategy=self.impute['strategy'], 
                fill_value=self.impute['fill_value']
                )
            self.y_ = imp.fit_transform(y.reshape(-1, 1)).squeeze()
        else:
            self.y_ = y
        p1_list,p2_list,p3_list,d1_list,d2_list,d3_list = self.pd_list
        input_params = {
            "p1_list":p1_list,
            "p2_list":p2_list,
            "d1_list":d1_list,
            "d2_list":d2_list,
            "p3_list":p3_list,
            "d3_list":d3_list,
            "lmbda_set":self.lmbda_set,
            "lmbda2_set":[0],
            ## Z_train is None, when not consider covariate
            "Z_train":self.Z_,
            "Y_train":self.y_,
            "R_list":self.R_list, "n_cores":self.n_cores,"max_iter":self.max_iter,"print_iter":self.print_iter}

        if self.run_type == 'standard':
            # Fit for both C and gamma values from SKPD
            input_params['X_train'] = self.X_
            self.a_hat,self.b_hat,gamma_hat,lmbda1,lmbda2,R,p1,p2,p3,d1,d2,d3,solver = skpdRegressor(**input_params)
            self.A,self.B,kron_ab = func_kron_ab(self.a_hat,self.b_hat,R,p1,p2,d1,d2,p3,d3)
            self.C_hat = kron_ab[-1]
            self.gamma_hat = gamma_hat
            self.hyperparameters = {'lmbda1': lmbda1, 'lmbda2': lmbda2, 'R': R, 'p': [p1, p2, p3], 'd': [d1, d2, d3]}
        elif self.run_type == 'multi':
            # Assume ditionary is passed through and is consistent for all records
            C = {}
            for _, c in enumerate(self.X_[0].keys()):
                C[c] = None
            gamma = {}
            self.hyperparameters = {}
            for _, c in enumerate(C.keys()):
                input_params['X_train'] = [x[c] for x in self.X_]
                self.a_hat,self.b_hat,gamma[c],lmbda1,lmbda2,R,p1,p2,p3,d1,d2,d3,solver = skpdRegressor(**input_params)
                self.A,self.B,kron_ab = func_kron_ab(self.a_hat,self.b_hat,R,p1,p2,d1,d2,p3,d3)
                C[c] = kron_ab[-1]
                self.hyperparameters[c] = {'lmbda1': lmbda1, 'lmbda2': lmbda2, 'R': R, 'p': [p1, p2, p3], 'd': [d1, d2, d3]}
            self.C_hat = C
            self.gamma_hat = gamma

        # Return the classifier
        return self
    
    def predict(self, X):
    # Check if fit has been called
        check_is_fitted(self)
        X_pred, Z_pred, impute = self.extract_features(X)
        if self.impute is not None:
            try:
                X_pred = X_pred[~impute]
                Z_pred = Z_pred[~impute]
            except:
                print('There was an error subsetting the values')

        if self.run_type == 'standard':
            y_hat = np.array([np.vdot(x, self.C_hat) for x in X_pred])
            if not Z_pred is None:
                y_hat += np.squeeze(Z_pred @ self.gamma_hat)
        elif self.run_type == 'multi':
            y_hat = np.zeros(len(X_pred))
            for i, c in enumerate(self.C_hat.keys()):
                y_hat += np.array([np.vdot(x[c], self.C_hat[c]) for x in X_pred])
                if not Z_pred is None:
                    y_hat += np.squeeze(Z_pred @ self.gamma_hat[c])
            y_hat = np.array(y_hat/(i+1))
        return y_hat