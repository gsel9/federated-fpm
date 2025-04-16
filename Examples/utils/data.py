import numpy as np 
from scipy.linalg import lstsq

from sksurv.util import Surv
from sksurv.linear_model import CoxPHSurvivalAnalysis

from sklearn.preprocessing import StandardScaler


def feature_scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def init_params_random(D, X, duration, event, seed: int = 42):
    
    np.random.seed(seed)
    beta = np.random.random(X.shape[1])
    gamma = np.random.random(D.shape[1])
    
    return beta[None, :], gamma[None, :]


def init_params_cox(D, X, duration, event):

    # Create structured array
    y = Surv.from_arrays(event=event.squeeze(), time=duration.squeeze())
    
    # Fit a Cox PH model
    cox = CoxPHSurvivalAnalysis(alpha=0.01, tol=1e-6)
    cox.fit(X, y)

    # Estimated cumulative hazard conditional on covarites
    cumulative_hazards = cox.predict_cumulative_hazard_function(X, return_array=True)
    # Average cumulative hazard for each subject over observed times
    mean_hazard = np.clip(np.mean(cumulative_hazards, axis=1), 1e-16, 1e16)

    # Fit linear regression to log cumulative hazard
    beta, _, _, _ = lstsq(X, np.log(mean_hazard))
    gamma, _, _, _ = lstsq(D, np.log(mean_hazard))
    
    return beta[None, :], gamma[None, :]