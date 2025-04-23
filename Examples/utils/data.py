import numpy as np 
from scipy.linalg import lstsq

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def feature_scaling(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


def train_test_splitting(idx, test_size, stratify, seed: int = 42):

    # Train-test splitting 
    train_idx, test_idx = train_test_split(
        idx, stratify=stratify, test_size=test_size, random_state=seed
    )
    return train_idx, test_idx


def init_knots(duration, event, n_knots):
    
    # Cast to integer numpy array 
    event = np.array(event).astype(int)
    
    # Logarithm of time-to-event 
    log_t = np.log(duration)
    
    # Knot locations: Centiles of the distribution of **uncensored** log event times
    # - Boundary knots: placed at the 0th and 100th centiles (min and max values)
    # - Internal knots: internal knots are placed at the centiles between the min and max   
    centiles = np.linspace(0, 1, n_knots) * 100
    knots = np.percentile(log_t[event == 1], centiles)
    
    return knots 


def init_beta(X, y):

    # Fit a Cox PH model
    cox = CoxPHSurvivalAnalysis(alpha=0.01, tol=1e-6)
    cox.fit(X, y)

    # Estimated cumulative hazard conditional on covarites
    cumulative_hazards = cox.predict_cumulative_hazard_function(X, return_array=True)
    # Average cumulative hazard for each subject over observed times
    mean_hazard = np.clip(np.mean(cumulative_hazards, axis=1), 1e-16, 1e16)
    # Fit linear regression to log cumulative hazard
    beta, _, _, _ = lstsq(X, np.log(mean_hazard))
    # Expand with one dimension    
    return beta[None, :]


def init_gamma(D, duration):
    # Solve least squares problem
    gamma = np.linalg.inv(D.T @ D) @ D.T @ np.log(duration)
    # Expand with one dimension    
    return gamma[None, :]
