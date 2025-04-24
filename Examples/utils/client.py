import numpy as np 
from sksurv.util import Surv

# Local imports 
from .model import Model 
#from .splines import spline_design_matrix
from .splines import bspline_design_matrix
from .data import (
    feature_scaling, train_test_splitting, init_knots, init_beta, init_gamma
)


class Client:

    def __init__(self, data, n_knots, n_epochs, event_col, duration_col):
        self.data = data 
        self.n_knots = n_knots
        self.n_epochs = n_epochs
        self.event_col = event_col
        self.duration_col = duration_col 

        # Client model  
        self.model = None 
        
        # Controls parameter initialization         
        self._is_first_call = True 
        
    def preprocess_data(self):
        
        # Indices for training and test sets
        train_idx, test_idx = train_test_splitting(
            np.arange(self.data.shape[0]),
            test_size=0.25, 
            stratify=self.data[self.event_col].squeeze().astype(int)
        )
        # Cast feature matrix to numpy 
        X = self.data.drop(columns=[self.event_col, self.duration_col]).to_numpy()
        
        # Scale training and test data
        self.X_train, self.X_test = feature_scaling(X[train_idx], X[test_idx])
        
        # Create structured array
        y = Surv.from_arrays(
            event=self.data[self.event_col].to_numpy().squeeze(), 
            time=self.data[self.duration_col].to_numpy().squeeze()
        )
        # Split structured array
        self.y_train = y[train_idx]
        self.y_test = y[test_idx]
        
    def init_model(self):
        # Unpack structured array 
        event, duration = zip(*self.y_train)
        
        # Set knot locations 
        knots = init_knots(duration, event, self.n_knots)
        
        # Create one spline equation per time point 
        #D = spline_design_matrix(np.log(duration), knots)
        D = bspline_design_matrix(np.log(duration), knots)
        # Initialize gamma coefficients
        gamma = init_gamma(D, duration)
        #gamma = init_gamma(D, self.X_train, duration)
        
        # Initialize beta coefficients
        beta = init_beta(self.X_train, self.y_train)
        
        # Initialize FPM   
        self.model = Model(
            epochs=self.n_epochs, knots=knots, learning_rate=0.01, l2_lambda=10
        )
        # Update model parameters 
        self.model.set_params({"beta": beta, "gamma": gamma})
            
    def fit_model(self):
        # Fit model 
        self.model.fit(self.X_train, self.y_train)

    def set_params(self, params: dict):
        self.model.set_params(params)
        
    def get_params(self) -> dict:
        return self.model.get_params()
    
    def risk_score(self, X):
        return self.model.risk_score(X)
    
    def survival_curve(self, X, times):
        return self.model.survival_curve(X, times)
        