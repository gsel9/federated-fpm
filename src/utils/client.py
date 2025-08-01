import numpy as np 
from sksurv.util import Surv

# Local imports 
from .model import Model 
from .splines import bspline_design_matrix
from .data import (
    feature_scaling, train_test_splitting, init_knots, init_beta, init_gamma
)


class Client:

    def __init__(self, data, n_knots, n_epochs, event_col, duration_col, rho=1):
        self.data = data 
        self.n_knots = n_knots
        self.n_epochs = n_epochs
        self.event_col = event_col
        self.duration_col = duration_col 
        self.rho = rho 

        # Client model  
        self.model = None 
        
    @property 
    def loss(self):
        # Final loss value 
        return float(self.model.losses[-1])
        
    @property
    def beta(self):
        return self.model.beta 
    
    @property
    def gamma(self):
        return self.model.gamma 
    
    def preprocess_data(self, train_test_split: bool):
        
        # Cast feature matrix to numpy 
        X = self.data.drop(columns=[self.event_col, self.duration_col]).to_numpy()
        
        # Create structured array
        y = Surv.from_arrays(
            event=self.data[self.event_col].to_numpy().squeeze(), 
            time=self.data[self.duration_col].to_numpy().squeeze()
        )
        if train_test_split:
            # Indices for training and test sets
            train_idx, test_idx = train_test_splitting(
                np.arange(self.data.shape[0]),
                test_size=0.2, 
                stratify=self.data[self.event_col].squeeze().astype(int)
            )
            
            # Scale training and test data
            self.X_train, self.X_test = feature_scaling(X[train_idx], X[test_idx])
        
            # Split structured array
            self.y_train = y[train_idx]
            self.y_test = y[test_idx]
        else:
            # Scale training  
            self.X_train = feature_scaling(X)
            # Split structured array
            self.y_train = y 
    
    def init_model(self, local_knots: bool, knots=None, learning_rate=0.01, l2_lambda=1):
        # Unpack structured array 
        event, duration = zip(*self.y_train)
        
        if local_knots:
            # Set knot locations 
            knots = init_knots(duration, event, self.n_knots)
        
        # Create one spline equation per time point 
        D = bspline_design_matrix(np.log(duration), knots)
        # Initialize gamma coefficients
        gamma = init_gamma(D, duration)
        
        # Initialize beta coefficients
        beta = init_beta(self.X_train, self.y_train)
    
        # Initialize FPM   
        self.model = Model(
            epochs=self.n_epochs, 
            knots=knots, 
            learning_rate=learning_rate, 
            l2_lambda=l2_lambda, 
            rho=self.rho
        )
        # Update model parameters 
        self.model.set_params({"beta": beta, "gamma": gamma})
        
        # Dual variables 
        self.u_beta = np.zeros_like(beta)
        self.u_gamma = np.zeros_like(gamma)
            
    def fit_model(self, z_beta, z_gamma, tol=None):
        # Update model parameters 
        self.model.set_params({"beta": z_beta, "gamma": z_gamma})
        # Fit model 
        self.model.fit(self.X_train, self.y_train, tol=tol)
        
    def gradients(self, z_beta, z_gamma):
        # Update model parameters 
        self.model.set_params({"beta": z_beta, "gamma": z_gamma})
        # Single update step 
        grads = self.model.gradients(self.X_train, self.y_train)
        return grads
    
    def model_loss(self):
        return self.model.loss(self.X_train, self.y_train)
        
    def set_params(self, params: dict):
        self.model.set_params(params)
        
    def get_params(self) -> dict:
        return self.model.get_params()
    