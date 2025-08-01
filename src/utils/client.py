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
    
    # TODO: Upon param init, do one round of server update (after init model) to init 
    # all clients with the exact same starting beta and gamma 
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
    
    def fit_model_fedadmm(self, z_beta, z_gamma):
        # Fit model 
        self.model.fit_fedadmm(
            self.X_train, self.y_train, z_beta, z_gamma, self.u_beta, self.u_gamma
        )
        
    def gradients(self, z_beta, z_gamma):
        # Update model parameters 
        self.model.set_params({"beta": z_beta, "gamma": z_gamma})
        # Single update step 
        grads = self.model.gradients(self.X_train, self.y_train)
        return grads
    
    def gradients_adjusted(self, z_beta, z_gamma, q_scale):
        # Update model parameters 
        self.model.set_params({"beta": z_beta, "gamma": z_gamma})
        # Single update step 
        grads = self.model.gradients_adjusted(self.X_train, self.y_train, q_scale)
        return grads
    
    def gradients_constrained(self, z_beta, z_gamma):
        # Update model parameters 
        self.model.set_params({"beta": z_beta, "gamma": z_gamma})
        # Single update step 
        grads = self.model.gradients_constrained(self.X_train, self.y_train)
        return grads
    
    def gradients_iterative(self, beta_global, gamma_global, epochs, tol=None):
        # Update model parameters 
        self.model.set_params({"beta": beta_global, "gamma": gamma_global})
        # Fitting steps 
        grads = self.model.gradients_iterative(self.X_train, self.y_train, epochs, tol=tol)
        return grads 
    
    def gradients_fedadmm(self, z_beta, z_gamma):
        grads = self.model.gradients_fedadmm(
            self.X_train, self.y_train, z_beta, z_gamma, self.u_beta, self.u_gamma
        )
        return grads
    
    def model_loss_fedadmm(self, z_beta, z_gamma):
        return self.model.loss_fedadmm(
            self.X_train, self.y_train, z_beta, z_gamma, self.u_beta, self.u_gamma
        )
    
    def model_loss(self):
        return self.model.loss(self.X_train, self.y_train)
        
    def update_duals(self, z_beta, z_gamma):
        # Update dual variables 
        self.u_beta += self.rho * (self.model.beta - z_beta)
        self.u_gamma += self.rho * (self.model.gamma - z_gamma)
        
    def set_params(self, params: dict):
        self.model.set_params(params)
        
    def get_params(self) -> dict:
        return self.model.get_params()
    
    def risk_score(self, X):
        return self.model.risk_score(X)
    
    def survival_curve(self, X, times):
        return self.model.survival_curve(X, times)
        