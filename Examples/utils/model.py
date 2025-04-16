from typing import Optional

import numpy as np 
import tensorflow as tf 

from .data import init_params_random, init_params_cox
from .splines import spline_design_matrix, spline_derivative_design_matrix


class Model:

    def __init__(self, epochs=3, alpha=0.01, l1_ratio=0, learning_rate=0.01, knots=None, n_knots: Optional[int] = 5):

        self.epochs = epochs
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.learning_rate = learning_rate
        self.knots = knots
        self.n_knots = n_knots
        
        self.D = None 
        self.D_prime = None 
        
        self.beta = None 
        self.gamma = None 
        self.losses = None
        
    def _check_init(self):
        # Logarithm of time-to-event 
        log_t = np.log(self.duration)
        
        if self.knots is None:
            # Sanity check
            assert self.n_knots is not None, "Should provide either knots or n_knots"            
            # Knot locations: Centiles of the distribution of **uncensored** log event times
            # - Boundary knots: placed at the 0th and 100th centiles (min and max values)
            # - Internal knots: internal knots are placed at the centiles between the min and max   
            centiles = np.linspace(0, 1, self.n_knots) * 100
            self.knots = np.percentile(log_t[self.event == 1], centiles)
            
        if self.D is None:
            self.D = spline_design_matrix(log_t, self.knots)
            
        if self.D_prime is None:
            self.D_prime = spline_derivative_design_matrix(log_t, self.knots)
                    
    def fit(self, X, y):
        
        # Unpack structured array 
        event, duration = zip(*y)
        # Cast to ndarray 
        self.event = np.array(event).astype(int)
        self.duration = np.array(duration).astype(float)
        
        self._check_init()
    
        if self.beta is None and self.gamma is None:
            #self.beta, self.gamma = init_params_random(self.D, X, self.duration, self.event)
            self.beta, self.gamma = init_params_cox(self.D, X, self.duration, self.event)

        def _neg_log_likelihood():
            # Linear terms
            phi = D @ tf.transpose(gamma) + X @ tf.transpose(beta)
            # Spline derivatives 
            ds_dt = tf.clip_by_value(D_prime @ tf.transpose(gamma), 1e-8, 1e8) 
            # Log-likelihood for each data sample (N x 1)
            log_likelihood = event * (phi + tf.math.log(ds_dt)) - tf.math.exp(phi)
            # Regularization
            reg_gamma = self.alpha * tf.norm(gamma, ord=2)
            reg_beta_l1 = self.l1_ratio * self.alpha * tf.norm(beta, ord=1)
            reg_beta_l2 = (1 - self.l1_ratio) * self.alpha * tf.norm(beta, ord=2)
            regularisers = reg_gamma + reg_beta_l1 + reg_beta_l2
            # Sum negative log-likelihood over data samples 
            return -1.0 * tf.reduce_sum(log_likelihood, axis=0) + regularisers
        
        # Optimization variables 
        beta = tf.Variable(self.beta, dtype=tf.float32)
        gamma = tf.Variable(self.gamma, dtype=tf.float32)
        
        # Cast to TF 
        X = tf.cast(X, dtype=tf.float32)
        D = tf.cast(self.D, dtype=tf.float32)
        D_prime = tf.cast(self.D_prime, dtype=tf.float32)
        event = tf.cast(self.event[:, None], dtype=tf.float32)
        
        self.losses = []
        optimiser = tf.keras.optimizers.Adam(learning_rate=float(self.learning_rate))
        
        for _ in tf.range(self.epochs):
            with tf.GradientTape() as tape:
                loss_value = _neg_log_likelihood()

            # Compute gradients
            gradients = tape.gradient(loss_value, [beta, gamma])
            # Apply gradients to update parameters 
            optimiser.apply_gradients(zip(gradients, [beta, gamma]))
            
            self.losses.append(float(loss_value))
            
        self.beta = beta.numpy()
        self.gamma = gamma.numpy()
        
    def risk_score(self, X):
        """
        Prognostic index.
        """
        # The linear predictor
        return X @ np.transpose(self.beta)
    
    def hazard(self, X, times):
        """
        Hazard at a given time.
        """
        # Create one spline equation per time point 
        D = spline_design_matrix(np.log(times), self.knots)
        # Create one spline equation per time point 
        D_prime = spline_derivative_design_matrix(np.log(times), self.knots)
        # Linear term
        phi = D @ np.transpose(self.gamma) + X @ np.transpose(self.beta)
        # Spline derivatives 
        ds_dt = D_prime @ np.transpose(self.gamma)
        # Transform to the hazard scale 
        return ds_dt * np.exp(phi)
    
    def baseline_survival(self, times):
        # Create one spline equation per time point 
        D = spline_design_matrix(np.log(times), self.knots)
        #return D @ np.transpose(self.gamma)
        return np.exp(-1.0 * np.exp(D @ np.transpose(self.gamma)))
    
    def survival_curve(self, X, times):
        """
        Survival probability over time.
        """
        # Create one spline equation per time point 
        D = spline_design_matrix(np.log(times), self.knots)
        # Linear term
        phi = D @ np.transpose(self.gamma) + X @ np.transpose(self.beta)
        # Transform to the survival scale 
        return np.exp(-1.0 * np.exp(phi))
        
    def set_params(self, params: dict):
        self.beta = params["beta"]
        self.gamma = params["gamma"]
        
    def get_params(self) -> dict:
        params = {"beta": self.beta, "gamma": self.gamma}
        return params 
