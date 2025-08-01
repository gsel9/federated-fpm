from typing import Optional
import numpy as np 
import tensorflow as tf 
from scipy.stats import wasserstein_distance

from .data import init_knots
from .splines import bspline_design_matrix, bspline_derivative_design_matrix

tf.random.set_seed(42)


class Model:
    
    def __init__(self, epochs=3, l2_lambda=0.01, learning_rate=0.01, knots=None, rho=1):

        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.knots = knots
        self.rho = rho / 2.0
        
        self.D = None 
        self.D_prime = None 
        
        self.beta = None 
        self.gamma = None 
        self.losses = None
        self.c_scores = None
           
    def fit(self, X, y, tol: Optional[float] = None):
        
        # Unpack structured array 
        event, duration = zip(*y)
        # Cast to ndarray 
        self.event = np.array(event).astype(int)
        self.duration = np.array(duration).astype(float)
        
        if self.D is None:
            self.D = bspline_design_matrix(np.log(self.duration), self.knots)
            
        if self.D_prime is None:
            self.D_prime = bspline_derivative_design_matrix(np.log(self.duration), self.knots)
        
        # Optimization variables 
        beta = tf.Variable(self.beta, dtype=tf.float32)
        gamma = tf.Variable(self.gamma, dtype=tf.float32)
        
        # For convergence  
        beta_prev = np.zeros_like(self.beta) 
        gamma_prev = np.zeros_like(self.gamma)
        
        # Cast to TF 
        X = tf.cast(X, dtype=tf.float32)
        D = tf.cast(self.D, dtype=tf.float32)
        D_prime = tf.cast(self.D_prime, dtype=tf.float32)
        event = tf.cast(self.event[:, None], dtype=tf.float32)
        
        def _neg_log_likelihood():
            # Linear terms
            phi = D @ tf.transpose(gamma) + X @ tf.transpose(beta)
            # Spline derivatives 
            ds_dt = tf.clip_by_value(D_prime @ tf.transpose(gamma), 1e-8, 1e8) 
            # Log-likelihood for each data sample (N x 1)
            log_likelihood = event * (phi + tf.math.log(ds_dt)) - tf.math.exp(phi)
            # Sum negative log-likelihood over data samples 
            nll = -1.0 * tf.reduce_sum(log_likelihood, axis=0)

            # Parameter regularization
            reg_beta = self.l2_lambda * tf.norm(beta, ord=2)
            reg_gamma = self.l2_lambda * tf.norm(gamma, ord=2)

            return nll + reg_gamma + reg_beta
        
        # Track training performance         
        self.losses, self.c_scores = [], []
        # Optimization algorithm 
        optimiser = tf.keras.optimizers.Adam(learning_rate=float(self.learning_rate))
        
        for i in tf.range(self.epochs):
            with tf.GradientTape() as tape:
                # Negative log-likelihood 
                loss_value = _neg_log_likelihood()

            # Derive gradients
            gradients = tape.gradient(loss_value, [beta, gamma])
            # Apply gradients to update parameters 
            optimiser.apply_gradients(zip(gradients, [beta, gamma]))
            
            # Track negative log-likelihood as loss  
            self.losses.append(float(loss_value))
            
            if tol is not None:
                if (
                    self.has_converged(beta_prev, beta.numpy(), tol) and 
                    self.has_converged(gamma_prev, gamma.numpy(), tol)
                ):
                    print(f"Converged after {i} iterations")
                    break 
                                      
                beta_prev = beta.numpy() 
                gamma_prev = gamma.numpy()
            
        # Update parameter estimates 
        self.beta = beta.numpy()
        self.gamma = gamma.numpy()
    
    @staticmethod
    def has_converged(params, params_other, tol):
        return np.linalg.norm(params - params_other) <= tol 
 
    def gradients(self, X, y):
        # Unpack structured array 
        event, duration = zip(*y)

        # Cast to ndarray 
        self.event = np.array(event).astype(int)
        self.duration = np.array(duration).astype(float)
        
        if self.D is None:
            self.D = bspline_design_matrix(np.log(self.duration), self.knots)
            
        if self.D_prime is None:
            self.D_prime = bspline_derivative_design_matrix(np.log(self.duration), self.knots)
        
        # Optimization variables 
        beta = tf.Variable(self.beta, dtype=tf.float32)
        gamma = tf.Variable(self.gamma, dtype=tf.float32)
        
        # Cast to TF 
        X = tf.cast(X, dtype=tf.float32)
        D = tf.cast(self.D, dtype=tf.float32)
        D_prime = tf.cast(self.D_prime, dtype=tf.float32)
        event = tf.cast(self.event[:, None], dtype=tf.float32)
        
        def _neg_log_likelihood():
            # Linear terms
            phi = D @ tf.transpose(gamma) + X @ tf.transpose(beta)
            # Spline derivatives 
            ds_dt = tf.clip_by_value(D_prime @ tf.transpose(gamma), 1e-8, 1e8) 
            # Log-likelihood for each data sample (N x 1)
            log_likelihood = event * (phi + tf.math.log(ds_dt)) - tf.math.exp(phi)
            # Sum negative log-likelihood over data samples 
            nll = -1.0 * tf.reduce_sum(log_likelihood, axis=0)

            # Parameter regularization
            reg_beta = self.l2_lambda * tf.norm(beta, ord=2)
            reg_gamma = self.l2_lambda * tf.norm(gamma, ord=2)

            return nll + reg_gamma + reg_beta

        with tf.GradientTape() as tape:
            # Negative log-likelihood 
            loss_value = _neg_log_likelihood()

        # Derive gradients
        gradients = tape.gradient(loss_value, [beta, gamma])
        return gradients
    
    def gradients_adjusted(self, X, y, q_scale):
        # Unpack structured array 
        event, duration = zip(*y)

        # Cast to ndarray 
        self.event = np.array(event).astype(int)
        self.duration = np.array(duration).astype(float)
        
        if self.D is None:
            self.D = bspline_design_matrix(np.log(self.duration), self.knots)
            
        if self.D_prime is None:
            self.D_prime = bspline_derivative_design_matrix(np.log(self.duration), self.knots)
        
        # Optimization variables 
        beta = tf.Variable(self.beta, dtype=tf.float32)
        gamma = tf.Variable(self.gamma, dtype=tf.float32)
        
        # Location of local knots to compare with global knot positions 
        knots_local = init_knots(duration, event, n_knots=5)
        # Gradient correction term 
        delta = q_scale * wasserstein_distance(self.knots, knots_local)
        delta = tf.cast(delta, dtype=tf.float32)
        
        # Cast to TF 
        X = tf.cast(X, dtype=tf.float32)
        D = tf.cast(self.D, dtype=tf.float32)
        D_prime = tf.cast(self.D_prime, dtype=tf.float32)
        event = tf.cast(self.event[:, None], dtype=tf.float32)
        
        def _neg_log_likelihood():
            # Linear terms
            phi = D @ tf.transpose(gamma) + X @ tf.transpose(beta)
            # Spline derivatives 
            ds_dt = tf.clip_by_value(D_prime @ tf.transpose(gamma), 1e-8, 1e8) 
            # Log-likelihood for each data sample (N x 1)
            log_likelihood = event * (phi + tf.math.log(ds_dt)) - tf.math.exp(phi)
            # Sum negative log-likelihood over data samples 
            nll = -1.0 * tf.reduce_sum(log_likelihood, axis=0)

            # Parameter regularization
            reg_beta = self.l2_lambda * tf.norm(beta, ord=2)
            reg_gamma = self.l2_lambda * tf.norm(gamma, ord=2)

            return nll + reg_gamma + reg_beta #+ delta

        with tf.GradientTape() as tape:
            # Negative log-likelihood 
            loss_value = _neg_log_likelihood()

        # Derive gradients
        gradients = tape.gradient(loss_value, [beta, gamma])
        return gradients
    
    def loss(self, X, y):
        
        # Unpack structured array 
        event, _ = zip(*y)
        # Cast to ndarray 
        event = np.array(event).astype(int)[:, None]
        
        def _loss():
            # Linear terms
            phi = self.D @ np.transpose(self.gamma) + X @ np.transpose(self.beta)
            ds_dt = np.clip(self.D_prime @ np.transpose(self.gamma), 1e-8, 1e8) 
            log_likelihood = event * (phi + np.log(ds_dt)) - np.exp(phi)
            nll = -1.0 * np.sum(log_likelihood, axis=0)

            # Regularization
            reg_beta = self.l2_lambda * np.linalg.norm(self.beta, ord=2)
            reg_gamma = self.l2_lambda * np.linalg.norm(self.gamma, ord=2)
          
            return nll + reg_beta + reg_gamma  

        return float(_loss())
    
    def set_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)
        
    def get_params(self) -> dict:
        params = {"beta": self.beta, "gamma": self.gamma}
        return params 
