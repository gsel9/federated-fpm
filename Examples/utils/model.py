from typing import Optional
import numpy as np 
import tensorflow as tf 
from sksurv.metrics import concordance_index_censored

#from .splines import spline_design_matrix, spline_derivative_design_matrix
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
        beta_prev = np.zeros_like(self.beta) #.copy()
        gamma_prev = np.zeros_like(self.gamma) #.copy()
        
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
        #return np.linalg.norm(params - params_other) / (np.linalg.norm(params) + 1e-12) <= tol

    def fit_fedadmm(
        self, X, y, z_beta, z_gamma, u_beta, u_gamma, tol: Optional[float] = None
    ):
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
        beta_prev = self.beta.copy()
        
        # Cast to TF 
        X = tf.cast(X, dtype=tf.float32)
        D = tf.cast(self.D, dtype=tf.float32)
        D_prime = tf.cast(self.D_prime, dtype=tf.float32)
        
        # Global parameters 
        z_beta = tf.cast(z_beta, dtype=tf.float32)
        z_gamma = tf.cast(z_gamma, dtype=tf.float32)
        
        # Local dual variables 
        u_beta = tf.cast(u_beta, dtype=tf.float32)
        u_gamma = tf.cast(u_gamma, dtype=tf.float32)

        # NOTE: Expanding dim
        event = tf.cast(self.event[:, None], dtype=tf.float32)
        
        def _augmented_lagrangian_loss():
            # Linear terms
            phi = D @ tf.transpose(gamma) + X @ tf.transpose(beta)
            ds_dt = tf.clip_by_value(D_prime @ tf.transpose(gamma), 1e-8, 1e8) 
            log_likelihood = event * (phi + tf.math.log(ds_dt)) - tf.math.exp(phi)
            nll = -1.0 * tf.reduce_sum(log_likelihood, axis=0)

            # Regularization
            reg_beta = self.l2_lambda * tf.norm(beta, ord=2)
            reg_gamma = self.l2_lambda * tf.norm(gamma, ord=2)
            
            # Completing the square for the augmented Lagrangian of consensus ADMM
            aug_beta = 0.5 * self.rho * tf.reduce_sum(tf.square(beta - z_beta + u_beta / self.rho))
            aug_gamma = 0.5 * self.rho * tf.reduce_sum(tf.square(gamma - z_gamma + u_gamma / self.rho))
            
            return nll + reg_beta + reg_gamma + aug_beta + aug_gamma

        # Track training performance         
        self.losses, self.c_scores = [], []
        # Optimization algorithm 
        optimiser = tf.keras.optimizers.Adam(learning_rate=float(self.learning_rate))

        for i in tf.range(self.epochs):
            with tf.GradientTape() as tape:
                loss_value = _augmented_lagrangian_loss()

            gradients = tape.gradient(loss_value, [beta, gamma])
            optimiser.apply_gradients(zip(gradients, [beta, gamma]))
            
            # Track negative log-likelihood as loss  
            self.losses.append(float(loss_value))
            
            if tol is not None:
                if self.has_converged(beta_prev, beta.numpy()):
                    print(f"Converged after {i} iterations")
                    break 

        # Update parameter estimates 
        self.beta = beta.numpy()
        self.gamma = gamma.numpy()
        
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
        
    def loss_fedadmm(self, X, y, z_beta, z_gamma, u_beta, u_gamma):
        
        # Unpack structured array 
        event, _ = zip(*y)
        # Cast to ndarray 
        event = np.array(event).astype(int)[:, None]
        
        def _augmented_lagrangian_loss():
            # Linear terms
            phi = self.D @ np.transpose(self.gamma) + X @ np.transpose(self.beta)
            ds_dt = np.clip(self.D_prime @ np.transpose(self.gamma), 1e-8, 1e8) 
            log_likelihood = event * (phi + np.log(ds_dt)) - np.exp(phi)
            nll = -1.0 * np.sum(log_likelihood, axis=0)

            # Regularization
            reg_beta = self.l2_lambda * np.linalg.norm(self.beta, ord=2)
            reg_gamma = self.l2_lambda * np.linalg.norm(self.gamma, ord=2)
            
            # Completing the square for the augmented Lagrangian of consensus ADMM
            aug_beta = 0.5 * self.rho * np.sum(np.square(self.beta - z_beta + u_beta / self.rho))
            aug_gamma = 0.5 * self.rho * np.sum(np.square(self.gamma - z_gamma + u_gamma / self.rho))
            
            return nll + reg_beta + reg_gamma + aug_beta + aug_gamma

        return float(_augmented_lagrangian_loss())
    
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
    
    def score(self, X, event, duration):
        # Estimate risk score
        risk = self.risk_score(X).squeeze()
        # Get C-index
        score, _, _, _ = concordance_index_censored(event.astype(bool), duration, risk)
    
        return score
        
    def risk_score(self, X):
        """
        Prognostic index.
        """
        # The linear predictor
        return X @ np.transpose(self.beta)

    def survival_curve(self, X, times):
        """
        Survival probability over time.
        """
        # Create one spline equation per time point 
        D = bspline_design_matrix(np.log(times), self.knots)
        # Linear term
        phi = D @ np.transpose(self.gamma) + X @ np.transpose(self.beta)
        # Transform to the survival scale 
        return np.exp(-1.0 * np.exp(phi))
    
    def hazard(self, X, times):
        """
        """
        # Create one spline equation per time point 
        D = bspline_design_matrix(np.log(times), self.knots)
        # Linear term
        phi = D @ np.transpose(self.gamma) + X @ np.transpose(self.beta)
        # Create one differentiated spline equation per time point 
        D_prime = bspline_derivative_design_matrix(np.log(times), self.knots)
        # Spline derivative 
        ds_dt = D_prime @ np.transpose(self.gamma)
        # Transform to the hazard scale 
        return ds_dt * np.exp(phi)
    
    def set_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)
        
    def get_params(self) -> dict:
        params = {"beta": self.beta, "gamma": self.gamma}
        return params 
