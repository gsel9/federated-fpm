import numpy as np 
import tensorflow as tf 
from sksurv.metrics import concordance_index_censored

#from .splines import spline_design_matrix, spline_derivative_design_matrix
from .splines import bspline_design_matrix, bspline_derivative_design_matrix


class Model:
    
    def __init__(self, epochs=3, l2_lambda=0.01, learning_rate=0.01, knots=None):

        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.learning_rate = learning_rate
        self.knots = knots
        
        self.D = None 
        self.D_prime = None 
        
        self.beta = None 
        self.gamma = None 
        self.losses = None
        self.c_scores = None
           
    def fit(self, X, y):
        
        # Unpack structured array 
        event, duration = zip(*y)

        # Cast to ndarray 
        self.event = np.array(event).astype(int)
        self.duration = np.array(duration).astype(float)
        
        if self.D is None:
            #self.D = spline_design_matrix(np.log(self.duration), self.knots)
            self.D = bspline_design_matrix(np.log(self.duration), self.knots)
            
        if self.D_prime is None:
            #self.D_prime = spline_derivative_design_matrix(np.log(self.duration), self.knots)
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
        
        # Track training performance         
        self.losses, self.c_scores = [], []
        # Optimization algorithm 
        optimiser = tf.keras.optimizers.Adam(learning_rate=float(self.learning_rate))
        
        for _ in tf.range(self.epochs):
            with tf.GradientTape() as tape:
                # Negative log-likelihood 
                loss_value = _neg_log_likelihood()

            # Derive gradients
            gradients = tape.gradient(loss_value, [beta, gamma])
            # Apply gradients to update parameters 
            optimiser.apply_gradients(zip(gradients, [beta, gamma]))
            
            # Track negative log-likelihood as loss  
            self.losses.append(float(loss_value))
            
            # TODO: 
            # Track risk estimation performance 
            #self.c_scores.append(self.score(X, self.event, self.duration))
        
        # Update parameter estimates 
        self.beta = beta.numpy()
        self.gamma = gamma.numpy()
        
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
        #D = spline_design_matrix(np.log(times), self.knots)
        D = bspline_design_matrix(np.log(times), self.knots)
        
        # Linear term
        phi = D @ np.transpose(self.gamma) + X @ np.transpose(self.beta)
        # Transform to the survival scale 
        return np.exp(-1.0 * np.exp(phi))
    
    def set_params(self, params: dict):
        for key, value in params.items():
            setattr(self, key, value)
        
    def get_params(self) -> dict:
        params = {"beta": self.beta, "gamma": self.gamma}
        return params 
