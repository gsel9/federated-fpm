from sksurv.metrics import concordance_index_censored

# Local imports 
from utils.data import feature_scaling, init_params_random, init_params_cox
from utils.splines import spline_design_matrix, spline_derivative_design_matrix



class Client:

    def __init__(self, cid, data, times, delta):

        self.cid = cid
        self.data = data 
        self.times = times
        self.delta = delta 

    def set_params(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma 

    def _train_test_split(self, test_size=0.25):

        # Train-test splitting 
        train_idx, test_idx = train_test_split(
            np.arange(self.data.shape[0]), 
            test_size=test_size, 
            random_state=42, 
            stratify=self.delta.squeeze().astype(int)
        )
        return train_idx, test_idx

    def train_test_split(self):

        # Create training and test splits 
        train_idx, test_idx = self._train_test_split()

        # Cast to <numpy>
        X = self.data.to_numpy().copy()
        
        # Split data
        self.d_train, self.d_test = self.delta[train_idx], self.delta[test_idx]
        self.t_train, self.t_test = self.times[train_idx], self.times[test_idx]

        # Scale training and test data
        self.X_train, self.X_test = feature_scaling(X[train_idx], X[test_idx])

    def initialize_params(self, knots):
        
        # Spline design matrices of log-time 
        self.D, self.dDdt = create_splines(log_t=np.log(self.t_train.squeeze()), knots=knots)

        # Initialize model parameters 
        self.beta, self.gamma = init_beta_gamma(self.D, self.X_train, y_train)

    def loss(self):
        phi = self.D @ self.gamma.T + self.X_train @ self.beta.T
    
        uncensored = np.exp(phi - np.exp(phi)) * (self.dDdt @ self.gamma.T)
        censored = np.exp(-1.0 * np.exp(phi))
        
        return float(np.sum(self.d_train * uncensored + (1 - self.d_train) * censored, axis=0))

    def gradient_gamma(self):
    
        phi = self.D @ self.gamma.T + self.X_train @ self.beta.T
        dsdt = self.dDdt @ self.gamma.T
        
        return (np.exp(phi) - self.d_train).T @ self.D - (self.d_train / dsdt).T @ self.dDdt

    def gradient_beta(self):
        phi = self.D @ self.gamma.T + self.X_train @ self.beta.T
        return (np.exp(phi) - self.d_train).T @ self.X_train

    def train_steps(self, local_steps=1, learning_rate=0.01):
        
        for i in range(local_steps):
            # Gradient descent steps 
            self.beta -= learning_rate * self.gradient_beta()
            self.gamma -= learning_rate * self.gradient_gamma()

    def c_score(self, beta=None):
        
        if beta is None:
            beta = self.beta 
    
        train_score, _, _, _, _ = concordance_index_censored(
            self.d_train.squeeze(), 
            np.log(self.t_train.squeeze()), 
            (self.X_train @ beta.T).squeeze()
        )
        test_score, _, _, _, _ = concordance_index_censored(
            self.d_test.squeeze(), 
            np.log(self.t_test.squeeze()), 
            (self.X_test @ beta.T).squeeze()
        )
        return train_score, test_score