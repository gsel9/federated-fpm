import numpy as np 


def knots(logtime, delta):

    # knot locations are at the centiles of the distribution of *uncensored* log event times
    knots_x = np.linspace(0, 1, 6)
    knots_y = np.ones(len(knots_x)) * float(np.nan)

    # uncensored event times 
    logtime_uncens = logtime[delta == 0]

    # interior points
    for i in range(1, len(knots_x) - 1):
        knots_y[i] = np.quantile(logtime_uncens, knots_x[i])

    # boundaries
    knots_y[0] = min(logtime_uncens)
    knots_y[-1] = max(logtime_uncens)

    return knots_x, knots_y

    
class NaturalCubicSpline:
    
    def __init__(self, order, knots, intercept=True):
        
        self.order = order
        self.knots = np.array(knots)
        self.intercept = intercept

        self.K = order + int(self.intercept)

    @property
    def n_knots(self):
        return len(self.knots)

    def fit(self, X, y, **fit_params):
        return self
    
    def transform(self, X, derivative=False, **transform_params):
        
        if derivative:
            return self._transform_derivative(X.squeeze())
            
        return self._transform(X.squeeze())

    def v_basis(self, j, X, power=3):
        
        g = lambda t: np.maximum(0, t) ** power
        l = (self.knots[-1] - self.knots[j]) / (self.knots[-1] - self.knots[0])
        
        return g(X - self.knots[j]) - l * g(X - self.knots[0]) - (1 - l) * g(X - self.knots[-1])
    
    def _transform(self, X, **transform_params):
        
        X_spl = np.zeros((X.shape[0], self.K + self.n_knots - 1))  
        
        # polynomial terms
        if self.intercept:

            for p in range(self.K):
                X_spl[:, p] = X ** p
        else:
            for p in range(self.K):
                X_spl[:, p] = X ** (p + 1)
            
        # basis expansion 
        for j in range(0, self.n_knots - 1):
            X_spl[:, j + self.K] = self.v_basis(j, X)
            
        return X_spl
    
    def derivative_v_basis(self, j, X, power=3):
        
        h = lambda t: np.maximum(0, t) ** (power - 1)
        m = (self.knots[-1] - self.knots[j]) / (self.knots[-1] - self.knots[0])
        
        return power * h(X - self.knots[j]) - power * m * h(X - self.knots[0]) - power * (1 - m) * h(X - self.knots[-1])
                    
    def _transform_derivative(self, X, **transform_params):
        # derivative of spline wrt. the covaraites (ie not gamma coefficients)
        
        X_spl = np.zeros((X.shape[0], self.K + self.n_knots - 2))  
        
        # derivative of polynomial terms
        for p in range(1, self.K):
            X_spl[:, p-1] = p * X ** (p - 1)
    
        # derivative of basis expansion 
        for j in range(0, self.n_knots - 1):
            X_spl[:, j + self.K - 1] = self.derivative_v_basis(j, X)
            
        return X_spl