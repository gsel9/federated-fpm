import numpy as np 
from scipy.interpolate import BSpline


def relu(x):
    return max(x, 0)


def spline_basis(x, k_j, k_min, k_max, derivative=False):
    """Computes the basis function S(x; k_j)."""
    # Scaling coefficient 
    phi_j = (k_max - k_j) / (k_max - k_min)

    if derivative:
        # Derivative of the spline basis function
        term1 = relu(x - k_j) ** 2
        term2 = phi_j * relu(x - k_min) ** 2
        term3 = (1 - phi_j) * relu(x - k_max) ** 2
        return 3 * (term1 - (term2 + term3))

    # Spline basis function 
    term1 = relu(x - k_j) ** 3
    term2 = phi_j * relu(x - k_min) ** 3
    term3 = (1 - phi_j) * relu(x - k_max) ** 3
    return term1 - (term2 + term3)


def spline_vector(ln_t, knots):
    """Computes the spline function s(x; \gamma, k)."""
    # Boundary knots
    k_min, k_max = knots[0], knots[-1]
    # Construct basis functions over internal knots 
    basis = [spline_basis(ln_t, k_j, k_min, k_max) for k_j in knots[1:-1]]
    # Design matrix 
    return np.array([ln_t] + basis)


def spline_derivative_vector(ln_t, knots):
    """Computes the spline function s(x; \gamma, k)."""
    # Boundary knots
    k_min, k_max = knots[0], knots[-1]
    # Construct basis functions over internal knots 
    basis = [spline_basis(ln_t, k_j, k_min, k_max, derivative=True) for k_j in knots[1:-1]]
    # Design matrix 
    return 1 / np.exp(ln_t) * np.array([1] + basis)


def spline_design_matrix(log_t, knots):
    D = []
    for log_time in log_t:
        D.append(spline_vector(log_time, knots))
    # Cast to <numpy.ndarray>
    return np.array(D)


def spline_derivative_design_matrix(log_t, knots):
    dDdt = []
    for log_time in log_t:
        dDdt.append(spline_derivative_vector(log_time, knots))
    # Cast to <numpy.ndarray>
    return np.array(dDdt)


def bspline_design_matrix(log_t, knots, degree=3):
    
    # Padding boundary knots
    t = np.concatenate((
        [log_t.min()] * (degree + 1), knots[1:-1], [log_t.max()] * (degree + 1)
    ))

    n_bases = len(t) - (degree + 1)
    design_matrix = np.zeros((len(log_t), n_bases))

    for i in range(n_bases):
        c = np.zeros(n_bases)
        c[i] = 1
        spline = BSpline(t, c, degree)
        design_matrix[:, i] = spline(log_t)

    design_matrix = np.array(design_matrix)
    
    return design_matrix
    
    
def bspline_derivative_design_matrix(log_t, knots, degree=3):

    # Padding boundary knots
    t = np.concatenate((
        [log_t.min()] * (degree + 1), knots[1:-1], [log_t.max()] * (degree + 1)
    ))

    n_bases = len(t) - (degree + 1)
    deriv_matrix = np.zeros((len(log_t), n_bases))

    for i in range(n_bases):
        c = np.zeros(n_bases)
        c[i] = 1
        basis = BSpline(t, c, degree)
        basis_deriv = basis.derivative()
        deriv_matrix[:, i] = basis_deriv(log_t)

    deriv_matrix = np.array(deriv_matrix)
    
    return deriv_matrix