import numpy as np 
from scipy.interpolate import BSpline


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