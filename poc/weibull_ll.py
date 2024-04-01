# This file replaces the stata ado file for the likelihood 
import jax.numpy as jnp   
# a set of utility programs for mlad
import mladutil as mu


# NOTE: the fuction name must always be `python_ll`
def logl(beta, X, wt, M):
    """TODO

    Args:
        beta (_type_): Vector of learnable parameters 
        X (_type_): The input data (k x p). The covariates are 
            automatically transferred to Python and stored 
            in a list. 
        wt (_type_): Sample weights (optional).
        M (_type_): A Python dictionary containing any 
            variables specified in the othervars() option 
            of mlad, matrices specified in the matrices() 
            option or scalars specified in the scalars() 
            option. Here the survival time (_t) and the 
            event indicator (_d) are needed to calculate 
            the likelihood function. Note that these will 
            be named t and d in the Python dictionary, M, 
            as defined in the othervarnames() option.
    """
    # linpred() is a utility function to calculate the current 
    # predicted value for the kth equation given X and beta
    
    # log lambda 
    lnlam =  mu.linpred(beta, X, 1)
    # log lambda 
    lngam  = mu.linpred(beta, X, 2)
    # gamma 
    gam = jnp.exp(lngam)

    lli = M["d"] * (lnlam + lngam + (gam - 1) * jnp.log(M["t"])) - jnp.exp(lnlam) * M["t"] ** (gam)
    return(jnp.sum(lli))