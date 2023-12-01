# Federated flexible parametric models 

- I mentioned about usability and the potential need to be able to transfer model estimates into Stata and/or R, so that the various postestimation tools can be used. 
- I wrote a Stata command (mlad) when I was stuck in England during COVID that combined Stata’s optimizer (ml) with performing computational intensive calculations in Python.
- The reason for the switch to Python was to use the JAX module (https://jax.readthedocs.io/en/latest/index.html) that could (1) make use of the automatic differentiation within JAX and (2) use all available CPUs (and in theory GPUs, but I never tested this).
- I mention this because although Python was doing the heavy work in terms of computation – the returned model was exactly as Stata expected to see, so immediately usable with various postestimation utilities.
- I will not go into details, but at the start of each iteration I passed the current estimates of the parameters from Stata to Python (a 1 x p vector). Python then calculated the following.
  - The value of the log likelihood (a scalar)
  - The score (gradient) vector (a 1 x p vector)
  - The Hessian (second derivatives) matrix (a p x p matrix).

    These were then passed to Stata.
    Stata then updated the parameter estimates (using the Newton-Raphson algorithm) and returned the updated vector of parameters to Python.
    In terms of federated analysis, I assume that you are using an optimizer in Python and probably need these 3 things anyway (or they are automatically calculated within the optimizer) so in theory these could be passed back to Stata in the same way that I have done. 
