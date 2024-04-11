import jax.numpy as jnp
from sfi import Data

## artificially create 3 datasets
## Very ugly - hard codes data needed for this logistic model

def mlad_setup(M):
  M["X_center1"] = [[0]]
  M["X_center1"].append(jnp.array(Data.get(["age", "smoke", "center2", "center3","cons"],selectvar="center1")))
  M["y_center1"] = jnp.array(Data.get("low",selectvar="center1"))[:,None]
  
  M["X_center2"] = [[0]]
  M["X_center2"].append(jnp.array(Data.get(["age", "smoke", "center2", "center3","cons"],selectvar="center2")))
  M["y_center2"] = jnp.array(Data.get("low",selectvar="center2"))[:,None]

  M["X_center3"] = [[0]]
  M["X_center3"].append(jnp.array(Data.get(["age", "smoke", "center2", "center3","cons"],selectvar="center3")))
  M["y_center3"] = jnp.array(Data.get("low",selectvar="center3"))[:,None]

  
