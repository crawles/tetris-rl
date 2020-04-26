import collections
from functools import reduce
import numpy as onp
import jax.numpy as jnp

Path = collections.namedtuple('Path', 'obs acs logps values rewards')

def convert_types(params):
    for k, v in params.items():
        try:
            params[k] = eval(v)
        except:
            pass

def flatten(x):
    output_shape = reduce(onp.multiply, x.shape)
    return onp.reshape(x, output_shape)

def path(obs, acs, logps, values, rewards):
    return Path(onp.array(obs),
                onp.array(acs),
                onp.array(logps),
                onp.array(values),
                onp.array(rewards))
