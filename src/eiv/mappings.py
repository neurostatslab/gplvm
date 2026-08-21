"""
Bijective mapping functions for probabilistic layers.

Each function takes in `(params, x)` and returns `x_next`.

`params` is anything jax can differentiate (i.e. a PyTree).

`x` and `x_next` are jax arrays, matrices with shape (num_particles, num_dims).
"""

import jax
from jax import jit
import jax.numpy as jnp
import numpy as np
from functools import partial
from collections import namedtuple
import itertools
import matplotlib.pyplot as plt

def identity(params, x):
    return x

def linear(A, x):
    return x @ A.T

def affine(params, x):
    A, b = params
    return x @ A.T + b[None, :]


class IdentityMapping:
    def __call__(self, params, x):
        return x

    def sample(self, key, params):
        return params

    def log_density(self, params):
        return 0

class EIVMapping:
    def __init__(self, mappings):
        self.mappings = mappings# if isinstance(mappings, list) else [mappings]
        self.params_per_neuron = mappings[0].params_per_neuron
        
    def __call__(self, params, xs):
        # TODO - this is HACKEY fix this
        params = [params, None]
        
        evals = [mapping(p, xs) for mapping, p in zip(self.mappings, params)]
        
        return evals

    def sample(self, key, params):
        # TODO - this is HACKEY fix this
        params = [params, None]
        keys = jax.random.split(key, num=len(params))
        samples = [mapping.sample(k, p) for mapping, k, p in zip(self.mappings, keys, params)]
        return samples

    def log_density(self, params):
        params = [params, None]
        dense = [mapping.log_density(p) for mapping, p in zip(self.mappings, params)]
        
        return jnp.sum(jnp.array(dense))

class CompoundMapping:
    def __init__(self, mappings):
        self.mappings = mappings# if isinstance(mappings, list) else [mappings]

        self.params_per_neuron = None 
        #TODO - fix this, auto determine from map types?
        
    def __call__(self, params, xs):
        
        evals = [mapping(p, xs) for mapping, p in zip(self.mappings, params)]
        return evals

    def sample(self, key, params):
        keys = jax.random.split(key, num=len(params))
        samples = [mapping.sample(k, p) for mapping, k, p in zip(self.mappings, keys, params)]
        return samples

    def log_density(self, params):
        dense = [mapping.log_density(p) for mapping, p in zip(self.mappings, params)]
        return jnp.sum(jnp.array(dense))

def make_fourier_freqs(d: int, K: int) -> jnp.ndarray:
    """Basis functions for dimension `d` with max frequency `K`"""

    all_freqs = itertools.product(range(-K, K + 1), repeat=d)

    def in_half_space(m):
        # Keep m if its first nonzero component is positive.
        for mi in m:
            if mi > 0:
                return True
            if mi < 0:
                return False

        # I assume we want to exclude the all-zeros vector as a basis function?
        # If that's the case, then we'd return False here, otherwise switch this
        # to return True.
        return False

    return jnp.array([m for m in all_freqs if in_half_space(m)])


class WeightedFourierBasisMapping:
    """
        Initializes the WeightedFourierBasisMapping from params dict.

        Args:
            params (dict): A dictionary containing hyperparameters:
                - 'max_freq' (int): Maximum frequency for Fourier basis.
                - 'num_dims' (int): Input space dimensionality.
                - 'len_scale' (float): Length scale for the kernel.
                - 'out_scale' (float): Output scale for the kernel.
                - 'num_neurons' (int): Number of neurons.
                - 'tol' (float): Threshold for truncating small frequencies.
                - 'nonlinearity' (callable): Nonlinear activation function.
    """

    def __init__(self, params):
        self.max_freq = params['max_freq']
        self.num_dims = params['num_dims']
        self.len_scale = params['len_scale']
        self.out_scale = params['out_scale']
        self.num_neurons = params['num_neurons']
        self.tol = params['tol']
        self.nonlinearity = params['nonlinearity']

        F = 2*jnp.pi *make_fourier_freqs(self.num_dims, self.max_freq)
        
        lam = jnp.sum(F ** 2, axis=1)

        # specify kernel
        kern = self.out_scale * jnp.exp(-0.5 * (self.len_scale ** 2) * lam)
        tau = jnp.sqrt(kern)

        # truncate frequencies below tolerance
        thres = jnp.max(tau) * self.tol
        idx = tau > thres
        self.tF, self.ttau = F[idx], tau[idx]
        self.params_per_neuron = 1 + 2 * len(self.ttau)

        
    @partial(jax.vmap, in_axes=(None,-1, None), out_axes=1)
    def __call__(self, params, x):
        """
        Computes the Fourier basis mapping at x.

        Args:
            params (jnp.ndarray): Flattened array of weights with shape (params_per_neuron,).
            x (jnp.ndarray): Input array with shape (observations, num_dims).

        Returns:
            jnp.ndarray: Tuning evaluated at x (transformed by nonlinearity).
        """
        """
        lattice.shape = [num_basis_funcs, dim_in]
        x.shape = [observations, dim_in]
        sin_coeffs.shape = [num_basis_funcs, dim_out]
        cos_coeffs.shape = [num_basis_funcs, dim_out]
        bias.shape = [dim_out]
        """
        # coeffs.shape = n_neurons x n_bases
        # x.shape = n_samples x n_dims
        # lattice.shape = n_bases x n_dims
        # z.shape = n_samples x n_bases
        bias = params[0]
        sin_coeffs, cos_coeffs = jnp.array_split(params[1:], 2)
        
        z = x @ self.tF.T # [observations, num_basis_funcs]
        
        wsin = jnp.sin(z) * self.ttau * sin_coeffs # [observations, dim_out]
        wcos = jnp.cos(z) * self.ttau * cos_coeffs # [observations, dim_out]
        
        return self.nonlinearity(bias + jnp.sum(
            wsin  + wcos, axis = -1
        )) # [observations, dim_out]


    def sample(self, key, params):
        # TODO - Params is a junk variable - remove, or make num_neurons/chains
        return jax.random.normal(
                        key, shape=(self.params_per_neuron, self.num_neurons)
                    )
        
    @partial(jit, static_argnums=(0,))
    def log_density(self, params):
        return jnp.sum(jax.scipy.stats.norm.logpdf(params[1:]))


class WeightedLinearMapping:
    """
    Linear mapping for PPCA comparison
    """
    def __init__(self, params):
        self.dim_in = params['dim_in']
        self.w_variance = params['w_variance']
        self.nonlinearity = params['nonlinearity']
        self.params_per_neuron = params['dim_in']

    def __call__(self, params, x):
        """
        x.shape = [observations, dim_in]
        """
        print(params.shape)
        wx = x @ params  # [observations, dim_in]
        return self.nonlinearity(wx) # [observations, dim_in]

    def sample(self, key, params):
        
        k0, k1, k2 = jax.random.split(key, num=3)
        params  = jax.random.normal(
                            k1, shape=(params['num_neurons'], params['dim_in'])
                        )
        
        print(params)
        return params

    def log_density(self, params):
        
        l0 = jax.scipy.stats.norm.logpdf(
            params, loc=0.0, scale=self.w_variance
        )

        return jnp.sum(l0)

class WeightedFourierUniformTruncation:
    """
    Basis mapping with uniform truncation
    Not rotation invariant
    """
    def __init__(self, params):
        self.max_freq = params['max_freq']
        self.dim_in = params['dim_in']
        self.tau_decay = params['tau_decay']
        self.tau_scale = params['tau_scale']
        self.bias_mean = params['bias_mean']
        self.bias_std = params['bias_std']
        self.nonlinearity = params['nonlinearity']

        grid = jnp.meshgrid(
            *[jnp.arange(self.max_freq + 1) for _ in range(self.dim_in)]
        )
        self.lattice = jnp.column_stack(
            [z.ravel() for z in grid]
        ).astype(float)[1:]
        self.num_basis_funcs = self.lattice.shape[0]

        self.tau = self.tau_scale * jnp.exp(
            -self.tau_decay * jnp.sum(
                self.lattice, axis=1
            )
        )
        

    def __call__(self, params, x):
        """
        lattice.shape = [num_basis_funcs, dim_in]
        x.shape = [observations, dim_in]
        sin_coeffs.shape = [num_basis_funcs, dim_out]
        cos_coeffs.shape = [num_basis_funcs, dim_out]
        bias.shape = [dim_out]
        """
        
        z = x @ self.lattice.T # [observations, num_basis_funcs]
        
        wsin = jnp.sin(z) @ params["sin_coeffs"].T # [observations, dim_out]
        wcos = jnp.cos(z) @ params["cos_coeffs"].T # [observations, dim_out]
        return self.nonlinearity(params["bias"] + wsin + wcos) # [observations, dim_out]

    def sample(self, key, params):
        k0, k1, k2 = jax.random.split(key, num=3)

        weight_params = {
            "cos_coeffs": jax.random.normal(
                            k1, shape=(params['num_neurons'], len(self.tau))
                        ) * self.tau,
            "sin_coeffs": jax.random.normal(
                            k2, shape=(params['num_neurons'], len(self.tau))
                        ) * self.tau,
            "bias": jax.random.normal(
                        k0, shape=(params['num_neurons'],)
                    ) * self.bias_std,
        }
        return weight_params

    def log_density(self, params):
        l0 = jax.scipy.stats.norm.logpdf(
            params['bias'],
            loc=self.bias_mean,
            scale=self.bias_std
        )
        l1 = jax.scipy.stats.norm.logpdf(
            params['cos_coeffs'], loc=0.0, scale=self.tau
        )
        l2 = jax.scipy.stats.norm.logpdf(
            params['sin_coeffs'], loc=0.0, scale=self.tau
        )
