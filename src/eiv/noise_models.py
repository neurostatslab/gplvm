
"""
noise_models.py
===============
Probability distributions used as observation noise models and priors.
 
Each class follows a common two-method interface:
 
    log_density(loc, x) -> scalar
        Log-probability of observing ``x`` given the distribution is centred
        at ``loc``.  Sums over the last axis so the return value is a scalar
        (or batch of scalars when leading batch dimensions are present).
 
    sample(key, loc) -> array
        Draw one sample from the distribution centred at ``loc``, consuming
        the JAX PRNG key ``key``.
 
Most ``log_density`` and ``sample`` methods are JIT-compiled via
``@partial(jit, static_argnums=(0,))``.
 
Univariate / isotropic distributions
--------------------------------------
- ``IsotropicGaussian``         – diagonal Gaussian with a single learned log-std
- ``Gaussian``                  – diagonal Gaussian (direct std, legacy; prefer others)
- ``Gaussian_tfd``              – diagonal Gaussian backed by TFP
- ``TruncatedGaussian``         – truncated Gaussian via TFP
- ``Poisson``                   – Poisson with log-rate parameterisation
- ``Beta``                      – Beta parameterised by mean and variance
- ``Uniform``                   – uniform over a fixed interval
 
Multivariate distributions
---------------------------
- ``MultivariateGaussian``      – full-covariance Gaussian via TFP
 
Circular / directional distributions
-------------------------------------- 
- ``VonMisesNormed``            – von Mises on the circle (normalised [0, 1])
 
Count distributions
-------------------
- ``NegativeBinomial``          – negative-binomial parameterised by mean and dispersion
 
Spatial / disc distributions
-----------------------------
- ``ConcentratedDiscDistribution_resample``
                                – isotropic Gaussian on a disc, with coordinate
                                  conversion helpers between plane [0, 0.8]² and
                                  disc space
 
Compound noise models
---------------------
- ``EIVNoiseModel``             – errors-in-variables: independent noise per variable, parameter handling
- ``CompoundNoiseModel``        – sums log-densities of multiple noise models
 
Dependencies
------------
- JAX / JAX NumPy
- TensorFlow Probability (JAX substrate)  – ``tfd``
- SciPy (Sobol engine, fallback stats)
 
Notes
-----
- All ``loc`` and ``x`` inputs are assumed to have a trailing feature axis that
  is summed over by ``log_density``.
- Circular distributions (``VonMisesNormed``) expect
  inputs normalised to [0, 1] and internally map to [-π, π].



# TODO - Latent space must norm to 1, unit check noise models
"""


import jax
import jax.numpy as jnp
from jax import random, lax
import numpy as np
from jax import jit
from jax.scipy.special import logsumexp
from jax.scipy.special import gamma
from scipy.stats.qmc import Sobol
import scipy
from functools import partial
import matplotlib.pyplot as plt
from collections import namedtuple
import tensorflow_probability.substrates.jax.distributions as tfd


# ---------------------------------------------------------------------------
# Univariate / isotropic Gaussian variants
# ---------------------------------------------------------------------------
 

class IsotropicGaussian:
    """Isotropic (diagonal) Gaussian noise model with a single learned log-std.
 
    The standard deviation is parameterised as ``softplus(log_std)`` to ensure
    positivity during gradient-based optimisation.
 
    Parameters
    ----------
    log_std : float or jnp.ndarray
        Pre-softplus log-standard-deviation.  Passed through ``jax.nn.softplus``
        before use, so it is unconstrained.
    """
    def __init__(self, log_std):
        self.log_std = log_std

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        s = jax.nn.softplus(self.log_std)
        s2 = s ** 2
        z = (x - loc) / s
        z2 = z ** 2
        return jnp.sum(-0.5 * (jnp.log(s2 * 2 * jnp.pi) + z2), axis=-1)
    
    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        return loc + jax.nn.softplus(self.log_std) * jax.random.normal(key, shape=loc.shape)

class Gaussian:
    """Diagonal Gaussian noise model with a fixed standard deviation.
 
    .. deprecated::
        Prefer ``IsotropicGaussian`` or ``Gaussian_tfd`` for new code.
        ``log_std`` is used directly as the standard deviation (the name is
        misleading).
 
    Parameters
    ----------
    log_std : float or array-like
        Standard deviation (despite the name, *not* in log space).
    """
    def __init__(self, scale):
        self.scale = jnp.array(scale)

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """Compute the log-probability of ``x`` under N(loc, log_std²).
 
        Parameters
        ----------
        loc : jnp.ndarray, shape (..., D)
            Distribution mean.
        x : jnp.ndarray, shape (..., D)
            Evaluation points.
 
        Returns
        -------
        jnp.ndarray, shape (...)
            Sum of log-probabilities over the last axis.
        """
        return jnp.sum(jax.scipy.stats.norm.logpdf(x, loc=loc, scale = self.scale), axis = -1)
    
    
    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        """Draw one sample from N(loc, log_std²).
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        loc : jnp.ndarray, shape (..., D)
            Distribution mean.
 
        Returns
        -------
        jnp.ndarray, shape (..., D)
        """
        return loc + self.scale * jax.random.normal(key, shape=loc.shape)


class MultivariateGaussian:
    """Multivariate Gaussian with a full covariance matrix.
 
    Backed by ``tfd.MultivariateNormalFullCovariance``.
 
    Parameters
    ----------
    covariance : array-like, shape (..., D, D)
        Symmetric positive-definite covariance matrix.
    """

    def __init__(self, covariance):
        """
        covariance: [..., D, D] symmetric positive-definite covariance matrix
        """
        self.covariance = jnp.array(covariance)

    @partial(jax.jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """Compute log p(x | loc, covariance) under a multivariate Gaussian.
 
        Parameters
        ----------
        loc : jnp.ndarray, shape (..., D)
            Distribution mean.
        x : jnp.ndarray, shape (..., D)
            Evaluation points.
 
        Returns
        -------
        jnp.ndarray, shape (...)
            Log-probability (not summed; TFP handles the full-D log-prob).
        """
        dist = tfd.MultivariateNormalFullCovariance(
            loc=loc,
            covariance_matrix=self.covariance
        )
        return dist.log_prob(x)

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key, loc):
        """Draw one sample from N(loc, covariance).
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        loc : jnp.ndarray, shape (..., D)
            Distribution mean.
 
        Returns
        -------
        jnp.ndarray, shape (..., D)
        """

        dist = tfd.MultivariateNormalFullCovariance(
            loc=loc,
            covariance_matrix=self.covariance
        )
        return dist.sample(seed=key)


class TruncatedGaussian:
    """Truncated Gaussian distribution with fixed bounds.
 
    Backed by ``tfd.TruncatedNormal``.
 
    Parameters
    ----------
    log_std : float or array-like
        Standard deviation of the underlying (untruncated) normal.
    low : float or array-like
        Lower truncation bound (same units as ``loc`` and ``x``).
    high : float or array-like
        Upper truncation bound.
    """

    def __init__(self, log_std, low, high):
        self.log_std = jnp.array(log_std)
        self.low = jnp.array(low)
        self.high = jnp.array(high)

    @partial(jax.jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """Compute log p(x | loc) under TruncatedNormal(loc, log_std, low, high).
 
        Parameters
        ----------
        loc : jnp.ndarray, shape (..., D)
            Distribution mean (before truncation).
        x : jnp.ndarray, shape (..., D)
            Evaluation points.
 
        Returns
        -------
        jnp.ndarray, shape (...)
            Sum of per-dimension log-probabilities over the last axis.
        """
        scale = self.log_std
        dist = tfd.TruncatedNormal(
            loc=loc,
            scale=scale,
            low=self.low,
            high=self.high
        )
        return jnp.sum(dist.log_prob(x), axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key, loc):
        """Draw one sample from TruncatedNormal(loc, log_std, low, high).
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        loc : jnp.ndarray, shape (..., D)
            Distribution mean.
 
        Returns
        -------
        jnp.ndarray, shape (..., D)
        """
        scale = self.log_std
        dist = tfd.TruncatedNormal(
            loc=loc,
            scale=scale,
            low=self.low,
            high=self.high
        )
        return dist.sample(seed=key)


class Poisson:
    """Poisson noise model.
 
    The rate parameter ``loc`` is interpreted directly as the Poisson mean λ.
    No learnable parameters; the class is stateless.
    """

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, k: int):
        """Compute the Poisson log-PMF log p(k | loc).
 
        Parameters
        ----------
        loc : jnp.ndarray, shape (..., D)
            Poisson rate λ (must be positive).
        k : jnp.ndarray or int, shape (..., D)
            Observed counts.
 
        Returns
        -------
        jnp.ndarray, shape (...)
            Sum of per-dimension log-PMF values over the last axis.
        """
        return jnp.sum(k * jnp.log(loc) - loc - jax.scipy.special.gammaln(k + 1), axis=-1)

    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        """Draw one Poisson sample with rate ``loc``.
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        loc : jnp.ndarray, shape (..., D)
            Poisson rate λ.
 
        Returns
        -------
        jnp.ndarray, shape (..., D)
            Integer-valued Poisson samples.
        """
        return jax.random.poisson(key, loc)


# ---------------------------------------------------------------------------
# Continuous bounded distributions
# ---------------------------------------------------------------------------
 

class Beta:
    """Beta noise model parameterised by mean and variance.
 
    The standard Beta(α, β) parameterisation is recovered from the mean ``loc``
    and ``var`` via method-of-moments:
 
        α = loc  * (loc*(1-loc)/var - 1)
        β = (1-loc) * (loc*(1-loc)/var - 1)
 
    Both ``loc`` and ``x`` are first rescaled by ``scale`` before computing
    α and β, so the distribution is defined on [0, scale].
 
    Parameters
    ----------
    var : float
        Target variance of the Beta distribution (in the *normalised* [0,1]
        space after dividing by ``scale``).
    scale : float
        Scaling factor; observations are assumed to lie in [0, scale].
    """

    def __init__(self, var, scale):
        self.var = var
        self.scale = scale

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """Compute log p(x | loc) under the Beta noise model.
 
        Parameters
        ----------
        loc : jnp.ndarray, shape (..., D)
            Mean of the distribution in [0, scale].
        x : jnp.ndarray, shape (..., D)
            Evaluation points in [0, scale].
 
        Returns
        -------
        jnp.ndarray, shape (...)
            Sum of per-dimension log-PDF values over the last axis.
        """
        loc =jnp.clip(loc/self.scale, 1e-5, 1 - 1e-5)
        x = jnp.clip(x/self.scale, 1e-5, 1 - 1e-5)
        
        alpha = jnp.clip(loc * ((loc * (1 - loc)) / self.var - 1), 1e-4)
        beta = jnp.clip((1 - loc) * ((loc * (1 - loc)) / self.var - 1), 1e-4)

        return jnp.sum(jax.scipy.stats.beta.logpdf(x, alpha,beta), axis=-1)
    
    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        """Draw one sample from the Beta noise model.
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        loc : jnp.ndarray, shape (..., D)
            Mean of the distribution in [0, scale].
 
        Returns
        -------
        jnp.ndarray, shape (..., D)
            Samples in [0, scale].
        """
        loc = jnp.clip(loc/self.scale, 1e-5, 1 - 1e-5)
        alpha = jnp.clip(loc * ((loc * (1 - loc)) / self.var - 1), 1e-4)
        beta = jnp.clip((1 - loc) * ((loc * (1 - loc)) / self.var - 1), 1e-4)
        samps = jax.random.beta(key, alpha, beta, shape=loc.shape)
        return samps*self.scale


class Uniform:
    """Uniform distribution over a fixed interval [minval, maxval].
 
    The ``loc`` argument to ``log_density`` and ``sample`` is accepted for
    interface compatibility but is ignored — the density and samples are
    determined entirely by the interval bounds.
 
    Parameters
    ----------
    minval : float
        Lower bound of the support.
    maxval : float
        Upper bound of the support.  Must satisfy ``maxval > minval``.
    """
    def __init__(self, minval, maxval):
        assert maxval > minval, "maxval must be greater than minval"
        self.minval = minval
        self.maxval = maxval
    
    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """Compute the log-density of the uniform distribution.
 
        The density is constant and does not depend on ``loc``.
 
        Parameters
        ----------
        loc : jnp.ndarray, shape (..., D)
            Unused location parameter (kept for interface compatibility).
        x : jnp.ndarray, shape (..., D)
            Evaluation points.
 
        Returns
        -------
        jnp.ndarray, shape (...)
            Constant log-density summed over the last axis.
        """
        num_dims = x.shape[1]
        return jnp.sum(jnp.full(
            x.shape,
            -num_dims * jnp.log(self.maxval - self.minval)
        ), axis=-1)

    @partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        """Draw one uniform sample in [minval, maxval].
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        loc : jnp.ndarray, shape (..., D)
            Unused; the sample shape is taken from ``loc.shape``.
 
        Returns
        -------
        jnp.ndarray, shape (..., D)
            Samples drawn uniformly from [minval, maxval].
        """
        return jax.random.uniform(
            key,
            minval=self.minval,
            maxval=self.maxval,
            shape=loc.shape
        )


# ---------------------------------------------------------------------------
# Circular / directional distributions
# ---------------------------------------------------------------------------

class VonMisesNormed:
    """Von Mises distribution on the circle, with inputs normalised to [0, 1].
 
    The von Mises distribution is the circular analogue of the Gaussian.
    Inputs and outputs are in the *normalised* range [0, 1], which is mapped
    internally to angles in [-π, π].
 
    Backed by ``tfd.VonMises``.
 
    Parameters
    ----------
    kappa : float or jnp.ndarray
        Concentration parameter.  ``kappa = 0`` gives a uniform distribution;
        larger values give a tighter concentration around ``loc``.
    """

    def __init__(self, kappa):
        self.kappa = kappa
        
    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """Compute log p(x | loc) under VonMises(loc, kappa).
 
        Parameters
        ----------
        loc : jnp.ndarray, shape (..., D)
            Mean direction(s) in [0, 1].
        x : jnp.ndarray, shape (..., D)
            Observation angle(s) in [0, 1].
 
        Returns
        -------
        jnp.ndarray, shape (...)
            Sum of per-dimension log-probabilities over the last axis.
        """
        x = (x *2*jnp.pi)-jnp.pi
        loc = (loc *2*jnp.pi)-jnp.pi
        
        vm_dist = tfd.VonMises(loc=loc, concentration=self.kappa)
        f = vm_dist.log_prob(x)
        
        return jnp.sum(f, axis=-1)

    
    def sample(self, key,loc):
        """Draw one von Mises sample.
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        loc : jnp.ndarray, shape (..., D)
            Mean direction(s) in [0, 1].
 
        Returns
        -------
        jnp.ndarray, shape (..., D)
            Sample angle(s) in [0, 1].
        """
        loc = (loc * 2*jnp.pi)-jnp.pi
        vm_dist = tfd.VonMises(loc=loc, concentration=self.kappa)
        samps = vm_dist.sample(seed = key)

        return (samps+jnp.pi)/(2*jnp.pi)


class NegativeBinomial:
    """Negative-binomial noise model parameterised by mean and dispersion.
 
    The NB distribution is reparameterised in terms of the mean ``loc`` and a
    fixed dispersion (overdispersion) parameter ``theta`` using the relation
 
        p = loc / (theta + loc)
 
    so that ``E[x] = loc`` and ``Var[x] = loc + loc² / theta``.  As
    ``theta → ∞`` the distribution converges to a Poisson.
 
    Parameters
    ----------
    theta : float or jnp.ndarray
        Dispersion parameter (also called *r* or *total_count* in TFP).
        Larger values → less overdispersion.
    """
    def __init__(self, theta):
        self.theta = theta
        
    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """Compute log p(x | loc) under NegativeBinomial(theta, loc/(theta+loc)).
 
        Parameters
        ----------
        loc : jnp.ndarray, shape (..., D)
            Expected count (mean of the distribution).
        x : jnp.ndarray, shape (..., D)
            Observed counts.
 
        Returns
        -------
        jnp.ndarray, shape (...)
            Sum of per-dimension log-PMF values over the last axis.
        """
        p = loc/(self.theta+loc)
        nb_dist = tfd.NegativeBinomial(total_count=self.theta, probs=p)
        f = nb_dist.log_prob(x)
        return jnp.sum(f, axis=-1)

    
    def sample(self, key,loc):
        """Draw one negative-binomial sample with mean ``loc``.
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            PRNG key.
        loc : jnp.ndarray, shape (..., D)
            Expected count.
 
        Returns
        -------
        jnp.ndarray, shape (..., D)
            Integer-valued samples.
        """
        p = loc/(self.theta+loc)
        nb_dist = tfd.NegativeBinomial(total_count=self.theta, probs=p)
        samps = nb_dist.sample(seed = key)
        return samps


class ConcentratedDiscDistribution_resample:
    def __init__(self, log_concentration):
        self.log_concentration = jnp.array(log_concentration)
        self.radius = 0.4  # Diameter 0.8 -> radius 0.4
        self.plane_min = 0.0
        self.plane_max = 0.8
        self.plane_center = 0.4  # Center of [0, 0.8] range
    
    @partial(jax.jit, static_argnums=(0,))
    def plane_to_disc(self, plane_coords):
        """Convert from [0, 0.8] x [0, 0.8] plane coordinates to disc coordinates (radius 0.4)"""
        # First clamp to plane bounds
        plane_coords = jnp.clip(plane_coords, self.plane_min, self.plane_max)
        
        # Center the coordinates: [0, 0.8] -> [-0.4, 0.4]
        centered_coords = plane_coords - self.plane_center
        
        # Now we have coordinates in [-0.4, 0.4] x [-0.4, 0.4]
        r_plane = jnp.linalg.norm(centered_coords, axis=-1, keepdims=True)
        r_plane = jnp.maximum(r_plane, 1e-8)
        
        # Max distance from center in plane is sqrt(0.4^2 + 0.4^2) = 0.4*sqrt(2)
        max_plane_r = jnp.sqrt(self.radius**2 + self.radius**2) # 0.4 * jnp.sqrt(2.0)
        
        # Map to disc radius (linearly scale to fit within disc)
        r_disc = jnp.minimum(r_plane, self.radius)  # Direct mapping since max_plane_r ≈ 0.566 > 0.4
        
        # If the plane point is outside the disc radius, scale it down
        r_disc = jnp.where(r_plane > self.radius, self.radius, r_plane)
        
        # Preserve direction
        direction = centered_coords / r_plane
        return direction * r_disc
    
    @partial(jax.jit, static_argnums=(0,))
    def disc_to_plane(self, disc_coords):
        """Convert from disc coordinates (radius 0.4) to [0, 0.8] x [0, 0.8] plane coordinates"""
        # Clamp to disc bounds using jnp.where instead of if statement
        r_disc = jnp.linalg.norm(disc_coords, axis=-1, keepdims=True)
        r_disc = jnp.maximum(r_disc, 1e-8)  # Avoid division by zero
        
        # Scale down if outside disc radius
        disc_coords = jnp.where(
            r_disc > self.radius,
            disc_coords * (self.radius / r_disc),
            disc_coords
        )
    
        # Disc coordinates are already in [-0.4, 0.4] range (centered)
        # Convert back to [0, 0.8] by adding center offset
        plane_coords = disc_coords + self.plane_center
        
        # Clamp to plane bounds
        return jnp.clip(plane_coords, self.plane_min, self.plane_max)
    
    @partial(jax.jit, static_argnums=(0,))
    def log_density(self, loc, x):
        """
        Args:
            loc: location parameter in plane coordinates [0, 0.8] x [0, 0.8]
            x: evaluation points in plane coordinates [0, 0.8] x [0, 0.8]
        Returns:
            log density
        """
        concentration = jnp.exp(self.log_concentration)
        
        # Convert plane coordinates to disc coordinates
        loc_disc = self.plane_to_disc(loc)
        x_disc = self.plane_to_disc(x)
        
        # Compute difference in disc space
        diff = x_disc - loc_disc
        
        # Check if points are within disc (should always be true due to our mapping)
        distance = jnp.linalg.norm(x_disc, axis=-1)
        in_disc = distance <= self.radius
        
        # Gaussian log probability in disc space
        log_prob = -0.5 * concentration * jnp.sum(diff**2, axis=-1)
        
        # Simple normalization
        log_normalizer = jnp.log(2 * jnp.pi) - self.log_concentration
        
        return jnp.where(in_disc, log_prob - log_normalizer, -jnp.inf)
    

    @partial(jax.jit, static_argnums=(0,))
    def resample_particles(self, key, init_particles, loc_disc, scale):
            """Resample until all particles are in [low, high]."""

            def cond_fun(state):
                key, particles = state
                sample_norm = jnp.linalg.norm(particles, axis=-1, keepdims=True)
                return jnp.any(sample_norm > self.radius)

            def body_fun(state):
                key, particles = state
                key, subkey = random.split(key)
                # propose new samples
                disc_offset = scale * jax.random.normal(subkey, shape=loc_disc.shape)
                new_particles = loc_disc + disc_offset
                sample_norm = jnp.linalg.norm(new_particles, axis=-1, keepdims=True)
                #keep valid ones
                particles = jnp.where(sample_norm > self.radius,
                                    particles, new_particles)
                return key, particles

            final_key, final_particles = lax.while_loop(cond_fun, body_fun, (key, init_particles))
            return final_key, final_particles

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key, loc):
        """
        Args:
            loc: location parameter in plane coordinates [0, 0.8] x [0, 0.8]
        Returns:
            sample in plane coordinates [0, 0.8] x [0, 0.8]
        """
        concentration = jnp.exp(self.log_concentration)
        scale = 1.0 / jnp.sqrt(concentration)
        
        # Convert loc to disc coordinates
        loc_disc = self.plane_to_disc(loc)
        
        # Sample offset in disc space
        key, subkey = jax.random.split(key)
        disc_offset = scale * jax.random.normal(subkey, shape=loc_disc.shape)
        
        # Add to location in disc space
        disc_sample = loc_disc + disc_offset
        

        _, disc_sample = self.resample_particles(key, disc_sample, loc_disc, scale)

        # Convert back to plane coordinates
        plane_sample = self.disc_to_plane(disc_sample)
        
        # Ensure within plane bounds [0, 0.8]
        return jnp.clip(plane_sample, self.plane_min, self.plane_max)


# ---------------------------------------------------------------------------
# Compound / errors-in-variables noise models
# ---------------------------------------------------------------------------
 

class CompoundNoiseModel:
    """Compound noise model that sums the log-densities of multiple noise models.
 
    More general than the ``EIVNoiseModel``, accepts either a single noise model or a
    list, making it more flexible for compositional model building.
 
    Parameters
    ----------
    noise_models : noise model or list of noise models
        One or more noise model instances, each implementing
        ``log_density(loc, x)`` and ``sample(key, loc)``.  A single instance
        is automatically wrapped in a list.
    """
    def __init__(self, noise_models):
        self.noise_models = noise_models if isinstance(noise_models, list) else [noise_models]
    
    @partial(jit, static_argnums=(0,))
    def log_density(self, locs, xs):
        """Compute the total log-density as a sum over all noise models.
 
        Parameters
        ----------
        locs : sequence of jnp.ndarray
            Location parameters, one array per noise model.
        xs : sequence of jnp.ndarray
            Observations, one array per noise model.
 
        Returns
        -------
        jnp.ndarray, shape ()
            Sum of per-model log-densities.
        """
        evals = jnp.array([noise.log_density(loc, x) for noise, loc, x in zip(self.noise_models, locs, xs)])
        
        return jnp.sum(evals, axis=0)
        
    def sample(self, key, locs):
        """Draw one sample from each noise model.
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            Master PRNG key, split into sub-keys (one per model).
        locs : sequence of jnp.ndarray
            Location parameters, one array per noise model.
 
        Returns
        -------
        list of jnp.ndarray
            One sample per noise model.
        """
        keys = jax.random.split(key, num=2)
        samples = [no.sample(k, l) for no, k, l in zip(self.noise_models, keys, locs)]     
        
        return samples
 
class EIVNoiseModel:
    """Errors-in-variables (EIV) noise model combining independent per-variable noise.
 
    Computes the joint log-density as the sum of independent noise models
    applied to each variable separately.  Intended for models where both the
    inputs and outputs are observed with noise (errors-in-variables regression).
 
    Handles parameter wrapping to avoid having to pass empty lists. 

    Parameters
    ----------
    noise_models : list
        Ordered list of noise model instances, one per variable.  Each must
        implement ``log_density(loc, x)`` and ``sample(key, loc)``.
    """

    def __init__(self, noise_models):
        self.noise_models = noise_models 

    @partial(jit, static_argnums=(0,))
    def log_density(self, locs, xs):
        """Compute the total log-density as a sum over independent noise models.
 
        Parameters
        ----------
        locs : sequence of jnp.ndarray
            Location parameters, one array per noise model.
        xs : sequence of jnp.ndarray
            Observations, one array per noise model.
 
        Returns
        -------
        jnp.ndarray, shape ()
            Sum of per-model log-densities.
        """
        evals = jnp.array([noise.log_density(loc, x) for noise, loc, x in zip(self.noise_models, locs, xs)])
        
        return jnp.sum(evals, axis=0)
        
    
    def sample(self, key, locs):
        """Draw one sample from each noise model.
 
        Parameters
        ----------
        key : jax.random.PRNGKey
            Master PRNG key, split into two sub-keys (one per model).
        locs : sequence of jnp.ndarray
            Location parameters, one array per noise model.
 
        Returns
        -------
        list of jnp.ndarray
            One sample per noise model.
        """
        keys = jax.random.split(key, num=2)
        samples = [no.sample(k, l) for no, k, l in zip(self.noise_models, keys, locs)]     
        
        return samples
 
class UniformSobol:

    def __init__(self, minval, maxval):
        assert maxval > minval
        self.minval = minval
        self.maxval = maxval

    @partial(jit, static_argnums=(0,))
    def log_density(self, loc, x):
        num_dims = x.shape[1]
        return jnp.sum(jnp.full(
                x.shape,
                -num_dims * jnp.log(self.maxval - self.minval)
            ), axis=-1)

    
    #@partial(jit, static_argnums=(0,))
    def sample(self, key, loc):
        """
        Parameters
        ----------
        minval,maxval : float64
            Range of samples

        Returns
        -------
        X: Array
            (num_mc_samples x num_dimensions)
        """
        n_samples, n_dims = loc.shape
        qrng = Sobol(n_dims, seed=0)
        xs = jnp.array(qrng.random(n=n_samples) * \
                        (self.maxval-self.minval))  + self.minval

        return xs
