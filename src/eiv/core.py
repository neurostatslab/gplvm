
"""
gplvm.py
========
Gaussian Process Latent Variable Model (GPLVM) family: model classes that tie
together a sampler/proposal over latent variables, an observation model
(mapping + noise), and an inference backend (optimizer or SMC sampler).
 
All concrete model classes subclass ``AbstractGPLVM`` and must implement
``marginal_log_likelihood_params`` and ``simulate``.  The common pattern is:
 
    model = GPLVM(observation=my_layer, sampler=Roberts())
    model.fit(Y, method="adam", opt_params={...})
    # -> model.params_, model.objhist_, etc. (set by the chosen inference.* class)
 
Model classes
-------------
- ``AbstractGPLVM``   – base class: shared posterior/likelihood machinery,
                        latent-grid evaluation helpers, and the ``fit`` dispatcher.
- ``GPLVM``           – static (non-dynamic) latent-variable model using
                        Monte-Carlo marginalisation over a fixed latent-space
                        sampler (e.g. Sobol/Roberts quasi-random points).
- ``GPGLM``           – degenerate case where the latent variable ``x`` is
                        directly observed (a GLM): no marginalisation over
                        latent needed.
- ``DynamicGPLVM``    – sequential (state-space) latent-variable model using
                        Sequential Monte Carlo (particle filtering) to estimate
                        the marginal likelihood through time.
  
Model classes
-------------
- ``EIV``             - Wrapper for static error-in-variables model, simplifies layer 
                        construction, only requires 3 args (Kappa, tuning output scale, 
                        tuning length scale). Defaults Poisson noise and VonMises 
                        distribution over errors.
- ``DynamicEIV``      - Wrapper for dynamic error-in-variables model, requires an additional
                        parameter to govern smoothness of behavioral trajectories. Transitional
                        and proposal noise also default to VonMises. 

Supporting classes
-------------------
- ``Layer``           – a probabilistic mapping ``f(x) + noise``: combines a
                        deterministic ``mapping`` with a ``noise`` model to
                        define both ``log_density`` and ``sample``.
- ``Proposal``        – SMC proposal distribution: wraps a ``Layer`` to
                        provide ``sample`` / ``initialize`` for particle filters.
 
Dependencies
------------
- JAX / JAX NumPy
- Internal modules: ``smc``, ``smc_adaptive``, ``mappings``, ``noise_models``,
  ``inference``, ``mc_samplers``
 
Notes
-----
- "params" throughout refers to a single pytree of model parameters
  (mapping weights, noise parameters, etc.), not separate per-component args.
- ``key`` arguments are JAX PRNG keys.
- Latent variables are denoted ``x`` (or ``X`` for a batch/grid of candidates);
  observations are denoted ``y`` / ``Y``.
"""

import jax
import jax.numpy as jnp
from jax import jit
from jax.scipy.special import logsumexp
from jax.nn import softplus

import inspect
from functools import partial


import smc
import mappings
import noise_models
import inference
import matplotlib.pyplot as plt
from mc_samplers import Roberts


class AbstractGPLVM:
    """Base class for GP-style latent-variable models.
 
    Defines the shared posterior/likelihood machinery used by all subclasses:
    the (unnormalised) posterior over parameters, conditional density of
    observations given latent positions, and a dispatcher to the
    ``inference`` module's optimizer/sampler classes.
 
    Subclasses must implement ``marginal_log_likelihood_params`` (how the
    latent variable is marginalised out) and ``simulate`` (how to generate
    synthetic data from the model).
 
    Parameters
    ----------
    observation : Layer
        Observation model combining a deterministic mapping ``f(x)`` with a
        noise model.  Must expose ``mapping`` (with ``params_per_neuron`` and
        ``log_density``) and ``log_density`` / ``sample`` methods.
    num_samples : int, optional
        Number of Monte-Carlo samples (or particles) used to approximate
        latent-variable integrals.  Default ``2**10``.
 
    Attributes
    ----------
    observation : Layer
        The observation model passed at construction.
    params_per_neuron : int
        Number of mapping parameters per neuron, read from
        ``observation.mapping.params_per_neuron``.
    num_samples : int
        Number of MC samples / particles used in marginalisation.
    """
    def __init__(
            self, observation=None,  num_samples=2**10
        ):

        self.observation = observation
        self.params_per_neuron = observation.mapping.params_per_neuron

        # TODO - maybe move this to inference class
        self.num_samples = num_samples
        

    def marginal_log_likelihood_params(self, params, y, key):
        """Compute the marginal log-likelihood of ``y`` under ``params``.
 
        Must be implemented by subclasses; the marginalisation strategy
        (plain Monte Carlo, SMC, or none at all) depends on whether the
        latent variable is static, sequential, or directly observed.
 
        Parameters
        ----------
        params : pytree
            Model parameters.
        y : array-like
            Observed data.
        key : jax.random.PRNGKey
            PRNG key for any stochastic estimator.
 
        Returns
        -------
        array-like
            Log-likelihood value(s), to be summed by ``log_posterior_params``.
 
        Raises
        ------
        NotImplementedError
            Always, unless overridden by a subclass.
        """
        
        raise NotImplementedError(
            "marginal_log_likelihood_params function must be implemented by subclass."
        )

    def simulate(self, params, y, key):
        raise NotImplementedError(
            "Simulate function must be implemented by subclass."
        )

    #@partial(jit, static_argnums=(0,))
    def logp_y_given_x(self, params, y, xs):
        
        log_pdf = self.observation.log_density(
            params, xs, y
        )
        
        return log_pdf

    #@partial(jit, static_argnums=(0,))
    def log_posterior_params(self, params, y, key):

        return (
            jnp.sum(self.marginal_log_likelihood_params(params, y, key)) +
            self.observation.mapping.log_density(params)
        )

    @partial(jax.vmap, in_axes=(None, None, 0, None), out_axes=0)
    def logp_x(self, params, y, X):
        eiv_flag = True if isinstance(self.observation.mapping, mappings.EIVMapping) else False

        if eiv_flag:
        
            y_pred = jnp.moveaxis(self.observation.mapping(params, X)[0],1,0)
            #self.observation.mapping(params, X)[0].T
            lps = self.observation.noise.noise_models[0].log_density(
                        jnp.expand_dims(y_pred, -1), y[:, *([jnp.newaxis] * X.shape[-1]), None]
                    )
        else:
            y_pred = jnp.moveaxis(self.observation.mapping(params, X),1,0)
            lps = self.observation.noise.log_density(
                        jnp.expand_dims(y_pred, -1), y[:, *([jnp.newaxis] * X.shape[-1]), None]
                    )
     
        # Sum over neurons (conditionally independent given x).
        logp_unnrm = jnp.sum(lps, axis=0)
        # Normalize log density so that the density sums to one.
        logp = logp_unnrm - logsumexp(logp_unnrm)
        
        return logp


    @partial(jax.vmap, in_axes=(None, None, 0, None), out_axes=0)
    def logp_x_map(self, params, y, X):
        eiv_flag = True if isinstance(self.observation.mapping, mappings.EIVMapping) else False

        if eiv_flag:
        
            y_pred = jnp.moveaxis(self.observation.mapping(params, X)[0],1,0)
            #self.observation.mapping(params, X)[0].T
            lps = self.observation.noise.noise_models[0].log_density(
                        jnp.expand_dims(y_pred, -1), y[:, *([jnp.newaxis] * X.shape[-1]), None]
                    )
        else:
            y_pred = jnp.moveaxis(self.observation.mapping(params, X),1,0)
            lps = self.observation.noise.log_density(
                        jnp.expand_dims(y_pred, -1), y[:, *([jnp.newaxis] * X.shape[-1]), None]
                    )
     
        # Sum over neurons (conditionally independent given x).
        logp_unnrm = jnp.sum(lps, axis=0)
        # Normalize log density so that the density sums to one.
        logp = logp_unnrm - logsumexp(logp_unnrm)
        
        max_inds = jnp.unravel_index(logp.argmax(), logp.shape)

        return X[max_inds]

    def random_init(self, key):
        return jax.random.normal(
                    key, shape=(self.params_per_neuron, self.num_neurons)
                )

    def fit(self, Y, method, opt_params, **kwargs):
        
        _fitting_methods = \
            dict(adam=inference.Adam,
                 sgd=inference.SGD
                 )

        if method not in _fitting_methods:
            raise Exception("Invalid method: {}. Options are {}".\
                            format(method, _fitting_methods.keys()))

        method = _fitting_methods[method](self, opt_params)

        return method.fit(Y,**kwargs)


class GPLVM(AbstractGPLVM):

    def __init__(self, *args, sampler = Roberts(),
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sampler = sampler

    def marginal_log_likelihood_params(self, params, y, key):
        xs = self.sampler.sample(key, self.num_samples)
        @partial(jax.vmap, in_axes=(None, 0, None), out_axes=0)
        def _marginal_log_likelihood_params(params, y, key):
            # might need to separate this out
            return self.logp_y_given_x(params, y, xs)
        
        return logsumexp(_marginal_log_likelihood_params(params, y, key), axis=1) - jnp.log(self.num_samples)
    
    def simulate(self, key, params, num_observations):
        k1, k2 = jax.random.split(key, 2)
        
        x = self.sampler.sample(
            k1, num_observations
        )

        y = self.observation.sample(
            k2, params, x
        )

        return x, y


class GPGLM(AbstractGPLVM):

    def __init__(self, *args, sampler = Roberts(),
                 **kwargs):
        super().__init__(*args, **kwargs)
        
        self.sampler = sampler

    def marginal_log_likelihood_params(self, params, y, key):
        ys, xs = y
        return self.logp_y_given_x(params, ys, xs)
    
    def simulate(self, key, params, num_observations):
        k1, k2 = jax.random.split(key, 2)
        
        x = self.sampler.sample(
            k1, num_observations
        )

        y = self.observation.sample(
            k2, params, x
        )

        return x, y

        
class DynamicGPLVM(AbstractGPLVM):

    def __init__(self, *args, transition=None, proposal=None,**kwargs):
        super().__init__(*args, **kwargs)
        self.transition = transition
        
        self.proposal = proposal

    def log_transition_prob(self, params, x, x_last):
        log_pdf = self.transition.log_density(
            params, x_last, x
        )
        return log_pdf

    def marginal_log_likelihood_params(self, params, y, key):
        

        log_avg_weights = smc.log_marginal_likelihood(
                    params, y, key, self.proposal.initialize,  
                    self.proposal.sample, self.log_transition_prob,
                    self.logp_y_given_x, self.num_samples,
            )
        return log_avg_weights


    def simulate(self, key, params, x_init, num_timesteps):
        
        def scanned_func(x_last, k):
            k1, k2 = jax.random.split(k, 2)
            x = self.transition.sample(
                k1, params, x_last
            )
            y = self.observation.sample(
                k2, params, x
            )
            return x, (x, y)
        return jax.lax.scan(
            scanned_func, x_init, jax.random.split(key, num_timesteps)
        )[1]


class MISGPLVM(AbstractGPLVM):

    def __init__(self, *args, num_is = 50, sampler = Roberts(), proposal = noise_models.UniformSobol(0.,1.),
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.observation.mapping, mappings.EIVMapping), "MIS is only implemented for EIV, 1D"
        # Not dynamic, have to specify sampling procedure
        self.sampler = sampler
        self.proposal = proposal
        self.num_is = num_is
        self.generate_samples = jax.vmap(lambda k, loc: self.proposal.sample(k, loc), in_axes=(0, None))

    def marginal_log_likelihood_params(self, params, y, key):
        # TODO - this is not general not going ot work for >1 noise model
        # TODO - Figure out where most efficient to generate smaples
        is_keys = jax.random.split(key, num=self.num_is)
        @partial(jax.vmap, in_axes=(None, 0, None), out_axes=0)
        def _marginal_log_likelihood_params_IS(params, y, keys):
            xs_ = self.generate_samples(keys, y[1])
            log_p = self.observation.noise.noise_models[1].log_density(y[1],xs_)
            log_q = self.proposal.log_density(y[1], xs_)
            log_w = log_p - log_q
            return self.logp_y_given_x(params, y, xs_)+log_w

        xs = self.sampler.sample(key, self.num_samples)
        @partial(jax.vmap, in_axes=(None, 0, None), out_axes=0)
        def _marginal_log_likelihood_params(params, y, key):
            # might need to separate this out
            return self.logp_y_given_x(params, y, xs)
        
        is_mll = logsumexp(_marginal_log_likelihood_params_IS(params, y, is_keys), axis=1)-jnp.log(self.num_is)
        mc_mll = logsumexp(_marginal_log_likelihood_params(params, y, key), axis=1)-jnp.log(self.num_samples)
        
        return logsumexp(jnp.array([is_mll,mc_mll]), axis = 0) - jnp.log(2)
        
    
    
    def simulate(self, key, params, num_observations):
        k1, k2 = jax.random.split(key, 2)
        
        x = self.sampler.sample(
            k1, num_observations
        )

        y = self.observation.sample(
            k2, params, x
        )

        return x, y

class Layer:
    """
    Probabilistic mapping layer, f(x) + noise.
    """

    def __init__(self, mapping, noise):
        self.mapping = mapping
        self.noise = noise
        
    def log_density(self,params, x, y):
        loc = self.mapping(params, x) 
        
        # if compound mapping, loc is going to return n_mappings versions of x
        return self.noise.log_density(loc, y)
    
    def sample(self, key, params, x):

        loc = self.mapping(params, x)
        return self.noise.sample(key, loc)

class Proposal:
    """
    Proposal distribution (used for sequential monte carlo).
    """
    #TODO - pretty much got rid of init params, do something about this
    def __init__(self, layer, params, init_params, init_loc):
        self.params = params
        self.layer = layer
        self.init_params = init_params
        self.init_loc = init_loc

    @partial(jax.jit, static_argnums=(0,))
    def sample(self, key, x_last):
        x = self.layer.sample(key, self.params, x_last)
        log_pdf = self.layer.log_density(self.init_params, x_last, x)

        return x, log_pdf

    @partial(jax.jit, static_argnums=(0,2))
    def initialize(self, key, num_particles):
        loc = jnp.tile(self.init_loc, (num_particles, 1))
        # loc = (n_particles x 1)
        
        x = self.layer.sample(key, self.init_params, loc)
        # x = (n_particles x n_dims)
        
        log_pdf = self.layer.log_density(self.init_params, loc, x)
        
        return x, log_pdf

    @partial(jax.jit, static_argnums=(0,))
    def initialize_samp(self, key, num_particles):
        
        loc = jnp.tile(self.init_loc, (num_particles, 1))
        # loc = (n_particles x 1)
        
        x = self.layer.sample(key, self.init_params, loc)
        # x = (n_particles x n_dims)

        return x

def _build_noise(model, args):
        
    if not inspect.isclass(model):
        return model  # already an instance
    
    if args is None:
        return model()
        
    return model(args)

class EIV(GPLVM):

    def __init__(self, 
        len_scale, out_scale, kappa,
        num_dims,num_neurons,
        *,
        max_freq=30, basis_tol=1e-3, 
        nonlinearity=lambda x: softplus(x),
        spike_noise = noise_models.Poisson,
        behav_noise = noise_models.VonMisesNormed,
        behav_obs_kwargs=None,
        spike_obs_kwargs= None,
        num_samples=512, sampler=None,
        sim_key = None
        ):

        self.num_neurons = num_neurons
        self.num_dims = num_dims
        
        self.basis_params = {
            "max_freq": max_freq,
            "num_dims": num_dims,
            "out_scale": out_scale,
            "len_scale": len_scale,
            "num_neurons": num_neurons,
            "tol": basis_tol,
            "nonlinearity": nonlinearity,
        }

        behav_obs_args = [kappa] + (behav_obs_kwargs or [])
        
        observation = Layer(
            mapping=mappings.EIVMapping(
                [
                    mappings.WeightedFourierBasisMapping(self.basis_params),
                    mappings.IdentityMapping(),
                ]
            ),
            noise=noise_models.EIVNoiseModel(
                [
                    _build_noise(spike_noise, spike_obs_kwargs),
                    _build_noise(behav_noise, behav_obs_args),
                ]
            ),
        )

        if sampler is None:
            sampler = Roberts(num_dims=num_dims)

        super().__init__(observation=observation, sampler=sampler, num_samples=num_samples)

        self._sim_key = sim_key if sim_key is not None else jax.random.PRNGKey(0)
        self.params_ = None

    


    def simulate(self, num_steps, true_params=None, key=None):

        if key is None:
            self._sim_key, key = jax.random.split(self._sim_key)

        if true_params is None:
            true_params = jax.random.normal(
                    key, shape=(self.params_per_neuron, self.num_neurons)
                )
            
        self.true_params = true_params
        # Base class `simulate` has signature (key, params, num_observations).
        return super().simulate(key, true_params, num_steps)



class DynamicEIV(DynamicGPLVM):

    def __init__(self, 
        len_scale, out_scale, kappa,
        num_dims,num_neurons,proposal_concentration,
        *,
        max_freq=30, basis_tol=1e-3, 
        nonlinearity=lambda x: softplus(x),
        spike_noise = noise_models.Poisson,
        behav_noise = noise_models.VonMisesNormed,
        proposal_noise = noise_models.VonMisesNormed,
        proposal_kwargs = None,
        behav_obs_kwargs = None,
        spike_obs_kwargs = None,
        num_samples=100, sim_key = None
        ):

        self.num_neurons = num_neurons
        self.num_dims = num_dims
        
        self.basis_params = {
            "max_freq": max_freq,
            "num_dims": num_dims,
            "out_scale": out_scale,
            "len_scale": len_scale,
            "num_neurons": num_neurons,
            "tol": basis_tol,
            "nonlinearity": nonlinearity,
        }

        behav_obs_args = [kappa] + (behav_obs_kwargs or [])
        proposal_args = [proposal_concentration] + (proposal_kwargs or [])
        
        proposal = Proposal(
                    layer=Layer(
                        mapping=mappings.identity,
                        noise=_build_noise(proposal_noise, proposal_args)
                    ),
                    params= jnp.array(proposal_concentration),
                    init_params=jnp.array(proposal_concentration),
                    init_loc = jnp.zeros(self.num_dims)                   
                )

        transition=Layer(
                    mapping=mappings.identity,
                    noise= _build_noise(proposal_noise, proposal_args)
                )

        observation = Layer(
            mapping=mappings.EIVMapping(
                [
                    mappings.WeightedFourierBasisMapping(self.basis_params),
                    mappings.IdentityMapping(),
                ]
            ),
            noise=noise_models.EIVNoiseModel(
                [
                    _build_noise(spike_noise, spike_obs_kwargs),
                    _build_noise(behav_noise, behav_obs_args),
                ]
            ),
        )

         
        super().__init__(transition=transition,
                        observation=observation, 
                        proposal=proposal,
                         num_samples=num_samples)

        self._sim_key = sim_key if sim_key is not None else jax.random.PRNGKey(0)
        self.params_ = None



    def simulate(self, num_steps, true_params=None, key=None):

        if key is None:
            self._sim_key, key = jax.random.split(self._sim_key)

        if true_params is None:
            true_params = jax.random.normal(
                    key, shape=(self.params_per_neuron, self.num_neurons)
                )
            
        self.true_params = true_params
        # Base class simulate(key, params, num_observations).

        xs_true, ys = super().simulate(
                    key=key,
                    params=self.true_params,
                    x_init=jnp.zeros(self.num_dims)[None,:], 
                    num_timesteps=num_steps 
                    )
        
        ys = tuple([jnp.squeeze(ys[0]), ys[1][:,0,:]])
        xs_true = jnp.squeeze(xs_true)

        return xs_true, ys

    
