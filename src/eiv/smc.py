from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def filter(
        params, ys, key, initialize_proposal, sample_proposal, log_transition_prob,
        logp_y_given_x, num_particles,
    ):
    """
    Use Sequential Monte Carlo (SMC) to estimate the log marginal likelihood
    of observations, p(y_2, ..., y_t), integrating over latent trajectories
    (x_1, ..., x_t) that evolve according to a nonlinear state-space model
    parameterized by `params`.

    Notation follows Naesseth CA, Lindsten F, Schon TB (2022) "Elements of
    Sequential Monte Carlo." In their exposition, the marginal likelihood
    is equal to the normalization constant, Z_t.

    Parameters
    ----------
    params : pytree
        Parameters that define log p(y_t | x_t) and log p(x_t | x_{t-1})

    ys : jax.array with shape (T, ...)
        Observations over T timebins
    
    proposal : Callabl
        A callable object that 
    
    transition : Callable
        Transition proposal distribution. This is a function that takes
        in jax.random.PRNGKey and a set of particles sampled at time
        (t - 1) and returns proposed particle positions at time t.
    
    
    """
    
    def scanned_func(carry, inputs):

        # Unpack carry from last iteration
        x_last, weights_last = carry

        # Unpack input for this iteration.
        (k1, k2), y_t = inputs
        # selection / resampling step
        i = jax.random.choice(
            k1, num_particles,
            p=weights_last,
            shape = (num_particles,)
        )

        # propagation step (Eq. 2.15 in Naesseth)
        x_t, log_q_t = sample_proposal(k2, x_last[i])
        
        # reweighting step (Eq. 2.16 in Naesseth)
        log_weights = ( # all n_samps x n_dims
            log_transition_prob(params, x_t, x_last[i])
            + logp_y_given_x(params, tuple([obs[None,:] for obs in y_t]), x_t)
            - log_q_t
                    ) 
       
        # Compute log sum of particle weights.
        lse_weights = logsumexp(log_weights)
        
        # Normalize particle weights to sum to one.
        weights = jnp.exp(log_weights - lse_weights)

        # Average weights in log space.
        log_avg_weight = jnp.log(1 / num_particles) + lse_weights

        # Carry over x_t and log_weights, emit average log weights.
        return (x_t, weights), (x_t, weights, log_avg_weight)

    # Initialize carry tuple.
    x_init, log_q_init = initialize_proposal(key, num_particles)
    
    log_weights_init = ( 
        logp_y_given_x(params, tuple([obs[:1,:] for obs in ys]), x_init) - log_q_init
    )
    # should be size: (num_particles)
    weights_init = jnp.exp(log_weights_init - logsumexp(log_weights_init))

    init_carry = (x_init, weights_init)


    # Specify inputs at each iteration.
    inputs = (
        jax.random.split(key, (len(ys[0]), 2)), ys
    )
    # Call scan
    return jax.lax.scan(scanned_func, init_carry, inputs)[1]

@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def log_marginal_likelihood(params, y, key, prop_init,  
                    prop_samp, log_transition_prob,
                    logp_y_given_x, num_samples):

    _, _, log_avg_weights = filter(
         params, y, key, prop_init,  
                    prop_samp, log_transition_prob,
                    logp_y_given_x, num_samples
    )
    return jnp.sum(log_avg_weights)

