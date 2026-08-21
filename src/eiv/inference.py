from mc_samplers import Roberts, Sobol
import jax
import optax
import smc
import time
import jax.numpy as jnp
import numpy as np
from tqdm import trange, tqdm
from jax.scipy.special import logsumexp
from functools import partial
from jax import jit

"""
inference.py
=============
Gradient-based and MCMC optimization routines for EIV model.
 
All optimizers share a common interface:
 
    opt = OptimizerClass(model, opt_params)
    opt.fit(Y)
 
After calling ``fit``, the trained parameters are stored in ``model.params_``
(point-estimate methods) or ``model.saved_params_`` (sampling methods), along
with diagnostic likelihood histories stored in ``model.objhist_`` and ``model.priorhist_``.
 
Point-estimate optimizers
-------------------------
- ``Adam``                  – single-trial Adam, supports batching and stopping conditions
- ``SGD``                   – SGD with Nesterov momentum and cosine-warmup schedule

Dependencies
------------
- JAX / JAXopt
- Optax
- tqdm

Notes
-----
- All optimizers expect ``model`` to expose a ``log_posterior_params(params, Y, key)``
  method returning a scalar log-posterior value.
- Random-number keys should be JAX ``PRNGKey`` objects.
- ``Y`` may be a single array or a tuple of arrays depending on whether the model is a GPLVM or an EIV model.

To - Do
-----
- Fix save syntax for sampling method params to be consistent
- Consolidate batching to be separate from inference methods, include random sampling method
- Simplify initialization procedure:
    if opt_params["init_params"][0]: use params else: initialize randomly
- Clarify where point estimate or full posterior
- Make pytree handling more consistent
- Make opt_params dictionary syntax more consistent
- ADD DIMENSIONS
"""


class Adam:
    """Adam optimizer for MAP estimation, with either full-batch or
     minibatch a fixed set of trial windows.

    ---> Primarily used for particle filter, smoothing over fixed windows of time
         smaller than the full recording duration

    Parameters
    ----------
    model : object
        Model instance that exposes:
        - ``log_posterior_params(params, Y_single, key) -> scalar``
        - ``observation.mapping.log_density(params) -> scalar``
    opt_params : dict
        Configuration dictionary with the following keys:

        save_prior : bool
            If ``True``, record ``log_density(params)`` at every step.
        opt_key : jax.random.PRNGKey
            Key used to generate per-step random keys and epoch shuffles.
        init_key : jax.random.PRNGKey
            Reserved initialisation key (unused when ``init_params`` is given).
        n_iters : int
            Number of gradient steps in full-batch mode, or number of
            *epochs* (full passes over all windows) in minibatch mode.
        batch_size : int or False
            Window length (in timesteps) used to reshape ``Y`` into
            ``(num_batch, batch_size, n)``. Required (not ``False``) when
            ``minibatch=True``.
        minibatch : bool
            If ``True``, train with classic minibatch SGD: each step uses
            exactly one randomly-selected window (no replacement), cycling
            through all windows once per epoch. If ``False`` (default),
            uses the original full-batch behavior: every step computes the
            gradient over *all* windows at once via ``jax.vmap``.
        scale_likelihood : bool
            Only used when ``minibatch=True``. If ``True`` (default), the
            single-window log-likelihood is scaled by ``num_batch`` so its
            expectation over a shuffled epoch matches the true full-data
            log-likelihood sum -- this keeps gradient step sizes and the
            relative weight of the prior comparable to full-batch training.
            Set ``False`` to use the raw, unscaled per-window log-likelihood.
        lr : float or optax schedule
            Learning rate for ``optax.adam``.
        tol_loss : float or False
            Relative loss-change stopping tolerance. In full-batch mode this
            is checked every step; in minibatch mode it's checked once per
            epoch (against the mean loss over that epoch), since per-step
            loss in SGD is inherently noisy. If ``False``, disables early
            stopping and runs for exactly ``n_iters`` steps/epochs.

    Attributes
    ----------
    model.params_ : pytree
        Estimated parameters after optimisation.
    model.objhist_ : list of float
        Per-step loss (full-batch mode) or per-window loss (minibatch mode).
    model.priorhist_ : list of float
        Log-prior at every step (only if ``save_prior=True``).
    model.g_norms_ : list of float
        Relative gradient norm per step, diagnostics only (full-batch,
        ``tol_loss`` set) -- not populated in minibatch mode.
    model.rel_loss_hist_ : list of float
        Relative loss change -- per step (full-batch) or per epoch (minibatch).
    model.iters_run_ : int
        Number of steps (full-batch) or epochs (minibatch) actually run.
    """

    DEFAULTS = {
            "init_params":None,
            "save_prior": True,
            "n_iters": 1000,
            "tol_loss": False,
            "minibatch": False,
            "scale_likelihood": True,
            "batch_size": False,
            "lr": 1e-1,
        }
    REQUIRED = ("opt_key", "init_key")

    def __init__(self, model, opt_params):

        allowed = set(self.DEFAULTS) | set(self.REQUIRED)
        unknown = set(opt_params) - allowed
        if unknown:
            raise ValueError(
                f"Unrecognized opt_params key(s): {sorted(unknown)}. "
                f"Allowed keys: {sorted(allowed)}"
            )

        missing = [k for k in self.REQUIRED if k not in opt_params]
        if missing:
            raise ValueError(f"opt_params is missing required key(s): {missing}")

        self.model = model
        self.opt_key = opt_params["opt_key"]
        self.init_key = opt_params["init_key"]

        for key, default in self.DEFAULTS.items():
            setattr(self, key, opt_params.get(key, default))
        
        if self.minibatch and self.batch_size == False:
            raise ValueError("minibatch=True requires a non-False `batch_size`.")

    def batch_time_series(self, x, window_size):
        """
        Reshape (t, n) -> (b, window_size, n), dropping any trailing
        timesteps that don't fill a full window.
        """
        t = x.shape[0]
        num_batch = t // window_size
        remainder = t - num_batch * window_size

        if remainder > 0:
            print(
                f"batch_time_series: dropping last {remainder} timestep(s) "
                f"(t={t} not divisible by window_size={window_size}); "
                f"kept {num_batch * window_size}/{t} samples across {num_batch} batches."
            )

        x = x[: num_batch * window_size]
        return x.reshape((num_batch, window_size) + x.shape[1:])

    def _make_optimizer_and_state(self, est_params):
        optimizer = optax.chain(
            optax.adam(learning_rate=self.lr),
            optax.scale(-1.0),
        )
        opt_state = optimizer.init(est_params)
        return optimizer, opt_state

    def fit(self, Y):
        """Run Adam optimisation (full-batch or minibatch) and store
        results on ``self.model``.

        Parameters
        ----------
        Y : tuple or array
            Batch of observed data. The first axis of each element is
            expected to index individual trials/windows.
        """

        if self.minibatch and self.batch_size == False:
            raise ValueError("minibatch=True requires a non-False `batch_size`.")


        # Full-batch mode): every step sees all windows at once via vmap.
        
        if not self.minibatch:

            if self.batch_size == False:
                objective = jax.value_and_grad(self.model.log_posterior_params)
            else:
                ys = self.batch_time_series(jnp.asarray(Y[0]), self.batch_size)
                s = self.batch_time_series(jnp.asarray(Y[1]), self.batch_size)
                Y = (ys, s)

                mapped_post = jax.vmap(
                    self.model.log_posterior_params, in_axes=(None, 0, None), out_axes=0
                )

                def total_mll(params, y, key):
                    temp = mapped_post(params, y, key)
                    return jnp.sum(temp) + self.model.observation.mapping.log_density(params)

                objective = jax.value_and_grad(total_mll)
            if self.init_params is None:
                est_params = self.model.random_init(self.init_key)
            else: 
                est_params = self.init_params
            optimizer, opt_state = self._make_optimizer_and_state(est_params)

            log_density_fn = self.model.observation.mapping.log_density
            save_prior = self.save_prior

            @jax.jit
            def step(est_params, opt_state, Y, key):
                val, grads = objective(est_params, Y, key)
                updates, opt_state = optimizer.update(grads, opt_state)
                est_params = optax.apply_updates(est_params, updates)

                prior_val = log_density_fn(est_params) if save_prior else None
                g_norm = optax.global_norm(grads)

                return est_params, opt_state, val, prior_val, g_norm

            objhist = []
            priorhist = []

            if self.tol_loss == False:
                for key in tqdm(jax.random.split(self.opt_key, self.n_iters)):
                    est_params, opt_state, val, prior_val, _ = step(
                        est_params, opt_state, Y, key
                    )
                    if save_prior:
                        priorhist.append(prior_val)
                    objhist.append(val)
            else:
                g_norms = []
                rel_loss_hist = []

                i = 0
                key = self.opt_key

                est_params, opt_state, val, prior_val, g_norm = step(
                    est_params, opt_state, Y, key
                )
                g_normer = g_norm

                objhist.append(val)
                if save_prior:
                    priorhist.append(prior_val)
                g_norms.append(g_norm / g_normer)
                rel_loss_hist.append(jnp.inf)

                prev_val = val
                
                converged = False

                while jnp.logical_and(i < self.n_iters, jnp.logical_not(converged)):
                    key, subkey = jax.random.split(key)

                    est_params, opt_state, val, prior_val, g_norm = step(
                        est_params, opt_state, Y, subkey
                    )

                    if save_prior:
                        priorhist.append(prior_val)
                    objhist.append(val)

                    rel_loss_change = jnp.abs(val - prev_val) / (jnp.abs(prev_val) + 1e-8)
                    rel_loss_hist.append(rel_loss_change)
                    converged = rel_loss_change < self.tol_loss

                    g_norms.append(g_norm / g_normer)

                    prev_val = val
                    i = i + 1

                self.model.g_norms_ = g_norms
                self.model.rel_loss_hist_ = rel_loss_hist
                self.model.iters_run_ = i

            self.model.params_ = est_params
            self.model.objhist_ = objhist
            if self.save_prior:
                self.model.priorhist_ = priorhist

            return

        # One window per step, shuffled once per epoch, no replacement, full pass = one epoch.
        
        ys = self.batch_time_series(jnp.asarray(Y[0]), self.batch_size)
        s = self.batch_time_series(jnp.asarray(Y[1]), self.batch_size)
        num_batch = ys.shape[0]

        log_posterior = self.model.log_posterior_params
        log_density_fn = self.model.observation.mapping.log_density
        save_prior = self.save_prior
        scale = float(num_batch) if self.scale_likelihood else 1.0

        def total_mll_minibatch(params, y, key):
            """Scaled single-window log-likelihood + log-prior.

            Scaling by `num_batch` makes E[this] over a shuffled epoch equal
            the true full-data log-likelihood sum, so gradient magnitude and
            prior/likelihood balance stay comparable to full-batch training.
            """
            return scale * log_posterior(params, y, key) + log_density_fn(params)

        objective = jax.value_and_grad(total_mll_minibatch)

        if self.init_params is None:
         
                est_params = self.model.random_init(self.init_key)
        else: 
            est_params = self.init_params
        optimizer, opt_state = self._make_optimizer_and_state(est_params)

        @jax.jit
        def step(est_params, opt_state, y_window, key):
            val, grads = objective(est_params, y_window, key)
            updates, opt_state = optimizer.update(grads, opt_state)
            est_params = optax.apply_updates(est_params, updates)

            prior_val = log_density_fn(est_params) if save_prior else None

            return est_params, opt_state, val, prior_val

        objhist = []
        priorhist = []
        rel_loss_hist = []

        key = self.opt_key
        epoch = 0
        prev_epoch_loss = None
        converged = False

        while epoch < self.n_iters and not converged:
            key, perm_key = jax.random.split(key)
            # Pull the shuffle to host once, then index with plain ints so
            # `step` never retraces (only index *values* change, not shapes).
            perm = np.asarray(jax.random.permutation(perm_key, num_batch))

            epoch_losses = []
            for b in tqdm(perm, desc=f"epoch {epoch}", leave=False):
                key, subkey = jax.random.split(key)
                y_window = (ys[b], s[b])

                est_params, opt_state, val, prior_val = step(
                    est_params, opt_state, y_window, subkey
                )

                objhist.append(val)
                epoch_losses.append(val)
                if save_prior:
                    priorhist.append(prior_val)

            mean_epoch_loss = jnp.mean(jnp.stack(epoch_losses))

            if self.tol_loss != False and prev_epoch_loss is not None:
                rel_loss_change = jnp.abs(mean_epoch_loss - prev_epoch_loss) / (
                    jnp.abs(prev_epoch_loss) + 1e-8
                )
                rel_loss_hist.append(rel_loss_change)
                converged = rel_loss_change < self.tol_loss
            else:
                rel_loss_hist.append(jnp.inf)

            prev_epoch_loss = mean_epoch_loss
            epoch += 1

        self.model.params_ = est_params
        self.model.objhist_ = objhist
        self.model.rel_loss_hist_ = rel_loss_hist
        self.model.iters_run_ = epoch
        if self.save_prior:
            self.model.priorhist_ = priorhist


class SGD:
    """SGD with Nesterov momentum and a cosine warm-up learning-rate schedule.
 
    Uses ``optax.sgd`` with ``nesterov=True`` and
    ``optax.warmup_cosine_decay_schedule`` to anneal the learning rate from
    ``init_value`` up to ``peak_value`` over ``warmup_steps`` steps, then
    decay back toward zero by ``total_steps``.
 
    Parameters
    ----------
    model : object
        Model instance exposing ``log_posterior_params(params, Y, key) -> scalar``.
    opt_params : dict
        Configuration dictionary with the following keys:
 
        save_prior : bool
            If ``True``, record ``log_density(params)`` at every iteration.
        opt_key : jax.random.PRNGKey
            Key for per-iteration random keys.
        init_key : jax.random.PRNGKey
            Reserved initialisation key.
        init_value : float
            Starting learning rate (before warm-up).
        peak_value : float
            Peak learning rate reached at the end of warm-up.
        warmup_steps : int
            Number of steps over which to linearly increase the LR.
        total_steps : int
            Total number of gradient steps (also the decay horizon).
 
    Attributes
    ----------
    model.params_ : pytree
        Estimated parameters after optimisation.
    model.objhist_ : list of float
        Log-posterior value at every iteration.
    model.priorhist_ : list of float
        Log-prior at every iteration (only if ``save_prior=True``).
    """

    def __init__(self, model, opt_params):
        
        self.model = model
        self.init_params = opt_params["init_params"]
        self.save_prior = opt_params["save_prior"]
        self.opt_key = opt_params["opt_key"]
        self.init_key = opt_params["init_key"]
        self.init_value = opt_params["init_value"]
        self.peak_value = opt_params["peak_value"]
        self.warmup_steps = opt_params["warmup_steps"]
        self.total_steps = opt_params["total_steps"]

    def fit(self, Y):
        """Run SGD optimisation and store results on ``self.model``.
 
        Parameters
        ----------
        Y : array-like
            Observed data passed to ``model.log_posterior_params``.
        """
        
        objective = jax.value_and_grad(self.model.log_posterior_params)
        
        warmup_exponential_decay_scheduler = \
        optax.warmup_cosine_decay_schedule(init_value=self.init_value, peak_value=self.peak_value,
                                                warmup_steps=self.warmup_steps,
                                                decay_steps=self.total_steps,
                                                end_value=0) 

        optimizer = optax.chain(
            optax.sgd(learning_rate=warmup_exponential_decay_scheduler, momentum=.99, nesterov=True),
            optax.scale(-1.0)
        )

        if self.init_params is None:
                est_params = self.model.random_init(self.init_key)
        else: 
            est_params = self.init_params

        opt_state = optimizer.init(est_params)

        objhist = []
        priorhist = []

        for key in tqdm(jax.random.split(self.opt_key, self.total_steps)):
            val, grads = objective(est_params, Y, key)
            updates, opt_state = optimizer.update(grads, opt_state)
            est_params = optax.apply_updates(est_params, updates)
                
            if self.save_prior:
                priorhist.append(self.model.observation.mapping.log_density(est_params))
            objhist.append(val)

        self.model.params_ = est_params
        self.model.objhist_ = objhist
        
        
        if self.save_prior:
            self.model.priorhist_ = priorhist
