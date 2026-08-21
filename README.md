# error-in-variables-garon-2026

> **Tracking the Fidelity of Internal Neural Representations with Error-In-Variables Regression**
> Isabel Garon, Stephen Keeley, Alex H. Williams
> bioRxiv 2026.04.22.720005; doi: [10.64898/2026.04.22.720005](https://doi.org/10.64898/2026.04.22.720005)

A JAX implementation of Error-in-Variables (EIV) Regression, a probabilistic framework for estimating neural tuning curves and latent population representations when behavioral or sensory measurements are imperfect proxies for the variables encoded by neural activity.

Unlike conventional encoding models, which assume behavioral observations are known exactly, EIV regression explicitly models uncertainty in the behavioral measurements. The method continuously interpolates between supervised encoding models and unsupervised latent variable models through a single interpretable hyperparameter, κ. The full pipeline yields:

- neuron-specific tuning curves
- latent neural representations
- `kappa` a description of representational fidelity between neural activity and behavior

# Overview

Suppose we record simultaneously

- neural activity **Y**
- behavioral observations **S**

Traditional tuning curve estimation assumes the observed behavior is noise-free, however internal neural representations may systematically deviate from measured behavior because of sensory noise, behavioral measurement error, or internal computation.

EIV regression introduces an unobserved latent representation


$$
S = X + \epsilon
$$

The latent variable **X** is constrained jointly by the neural population activity and the observed behavior. The strength of this coupling is governed by the **representational fidelity parameter** κ.

As

- **κ → ∞:** the model reduces to a conventional supervised encoding model (GLM with nonlinear basis functions).
- **κ → 0:** behavioral observations become uninformative and the model becomes an unsupervised Gaussian Process Latent Variable Model (GPLVM).
- **Intermediate κ:** neural activity and behavioral observations jointly constrain the latent representation.

This provides a statistically principled framework for quantifying when neural representations diverge from externally measured variables.

There are two models included in this implementation:

**EIV** — An efficient, static implementation of the latent-variable model, where time points are treated independently. Latent variables are marginalized using quasi-Monte Carlo integration.

**DynamicEIV** — A dynamic extension that incorporates temporal structure through Sequential Monte Carlo (particle filtering).
Useful when, firing rates are low, time bins are small, and latent trajectories evolve smoothly over time.


## Installation
```bash
git clone https://github.com/neurostatslab/error-in-variables-garon-2026.git
cd error-in-variables-garon-2026
pip install -r requirements.txt
```

## Quickstart

To set up the model
```python
# Generative Hyperparams
num_neurons = 100
num_dims = 1
num_steps = 3000

# Construct Model
model = EIV(len_scale = .2,
            out_scale = 50.,
            kappa = 7.,
            num_dims=num_dims,
            num_neurons=num_neurons)


```

Simulate data, or plug in your own - model is fit to `ys`, a tuple of neural observations `(T x N)` and behavioral observations `(T x D)` where `T` is the # of time points, `N` the # of neurons, and `D` is the dimensionality of the behavior.


```python

# Simulate Data
xs_true, ys = model.simulate(
    num_steps=num_steps
)

# Visualize Data
utils.plot_simulated_data_1D(xs_true, model.true_params, ys, model);
```

Fit Model 
```python
# Adjust run params, select keys for initialization & reproducibility
opt_params = {
        "opt_key": OPT_KEY,
        "init_key": INIT_KEY,
    }
model.fit(ys, "adam", opt_params)
```

Plot results

```python
utils.plot_objhist(model);
utils.plot_real_tuning(model, true_tunings)
utils.plot_latent_recon_real(model, ys, grid_reso = 100, window = 500, grid_max = 1)
```

# Choosing Hyperparameters

## Representational Fidelity (`kappa`)

The central hyperparameter of the model is the representational fidelity parameter **κ**, which determines how strongly the latent representation is coupled to the observed behavior.

- Large κ → supervised regression
- Small κ → unsupervised manifold learning
- Intermediate κ → semi-supervised latent variable model

In practice, κ should be selected by maximizing the cross-validated marginal likelihood, as described in the accompanying paper.

<p align="center">
<img src="figures/kappa_schematic.png" width="650">
</p>

## Gaussian Process Prior

Tuning curves are represented using a **weight-space Gaussian process approximation** based on a truncated Fourier basis.

Two hyperparameters govern this prior.

### `len_scale`

Controls the smoothness of tuning curves.

Smaller values permit more rapidly varying tuning functions.

### `out_scale`

Controls the prior variance (response magnitude).

<p align="center">
<img src="figures/tuning_prior.png" width="650">
</p>

---

## Dynamic Model

`DynamicEIV` introduces one additional hyperparameter,

```python
proposal_concentration
```

which controls how strongly neighboring latent states are coupled during particle filtering.

Higher values produce smoother latent trajectories, while lower values allow greater flexibility.

<p align="center">
<img src="figures/prop_concentration_fig.png" width="650">
</p>

---
### Optimization

## Structure
```
src/eiv/
  ├── __init__.py      
  ├── core.py             # Abstract model class, fit methods, layer and proposal structure
  ├── inference.py        # Implemented inference methods and batching
  ├── loader.py           # Data loader for example datasets
  ├── mappings.py         # Mappings - fourier for GP prior
  ├── mc_samplers.py      # Samplers for marginalizing latent space
  ├── noise_models.py     # Noise models for behavioral observations and spiking activity
  ├── smc.py              # Particle filter for modeling dependencies over time points
  └── utils.py            # Plotting and helper functions
```
## Citation

If you use this code, please cite:

```bibtex
@article{garon2026trackingfidelity,
  title   = {Tracking the Fidelity of Internal Neural Representations with
             Error-In-Variables Regression},
  author  = {Garon, Isabel and Keeley, Stephen and Williams, Alex H.},
  journal = {bioRxiv},
  year    = {2026},
  doi     = {10.64898/2026.04.22.720005},
  url     = {https://doi.org/10.64898/2026.04.22.720005}
}
```

## License
