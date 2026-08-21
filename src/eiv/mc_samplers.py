import jax
import jax.numpy as jnp
from scipy.stats.qmc import Sobol
from jax import lax

class Roberts:
    """
    Creates random numbers tiling a hybercube [0, 1]^d where d is `num_dims`.

    Code modified from:
    https://gist.github.com/carlosgmartin/1fd4e60bed526ec8ae076137ded6ebab
    """

    def __init__(self, num_dims=1,mc_scale=1, 
                root_iters=10_000):

        self.num_dims = num_dims
        self.root_iters = root_iters
        self.mc_scale = mc_scale

        root = self._compute_root()
        # Compute basis parameter
        self.basis = 1 - (1 / root ** (1 + jnp.arange(self.num_dims)))

        
    # Compute the unique positive root of f using the Newton-Raphson method.
    def f(self,x):
        return x ** (self.num_dims + 1) - x - 1

    def _compute_root(self):
        """
        Compute the unique positive root of the function f(x) using the Newton-Raphson method.
        """
        def newton_raphson_update(x, _):
            f_val = self.f(x)
            f_grad = jax.grad(self.f)(x)
            return x - f_val / f_grad, None

        # Perform Newton-Raphson iterations using lax.scan
        # TODO - Something is deprecated here! Update jax. argument length should take the place of length of xs if no
        # xs is needed, but getting an error if not included
        root, _ = lax.scan(newton_raphson_update, 1.0, xs = jnp.arange(self.root_iters), length=self.root_iters)
        return root

    def sample(self, key, num_points):
        # Define sequence without taking modulo 1
        sequence = jnp.arange(num_points)[:, None] * self.basis[None, :]
        
        # TODO - issue with random seeding - self consistent but doesnt match demo colab, 
        # even with same seed. sequence etc match
        # print(jax.random.uniform(jax.random.PRNGKey(123), shape=[self.num_dims]))
        return jnp.modf(sequence + jax.random.uniform(key, shape=[self.num_dims]))[0] * self.mc_scale

class UniformSobol:
    ''' this is in the wrong location'''

    def __init__(self, minval=0, maxval=1, num_dims=1):
        assert maxval > minval
        self.minval = minval
        self.maxval = maxval
        self.num_dims = num_dims

    def log_density(self, params, loc):
        def log_pdf(x):
            num_dims = x.shape[1]
            return jnp.sum(jnp.full(
                x.shape,
                -num_dims * jnp.log(self.maxval - self.minval)
            ), axis=-1)
        return log_pdf
    
    def sample(self, key, num_points):
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
        # TODO - thisis wrong setting key to fixed val
        qrng = Sobol(self.num_dims, seed=0)
        xs = jnp.array(qrng.random(n=num_points) * \
                        (self.maxval-self.minval))  + self.minval
        return xs
