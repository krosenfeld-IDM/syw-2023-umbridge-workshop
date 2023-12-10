"""
Outbreak estimate given uncertainty in R0 using pymc

Reference:
- https://www.pymc.io/projects/examples/en/latest/howto/blackbox_external_likelihood_numpy.html
"""
import json
import arviz as az
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import scipy.optimize     as spopt

import paths

print(f"Running on PyMC v{pm.__version__}")

# Reference trajectory (Kermack-McKendric analytic solution)
def KMlimt(theta, x):
    """ Return R0 given Z """
    Z = theta
    return -np.log(1-Z)/Z

def my_loglike(theta, x, data, sigma):
    model = KMlimt(theta, x)
    return -(0.5 / sigma**2) * np.sum((data - model) ** 2)

# define a pytensor Op for our likelihood function
class LogLike(pt.Op):

    """
    Specify what type of object will be passed and returned to the Op when it is
    called. In our case we will be passing it a vector of values (the parameters
    that define our model) and returning a single "scalar" value (the
    log-likelihood)
    """

    itypes = [pt.dvector]  # expects a vector of parameter values when called
    otypes = [pt.dscalar]  # outputs a single scalar value (the log likelihood)

    def __init__(self, loglike, data, x, sigma):
        """
        Initialise the Op with various things that our log-likelihood function
        requires. Below are the things that are needed in this particular
        example.

        Parameters
        ----------
        loglike:
            The log-likelihood (or whatever) function we've defined
        data:
            The "observed" data that our log-likelihood function takes in
        x:
            The dependent variable (aka 'x') that our model requires
        sigma:
            The noise standard deviation that our function requires.
        """

        # add inputs as class attributes
        self.likelihood = loglike
        self.data = data
        self.x = x
        self.sigma = sigma

    def perform(self, node, inputs, outputs):
        # the method that is used when calling the Op
        (theta,) = inputs  # this will contain my variables

        # call the log-likelihood function
        logl = self.likelihood(theta, self.x, self.data, self.sigma)

        outputs[0][0] = np.array(logl)  # output the log-likelihood

R0_ranges = [[1.3, 1.8], [1.4, 2.2], [1.5, 2], [1.7, 1.8], [1.5, 2]]
names = ['Flu', 'Mumps', 'Smallpox', 'Diptheria', 'Rubella']

res = {}
for ii in range(len(R0_ranges)):
    # get R0 range and name
    R0_range = R0_ranges[ii]
    name = names[ii]

    # set up our data
    N = 1  # number of data points (just the one R0)
    sigma = (R0_range[1]-R0_range[0])/3  # standard deviation of noise
    R0_true = np.mean(R0_range) # true R0

    # make data
    x = np.random.random(N)
    rng = np.random.default_rng(716743)
    data = sigma * rng.normal(size=N) + R0_true

    # create our Op
    logl = LogLike(my_loglike, data, x, sigma)

    # use PyMC to sampler from log-likelihood
    with pm.Model():
        # uniform prior
        Z = pm.Uniform("Z", lower=0.0, upper=1.0)
        # Z = pm.Normal("Z", mu=0.7, sigma=0.1)
        # convert Z to a tensor vector
        theta = pt.as_tensor_variable([Z])

        # use a Potential to "call" the Op and include it in the logp computation
        pm.Potential("likelihood", logl(theta))

        # Use custom number of draws to replace the HMC based defaults
        idata_mh = pm.sample(3000, tune=1000, chains=4)

    # save data
    idata_mh.to_json(paths.output / "idata_{}.json".format(name))

    print(az.summary(idata_mh, round_to=4))

# plot the traces
# az.plot_trace(idata_mh)
ax = az.plot_posterior(idata_mh)
ax.set_xlim(0, 1)
plt.savefig(paths.figures / "tmp.png")


