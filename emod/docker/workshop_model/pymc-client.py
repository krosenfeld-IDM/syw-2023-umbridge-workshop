'''
python pymc-client.py  http://localhost:4243
'''
import argparse
import numpy as np
import pymc as pm
import aesara.tensor as at
from aesara.gradient import verify_grad
import arviz as az
import matplotlib.pyplot as plt
from umbridge.pymc import UmbridgeOp


if __name__ == "__main__":
    # Read URL from command line argument
    parser = argparse.ArgumentParser(description='Minimal HTTP model demo.')
    parser.add_argument('url', metavar='url', type=str,
                        help='the ULR on which the model is running, for example http://localhost:4242')
    args = parser.parse_args()
    print(f"Connecting to host URL {args.url}")

    # Set up an aesara op connecting to UM-Bridge model
    op = UmbridgeOp(args.url, "posterior")

    # # Define input parameter
    input_dim = 3

    with pm.Model() as model:
        # UM-Bridge models with a single 1D output implementing a PDF
        # may be used as a PyMC density that in turn may be sampled
        posterior = pm.DensityDist('posterior',logp=op,shape=input_dim)

        inferencedata = pm.sample(tune=500, draws=500, step=pm.Metropolis(), cores=1, chains=1)

        az.plot_pair(inferencedata);

        plt.savefig('inference.png')

        plt.show()