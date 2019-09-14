from __future__ import print_function

import os, signal
import torch
import torch.distributions.constraints as consts
import torch.nn as nn
import torch.nn.functional as F 


import pyro
import pyro.distributions as dist
# Pyro also has a reparameterized Beta distribution so we import
# the non-reparameterized version to make our point
from pyro.distributions.testing.fakes import NonreparameterizedBeta
import pyro.optim as optim
from pyro.infer import SVI, TraceGraph_ELBO
import sys

import matplotlib.pyplot as plt

# enable validation (e.g. validate parameters of distributions)
pyro.enable_validation(True)

# this is for running the notebook in our testing framework
smoke_test = ('CI' in os.environ)
max_steps = 2 if smoke_test else 10000



def param_abs_error(name, target):
    return torch.sum(torch.abs(target - pyro.param(name))).item()

class BaselineNN(nn.Module):

    def __init__(self):
        super(BaselineNN, self).__init__()
        self.fc1 = nn.Linear(1,10)
        self.fc2 = nn.Linear(10,1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class BernoulliBetaExample(object):

    def __init__(self, max_steps):
        self.max_steps = max_steps

        # alpha, beta params of prior
        self.alpha0 = 10.
        self.beta0 = 10.

        # measurements
        self.data = torch.zeros(10)
        self.data[0:6] = torch.ones(6)
        self.n_data = self.data.size(0)

        # alpha, beta params of exact beta posterior
        self.alpha_n = self.data.sum() + self.alpha0
        self.beta_n = -self.data.sum() + torch.tensor(self.beta0 + self.n_data)

        # initial values for alpha, beta params in the guide (variational dist)
        self.alpha_q_0 = 15.
        self.beta_q_0 = 15.

        self.baseline_module = BaselineNN()

    def model(self, use_decaying_avg_baseline , x):
        f = pyro.sample("latent_fairness", dist.Beta(self.alpha0, self.beta0))

        with pyro.plate("data_plate"):
            pyro.sample("obs", dist.Bernoulli(f), obs=self.data)


    def guide(self, use_decaying_avg_baseline):

        alpha_q = pyro.param("alpha_q", torch.tensor(self.alpha_q_0), constraint = consts.positive)
        beta_q = pyro.param("beta_q", torch.tensor(self.beta_q_0), constraint = consts.positive)

        # build baseline for non parametrizable distributions
        baseline_dict = {'use_decaying_avg_baseline': use_decaying_avg_baseline,
                         'baseline_beta': 0.90}
        pyro.sample("latent_fairness", NonreparameterizedBeta(alpha_q, beta_q), infer = dict(baseline = baseline_dict))

    def nnguide(self, use_decaying_avg_baseline, x):
        pyro.module("nnbaseline", self.baseline_module)
        alpha_q = pyro.param("alpha_q", torch.tensor(self.alpha_q_0), constraint = consts.positive)
        beta_q = pyro.param("beta_q", torch.tensor(self.beta_q_0), constraint = consts.positive)
        pyro.sample("latent_fairness", NonreparameterizedBeta(alpha_q, beta_q), infer=dict(baseline={'nn_baseline': self.baseline_module,
                                                                                                     'nn_baseline_input': x}))


    def do_inference(self, use_decaying_avg_baseline, tolerance = 0.8):
        pyro.clear_param_store()
        optimizer_params = {"lr": 0.0005, "betas" : (0.93, 0.999)}
        #optimizer = optim.Adam(optimizer_params)
        optimizer = optim.Adam(self.per_param_args)
        svi = SVI(self.model, self.nnguide, optimizer, loss = TraceGraph_ELBO())
        print("Doing inference with use_decaying_avg_baseline=%s" % use_decaying_avg_baseline)

        ae, be = [], []
        # do up to this many steps of inference
        for k in range(self.max_steps):
            svi.step(use_decaying_avg_baseline, torch.tensor([1.0]))
            if k % 100 == 0:
                print('.', end='')
                sys.stdout.flush()

            # compute the distance to the parameters of the true posterior
            alpha_error = param_abs_error("alpha_q", self.alpha_n)
            beta_error = param_abs_error("beta_q", self.beta_n)

            ae.append(alpha_error)
            be.append(beta_error)
            # stop inference early if we're close to the true posterior
            if alpha_error < tolerance and beta_error < tolerance:
                print("Stopped early at step: {}".format(k))
                break
        return (ae,be)

    def per_param_args(self,module_name, param_name):
        if param_name == "alpha_q":
            return {"lr" : 0.0005, "betas": (0.93, 0.999)}
        elif param_name == "beta_q":
            return {"lr" : 0.0005, "betas": (0.93, 0.999)}
        if "nnbaseline" in param_name or "nnbaseline" in module_name:
            return {"lr" : 0.001 }
        else:
            return {"lr": 0.001}


# do the experiment
bbe = BernoulliBetaExample(max_steps=max_steps)
e1 = bbe.do_inference(use_decaying_avg_baseline=True)
e2 = bbe.do_inference(use_decaying_avg_baseline=False)
print("alpha: inferred: {:.4f}  real: {:.4f}".format(pyro.param("alpha_q").item(), bbe.alpha_n))
print("beta: inferred: {:.4f}  real: {:.4f}".format(pyro.param("beta_q").item(), bbe.beta_n))

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

ax1.set_title("Alpha Inference Error")
ax1.plot(e1[0], color = 'r', label = 'w/baseline')
ax1.plot(e2[0], color = 'b', label = 'wo/baseline')
ax1.set_xlabel('step')
ax1.set_ylabel('value')
ax1.legend(loc = 'upper right')

ax2.set_title("Beta Inference Error")
ax2.plot(e1[1], color = 'r', label = 'w/baseline')
ax2.plot(e2[1], color = 'b', label = 'wo/baseline')
ax2.set_xlabel('step')
ax2.set_ylabel('value')
ax2.legend(loc = 'upper right')

plt.show()

# Kill process on exit
print("\nKilling Process... bye!")
os.kill(os.getpid(), signal.SIGKILL)