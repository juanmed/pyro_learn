import pyro
from pyro.infer import SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist


import torch
from torch.distributions import constraints as consts

import numpy as np
import matplotlib.pyplot as plt

# defining optimizer parameters for all the parameters of a guide
adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
optimizer = Adam(adam_params)


# define optimizer parameters for specific parameters of the guide
# This allows for fine control, for example, customize learning 
# rates for different parameter
def per_param_callable(module_name, param_name):
    if param_name == "alpha_q":
        return {"lr" : 0.0001, "betas": (0.9, 0.999)}
    elif param_name == "beta_q":
        return {"lr" : 0.0001, "betas": (0.9, 0.999)}
    else:
        return {"lr": 0.001}

optimizer = Adam(per_param_callable)


# enable validation (e.g. validate parameters of distributions)
assert pyro.__version__.startswith('0.4')
pyro.enable_validation(True)

# clear the param store in case we're in a REPL
pyro.clear_param_store()

def get_coin_samples(n):
    samples = []
    for i in range(n):
        samples.append(pyro.sample("s", dist.Bernoulli(0.5)))
    return samples

def model(data):
    # model a beta distribution, these are the parameters
    alpha = torch.tensor(10.)
    beta = torch.tensor(10.)
    # the fairness f of the coin
    f = pyro.sample("latent_fairness", dist.Beta(alpha, beta))
    
    # loop over each data point, remember SVI will maximize over the
    # observation distribution
    for i in range(len(data)):
        pyro.sample("obs_{}".format(i), dist.Bernoulli(f), obs=data[i])

def guide(data):
    # register parameters of the guide
    alpha_q = pyro.param("alpha_q", torch.tensor(15.), constraint = consts.positive)
    beta_q = pyro.param("beta_q", torch.tensor(15.), constraint = consts.positive)

    # use Beta distribution as variotional distribution
    pyro.sample("latent_fairness", dist.Beta(alpha_q, beta_q))


def get_inferred_mean_std(alpha_q, beta_q):
    # here we use some facts about the beta distribution
    # compute the inferred mean of the coin's fairness
    inferred_mean = alpha_q / (alpha_q + beta_q)
    # compute inferred standard deviation
    factor = beta_q / (alpha_q * (1.0 + alpha_q + beta_q))
    inferred_std = inferred_mean * np.sqrt(factor)    
    return inferred_mean, inferred_std

# generate some sample measurements
data = get_coin_samples(10)
data = []
for _ in range(7):
    data.append(torch.tensor(1.0))
for _ in range(3):
    data.append(torch.tensor(0.0))

# define optimizer
adam_params = {"lr" : 0.0005, "betas": (0.9, 0.999)}
optimizer = Adam(per_param_callable)

# setup SVI algorithm
svi = SVI(model, guide, optimizer, loss = Trace_ELBO())



# do gradient steps
n = 5000
losses, a, b, ims, iss  = [], [], [], [], []
for step in range(n):
    losses.append(svi.step(data))
    alpha_q = pyro.param("alpha_q").item() 
    beta_q = pyro.param("beta_q").item()
    a.append(alpha_q)
    b.append(beta_q)
    im, istd = get_inferred_mean_std(alpha_q, beta_q)
    ims.append(im)
    iss.append(istd)


# grab the learned variational parameters
alpha_q = pyro.param("alpha_q").item()
beta_q = pyro.param("beta_q").item()
print('alpha_q = ',alpha_q)
print('beta_q = ', beta_q)

inferred_mean, inferred_std = get_inferred_mean_std(alpha_q, beta_q)

print("\nBased on the data and our prior belief, the fairness " +
      "of the coin is %.3f +- %.3f" % (inferred_mean, inferred_std))

fig = plt.figure()
ax1 = fig.add_subplot(2,3,1)
ax2 = fig.add_subplot(2,3,2)
ax3 = fig.add_subplot(2,3,3)
ax4 = fig.add_subplot(2,3,4)
ax5 = fig.add_subplot(2,3,5)


ax1.plot(losses)
ax1.set_title("ELBO")
ax1.set_xlabel("step")
ax1.set_ylabel("loss")

#ax2.plot([0,n],[0.53,0.53], 'k:')
ax2.plot(a)
ax2.set_ylabel('alpha_q')

#ax3.plot([0,num_steps],[0.6,0.6], 'k:')
ax3.plot(b)
ax3.set_ylabel('beta_q')

ax4.plot(ims)
ax4.set_ylabel('inferred mean')

ax5.plot(iss)
ax5.set_ylabel('inferred std')

plt.tight_layout()
plt.show()
