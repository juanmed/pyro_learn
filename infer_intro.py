import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import constraints


import pyro
import pyro.infer
import pyro.optim
import pyro.distributions as dist

pyro.set_rng_seed(101)

def scale(guess):
	weight = pyro.sample("weight", dist.Normal(guess, 1.0))
	measurement = pyro.sample("measurement", dist.Normal(weight, 0.75))
	return measurement

def scale_obs(guess):  # equivalent to conditioned_scale below
    weight = pyro.sample("weight", dist.Normal(guess, 1.))
     # here we condition on measurement == 9.5
    return pyro.sample("measurement", dist.Normal(weight, 1.), obs=9.5)

def deferred_conditioned_scale(m, g):
	return pyro.condition(scale, data={"measurement": m})(g)

def perfect_guide(guess):
    loc =(0.75**2 * guess + 9.5) / (1 + 0.75**2) # 9.14
    scale = np.sqrt(0.75**2/(1 + 0.75**2)) # 0.6
    return pyro.sample("weight", dist.Normal(loc, scale))

def scale_parametrized_guide_constrained(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.), constraint=constraints.positive)
    return pyro.sample("weight", dist.Normal(a, b))  # no more torch.abs

def scale_parametrized_guide(guess):
    a = pyro.param("a", torch.tensor(guess))
    b = pyro.param("b", torch.tensor(1.))
    return pyro.sample("weight", dist.Normal(a, torch.abs(b)))

m = 9.5
conditioned_scale = pyro.condition(scale, data={"measurement": m})

guess = 8.5

pyro.clear_param_store()
svi = pyro.infer.SVI(model=conditioned_scale,
                     guide=scale_parametrized_guide,
                     optim=pyro.optim.SGD({"lr": 0.001, "momentum":0.01}),
                     loss=pyro.infer.Trace_ELBO())


losses, a,b  = [], [], []
num_steps = 2500
for t in range(num_steps):
    losses.append(svi.step(guess))
    a.append(pyro.param("a").item())
    b.append(pyro.param("b").item())

fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax2 = fig.add_subplot(2,2,2)
ax3 = fig.add_subplot(2,2,3)


ax1.plot(losses)
ax1.set_title("ELBO")
ax1.set_xlabel("step")
ax1.set_ylabel("loss")

ax2.plot([0,num_steps],[9.14,9.14], 'k:')
ax2.plot(a)
ax2.set_ylabel('a')

ax3.plot([0,num_steps],[0.6,0.6], 'k:')
ax3.plot(b)
ax3.set_ylabel('b')

plt.tight_layout()
plt.show()

print('a = ',pyro.param("a").item())
print('b = ', pyro.param("b").item())