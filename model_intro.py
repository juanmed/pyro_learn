import torch
import pyro
import matplotlib.pyplot as plt

#pyro.set_rng_seed(101)

def weather():
	cloudy = pyro.sample("cloudy", pyro.distributions.Bernoulli(0.3))  # returns torch.Tensor()
	#print(type(cloudy))
	cloudy = 'cloudy' if cloudy.item() == 1.0 else 'sunny'
	mean_temp = 55. if cloudy == 'cloudy' else 75.
	var_temp = 10. if cloudy == 'cloudy' else 15. 
	temp = pyro.sample("temp", pyro.distributions.Normal(mean_temp, var_temp))
	return cloudy, temp.item()

def ice_cream_sales():
    cloudy, temp = weather()
    expected_sales = 200. if cloudy == 'sunny' and temp > 80.0 else 50.
    ice_cream = pyro.sample('ice_sales', pyro.distributions.Normal(expected_sales, 10.0))
    return ice_cream.item()

def geometric(p, t=None):
    if t is None:
        t = 0
    x = pyro.sample("x_{}".format(t), pyro.distributions.Bernoulli(p))
    if x.item() == 1:
        return 0
    else:
        return 1 + geometric(p, t + 1)

def normal_product(loc, scale):
    z1 = pyro.sample("z1", pyro.distributions.Normal(loc, scale))
    z2 = pyro.sample("z2", pyro.distributions.Normal(loc, scale))
    y = z1 * z2
    return y.item()

def make_normal_normal():
    mu_latent = pyro.sample("mu_latent", pyro.distributions.Normal(0, 1))
    fn = lambda scale: normal_product(mu_latent, scale)
    return fn


mean = 0.
var = 1.
normal = torch.distributions.Normal(mean, var)
x = normal.rsample()
print("Sample: {}".format(x))
print("Log Prob: {}".format(normal.log_prob(x)))

ss = []
for _ in range(1000):
	#c, t = weather()
	#print("sky: {}, temp: {}".format(c,t))

	#print("{} sales: {}".format(_, ice_cream_sales()))

	#print("{} failures: {}".format(_, geometric(0.5)))
	ss.append(make_normal_normal()(1.))


plt.hist(ss)
plt.show()