---
title: L, epsilon, and other tweaking
shorttitle: hmctweaking
notebook: hmctweaking.ipynb
noline: 1
summary: ""
keywords: ['energy', 'hamiltonian monte carlo', 'nuts', 'leapfrog', 'canonical distribution', 'microcanonical distribution', 'transition distribution', 'marginal energy distribution', 'data augmentation', 'classical mechanics', 'detailed balance', 'statistical mechanics', 'divergences', 'step-size', 'non-centered hierarchical model', 'hierarchical']
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}




```python
%matplotlib inline
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("poster")
import pymc3 as pm
```




## Installation woes

To follow along on this notebook, we are going to use a cutting-edge version of pymc3: the developers add features very fast, and a feature we'd like to use here is the ability of pymc3 to tell us which integrations diverged.

This is what you need to do:

```
pip install theano==0.9
pip install pymc3==3.1rc2
```



```python
pm.__version__
```





    '3.1.rc2'



## Centered parametrization

We'll set up the modelled in what is called a "Centered" parametrization which tells us how $\theta_i$ is modelled: it is written to be directly dependent as a normal distribution from the hyper-parameters. 



```python
J = 8
y = np.array([28,  8, -3,  7, -1,  1, 18, 12])
sigma = np.array([15, 10, 16, 11,  9, 11, 10, 18])
```


We set up our priors in a Hierarchical model to use this centered parametrization. We can say: the $\theta_j$ is drawn from a Normal hyper-prior distribution with parameters $\mu$ and $\tau$. Once we get a $\theta_j$ then can draw the means from it given the data $\sigma_j$ and one such draw corresponds to our data.

$$
\mu \sim \mathcal{N}(0, 5)\\
\tau \sim \text{Half-Cauchy}(0, 5)\\
\theta_{j} \sim \mathcal{N}(\mu, \tau)\\
\bar{y_{j}} \sim \mathcal{N}(\theta_{j}, \sigma_{j})
$$

where $j \in \{1, \ldots, 8 \}$ and the
$\{ y_{j}, \sigma_{j} \}$ are given as data



```python
with pm.Model() as schools1:

    mu = pm.Normal('mu', 0, sd=5)
    tau = pm.HalfCauchy('tau', beta=5)
    theta = pm.Normal('theta', mu=mu, sd=tau, shape=J)
    obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)
```




```python
with schools1:
    trace1 = pm.sample(5000, init=None, njobs=2, tune=1000)
```


    100%|██████████| 5000/5000 [00:29<00:00, 191.81it/s]   | 1/5000 [00:00<08:32,  9.76it/s]




```python
pm.diagnostics.gelman_rubin(trace1), pm.diagnostics.effective_n(trace1)
```





    ({'mu': 1.0005631416817737,
      'tau': 1.000052543085129,
      'tau_log_': 1.0006214172273011,
      'theta': array([ 0.99990364,  0.99990547,  1.00000407,  1.0005178 ,  1.00064279,
              1.0006689 ,  1.00002825,  0.99999665])},
     {'mu': 1482.0,
      'tau': 732.0,
      'tau_log_': 189.0,
      'theta': array([ 2406.,  2950.,  2901.,  3135.,  2428.,  3019.,  1768.,  3109.])})



The Gelman-Rubin statistic seems fine, but notice how small the effective-n's are? Something is not quite right. Lets see traceplots.



```python
pm.traceplot(trace1);
```



![png](hmctweaking_files/hmctweaking_12_0.png)


Its hard to pick the thetas out but $\tau$ looks not so white-noisy. Lets zoom in:



```python
pm.traceplot(trace1, varnames=['tau_log_'])
```





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x1201da710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x120397a90>]], dtype=object)




![png](hmctweaking_files/hmctweaking_14_1.png)


There seems to be some stickiness at lower values in the trace. Zooming in even more helps us see this better:



```python
plt.plot(trace1['tau_log_'], alpha=0.6)
plt.axvline(5000, color="r")
#plt.plot(short_trace['tau_log_'][5000:], alpha=0.6);
```





    <matplotlib.lines.Line2D at 0x12040f2b0>




![png](hmctweaking_files/hmctweaking_16_1.png)


### Tracking divergences



```python
divergent = trace1['diverging']
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size/len(trace1)
print('Percentage of Divergent %.5f' % divperc)
```


    Number of Divergent 71
    Percentage of Divergent 0.01420




```python
def biasplot(trace):
    logtau = trace['tau_log_']
    mlogtau = [np.mean(logtau[:i]) for i in np.arange(1, len(logtau))]
    plt.figure(figsize=(8, 2))
    plt.axhline(0.7657852, lw=2.5, color='gray')
    plt.plot(mlogtau, lw=2.5)
    plt.ylim(0, 2)
    plt.xlabel('Iteration')
    plt.ylabel('MCMC mean of log(tau)')
    plt.title('MCMC estimation of cumsum log(tau)')
```




```python
biasplot(trace1)
```



![png](hmctweaking_files/hmctweaking_20_0.png)




```python
def funnelplot(trace):
    logtau = trace['tau_log_']
    divergent = trace['diverging']
    theta_trace = trace['theta']
    theta0 = theta_trace[:, 0]
    plt.figure(figsize=(5, 3))
    plt.scatter(theta0[divergent == 0], logtau[divergent == 0], color='r')
    plt.scatter(theta0[divergent == 1], logtau[divergent == 1], color='g')
    plt.axis([-20, 50, -6, 4])
    plt.ylabel('log(tau)')
    plt.xlabel('theta[0]')
    plt.title('scatter plot between log(tau) and theta[0]')
    plt.show()
```




```python
funnelplot(trace1)
```



![png](hmctweaking_files/hmctweaking_22_0.png)




```python
np.mean(trace1['mean_tree_accept'])
```





    0.7829179856279076



### Where are the divergences coming from?

Divergences can be a sign of the symplectic integration going off to infinity, or a false positive. False positives occur because instead of waiting for infinity, some heuristics are used. This is typically true of divergences not deep in the funnel, where the curvature of the target distribution is high.





## The effect of step-size

Looking at the docs for the `NUTS` sampler at https://pymc-devs.github.io/pymc3/api/inference.html#module-pymc3.step_methods.hmc.nuts , we see that we can co-erce a smaller step-size $\epsilon$, and thus an ok symplectic integration from our sampler by increasing the target acceptance rate.

If we do this, then we have geometric ergodicity (we go everywhere!) between the Hamiltonian transitions (ie in the leapfrogs) and the target distribution. This should result in the divergence rate decreasing.

But if for some reason we do not have geometric ergodicity, then divergences will persist. This can happen deep in the funnel, where even drastic decreases in the step size are not able to explore the highly curved geometry.




```python
with schools1:
    step = pm.NUTS(target_accept=.85)
    trace1_85 = pm.sample(5000, step=step, init=None, njobs=2, tune=1000)
with schools1:
    step = pm.NUTS(target_accept=.90)
    trace1_90 = pm.sample(5000, step=step, init=None, njobs=2, tune=1000)
with schools1:
    step = pm.NUTS(target_accept=.95)
    trace1_95 = pm.sample(5000, step=step, init=None, njobs=2, tune=1000)
with schools1:
    step = pm.NUTS(target_accept=.99)
    trace1_99 = pm.sample(5000, step=step, init=None, njobs=2, tune=1000)
```


    100%|██████████| 5000/5000 [00:38<00:00, 130.79it/s]   | 10/5000 [00:00<00:50, 99.63it/s]
    100%|██████████| 5000/5000 [00:56<00:00, 88.53it/s] 
    100%|██████████| 5000/5000 [00:50<00:00, 98.55it/s] 
    100%|██████████| 5000/5000 [12:51<00:00,  3.38it/s]




```python
for t in [trace1_85, trace1_90, trace1_95, trace1_99]:
    print("Acceptance", np.mean(t['mean_tree_accept']), "Step Size", np.mean(t['step_size']), "Divergence", np.sum(t['diverging']))
```


    Acceptance 0.804601458758 Step Size 0.203087336483 Divergence 39
    Acceptance 0.873340820433 Step Size 0.159223726996 Divergence 18
    Acceptance 0.923346597897 Step Size 0.126824682121 Divergence 9
    Acceptance 0.990173791609 Step Size 0.0164237997757 Divergence 5




```python
for t in [trace1_85, trace1_90, trace1_95, trace1_99]:
    biasplot(t)
    funnelplot(t)
```



![png](hmctweaking_files/hmctweaking_28_0.png)



![png](hmctweaking_files/hmctweaking_28_1.png)



![png](hmctweaking_files/hmctweaking_28_2.png)



![png](hmctweaking_files/hmctweaking_28_3.png)



![png](hmctweaking_files/hmctweaking_28_4.png)



![png](hmctweaking_files/hmctweaking_28_5.png)



![png](hmctweaking_files/hmctweaking_28_6.png)



![png](hmctweaking_files/hmctweaking_28_7.png)




```python
plt.plot(trace1_99['tau_log_'])
```





    [<matplotlib.lines.Line2D at 0x11cc13cf8>]




![png](hmctweaking_files/hmctweaking_29_1.png)


The divergences decrease, but dont totally go away, showing that we have lost some geometric ergodicity. And as we get to a very small step size we explore the funnel much better, but we are now taking our sampler more into a MH like random walk regime, and our sampler looks very strongly autocorrelated.

We know the fix, it is to move to a

## Non-centered paramerization

We change our model to:

$$
\mu \sim \mathcal{N}(0, 5)\\
\tau \sim \text{Half-Cauchy}(0, 5)\\
\nu_{j} \sim \mathcal{N}(0, 1)\\
\theta_{j} = \mu + \tau\nu_j \\
\bar{y_{j}} \sim \mathcal{N}(\theta_{j}, \sigma_{j})
$$

Notice how we have factored the dependency of $\theta$ on $\phi = \mu, \tau$ into a deterministic
transformation between the layers, leaving the
actively sampled variables uncorrelated. 

This does two things for us: it reduces steepness and curvature, making for better stepping. It also reduces the strong change in densities, and makes sampling from the transition distribution easier.



```python
with pm.Model() as schools2:
    mu = pm.Normal('mu', mu=0, sd=5)
    tau = pm.HalfCauchy('tau', beta=5)
    nu = pm.Normal('nu', mu=0, sd=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * nu)
    obs = pm.Normal('obs', mu=theta, sd=sigma, observed=y)
```




```python
with schools2:
    trace2 = pm.sample(5000, init=None, njobs=2, tune=1000)
```


    100%|██████████| 5000/5000 [00:11<00:00, 426.83it/s]   | 1/5000 [00:00<15:26,  5.40it/s]


And we reach the true value better as the number of samples increases, decreasing our bias



```python
biasplot(trace2)
```



![png](hmctweaking_files/hmctweaking_35_0.png)


How about our divergences? They have decreased too.



```python
divergent = trace2['diverging']
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size/len(trace2)
print('Percentage of Divergent %.5f' % divperc)
```


    Number of Divergent 13
    Percentage of Divergent 0.00260




```python
funnelplot(trace2)
```



![png](hmctweaking_files/hmctweaking_38_0.png)


The divergences are infrequent and do not seem to concentrate anywhere, indicative of false positives. Lowering the step size should make them go away.

### A smaller step size



```python
with schools2:
    step = pm.NUTS(target_accept=.95)
    trace2_95 = pm.sample(5000, step=step, init=None, njobs=2, tune=1000)
```


    100%|██████████| 5000/5000 [00:19<00:00, 256.73it/s]   | 14/5000 [00:00<00:36, 134.85it/s]




```python
biasplot(trace2_95)
```



![png](hmctweaking_files/hmctweaking_42_0.png)




```python
funnelplot(trace2_95)
```



![png](hmctweaking_files/hmctweaking_43_0.png)


Indeed at a smaller step-size our false-positive divergences go away, and the lower curvature in our parametrization ensures geometric ergodicity deep in our funnel

## Path length L

If we choose too small a $L$ we are returning our HMC sampler to a random walk. How long must a leapfrog run explore a level set of the Hamiltonian (ie of the canonical distribution $p(p,q)$ beofre we force an accept-reject step and a momentum resample?

Clearly if we go too long we'll be coming back to the neighborhood of areas we might have reached in smaller trajectories. NUTS is one approach to adaptively fix this by not letting trajectories turn on themselves.

In the regular HMC sampler, for slightly complex problems, $L=100$ maybe a good place to start. For a fixed step-size $\epsilon$, we can now check the level of autocorrelation. If it is too much, we want a larger $L$.

Now, the problem with a fixed $L$ is that one $L$ does not work everywhere in a distribution. To see this, note that tails correspond to much higher energies. Thus the level-set surfaces are larger, and a fixed length $L$ trajectory only explores a small portion of this set before a momentum resampling takes us off. This is why a dynamic method like NUTS is a better choice.

## Tuning HMC(NUTS)

This requires preliminary runs. In `pymc3` some samples are dedicated to this, and an adaptive tuning is carried out according to algorithm 6 in the original NUTS paper: http://www.stat.columbia.edu/~gelman/research/published/nuts.pdf .

But we have seem how to play with step-size within the context of the NUTS sampler, something which we might need to do for tetchy models. Clearly too large an $\epsilon$ can lead to loss of symplecticity. And too small a step-size will get us to a random walk for finite sized $L$. Clearly the adaptive approach in NUTS is a good idea.

In pymc3's regular HMC sampler a knob `step_rand` is available which allows us a distribution to sample a step-size from. In the case that we are not adaptively tuning as in NUTS, allowing for this randomness allows for occasionally small values of $\epsilon$ even for large meaned distributions. Note that this should be done at the star of leapfrog, not in-between.

Another place where this is useful is where the exact solution to the Hamiltonian equations (gaussian distribution, harmonic oscillator hamiltonian resulting) has periodicity. If $L\epsilon$ is chosen to be $2\pi$ then our sampler will lack ergodicity. In such a case choose $\epsilon$ from a distribution (or for that matter, $L$).

Finally there are multiple ways to tweak the mass matrix. One might use a variational posterior to obtain a approximate covariance matrix for the target distribution $p(q)$. Or one could use the tuning samples for this purpose. But choosing the mass matrix as the inverse of the covariance matrix of the target is highly recommended, as it will maximally decorrelate parameters of the target distribution.

The covariance matrix also establishes a scale for each parameter. This scale can be used to tweak step size as well. Intuitively the variances are measures of curvature along a particular dimension, and choosing a stepsize in each parameter which accomodates this difference is likely to help symplecticity. I do not believe this optimization is available within pymc3. This does not mean you are out of luck: you could simply redefine the parameters in a scaled form.

If you are combining HMC with other samplers, such as MH for discrete parameters in a gibbs based conditional scheme, then you might prefer smaller $L$ parameters to allow for the other parameters to be updated faster.


## Efficiency of momentum resampling

When we talked about the most Hamiltonian-trajectory momentum resampling, we talked about its efficiency. The point there was that you want the marginal energy distribution to match the transition distribution induced by momentum resampling.

`pymc3` gives us some handy stats to calculate this:



```python
def resample_plot(t):
    sns.distplot(t['energy']-t['energy'].mean(), label="P(E)")
    sns.distplot(np.diff(t['energy']), label = "p(E | q)")
    plt.legend();
    plt.xlabel("E - <E>")
    
```


So let us see this for our original trace



```python
resample_plot(trace1);
```


    //anaconda/envs/py35/lib/python3.5/site-packages/numpy/lib/function_base.py:564: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      n = np.zeros(bins, ntype)
    //anaconda/envs/py35/lib/python3.5/site-packages/numpy/lib/function_base.py:600: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      n += np.bincount(indices, weights=tmp_w, minlength=bins).astype(ntype)
    //anaconda/envs/py35/lib/python3.5/site-packages/statsmodels/nonparametric/kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j



![png](hmctweaking_files/hmctweaking_50_1.png)


Awful. The momentum resamples here will do a very slow job of traversing this distribution. This is indicative of the second issue we were having with this centered model (the first was a large step size for the curvature causing loss of symplectivity): the momentum resampling simply cannot provide enough energy to traverse the large energy changes that occur in this hierarchical model.



```python
resample_plot(trace1_95)
```


    //anaconda/envs/py35/lib/python3.5/site-packages/statsmodels/nonparametric/kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j



![png](hmctweaking_files/hmctweaking_52_1.png)


Note the caveat with such a plot obtained from our chains: it only tells us about the energies it explored: not the energies it ought to be exploring, as can be seen in the plot with `trace1_99` below. Still, a great diagnostic.



```python
resample_plot(trace1_99)
```


    //anaconda/envs/py35/lib/python3.5/site-packages/numpy/lib/function_base.py:564: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      n = np.zeros(bins, ntype)
    //anaconda/envs/py35/lib/python3.5/site-packages/numpy/lib/function_base.py:600: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      n += np.bincount(indices, weights=tmp_w, minlength=bins).astype(ntype)
    //anaconda/envs/py35/lib/python3.5/site-packages/statsmodels/nonparametric/kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j



![png](hmctweaking_files/hmctweaking_54_1.png)


The match is much better for the non-centered version of our model.



```python
resample_plot(trace2)
```


    //anaconda/envs/py35/lib/python3.5/site-packages/statsmodels/nonparametric/kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j



![png](hmctweaking_files/hmctweaking_56_1.png)




```python
resample_plot(trace2_95)
```


    //anaconda/envs/py35/lib/python3.5/site-packages/statsmodels/nonparametric/kdetools.py:20: VisibleDeprecationWarning: using a non-integer number instead of an integer will result in an error in the future
      y = X[:m/2+1] + np.r_[0,X[m/2+1:],0]*1j



![png](hmctweaking_files/hmctweaking_57_1.png)




```python

```

