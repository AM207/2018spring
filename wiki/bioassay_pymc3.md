---
title: Lab  7 - Bayesian inference with PyMC3.
shorttitle: bioassay_pymc3
notebook: bioassay_pymc3.ipynb
noline: 1
summary: ""
keywords: ['bioassay', 'hierarchical', 'binomial regression', 'regression', 'gewecke', 'gelman-rubin']
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}


## Learning Aims

- Using PyMC3
- Building probability models
- MAP estimates with PyMC3
- Sampling with PyMC3
- Convergence tests
$\newcommand{\Ex}{\mathbb{E}}$
$\newcommand{\Var}{\mathrm{Var}}$
$\newcommand{\Cov}{\mathrm{Cov}}$
$\newcommand{\SampleAvg}{\frac{1}{N({S})} \sum_{s \in {S}}}$
$\newcommand{\indic}{\mathbb{1}}$
$\newcommand{\avg}{\overline}$
$\newcommand{\est}{\hat}$
$\newcommand{\trueval}[1]{#1^{*}}$
$\newcommand{\Gam}[1]{\mathrm{Gamma}#1}$




```python
%matplotlib inline
import numpy as np
from scipy import stats
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('paper')
import pandas as pd

import pymc3 as pm
from pymc3 import Model, Normal, HalfNormal
import time
```


# A few words about PyMC3

PyMC3 is an open source probabilistic programming tool that uses a very natural syntax. It features state-of-the-art MCMC sampling methods such as the No-U-Turn-Sampler (NUTS) and Hamiltonian Monte Carlo (HMC). PyMC3 uses Theano to perform authomatic differentiation, which makes it very efficient in calculating the gradients used in these methods. In this lab we will be using PyMC3 to make inference the bioassay experiment. We blatantly steal some text and ideas from fonnesbeck's notebooks for bioinformatics. We also use material from the PyMC3 official tutorials: http://pymc-devs.github.io/pymc3/notebooks/getting_started.html

The idea behind PyMC3 is to make Bayesian inference and sampling as simple as possible. Some of the features of PyMC3 are:

* Fits Bayesian statistical models with Markov chain Monte Carlo, variational inference and other algorithms.
* Includes a large suite of well-documented statistical distributions.
* Creates summaries including tables and plots.
* Traces can be saved to the disk as plain text, SQLite or pandas dataframes.
* Several convergence diagnostics and model checking methods are available.
* Extensible: easily incorporates custom step methods and unusual probability distributions.
* MCMC loops can be embedded in larger programs, and results can be analyzed with the full power of Python.

# Bioassay experiment

This code follows an example from Gelman's "Bayesian Data Analysis".

We demonstrate this with an example and examine the convergence of the resulting samples. 

Bioassay (commonly used shorthand for biological assay), or biological standardisation is a type of scientific experiment. Bioassays are typically conducted to measure the effects of a substance on a living organism and are essential in the development of new drugs and in monitoring environmental pollutants. Both are procedures by which the potency (pharmacology) or the nature of a substance is estimated by studying its effects on living matter.

In this experiment, various drug doses are administered to animals and a binary outcome (death) is noted. There are 4 groups of 5 animals each, different doses administered, and deaths recorded. We construct a model for $\theta$ the binomial probabiliy of death, as a regression on dose through the logit$^{-1}$ function with two parameters (see below). We set imprecise normal priors on the regression coefficients, and pass the linear regression through the inverse logit function into a binomial likelihood.

| Dose, x_i  | Number of   | Number of    | 
| log(g/ml)  | animals,n_i |  deaths, y_i |
|:-----------|------------:|:------------:|
| -0.86      |     5       |     0        |
| -0.30      |     5       |     1        |
| -0.05      |     5       |     3        |
|  0.73      |     5       |     5        |

Let us enter these data:



```python
# Use a theano shared variable to be able to exchange the data the model runs on
from theano import shared

# Log dose in each group
log_dose = np.array([-.86, -.3, -.05, .73])

# Let's make this a theano shared variable so that we can make predictions for new values later
log_dose_shared = shared(log_dose)

# Sample size in each group
n = 5 * np.ones(4, dtype=int)

# The sample size has to be a shared variable too
n_shared = shared(n)

# Outcomes
deaths = np.array([0, 1, 3, 5])
```


The likelihood since is a success/fail experiment is expressed as a Binomial:

$$ P(D_i|\theta_i) = p(y_i, n_i| \theta_i) = {\rm Bionomial}( y_i, n_i \vert \theta_i)  \,\,\,\,    \rm{for}\,\, i=1, \ldots, 4$$

where $\theta$ is the rate of deaths which is modeled as a $\rm{logit}^{-1}$:  

$$ \theta_i = {\rm logit}^{-1}(\alpha+\beta x_i) \,\,\,\, \rm{for} \,\, i=1, \ldots, 4$$

We use flat priors for $\alpha,\beta$: 

$$ p(\alpha, \beta) \propto 1 $$

Posterior: 

$$ p(\alpha, \beta \vert y) \propto p(\alpha, \beta) \prod_{i=1}^{k} p(y_i \vert \alpha, \beta, n_i, x_i) $$
$$ = 1 \prod_{i=1}^{k} [{\rm logit}^{-1}(\alpha+\beta x_i)]^{y_i} [ 1-{\rm logit}^{-1}(\alpha+\beta x_i)]^{n_i-y_i} $$

We can first calculate this posterior by frute force (assuming uniform priors), and also plot some of the inverse logistic as a function of different $\alpha$ and $\beta$ values in order to have some intuition about the model:



```python
## invLogit this is the same as a sigmoid
logitInv= lambda x: np.exp(x)/(1.0+np.exp(x))

## posterior
def calc_posterior(a, b, y=deaths, x=log_dose):
    # Calculate joint posterior, given values for a, b 
    # x: dosage
    # y: number of deaths
    # a + b: parameters of the model
    p = np.product((logitInv(a+b*x)**y)*(1.0-logitInv( a+b*x))**(n-y))
    
    return p
```




```python
# some intuition about this model
xx = np.linspace(-10,10,100)
plt.plot(xx, logitInv(xx), label="alpha=1, beta=0")
plt.plot(xx, logitInv(0.4*xx), label="alpha=0.4, beta=0")
plt.plot(xx, logitInv(xx+3), label="alpha=1, beta=3")
plt.xlabel("dosage")
plt.ylabel("all dead")
plt.legend(loc=4)
```





    <matplotlib.legend.Legend at 0x111a224e0>




![png](bioassay_pymc3_files/bioassay_pymc3_11_1.png)


We see that for a given dosage, different values of $\alpha$ and $\beta$ can yield very different death rates. Now let us plot the posterior probability as a function of these two parameters:



```python
# check my posterior. I do a brute force just to make sure I do the right thing
X1 = np.linspace(-3,7,101) # alpha
X2 = np.linspace(-10, 30,100) # beta
k=0;j=0
pp=np.zeros((101,100))
for x1 in X1:
    j=0
    for x2 in X2:
        pp[k,j]=calc_posterior(x1,x2)
        j +=1
    k +=1
```




```python
# look at the posterior distribution
plt.contourf(X1,X2,pp.T)
plt.xlabel(r"$\alpha$",size=20)
plt.ylabel(r"$\beta$",size=20)
plt.colorbar()
```





    <matplotlib.colorbar.Colorbar at 0x111cae780>




![png](bioassay_pymc3_files/bioassay_pymc3_14_1.png)


## Setting up the model in PyMC3

The first step is to specify the probabilistic model in PyMC3. A probabilistic model relates unknown variables to data. PyMC3 provides the basic building blocks for Bayesian probability models: stochastic random variables and deterministic variables. A **stochastic random variable** is a factor whose value is not completely determined by its parents, while the value of a **deterministic random variable** is entirely determined by its parents. Most models can be constructed using only these two variable types. Typically, after instantiating our model object, we first set up the priors, then we write expressions for the deterministic variables, and finally we write the data likelihood as a function of the variables:



```python
from pymc3 import Normal, Binomial, sample, Model # Import relevant distributions
from pymc3.math import invlogit

with Model() as bioassay_model:

    # Logit-linear model parameters
    alpha = Normal('alpha', 0, sd=100)
    beta = Normal('beta', 0, sd=100)

    # Calculate probabilities of death
    theta = invlogit(alpha + beta * log_dose_shared)

    # Data likelihood
    obs_deaths = Binomial('obs_deaths', n=n_shared, p=theta, observed=deaths)
```


At the model-specification stage (before the data are observed), $\alpha$, $\beta$, $\theta$, and $y$ (the observed number of deaths) are all random variables. 

The only deterministic variable in this model is $\theta$. If we knew the values of $\theta$'s parents ($\alpha$ and $\beta$), we could compute the value of $\theta$ exactly. A deterministic like $\theta$ is defined by a mathematical function that returns its value given values for its parents. On the other hand, even if the values of the parents of variables $\alpha$, $\beta$ and $y$ (before observing the data) were known, we would still be uncertain of their values. These variables are stochastic, characterized by probability distributions that express how plausible their candidate values are, given values for their parents.

Using the model we have specified above, we can now do a number of things. For example, we can obtain maximum a posteriori (MAP) point estimate for the parameters. But we want to know about the uncertainties of these parameters, and we can only get that information by getting samples from the posterior (assuming we can not solve this analytically). Therefore, we can obtain samples from the posterior using PyMC3's ```sample``` function. It will fit probability models (linked collections of variables) like ours using Markov chain Monte Carlo (MCMC) sampling. Unless we manually assign particular algorithms to variables in our model, PyMC will assign algorithms that it deems appropriate (it usually does a decent job of this):

### Finding MAP point estimates
The maximum a posteriori (MAP) estimate for a model, is the mode of the posterior distribution and is generally found using numerical optimization methods. This is often fast and easy to do, but only gives a point estimate for the parameters and can be biased if the mode isn’t representative of the distribution. PyMC3 provides this functionality with the ```find_MAP```  function.



```python
# Obtain a MAP point estimate
from pymc3 import find_MAP

map_estimate = find_MAP(model=bioassay_model)

print(map_estimate)

```


    logp = -13.034, ||grad|| = 0.00043389: 100%|██████████| 14/14 [00:00<00:00, 99.51it/s]  


    {'alpha': array(0.8437566864565978), 'beta': array(7.730204614414374)}


By default, ```find_MAP``` uses the Broyden–Fletcher–Goldfarb–Shanno (BFGS) optimization algorithm (an approximation of Newton's method that approximates the Hessian and assumes expands the function as a Taylor expansion) to find the maximum of the log-posterior, but also allows selection of other optimization algorithms from the ```scipy.optimize``` module. For example, below we use Powell’s method to find the MAP.



```python
# Obtain a MAP point estimate with Powell's method
from scipy import optimize

map_estimate = find_MAP(model=bioassay_model, fmin=optimize.fmin_powell)

print(map_estimate)

```


    logp = -13.107:   2%|▏         | 79/5000 [00:00<00:04, 1008.75it/s]

    Optimization terminated successfully.
             Current function value: 13.033671
             Iterations: 3
             Function evaluations: 79
    {'alpha': array(0.8409685695516148), 'beta': array(7.724974670803666)}


    


In Powell's method we move along one orthogonal base direction at at time, needing to minimize only in one dimension, using methods that do not require a derivative to be calculated.

![](images/powell.png)

### Sampling

We can now sample using the ```sample``` function, to which we can optionally pass a step method and a starting value, than can naturally be one of our MAP point estimates.



```python
from pymc3 import find_MAP

with bioassay_model:
    
    # obtain starting values via MAP
    start = find_MAP(model=bioassay_model)
    
    # instantiate sampler
    step = pm.Metropolis()
    
    # draw 2000 posterior samples
    bioassay_trace = sample(50000, step=step, start=start)


#with bioassay_model:
#    bioassay_trace = sample(1000)
```


    logp = -13.034, ||grad|| = 0.00043389: 100%|██████████| 14/14 [00:00<00:00, 517.62it/s]   
    Multiprocess sampling (2 chains in 2 jobs)
    CompoundStep
    >Metropolis: [beta]
    >Metropolis: [alpha]
    100%|██████████| 50500/50500 [00:27<00:00, 1816.59it/s]
    The number of effective samples is smaller than 10% for some parameters.


The first argument of ```sample``` is the number of samples to be drawn. Note that we have also used something called ```step``` as an argument. This refers to the MCMC method that we will use. Possible choices are Metropolis, slice sampling, HMC and NUTS algorithms. For this lab we will stick to the Metropolis and NUTS steps. Interestingly, by using the ```step``` argument, we can use the MAP estimate that we have used as the starting point for the MCMCM sampling.

After we run the model, the chain of draws from it will be stored in a data structure called ```trace```. We can access the trace and plot it, as well as the smoothed histograms of the parameters. In this case, we just sample the last half of the samples, the first half is our burnin period:



```python
from pymc3 import traceplot

#traceplot(bioassay_trace[500:], varnames=['alpha'])
traceplot(bioassay_trace[25000:])
tr1 = bioassay_trace[25000:]
```



![png](bioassay_pymc3_files/bioassay_pymc3_26_0.png)




```python
from pymc3 import autocorrplot
autocorrplot(tr1)
```





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x11290f710>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x11281aeb8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x1124ac278>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1125b3550>]], dtype=object)




![png](bioassay_pymc3_files/bioassay_pymc3_27_1.png)


We can also plot the 2D posterior and compare with the analytical plot above.



```python
sns.kdeplot(bioassay_trace['alpha'][25000:], bioassay_trace['beta'][25000:])
plt.xlabel(r"$\alpha$",size=20)
plt.ylabel(r"$\beta$",size=20)
```





    <matplotlib.text.Text at 0x116824128>




![png](bioassay_pymc3_files/bioassay_pymc3_29_1.png)


We can print a summary for our variables:



```python
from pymc3 import summary

summary(tr1, varnames=['alpha', 'beta'])
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>mc_error</th>
      <th>hpd_2.5</th>
      <th>hpd_97.5</th>
      <th>n_eff</th>
      <th>Rhat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha</th>
      <td>1.322962</td>
      <td>1.111251</td>
      <td>0.018051</td>
      <td>-0.731014</td>
      <td>3.570176</td>
      <td>3852.0</td>
      <td>1.000150</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>11.728286</td>
      <td>5.851554</td>
      <td>0.093292</td>
      <td>2.474921</td>
      <td>23.219954</td>
      <td>3557.0</td>
      <td>1.000131</td>
    </tr>
  </tbody>
</table>
</div>



## Model checking

After we have obtained the samples, we need to perform model checking in order to ensure that the inferences we have derived from the sampling are actually valid. Model checking consists of two parts: 1) Convergence diagnostics and 2) goodness of the fit.

It is in general impossible to guarantee that a MCMC has indeed reached convergenge,  but convergence diagnostics can detect lack of convergence. Model checking allows us to check the internal validity of the model by comparing predictions of the model with data used to fit it (i.e. posterior predictive issues).

### Convergence diagnostics

An informal approach involves plotting a histogram for every set of $k$ iterations (perhaps 50-100) beyond some burn in threshold $n$; if the histograms are not visibly different among the sample intervals, this may be considered some evidence for convergence. Note that such diagnostics should be carried out for each stochastic estimated by the MCMC algorithm, because convergent behavior by one variable does not imply evidence for convergence for other variables in the analysis.



```python
beta_trace = bioassay_trace['beta'][25000:]

fig, axes = plt.subplots(1, 5, figsize=(14,6))
axes = axes.ravel()
for i in range(5):
    axes[i].hist(beta_trace[500*i:500*(i+1)])
plt.tight_layout()
```



![png](bioassay_pymc3_files/bioassay_pymc3_33_0.png)


Another *ad hoc* method to detect lack of convergence involves plotting the traces of chains initialized with different starting conditions. We can run more than one chain using the argument ```njobs``` of the  ```sample``` function. If convergence has occurred, we would expect the two chains to converge to the same value, and to have approximately the same variance. This convergence to the same equilibrium state is what we call *ergodicity*, but its fullfilment it is not guaranteed by these heuristic methods, since the chain can be in a state of *metastability* and reian there for a large number of iterations before moving to a more stable state.



```python
with bioassay_model:
    step = pm.Metropolis()
    bioassay_trace1 = sample(50000, step=step, njobs=4, start=[{'alpha':0.5}, {'alpha':5}, {'alpha':1.5}, {'alpha':3}])
```


    Multiprocess sampling (4 chains in 4 jobs)
    CompoundStep
    >Metropolis: [beta]
    >Metropolis: [alpha]
    100%|██████████| 50500/50500 [00:48<00:00, 1033.33it/s]
    The number of effective samples is smaller than 10% for some parameters.




```python
bioassay_trace1.get_values('alpha', chains=0)[0]
```





    0.64718323917200782





```python
np.array(range(100))[:20]
```





    array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
           17, 18, 19])





```python
plt.hist(bioassay_trace1.get_values('alpha', chains=0)[25000:], histtype="step", bins=20)
plt.hist(bioassay_trace1.get_values('alpha', chains=1)[25000:], histtype="step", bins=20)
plt.hist(bioassay_trace1.get_values('alpha', chains=2)[25000:], histtype="step", bins=20)
plt.hist(bioassay_trace1.get_values('alpha', chains=3)[25000:], histtype="step", bins=20)
```





    (array([    8.,    36.,   153.,   464.,  1320.,  2317.,  3116.,  4194.,
             3963.,  3218.,  2476.,  1475.,   933.,   576.,   351.,   201.,
               86.,    62.,    36.,    15.]),
     array([-2.31056645, -1.88523499, -1.45990353, -1.03457207, -0.60924061,
            -0.18390915,  0.24142231,  0.66675377,  1.09208523,  1.51741669,
             1.94274814,  2.3680796 ,  2.79341106,  3.21874252,  3.64407398,
             4.06940544,  4.4947369 ,  4.92006836,  5.34539982,  5.77073128,
             6.19606274]),
     <a list of 1 Patch objects>)




![png](bioassay_pymc3_files/bioassay_pymc3_38_1.png)


### Goodness of fit: posterior predictive

The posterior predictive is the distribution of unobserved data conditioned on the observed data. It is constructed by averaging the likelihood of new unseen datapoints over all possible parameter values, weighted by their posterior probability:

$$
p(\tilde{y}|y) = \int p(\tilde{y}|\theta) p(\theta|y) d\theta
$$

This is: the probability of observing an hypotetical new datapoint given the observed data equals the likelihood of the datapoint given the model parameters, times the probability of that particular choice of parameters, marginalized over all possible parameter values.

We would expect that the posterior predictive distribution is indistinguishable from the actual observed distribution of datapoints. It is easy to sample from the posterior predictive in PyMC3:



```python
from pymc3 import sample_ppc

with bioassay_model:
    deaths_sim = sample_ppc(bioassay_trace, samples=500)
```


    100%|██████████| 500/500 [00:02<00:00, 178.77it/s]


If the model fit is good, we would expect the observed true data to fall within a region of high probability of the posterior predictive distribution. Otherwise, if the fit is not so good, we would expect to see the observed true data in the tails if the posterior predictive. Let us plot and see:



```python
fig, axes = plt.subplots(1, 4, figsize=(12, 6))

for obs, sim, ax in zip(deaths, deaths_sim['obs_deaths'].T, axes):
    ax.hist(sim, bins=range(7))
    ax.plot(obs+0.5, 1, 'ro')
```



![png](bioassay_pymc3_files/bioassay_pymc3_42_0.png)


But of course, what we want is being able to predict new observations. We can create an array of new hypotetical doses:



```python
log_dose_to_predict = np.random.uniform(-0.8,0.7,size=50)
#log_dose_to_predict =  np.array([-0.5])
log_dose_to_predict
n_predict = n = 5 * np.ones(50, dtype=int)
n_predict
```





    array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
           5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
           5, 5, 5, 5])



We now update the values of the shared variable we have created with the values for which we want to predict:



```python
# Changing values here will also change values in the model
log_dose_shared.set_value(log_dose_to_predict)
n_shared.set_value(n_predict)
```




```python
# Simply running PPC will use the updated values and do prediction
ppc = pm.sample_ppc(tr1, model=bioassay_model, samples=500)
```


    100%|██████████| 500/500 [00:00<00:00, 1779.69it/s]




```python
ppc['obs_deaths'].shape
```





    (500, 50)





```python
plt.errorbar(x=log_dose_to_predict, y=np.asarray(ppc['obs_deaths']).mean(axis=0), yerr=np.asarray(ppc['obs_deaths']).std(axis=0), linestyle='', marker='o', alpha=0.5)
plt.plot(log_dose, deaths, 'o',color='r')
plt.xlabel('log_dose',size=15)
plt.ylabel('number of rats with tumors',size=15)
```





    <matplotlib.text.Text at 0x112800358>




![png](bioassay_pymc3_files/bioassay_pymc3_49_1.png)


The posterior predictives give us something nice: uncertainties in our predcictions.
