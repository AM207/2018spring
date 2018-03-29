---
title: Gelman Schools Theory
shorttitle: gelmanschoolstheory
notebook: gelmanschoolstheory.ipynb
noline: 1
summary: ""
keywords: ['normal-normal model', 'hierarchical normal-normal model', 'hierarchical', 'divergences', 'non-centered hierarchical model', 'sampling distribution']
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




From Gelman:

>a simple hierarchical model based on the normal distribu- tion, in which observed data are normally distributed with a different mean for each ‘group’ or ‘experiment,’ with known observation variance, and a normal population distribution for the group means. This model is sometimes termed the one-way normal random-effects model with known data variance and is widely applicable, being an important special case of the hierarchical normal linear model,...

## Statement of the model

The particular example we will deal with is called the 8-schools example, and is described thus:

>A study was performed for the Educational Testing Service to analyze the effects of special coaching programs on test scores. Separate randomized experiments were performed to estimate the effects of coaching programs for the SAT-V (Scholastic Aptitude Test- Verbal) in each of eight high schools. The outcome variable in each study was the score on a special administration of the SAT-V, a standardized multiple choice test administered by the Educational Testing Service and used to help colleges make admissions decisions; the scores can vary between 200 and 800, with mean about 500 and standard deviation about 100. The SAT examinations are designed to be resistant to short-term efforts directed specifically toward improving performance on the test; instead they are designed to reflect knowledge acquired and abilities developed over many years of education. Nevertheless, each of the eight schools in this study considered its short-term coaching program to be successful at increasing SAT scores. Also, there was no prior reason to believe that any of the eight programs was more effective than any other or that some were more similar in effect to each other than to any other.

>the estimated coaching effects are $\bar{y}_j$, and their sampling variances, $\sigma_j^2$... The estimates $\bar{y}_j$ are obtained by independent experiments and have approximately normal sampling distributions with sampling variances that are known, for all practical purposes, because the sample sizes in all of the eight experiments were relatively large, over thirty students in each school 

![](images/8schools.png)

Notice the bar on the y's and the mention of standard errors (rather than standard deviations) in the third column in the table above. Why is this?

The answer is that these are means taken over many (> 30) students in each of the schools. The general structure of this model can be written thus:

>Consider $J$ independent experiments, with experiment $j$ estimating the parameter $\theta_j$ from $n_j$ independent normally distributed data points, $y_{ij}$, each with known error variance $\sigma^2$; that is,

$$y_{ij} \vert \theta_j \sim N(\theta_j, \sigma^2), \, i = 1,...,n_j; j = 1,...,J.$$

![](images/school_model.png)

Thus there are $J=8$ schools, and in the $jth$ school, there are $n_j$ students. We are not given the scores of individual students, just the average score in the school: the so-called "sample-mean" (after all this is a sample of students from the school).

We label the sample mean of each group $j$ as

$$\bar{y_j} = \frac{1}{n_j} \sum_{i=1}^{n_j} y_{ij}$$

with sampling variance:

$$\sigma_j^2 = \sigma^2/n_j$$

  
>We can then write the likelihood for each $\theta_j$ using the sufficient statistics, $\bar{y}_j$:

$$\bar{y_j} \vert \theta_j \sim N(\theta_j,\sigma_j^2).$$

Since we are assuming the variance $\sigma^2$ is known from all the schools we have $$\sigma_j^2 = \sigma^2/n_j$$ to be the standard error of the sample-mean.

The idea is that if a particular school is very likely to have systematically positive treatment effect, we should be able to estimate that $\theta_j$ is relatively large and $\sigma_j^2$ is relatively small. If on the other hand, a school giving both positive and negative treatments we'll estimate $\theta_j$ around 0 and a relatively large variance $\sigma_j^2$.

This is

>a notation that will prove useful later because of the flexibility in allowing a separate variance $\sigma_j^2$ for the mean of each group $j$. ...all expressions will be implicitly conditional on the known values $\sigma_j^2$.... Although rarely strictly true, the assumption of known variances at the sampling level of the model is often an adequate approximation.

>The treatment of the model provided ... is also appropriate for situations in which the variances differ for reasons other than the number of data points in the experiment. In fact, the likelihood  can appear in much more general contexts than that stated here. For example, if the group sizes $n_j$ are large enough, then the means $\bar{y_j}$ are approximately normally distributed, given $\theta_j$, even when the data $y_{ij}$ are not. 

In other problems, like the one on your homework where we will use this model, you are given a $\sigma_2$ calculated from each group or unit. But since you will want  the variance of the sample mean, you will have to calculate the standard error by dividing out by the count in that unit.

![](images/restuarant_model.png)

Let us choose a prior:

$$
\theta_j \sim N(\mu, \tau^2)
$$

$\theta_j$ is the parameter we were estimating by the review-topic mean earlier.

The second of the formulae above will allow us to share information between reviews within each restaurant.

After doing some math, we can calculate the posterior distribution:

$$
p(\theta_j\, \vert \,\bar{y}_{j})\propto p(\bar{y}_{j}\, \vert \,\theta_j) p(\theta_j)
\propto \exp\left(-\frac{1}{2 \sigma_j^2} \left(\bar{y}_{j}-\theta_j\right)^2\right)  \exp\left(-\frac{1}{2 \tau^2} \left(\theta_j-\mu\right)^2\right)
$$

After some amount of algebra you'll find that this is the kernel of a normal distribution with mean 

$\frac{1}{\sigma^2_{\text{post}}}\left(\frac{\mu}{\tau^2} + \frac{\bar{y}_{j}}{\sigma^2_{j}}\right)$ 

and variance 

$ \sigma^2_{\text{post}} = \left(\frac{1}{\tau^2} + \frac{1}{\sigma^2_{j}}\right)^{-1}$. 

We can simplify the mean further to see a familiar form:

$$
\mathbb{E}[\theta_j\, \vert \,\bar y_j, \mu, \sigma_j^2, \tau^2] = \frac{\sigma_j^2}{\sigma_j^2 + \tau^2} \mu + \frac{\tau^2}{\sigma_j^2 + \tau^2}\bar{y}_{j}.
$$

The _posterior mean_ is a weighted average of the prior mean and the observed average. 

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



## Setting up the hierarchical model

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

### An equivalent model in a non-centered parametrization

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

## Sampling

One can work out this problem using Gaussian Formulae and EB (whch requires some root-finding!), but we love our samplers so we will do sampling!



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
    trace2 = pm.sample(10000, init=None, njobs=2, tune=1000)
```


    100%|██████████| 10000/10000 [00:26<00:00, 372.42it/s]  | 1/10000 [00:00<20:41,  8.05it/s]




```python
pm.traceplot(trace2)
```





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x11d621630>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x10fd424a8>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x10fd59a20>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1206c8160>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x120822828>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x12089cef0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x121331128>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1213abcc0>]], dtype=object)




![png](gelmanschoolstheory_files/gelmanschoolstheory_17_1.png)


Ok, so this seems to look better!



```python
plt.plot(trace2['tau_log_'], alpha=0.6)
plt.axvline(5000, color="r")
```





    <matplotlib.lines.Line2D at 0x121cbe860>




![png](gelmanschoolstheory_files/gelmanschoolstheory_19_1.png)


And the effective number of iterations hs improved as well:



```python
pm.diagnostics.gelman_rubin(trace2), pm.diagnostics.effective_n(trace2)
```





    ({'mu': 1.0000308061571239,
      'nu': array([ 0.99995748,  0.99995994,  0.99995388,  0.99995464,  0.99995733,
              0.99995898,  1.00001938,  0.99999721]),
      'tau': 0.99998481746202439,
      'tau_log_': 0.9999543464224343,
      'theta': array([ 0.9999525 ,  0.99995125,  1.00000408,  0.99998158,  1.00023177,
              0.99997517,  0.99996997,  0.99995046])},
     {'mu': 20000.0,
      'nu': array([ 20000.,  20000.,  20000.,  20000.,  20000.,  20000.,  20000.,
              20000.]),
      'tau': 9510.0,
      'tau_log_': 8571.0,
      'theta': array([ 19246.,  20000.,  20000.,  20000.,  20000.,  20000.,  20000.,
              20000.])})



And we reach the true value better as the number of samples increases, decreasing our bias



```python
# plot the estimate for the mean of log(τ) cumulating mean
logtau = trace2['tau_log_']
mlogtau = [np.mean(logtau[:i]) for i in np.arange(1, len(logtau))]
plt.figure(figsize=(15, 4))
plt.axhline(0.7657852, lw=2.5, color='gray')
plt.plot(mlogtau, lw=2.5)
plt.ylim(0, 2)
plt.xlabel('Iteration')
plt.ylabel('MCMC mean of log(tau)')
plt.title('MCMC estimation of cumsum log(tau)')
```





    <matplotlib.text.Text at 0x1226a7a20>




![png](gelmanschoolstheory_files/gelmanschoolstheory_23_1.png)


How about our divergences? They seem to be more than what we saw in class but note that we have double the number of samples and the divergences are distributed fairly uniformly so we are sure they are false positives.



```python
divergent = trace2['diverging']
print('Number of Divergent %d' % divergent.nonzero()[0].size)
divperc = divergent.nonzero()[0].size/len(trace2)
print('Percentage of Divergent %.5f' % divperc)
```


    Number of Divergent 22
    Percentage of Divergent 0.00220




```python
theta_trace = trace2['theta']
theta0 = theta_trace[:, 0]
plt.figure(figsize=(10, 6))
plt.scatter(theta0[divergent == 0], logtau[divergent == 0], color='r')
plt.scatter(theta0[divergent == 1], logtau[divergent == 1], color='g')
plt.axis([-20, 50, -6, 4])
plt.ylabel('log(tau)')
plt.xlabel('theta[0]')
plt.title('scatter plot between log(tau) and theta[0]')
plt.show()
```



![png](gelmanschoolstheory_files/gelmanschoolstheory_26_0.png)


Look how much longer the funnel actually is. And we have explored this much better.
