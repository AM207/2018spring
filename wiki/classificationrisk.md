---
title: Classification Risk
shorttitle: classificationrisk
notebook: classificationrisk.ipynb
noline: 1
summary: ""
keywords: ['classification', 'supervised learning', 'decision risk', 'decision theory', 'bayes risk']
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
```


## The multiple risks

There are *two risks in learning* that we must consider, one to *estimate probabilities*, which we call **estimation risk**, and one to *make decisions*, which we call **decision risk**. 

What do we mean by a "decision" exactly? We'll use the letter $a$ here to indicate a decision, in both the regression and classification problems. In the classification problem, one example of a decision is the process used to choose the class of a sample, given the probability of being in that class.  We must mix these probabilities with "business knowledge" or "domain knowledge" to make a decision. 

What we must additionally supply is the **decision loss** $l(y,a)$ or **utility** $u(l,a)$ (profit, or benefit) in making a decision $a$ when the predicted variable has value $y$. For example, we must provide all of the losses $l$(no-cancer, biopsy), $l$(cancer, biopsy), $l$(no-cancer, no-biopsy), and $l$(cancer, no-biopsy). One set of choices for these losses may be 20, 0, 0, 200 respectively.

To simplify matters though, lets presently insist that the **decision space** from which the decision $a$ is chosen is the same as the space from which $y$ is chosen. In other words, the decision to be made is a classification. Then we can use these losses to penalize mis-classification asymmetrically if we desire. In the cancer example, we then set $l$(observed-no-cancer, predicted cancer) to be 20 and $l$(observed-cancer, predicted-no-cancer) to be 200. This is the situation we talked about much earlier in the class where we penalize the false negative(observed cancer not predicted to be cancer) much more than the false positive(observed non-cancer predicted to be cancer). These estimates were obtained from the confusion matrix.

## Predictive averaged risk for classification

 We simply weigh each combinations loss by the predictive probability that that combination can happen, the integral from the risks notes reducing to a sum:



$$ R_{a}(x) = \sum_y l(y,a(x)) p(y|x)$$

That is, we calculate the **average risk** over all choices y, of making choice a for a given data point.

Then, if we want to calculate the overall risk, given all the samples in our set, we calculate:

$$R(a) = \int dx p(x) R_{a}(x)$$

(Since we usually assume fixed but unknown-distributed covariates, we can replace this integral by a sum over the empirical distribution, ie a sum over the data points)

It is sufficient to minimize the risk at each point or sample to minimize the overall risk since $p(x)$ is always positive.

Consider the two class classification case. Say we make a "decision a about which class" at a sample x. Then:

$$R_a(x) = l(1, g)p(1|x) + l(0, g)p(0|x).$$

Then for the "decision" $a=1$ we have:

$$R_1(x) = l(1,1)p(1|x) + l(0,1)p(0|x),$$

and for the "decision" $a=0$ we have:

$$R_0(x) = l(1,0)p(1|x) + l(0,0)p(0|x).$$

Now, we'd choose $1$ for the sample at $x$ if:

$$R_1(x) \lt R_0(x).$$

$$ P(1|x)(l(1,1) - l(1,0)) \lt p(0|x)(l(0,0) - l(0,1))$$

This gives us a ratio `r` between the probabilities to make a prediction. We assume this is true for all samples.

So, to choose '1', the Bayes risk can be obtained by setting:

$$p(1|x) \gt r P(0|x) \implies r=\frac{l(0,1) - l(0,0)}{l(1,0) - l(1,1)} =\frac{c_{FP} - c_{TN}}{c_{FN} - c_{TP}}$$

This may also be written as:

$$P(1|x) \gt t = \frac{r}{1+r}$$.

If you assume that True positives and True negatives have no cost, and the cost of a false positive is equal to that of a false positive, then $r=1$ and the threshold is the usual intuitive $t=0.5$.

### The symmetric case with 1-0 risk

For the 1-0 loss, $l(1,1) = l(0,0) =0$ and $l(1,0) = l(0,1) = 1$, and we get:

$$R_1(x) = p(0|x), R_0(x) = p(1|x).$$

We'd choose $1$ if:

$$R_1(x) \le R_0(x)$$

for a given sample $x$. Thus we get back the "intuitive" prescrription for classification we have been using so far: **choose $1$ if**:

$$p(1|x) \ge p(0|x).$$

Since these add to 1, this is equivalent to saying **choose $1$ if $p(1|x) \ge 0.5$**



```python

```

