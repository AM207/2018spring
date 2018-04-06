---
title: Prosocial Chimps
shorttitle: prosocialchimps
notebook: prosocialchimps.ipynb
noline: 1
summary: ""
keywords: ['glm', 'varying intercept', 'multiple varying intercept', 'posterior predictive', 'model averaging', 'model comparison']
data: ['data/chimpanzees2.csv']
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}



>The data for this example come from an experiment140 aimed at evaluating the prosocial tendencies of chimpanzees (Pan troglodytes). The experimental structure mimics many common experiments conducted on human students (Homo sapiens studiensis) by economists and psychologists. A focal chimpanzee sits at one end of a long table with two levers, one on the left and one on the right in FIGURE 10.1. On the table are four dishes which may contain desirable food items. The two dishes on the right side of the table are attached by a mechanism to the right-hand lever. The two dishes on the left side are similarly attached to the left-hand lever.

>When either the left or right lever is pulled by the focal animal, the two dishes on the same side slide towards opposite ends of the table. This delivers whatever is in those dishes to the opposite ends. In all experimental trials, both dishes on the focal animal's side contain food items. But only one of the dishes on the other side of the table contains a food item. Therefore while both levers deliver food to the focal animal, only one of the levers delivers food to the other side of the table.

>There are two experimental conditions. In the partner condition, another chimpanzee is seated at the opposite end of the table, as pictured in FIGURE 10.1. In the control condition, the other side of the table is empty. Finally, two counterbalancing treatments alternate which side, left or right, has a food item for the other side of the table. This helps detect any handedness preferences for individual focal animals. 

>When human students participate in an experiment like this, they nearly always choose the lever linked to two pieces of food, the prosocial option, but only when another student sits on the opposite side of the table. The motivating question is whether a focal chimpanzee behaves similarly, choosing the prosocial option more often when another animal is present. In terms of linear models, we want to estimate the interaction between condition (presence or absence of another animal) and option (which side is prosocial). (McElreath 292-293)



![](images/pchimps.png)

>Chimpanzee prosociality experiment, as seen from the perspective of the focal animal. The left and right levers are indicated in the foreground. Pulling either expands an accordion device in the center, pushing the food trays towards both ends of the table. Both food trays close to the focal animal have food in them. Only one of the food trays on the other side contains food. The partner condition means another animal, as pictured, sits on the other end of the table. Otherwise, the other end was empty. (McElreath 293)



## Seeing the Data



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




```python
df=pd.read_csv("data/chimpanzees2.csv", sep=";")
df.head(100)
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
      <th>actor</th>
      <th>recipient</th>
      <th>condition</th>
      <th>block</th>
      <th>trial</th>
      <th>prosoc_left</th>
      <th>chose_prosoc</th>
      <th>pulled_left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>44</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>46</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>50</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>56</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>58</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>70</th>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>6</td>
      <td>69</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>71</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>6</td>
      <td>71</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>72</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>73</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>74</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>75</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>76</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>77</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>78</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>13</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>79</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>15</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>80</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>17</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>81</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>82</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>21</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>83</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>84</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>85</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>86</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>29</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>87</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>31</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>88</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>33</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>89</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>35</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>90</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>37</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>91</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>39</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>92</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>41</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>93</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>43</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>94</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>45</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>95</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>47</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>96</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>49</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>97</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>51</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>98</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>53</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>99</th>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>55</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>100 rows × 8 columns</p>
</div>



>We're going to focus on `pulled_left` as the outcome to predict, with `prosoc_left` and `condition` as predictor variables. The outcome `pulled_left` is a 0 or 1 indicator that the focal animal pulled the left-hand lever. The predictor `prosoc_left` is a 0/1 indicator that the left-hand lever was (1) or was not (0) attached to the prosocial option, the side with two pieces of food. The `condition` predictor is another 0/1 indicator, with value 1 for the partner condition and value 0 for the control condition. (McElreath 293)



```python
df.shape
```





    (504, 8)



Lets explore the data a bit...



```python
gd={}
for k, v in df.groupby('actor'):
    temp = v.groupby(['condition', 'prosoc_left'])['pulled_left'].mean()
    gd[k] = temp.values
    #print(k, ldf.values)
```


For each actor we get the 4 combinations of condition/prosoc_left and see what fraction of times times that chimp pulled the left lever.



```python
gd
```





    {1: array([ 0.33333333,  0.5       ,  0.27777778,  0.55555556]),
     2: array([1, 1, 1, 1]),
     3: array([ 0.27777778,  0.61111111,  0.16666667,  0.33333333]),
     4: array([ 0.33333333,  0.5       ,  0.11111111,  0.44444444]),
     5: array([ 0.33333333,  0.55555556,  0.27777778,  0.5       ]),
     6: array([ 0.77777778,  0.61111111,  0.55555556,  0.61111111]),
     7: array([ 0.77777778,  0.83333333,  0.94444444,  1.        ])}



## 3 different Logistic regression models

Let $P$ be the indicator for `prosoc_left`, ie is the two-food or prosocial side is the left side(1) or the right side(0). Let $C$ be the indicator for `condition`, with 1 indicating the partner condition, ie a chimp at the other end, and a 0 indicating no animal. Let $L$ (`pulled_left`) indicate with a 1 value that the left side lever is pulled and with a 0 that the right one is pulled.

### Full Model

![](images/modelfull.png)



```python
def full_model():
    with pm.Model() as ps1:
        betapc = pm.Normal("betapc", 0, 10)
        betap = pm.Normal("betap", 0, 10)
        alpha = pm.Normal('alpha', 0, 10)
        logitpi = alpha + (betap + betapc*df.condition)*df.prosoc_left
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
        
    return ps1
```


>note that there is no main effect of $C_i$ itself, no plain beta-coefficient for condition. Why? Because there is no reason to hypothesize that the presence or absence of another animal creates a tendency to pull the left-hand lever. This is equivalent to assuming that the main effect of condition is exactly zero. You can check this assumption later, if you like.

>The priors above are chosen for lack of informativeness—they are very gently regularizing, but will be overwhelmed by even moderate evidence. So the estimates we'll get from this model will no doubt be overfit to sample somewhat. To get some comparative measure of that overfitting, we'll also fit two other models with fewer predictors. (McElreath 293-294)

### Intercept-Only Model

![](images/modelicept.png)



```python
def ionly_model():
    with pm.Model() as ps0:
        alpha = pm.Normal('alpha', 0, 10)
        logitpi = alpha 
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
    return ps0
```


### Model using `prosoc_left` only

![](images/modelnocong.png)




```python
def plonly_model():
    with pm.Model() as plonly:
        betap = pm.Normal("betap", 0, 10)
        alpha = pm.Normal('alpha', 0, 10)
        logitpi = alpha + betap*df.prosoc_left
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
    return plonly
```


### Sampling

Lets sample from these models



```python
ionly = ionly_model()
with ionly:
    trace_ionly=pm.sample(5000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha]
    100%|██████████| 6000/6000 [00:04<00:00, 1373.69it/s]




```python
pm.autocorrplot(trace_ionly)
```





    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x11038ea58>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x1106f2c50>]], dtype=object)




![png](prosocialchimps_files/prosocialchimps_23_1.png)




```python
pm.summary(trace_ionly)
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
      <td>0.319931</td>
      <td>0.088977</td>
      <td>0.00139</td>
      <td>0.143678</td>
      <td>0.491558</td>
      <td>4637.0</td>
      <td>0.99997</td>
    </tr>
  </tbody>
</table>
</div>





```python
full = full_model()
with full:
    trace_full=pm.sample(5000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betap, betapc]
    100%|██████████| 6000/6000 [00:18<00:00, 322.03it/s]




```python
plonly = plonly_model()
with plonly:
    trace_plonly=pm.sample(5000, tune=1000)
```


    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha, betap]
    100%|██████████| 6000/6000 [00:12<00:00, 490.45it/s]




```python
pm.summary(trace_plonly)
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
      <th>betap</th>
      <td>0.557852</td>
      <td>0.185852</td>
      <td>0.002627</td>
      <td>0.190106</td>
      <td>0.912260</td>
      <td>3869.0</td>
      <td>1.000059</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>0.048212</td>
      <td>0.127505</td>
      <td>0.001982</td>
      <td>-0.202406</td>
      <td>0.293146</td>
      <td>3624.0</td>
      <td>0.999904</td>
    </tr>
  </tbody>
</table>
</div>



## Model Comparison for these models



```python
def make_compare(names, traces, models, ic='WAIC'):
    comparedf=pm.compare(traces,models, method='pseudo-BMA')
    temp=comparedf.sort_index()
    temp['Model']=names
    comparedf = temp.sort_values(ic).set_index('Model')
    return comparedf
```




```python
names=['intercept only', 'full', 'prosocial left only']
dfc=make_compare(names, [trace_ionly, trace_full, trace_plonly],[ionly, full, plonly])
dfc
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
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>prosocial left only</th>
      <td>680.58</td>
      <td>2.04</td>
      <td>0</td>
      <td>0.69</td>
      <td>9.32</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>full</th>
      <td>682.26</td>
      <td>2.96</td>
      <td>1.68</td>
      <td>0.3</td>
      <td>9.46</td>
      <td>0.81</td>
      <td>0</td>
    </tr>
    <tr>
      <th>intercept only</th>
      <td>687.89</td>
      <td>0.97</td>
      <td>7.31</td>
      <td>0.02</td>
      <td>7.11</td>
      <td>6.15</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Notice that the full model is worse by a small amount from the `prosoc_left` only model. But see the standard error for the difference. Even if it were doubled, there is no way it would make up the distance between the two models. Why is this the case?



```python
with sns.plotting_context('poster'):
    pm.compareplot(dfc)
```



![png](prosocialchimps_files/prosocialchimps_32_0.png)




```python
pm.summary(trace_full)
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
      <th>betapc</th>
      <td>-0.105089</td>
      <td>0.256749</td>
      <td>0.003268</td>
      <td>-0.589178</td>
      <td>0.398558</td>
      <td>6555.0</td>
      <td>0.999903</td>
    </tr>
    <tr>
      <th>betap</th>
      <td>0.615898</td>
      <td>0.224468</td>
      <td>0.002960</td>
      <td>0.176292</td>
      <td>1.056009</td>
      <td>6142.0</td>
      <td>0.999957</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>0.048474</td>
      <td>0.127385</td>
      <td>0.001556</td>
      <td>-0.198057</td>
      <td>0.295233</td>
      <td>6336.0</td>
      <td>1.000004</td>
    </tr>
  </tbody>
</table>
</div>



>The estimated interaction effect bpC is negative, with a rather wide posterior on both sides of zero. So regardless of the information theory ranking, the estimates suggest that the chimpanzees did not care much about the other animal's presence. But they do prefer to pull the prosocial option, as indicated by the estimate for bp. (McElreath 296)

>First, let's consider the relative effect size of prosoc_left and its parameter bp. The customary measure of relative effect for a logistic model is the PROPORTIONAL CHANGE IN ODDS. You can compute the proportional odds by merely exponentiating the parameter estimate. Remember, odds are the ratio of the probability an event happens to the probability it does not happen. So in this case the relevant odds are the odds of pulling the left-hand lever (the outcome variable). If changing the predictor prosoc_left from 0 to 1 increases the log-odds of pulling the left-hand lever by 0.61 (the MAP estimate above), then this also implies that the odds are multiplied by: (McElreath 296)





```python
def invlogit(x):
    return np.exp(x) / (1 + np.exp(x))
```




```python
np.exp(0.61)
```





    1.8404313987816374



This is a 84% change in the log odds



```python
invlogit(0.04), invlogit(0.04+0.61), invlogit(0.04+0.61-0.1)
```





    (0.50999866687996553, 0.65701046267349883, 0.63413559101080075)



## Posteriors and Posterior predictives

First we create a trace function that takes into account the fact that we are using "nested" models, and that the full trace for $$logit(p)$$ can be obtained by setting some coefficients to 0



```python
def trace_or_zero(trace, name):
    if name in trace.varnames:
        return trace[name]
    else:
        return np.zeros(2*len(trace))
```


Next we write a function for this trace



```python
def model_pp(gridx, tracedict):
    temp = tracedict['alpha'] + gridx['P']*(tracedict['betap'] + tracedict['betapc']*gridx['C'])
    return temp
```


Now to compute the predictive, we get the trace of the logit, inverse logit it, and pass it through the sampling distribution.



```python
def compute_pp(lpgrid, trace, tsize, paramnames, sampdistrib, invlink, inner_pp):
    tdict={}
    for pn in paramnames:
        tdict[pn] = trace_or_zero(trace, pn)
    print(tdict.keys(), tsize)
    tl=tsize
    gl=len(lpgrid)
    pp = np.empty((gl, tl))
    for i, v in enumerate(lpgrid):
        temp = inner_pp(lpgrid[i], tdict)
        pp[i,:] = sampdistrib(invlink(temp))
    return pp
```


We construct the grid we want the posterior predictive on:



```python
import itertools
psleft = [0,1]
condition = [0,1]
xgrid = [{'C':v[0], 'P':v[1]} for v in itertools.product(condition, psleft)]
```




```python
xgrid
```





    [{'C': 0, 'P': 0}, {'C': 0, 'P': 1}, {'C': 1, 'P': 0}, {'C': 1, 'P': 1}]



And then get the posterior predictive. But which one? Notice that in modelling this problem as a logistic regression, we are modeling each row of the data. But in the binomial below, we are modelling the story of 7 chimps. We could do 10, 100, and so on and so off. What would happen?

Which should you use? The code below is for illustration, and for you to explore and correct.



```python
from scipy.stats import bernoulli, binom
```




```python
ppdivisor=7
def like_sample(p_array):
    ppdivisor=7
    return binom.rvs(ppdivisor, p=p_array)
```




```python
ppfull = compute_pp(xgrid, trace_full, 2*len(trace_full), trace_full.varnames, like_sample, invlogit, model_pp)
```


    dict_keys(['betapc', 'betap', 'alpha']) 10000




```python
ppfull
```





    array([[ 3.,  4.,  5., ...,  1.,  4.,  6.],
           [ 6.,  4.,  4., ...,  3.,  5.,  4.],
           [ 5.,  6.,  2., ...,  3.,  5.,  6.],
           [ 5.,  5.,  5., ...,  4.,  7.,  6.]])





```python
meanpp, stdpp = ppfull.mean(axis=1), ppfull.std(axis=1)
```




```python
with sns.plotting_context('poster'):
    fmt = lambda d: ",".join([e+"="+str(d[e]) for e in d])
    plt.plot(range(4),meanpp/ppdivisor, lw=3, color="black")
    for i, chimp in enumerate(gd):
        plt.plot(range(4), gd[chimp], label=str(chimp))
    plt.fill_between(range(4), (meanpp-stdpp)/ppdivisor, (meanpp+stdpp)/ppdivisor, alpha=0.3, color="gray")
    plt.ylim([0,1.2])
    plt.xticks(range(4),[fmt(e) for e in xgrid])
    plt.legend();
```



![png](prosocialchimps_files/prosocialchimps_55_0.png)


And this second likelihood gives us what happens for any one row of chimps.



```python
def ls2(p_array):
    return bernoulli.rvs(p=p_array)
```




```python
ppfull2 = compute_pp(xgrid, trace_full, 2*len(trace_full), trace_full.varnames, ls2, invlogit, model_pp)
meanpp2, stdpp2 = ppfull2.mean(axis=1), ppfull2.std(axis=1)
```


    dict_keys(['betapc', 'betap', 'alpha']) 10000




```python
ppfull2
```





    array([[ 1.,  1.,  0., ...,  0.,  1.,  1.],
           [ 1.,  1.,  1., ...,  1.,  1.,  1.],
           [ 1.,  0.,  1., ...,  0.,  1.,  1.],
           [ 1.,  1.,  0., ...,  1.,  0.,  1.]])





```python
ppfull2.mean(axis=1)
```





    array([ 0.5126,  0.6652,  0.5115,  0.6349])





```python
with sns.plotting_context('poster'):
    fmt = lambda d: ",".join([e+"="+str(d[e]) for e in d])
    plt.plot(range(4),meanpp2, lw=3, color="black")
    for i, chimp in enumerate(gd):
        plt.plot(range(4), gd[chimp], label=str(chimp))
    plt.fill_between(range(4), (meanpp2-stdpp2), (meanpp2+stdpp2), alpha=0.3, color="gray")
    plt.ylim([0,1.2])
    plt.xticks(range(4),[fmt(e) for e in xgrid])
    plt.legend();
```



![png](prosocialchimps_files/prosocialchimps_61_0.png)


>The colored lines display the empirical averages for each of the seven chimpanzees who participated in the experiment. The black line shows the average predicted probability of pulling the left-hand lever, across treatments. The zig-zag pattern arises from more left-hand pulls when the prosocial option is on the left. So the chimpanzees were, at least on average, attracted to the prosocial option. But the partner condition, shown by the last two treatment on the far right of the figure, are no higher than the first two treatments from the control condition. So it made little difference whether or not another animal was present to receive the food on the other side of the table. (McElreath 297-298)


## Ensemble the model



```python
for m in dfc.index:
    print(m)
```


    prosocial left only
    full
    intercept only




```python
modeldict={
    "prosocial left only": trace_plonly,
    "full": trace_full,
    "intercept only": trace_ionly
}
```




```python
def ensemble(grid, modeldict, paramnames, comparedf):
    accum_pp=0
    accum_weight=0
    for m in comparedf.index:
        weight = comparedf.loc[m]['weight']
        print(m, "size", len(modeldict[m]))
        pp=compute_pp(grid, modeldict[m], 2*len(modeldict[m]), paramnames, like_sample, invlogit, model_pp)
        print(m, weight, np.median(pp))
        accum_pp += pp*weight
        accum_weight +=weight
    return accum_pp/accum_weight
```




```python
ppens = ensemble(xgrid, modeldict, ['alpha', 'betap', 'betapc'], dfc)
```


    prosocial left only size 5000
    dict_keys(['alpha', 'betap', 'betapc']) 10000
    prosocial left only 0.88 4.0
    full size 5000
    dict_keys(['alpha', 'betap', 'betapc']) 10000
    full 0.0 4.0
    intercept only size 5000
    dict_keys(['alpha', 'betap', 'betapc']) 10000
    intercept only 0.12 4.0




```python
with sns.plotting_context('poster'):
    meanpp, stdpp = ppens.mean(axis=1), ppens.std(axis=1)
    fmt = lambda d: ",".join([e+"="+str(d[e]) for e in d])
    plt.plot(range(4),meanpp/ppdivisor, lw=3, color="black")
    for i, chimp in enumerate(gd):
        plt.plot(range(4), gd[chimp], label="actor {}".format(chimp), lw=3)
    plt.fill_between(range(4), (meanpp-stdpp)/ppdivisor, (meanpp+stdpp)/ppdivisor, alpha=0.4, color="gray")
    plt.ylim([0,1.1])
    plt.xticks(range(4),[fmt(e) for e in xgrid])
    plt.legend();
```



![png](prosocialchimps_files/prosocialchimps_68_0.png)


## Modeling as a binomial

>In the chimpanzees data context, the models all calculated the likelihood of observing either zero or one pulls of the left-hand lever. The models did so, because the data were organized such that each row describes the outcome of a single pull. But in principle the same data could be organized differently. As long as we don't care about the order of the individual pulls, the same information is contained in a count of how many times each individual pulled the left-hand lever, for each combination of predictor variables. (McElreath 303)



## A heirarchical model

>Now back to modeling individual variation. There is plenty of evidence of handedness in these data. Four of the individuals tend to pull the right-hand lever, across all treatments. Three individuals tend to pull the left across all treatments. One individual, actor number 2, always pulled the left-hand lever, regardless of treatment. That's the horizontal green line at the top (McElreath 299)

>Think of handedness here as a masking variable. If we can model it well, maybe we can get a better picture of what happened across treatments. So what we wish to do is estimate handedness as a distinct intercept for each individual, each actor. You could do this using a dummy variable for each individual. But it'll be more convenient to use a vector of intercepts, one for each actor. This form is equivalent to making dummy variables, but it is more compact  (McElreath 299)


Here we have a varying intercepts model

![](images/multichimp.png)



```python
def vi_model():
    with pm.Model() as vi:
        betapc = pm.Normal("betapc", 0, 10)
        betap = pm.Normal("betap", 0, 10)
        alpha = pm.Normal('alpha', 0, 10)
        sigma_actor = pm.HalfCauchy("sigma_actor", 1)
        alpha_actor = pm.Normal('alpha_actor', 0, sigma_actor, shape=7)
        logitpi = alpha + alpha_actor[df.index//72] + (betap + betapc*df.condition)*df.prosoc_left
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
        
    return vi
```




```python
vi = vi_model()
with vi:
    step=pm.NUTS(target_accept=0.95)
    vi_trace=pm.sample(10000, tune=4000, step=step)
```


    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha_actor, sigma_actor_log__, alpha, betap, betapc]
    100%|██████████| 14000/14000 [03:53<00:00, 59.84it/s]
    The number of effective samples is smaller than 25% for some parameters.




```python
pm.traceplot(vi_trace);
```



![png](prosocialchimps_files/prosocialchimps_74_0.png)




```python
pm.autocorrplot(vi_trace);
```



![png](prosocialchimps_files/prosocialchimps_75_0.png)




```python
pm.plot_posterior(vi_trace, kde_plot=True);
```



![png](prosocialchimps_files/prosocialchimps_76_0.png)


Positive values of `alpha_actor` indicate a preference for the left side.

>You can see that there is strong skew here. Plausible values of `alpha_actor__1` are always positive, indicating a left-hand bias. But the range of plausible values is truly enormous. What has happened here is that many very large positive values are plausible, because actor number 2 always pulled the left-hand lever (McElreath 300)





```python
pm.forestplot(vi_trace);
```



![png](prosocialchimps_files/prosocialchimps_78_0.png)




```python
pm.summary(vi_trace)
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
      <th>betapc</th>
      <td>-0.134896</td>
      <td>0.301527</td>
      <td>0.003191</td>
      <td>-0.741355</td>
      <td>0.437579</td>
      <td>8163.0</td>
      <td>1.000008</td>
    </tr>
    <tr>
      <th>betap</th>
      <td>0.823672</td>
      <td>0.263390</td>
      <td>0.002814</td>
      <td>0.305391</td>
      <td>1.330417</td>
      <td>8451.0</td>
      <td>0.999964</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>0.449606</td>
      <td>0.940080</td>
      <td>0.017642</td>
      <td>-1.392269</td>
      <td>2.315823</td>
      <td>2870.0</td>
      <td>1.001231</td>
    </tr>
    <tr>
      <th>alpha_actor__0</th>
      <td>-1.162653</td>
      <td>0.960387</td>
      <td>0.017437</td>
      <td>-3.081355</td>
      <td>0.697052</td>
      <td>2999.0</td>
      <td>1.001239</td>
    </tr>
    <tr>
      <th>alpha_actor__1</th>
      <td>4.134793</td>
      <td>1.598422</td>
      <td>0.022589</td>
      <td>1.485586</td>
      <td>7.402799</td>
      <td>5332.0</td>
      <td>0.999975</td>
    </tr>
    <tr>
      <th>alpha_actor__2</th>
      <td>-1.468864</td>
      <td>0.959703</td>
      <td>0.017903</td>
      <td>-3.378610</td>
      <td>0.416878</td>
      <td>2989.0</td>
      <td>1.001397</td>
    </tr>
    <tr>
      <th>alpha_actor__3</th>
      <td>-1.466724</td>
      <td>0.962787</td>
      <td>0.017766</td>
      <td>-3.439681</td>
      <td>0.400624</td>
      <td>3003.0</td>
      <td>1.001096</td>
    </tr>
    <tr>
      <th>alpha_actor__4</th>
      <td>-1.163595</td>
      <td>0.959745</td>
      <td>0.017693</td>
      <td>-3.109661</td>
      <td>0.691084</td>
      <td>2971.0</td>
      <td>1.001247</td>
    </tr>
    <tr>
      <th>alpha_actor__5</th>
      <td>-0.221391</td>
      <td>0.956959</td>
      <td>0.017693</td>
      <td>-2.133475</td>
      <td>1.640277</td>
      <td>2972.0</td>
      <td>1.001214</td>
    </tr>
    <tr>
      <th>alpha_actor__6</th>
      <td>1.314492</td>
      <td>0.981045</td>
      <td>0.017395</td>
      <td>-0.674050</td>
      <td>3.212776</td>
      <td>3166.0</td>
      <td>1.001018</td>
    </tr>
    <tr>
      <th>sigma_actor</th>
      <td>2.256800</td>
      <td>0.914543</td>
      <td>0.013069</td>
      <td>0.926622</td>
      <td>3.996550</td>
      <td>4610.0</td>
      <td>1.000306</td>
    </tr>
  </tbody>
</table>
</div>





```python
vi_trace['alpha_actor'][:,1].shape
```





    (20000,)



### Predictives are on individuals now

>You can best appreciate the way these individual intercepts influence fit by plotting posterior predictions again. The code below just modifies the code from earlier to show only a single individual, the one specified by the first line.  (McElreath 301)





```python
def like_sample_hier(p_array):
    return bernoulli.rvs(p=p_array)
```




```python
def model_pp_hier(gridx, tracedict, ix):
    temp = tracedict['alpha'] + tracedict['alpha_actor'][:,ix]+gridx['P']*(tracedict['betap'] + tracedict['betapc']*gridx['C'])
    return temp
```




```python
def compute_pp2(lpgrid, trace, paramnames, sampdistrib, invlink, inner_pp, ix):
    tdict=trace
    tl=2*len(trace)
    gl=len(lpgrid)
    pp = np.empty((gl, tl))
    for i, v in enumerate(lpgrid):
        temp = inner_pp(lpgrid[i], tdict, ix)
        pp[i,:] = invlink(temp)
    return pp
```




```python
vi_trace.varnames
```





    ['betapc', 'betap', 'alpha', 'sigma_actor_log__', 'alpha_actor', 'sigma_actor']





```python
vnames=['betapc', 'betap', 'alpha', 'alpha_actor']
pphier0=compute_pp2(xgrid, vi_trace, vnames, like_sample_hier, invlogit, model_pp_hier, 0)
```




```python
ppdivisor=1
meanpp, stdpp = pphier0.mean(axis=1), pphier0.std(axis=1)
fmt = lambda d: ",".join([e+"="+str(d[e]) for e in d])
plt.plot(range(4),meanpp/ppdivisor, lw=3, color="black")
plt.plot(range(4), gd[1], label="actor{}".format(1), lw=3)
plt.fill_between(range(4), (meanpp-stdpp)/ppdivisor, (meanpp+stdpp)/ppdivisor, alpha=0.4, color="gray")
plt.ylim([0,1.1])
plt.xticks(range(4),[fmt(e) for e in xgrid])
plt.legend();
```



![png](prosocialchimps_files/prosocialchimps_87_0.png)




```python
pphier6=compute_pp2(xgrid, vi_trace, vnames, like_sample_hier, invlogit, model_pp_hier, 6)
```




```python
ppdivisor=1
meanpp, stdpp = pphier6.mean(axis=1), pphier6.std(axis=1)
fmt = lambda d: ",".join([e+"="+str(d[e]) for e in d])
plt.plot(range(4),meanpp/ppdivisor, lw=3, color="black")
plt.plot(range(4), gd[7], label="actor{}".format(7), lw=3)
plt.fill_between(range(4), (meanpp-stdpp)/ppdivisor, (meanpp+stdpp)/ppdivisor, alpha=0.4, color="gray")
plt.ylim([0,1.1])
plt.xticks(range(4),[fmt(e) for e in xgrid])
plt.legend();
```



![png](prosocialchimps_files/prosocialchimps_89_0.png)


>Notice that these individual intercepts do help the model fit the overall level for each chimpanzee. But they do not change the basic zig-zag prediction pattern across treatments. (McElreath 302)

## Varying experimental blocks as well

>The kind of data structure here is usually called a CROSS-CLASSIFIED multilevel model. It is cross-classified, because actors are not nested within unique blocks. If each chimpanzee had instead done all of his or her pulls on a single day, within a single block, then the data structure would instead be hierarchical. However, the model specification would typically be the same. So the model structure and code you'll see below will apply both to cross-classified designs and hierarchical designs. Other software sometimes forces you to treat these differently, on account of using a conditioning engine substantially less capable than MCMC. There are other types of “hierarchical” multilevel models, types that make adaptive priors for adaptive priors. It's turtles all the way down, recall (page 13). You'll see an example in the next chapter. But for the most part, people (or their software) nearly always use the same kind of model in both cases. (McElreath 371)

>Each pull is within a cluster of pulls belonging to an individual chimpanzee. But each pull is also within an experimental block, which represents a collection of observations that happened on the same day. So each observed pull belongs to both an actor (1 to 7) and a block (1 to 6). There may be unique intercepts for each actor as well as for each block. (McElreath 370-371)

>To add the second cluster type, block, we merely replicate the structure for the actor cluster. This means the linear model gets yet another varying intercept, $α_{BLOCK[i]}$, and the model gets another adaptive prior and yet another standard deviation parameter.  Here is the mathematical form of the model, with the new pieces of the machine highlighted in blue: (McElreath 372-373)


![](images/multichimpcluster.png)

>Each cluster variable needs its own standard deviation parameter that adapts the amount of pooling across units, be they actors or blocks. These are αACTOR and αBLOCK, respectively. Finally, note that there is only one global mean parameter α, and both of the varying intercept parameters are centered at zero. We can't identify a separate mean for each varying intercept type, because both intercepts are added to the same linear prediction. So it is conventional to define varying intercepts with a mean of zero, so there's no risk of accidentally creating hard-to-identify parameters.  (McElreath 373)




```python
df.head(50)
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
      <th>actor</th>
      <th>recipient</th>
      <th>condition</th>
      <th>block</th>
      <th>trial</th>
      <th>prosoc_left</th>
      <th>chose_prosoc</th>
      <th>pulled_left</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>8</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>10</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>14</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>16</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>18</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>20</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>24</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>28</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>30</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>32</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>34</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>36</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>38</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>40</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>44</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>46</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>4</td>
      <td>48</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>50</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>52</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>54</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>56</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>58</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>5</td>
      <td>60</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>62</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>64</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>66</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>68</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>70</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>6</td>
      <td>72</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>1</td>
      <td>7</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>9</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>2</td>
      <td>13</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>2</td>
      <td>15</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1</td>
      <td>5</td>
      <td>1</td>
      <td>2</td>
      <td>17</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1</td>
      <td>7</td>
      <td>1</td>
      <td>2</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1</td>
      <td>8</td>
      <td>1</td>
      <td>2</td>
      <td>21</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>2</td>
      <td>23</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1</td>
      <td>6</td>
      <td>1</td>
      <td>3</td>
      <td>27</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>





```python
def viplusblock_model():
    with pm.Model() as vipb:
        betapc = pm.Normal("betapc", 0, 10)
        betap = pm.Normal("betap", 0, 10)
        alpha = pm.Normal('alpha', 0, 10)
        sigma_actor = pm.HalfCauchy("sigma_actor", 1)
        sigma_block = pm.HalfCauchy("sigma_block", 1)
        alpha_actor = pm.Normal('alpha_actor', 0, sigma_actor, shape=7)
        alpha_block = pm.Normal('alpha_block', 0, sigma_block, shape=6)
        logitpi = alpha + alpha_actor[df.index//72] + alpha_block[df.block.values -1]+ (betap + betapc*df.condition)*df.prosoc_left
        o = pm.Bernoulli("pulled_left", p=pm.math.invlogit(logitpi), observed=df.pulled_left)
        
    return vipb
```




```python
vipb=viplusblock_model()
with vipb:
    step=pm.NUTS(target_accept=0.95)
    trace_vipb = pm.sample(10000, tune=4000, step=step)
```


    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [alpha_block, alpha_actor, sigma_block_log__, sigma_actor_log__, alpha, betap, betapc]
    100%|██████████| 14000/14000 [06:52<00:00, 33.93it/s]
    There were 5 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.90614233342, but should be close to 0.95. Try to increase the number of tuning steps.
    There were 4 divergences after tuning. Increase `target_accept` or reparameterize.
    The number of effective samples is smaller than 25% for some parameters.




```python
pm.autocorrplot(trace_vipb);
```



![png](prosocialchimps_files/prosocialchimps_97_0.png)




```python
pm.summary(trace_vipb)
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
      <th>betapc</th>
      <td>-0.136885</td>
      <td>0.298762</td>
      <td>0.002316</td>
      <td>-0.733272</td>
      <td>0.439749</td>
      <td>14285.0</td>
      <td>0.999955</td>
    </tr>
    <tr>
      <th>betap</th>
      <td>0.829279</td>
      <td>0.260229</td>
      <td>0.002159</td>
      <td>0.321858</td>
      <td>1.333908</td>
      <td>14512.0</td>
      <td>0.999950</td>
    </tr>
    <tr>
      <th>alpha</th>
      <td>0.434569</td>
      <td>0.957979</td>
      <td>0.014928</td>
      <td>-1.355945</td>
      <td>2.459922</td>
      <td>3480.0</td>
      <td>1.000570</td>
    </tr>
    <tr>
      <th>alpha_actor__0</th>
      <td>-1.152940</td>
      <td>0.970981</td>
      <td>0.015013</td>
      <td>-3.097572</td>
      <td>0.765036</td>
      <td>3535.0</td>
      <td>1.000562</td>
    </tr>
    <tr>
      <th>alpha_actor__1</th>
      <td>4.182226</td>
      <td>1.636061</td>
      <td>0.021415</td>
      <td>1.508208</td>
      <td>7.465532</td>
      <td>5782.0</td>
      <td>1.000205</td>
    </tr>
    <tr>
      <th>alpha_actor__2</th>
      <td>-1.457689</td>
      <td>0.974489</td>
      <td>0.015160</td>
      <td>-3.453729</td>
      <td>0.414437</td>
      <td>3478.0</td>
      <td>1.000757</td>
    </tr>
    <tr>
      <th>alpha_actor__3</th>
      <td>-1.459539</td>
      <td>0.972212</td>
      <td>0.014664</td>
      <td>-3.453009</td>
      <td>0.371960</td>
      <td>3542.0</td>
      <td>1.000578</td>
    </tr>
    <tr>
      <th>alpha_actor__4</th>
      <td>-1.152205</td>
      <td>0.971254</td>
      <td>0.015102</td>
      <td>-3.179855</td>
      <td>0.679607</td>
      <td>3539.0</td>
      <td>1.000724</td>
    </tr>
    <tr>
      <th>alpha_actor__5</th>
      <td>-0.201399</td>
      <td>0.969822</td>
      <td>0.014917</td>
      <td>-2.222166</td>
      <td>1.630771</td>
      <td>3517.0</td>
      <td>1.000568</td>
    </tr>
    <tr>
      <th>alpha_actor__6</th>
      <td>1.334859</td>
      <td>0.996828</td>
      <td>0.014689</td>
      <td>-0.617785</td>
      <td>3.316638</td>
      <td>3793.0</td>
      <td>1.000734</td>
    </tr>
    <tr>
      <th>alpha_block__0</th>
      <td>-0.184579</td>
      <td>0.232796</td>
      <td>0.003477</td>
      <td>-0.676070</td>
      <td>0.181351</td>
      <td>4585.0</td>
      <td>1.000484</td>
    </tr>
    <tr>
      <th>alpha_block__1</th>
      <td>0.036786</td>
      <td>0.189143</td>
      <td>0.001817</td>
      <td>-0.360202</td>
      <td>0.435598</td>
      <td>10873.0</td>
      <td>1.000365</td>
    </tr>
    <tr>
      <th>alpha_block__2</th>
      <td>0.053597</td>
      <td>0.190420</td>
      <td>0.001975</td>
      <td>-0.318322</td>
      <td>0.472124</td>
      <td>10461.0</td>
      <td>1.000768</td>
    </tr>
    <tr>
      <th>alpha_block__3</th>
      <td>0.006276</td>
      <td>0.190338</td>
      <td>0.001863</td>
      <td>-0.361058</td>
      <td>0.431848</td>
      <td>10072.0</td>
      <td>0.999986</td>
    </tr>
    <tr>
      <th>alpha_block__4</th>
      <td>-0.032478</td>
      <td>0.190520</td>
      <td>0.001609</td>
      <td>-0.440451</td>
      <td>0.355683</td>
      <td>11512.0</td>
      <td>0.999972</td>
    </tr>
    <tr>
      <th>alpha_block__5</th>
      <td>0.115411</td>
      <td>0.204691</td>
      <td>0.002556</td>
      <td>-0.225687</td>
      <td>0.579629</td>
      <td>6080.0</td>
      <td>1.000400</td>
    </tr>
    <tr>
      <th>sigma_actor</th>
      <td>2.260354</td>
      <td>0.917322</td>
      <td>0.012141</td>
      <td>0.936306</td>
      <td>3.994636</td>
      <td>5508.0</td>
      <td>1.000018</td>
    </tr>
    <tr>
      <th>sigma_block</th>
      <td>0.228169</td>
      <td>0.181730</td>
      <td>0.003623</td>
      <td>0.008308</td>
      <td>0.560449</td>
      <td>2339.0</td>
      <td>1.002590</td>
    </tr>
  </tbody>
</table>
</div>



>First, notice that the number of effective samples, n_eff, varies quite a lot across parameters. This is common in complex models. Why? There are many reasons for this. But in this sort of model the most common reason is that some parameter spends a lot of time near a boundary. Here, that parameter is sigma_block. It spends a lot of time near its minimum of zero. As a consequence, you may also see a warning about “divergent iterations.”  (McElreath 374)



```python
pm.forestplot(trace_vipb)
```





    <matplotlib.gridspec.GridSpec at 0x115db1fd0>




![png](prosocialchimps_files/prosocialchimps_100_1.png)




```python
pm.traceplot(trace_vipb);
```



![png](prosocialchimps_files/prosocialchimps_101_0.png)


>While there's uncertainty about the variation among actors, this model is confident that actors vary more than blocks. You can easily see this variation in the varying intercept estimates: the a_actor distributions are much more scattered than are the a_block distributions. (McElreath 374-375)

Notice that the trace of the $\sigma_{ACTOR}$ is way wider.

>compare sigma_actor to sigma_block and notice that the estimated variation among actors is a lot larger than the estimated variation among blocks. This is easy to appreciate, if we plot the marginal posterior distributions of these  (McElreath 374)



```python
sns.distplot(trace_vipb['sigma_block'])
sns.distplot(trace_vipb['sigma_actor']);
plt.xlim([-1,10])
```





    (-1, 10)




![png](prosocialchimps_files/prosocialchimps_103_1.png)


## Model Comparison

Lets compare the WAICs of the actor block and cross-classified model.

>While the cross-classified model has 7 more parameters than the actor block model does, it has only about 2.5 more effective parameters. Why? Because the posterior distribution for sigma_block ended up close to zero. This means each of the 6 a_block parameters is strongly shrunk towards zero—they are relatively inflexible. In contrast, the a_actor parameters are shrunk towards zero much less, because the estimated variation across actors is much larger, resulting in less shrinkage. But as a consequence, each of the a_actor parameters contributes much more to the pWAIC value. (McElreath 375)





```python
dfc2=make_compare(['actor-multilevel', 'actor-block-crossclassified'], [vi_trace, trace_vipb],[vi, vipb])
```




```python
dfc2
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
      <th>WAIC</th>
      <th>pWAIC</th>
      <th>dWAIC</th>
      <th>weight</th>
      <th>SE</th>
      <th>dSE</th>
      <th>warning</th>
    </tr>
    <tr>
      <th>Model</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>actor-multilevel</th>
      <td>531.54</td>
      <td>8.2</td>
      <td>0</td>
      <td>1</td>
      <td>19.49</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>actor-block-crossclassified</th>
      <td>532.65</td>
      <td>10.4</td>
      <td>1.1</td>
      <td>0</td>
      <td>19.67</td>
      <td>1.78</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



Notice that while the cross-classified 



```python
with sns.plotting_context('poster'):
    pm.compareplot(dfc2)
```



![png](prosocialchimps_files/prosocialchimps_108_0.png)


>You might also notice that the difference in WAIC between these models is small, only 1.2. This is especially small compared the standard deviation of the difference, 1.94. These two models imply nearly identical predictions, and so their expected out-of-sample accuracy is nearly identical. The block parameters have been shrunk so much towards zero that they do very little work in the model.

>If you are feeling the urge to “select” the actor block model as the best model, pause for a moment. There is nothing to gain here by selecting either model. The comparison of the two models tells a richer story—whether we include block or not hardly matters, and the a_block and sigma_block estimates tell us why. Furthermore, the standard error of the difference in WAIC between the models is twice as large as the difference itself. By retaining and reporting both models, we and our readers learn more about the experiment. (McElreath 375-376)

### Posterior Predictives

There are now 2 kinds of posterior predicties here, just as we saw in the rat tumor problem.

>Before computing predictions, note that we should no longer expect the posterior predictive distribution to match the raw data, even when the model worked correctly. Why? The whole point of partial pooling is to shrink estimates towards the grand mean. So the estimates should not necessarily match up with the raw data, once you use pooling. (McElreath 377)

The first kind is the usual, say within an actor block, new posterior predictives.

The second kind, is for a new blockor new chimp (like the 71st experiment)

>Often, the particular clusters in the sample are not of any enduring interest. In the chimpanzees data, for example, these particular 7 chimpanzees are just seven individuals. We'd like to make inferences about the whole species, not just those seven individuals. So the individual actor intercepts aren't of interest, but the distribution of them definitely is.

>One way to grasp the task of construction posterior predictions for new clusters is to imagine leaving out one of the clusters when you fit the model to the data. For example, suppose we leave out actor number 7 when we fit the chimpanzees model. Now how can we assess the model's accuracy for predicting actor number 7's behavior? We can't use any of the a_actor parameter estimates, because those apply to other individuals. But we can make good use of the a and sigma_actor parameters, because those describe the population of actors. (McElreath 378)

Here too, there are multiple predictives you can create:

(1) predictive for an average actor.

>First, let's see how to construct posterior predictions for a now, previously unobserved average actor. By “average,” I mean an individual chimpanzee with an intercept exactly at a (α), the population mean. This simultaneously implies a varying intercept of zero. Since there is uncertainty about the population mean, there is still uncertainty about this average individual's intercept. But as you'll see, the uncertainty is much smaller than it really should be, if we wish to honestly represent the problem of what to expect from a new individual. (McElreath 378)

(2) prediction for an individual actor.

Draw a new $\alpha_{ACTOR} \sim N(0, \sigma_{ACTOR}$. Thus given a $\sigma_{ACTOR}$ trace we can get new probability traces. Here we can make two plots

1. A line for each sample over the grid. Since we are sampling $\alpha_{ACTOR}$ separately each time, we get a "new" chimp each time. Note that what we are doing here is removing the varying effects term and replacing it by this sample.
2. a marginal which now takes all these actor samples, and adds them in to get probability samples and then just passes these through the link to get proportions pulling left. This is the usual posterior predictive.
