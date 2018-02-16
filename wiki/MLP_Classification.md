---
title: Classification using Multi-Layer Perceptrons
shorttitle: MLP_Classification
notebook: MLP_Classification.ipynb
noline: 1
summary: ""
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
import seaborn.apionly as sns
sns.set_context("poster")
```


Two additional imports here, seaborn and tqdm. Install via pip or conda



```python
c0=sns.color_palette()[0]
c1=sns.color_palette()[1]
c2=sns.color_palette()[2]
```




```python
from matplotlib.colors import ListedColormap
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

def points_plot(ax, Xtr, Xte, ytr, yte, clf_predict, colorscale=cmap_light, cdiscrete=cmap_bold, alpha=0.3, psize=20):
    h = .02
    X=np.concatenate((Xtr, Xte))
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))


    Z = clf_predict(np.c_[xx.ravel(), yy.ravel()])
    ZZ = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, ZZ, cmap=cmap_light, alpha=alpha, axes=ax)
    showtr = ytr
    showte = yte
    ax.scatter(Xtr[:, 0], Xtr[:, 1], c=showtr-1, cmap=cmap_bold, s=psize, alpha=alpha,edgecolor="k")
    # and testing points
    ax.scatter(Xte[:, 0], Xte[:, 1], c=showte-1, cmap=cmap_bold, alpha=alpha, marker="s", s=psize+10)
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    return ax,xx,yy
```


## Create some noisy moon shaped data

In order to illustrate classification by a MLP, we first create some noisy moon shaped data. The *noise level* here and the *amount of data* is the first thing you might want to experiment with to understand the interplay of amount of data, noise level, number of parameters in the model we use to fit, and overfitting as illustrated by jagged boundaries.

We standardize the data so that it is distributed about 0 as well



```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
dataX, datay = make_moons(noise=0.35, n_samples=400)
dataX = StandardScaler().fit_transform(dataX)
X_train, X_test, y_train, y_test = train_test_split(dataX, datay, test_size=.4)
```




```python
h=.02
x_min, x_max = dataX[:, 0].min() - .5, dataX[:, 0].max() + .5
y_min, y_max = dataX[:, 1].min() - .5, dataX[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
ax = plt.gca()
# Plot the training points
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, alpha=0.5, s=30)
# and testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.5, s=30)
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())
```





    (-3.0442250970624234, 2.8557749029375823)




![png](MLP_Classification_files/MLP_Classification_7_1.png)




```python
import torch
import torch.nn as nn
from torch.nn import functional as fn
from torch.autograd import Variable
import torch.utils.data
```


## Writing a Multi-Layer Perceptron class

We wrap the construction of our network 



```python
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nonlinearity = fn.tanh, additional_hidden_wide=0):
        super(MLP, self).__init__()
        self.fc_initial = nn.Linear(input_dim, hidden_dim)
        self.fc_mid = nn.ModuleList()
        self.additional_hidden_wide = additional_hidden_wide
        for i in range(self.additional_hidden_wide):
            self.fc_mid.append(nn.Linear(hidden_dim, hidden_dim))
        self.fc_final = nn.Linear(hidden_dim, output_dim)
        self.nonlinearity = nonlinearity

    def forward(self, x):
        x = self.fc_initial(x)
        x = self.nonlinearity(x)
        for i in range(self.additional_hidden_wide):
            x = self.fc_mid[i](x)
            x = self.nonlinearity(x)
        x = self.fc_final(x)
        return x
```


We use it to train. Notice the double->float casting. Numpy defautlts to double but torch defaulta to float to enable memory efficient GPU usage.



```python
np.dtype(np.float).itemsize, np.dtype(np.double).itemsize
```





    (8, 8)



But torch floats are 4 byte as can be seen from here: http://pytorch.org/docs/master/tensors.html


### Training the model

Points to note:

- printing a model prints its layers, handy. Note that we implemented layers as functions. The autodiff graph is constructed on the fly on the first forward pass and used in backward.
- we had to cast to float
- `model.parameters` gives us params, `model.named_parameters()` gives us assigned names. You can set your own names when you create a layer
- we create an iterator over the data, more precisely over batches by doing `iter(loader)`. This dispatches to the `__iter__` method of the dataloader. (see https://github.com/pytorch/pytorch/blob/4157562c37c76902c79e7eca275951f3a4b1ef78/torch/utils/data/dataloader.py#L416) Always explore source code to understand what is going on



```python
model2 = MLP(input_dim=2, hidden_dim=2, output_dim=2, nonlinearity=fn.tanh, additional_hidden_wide=1)
print(model2)
criterion = nn.CrossEntropyLoss(size_average=True)
dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
lr, epochs, batch_size = 1e-1 , 1000 , 64
optimizer = torch.optim.SGD(model2.parameters(), lr = lr )
accum=[]
for k in range(epochs):
    localaccum = []
    for localx, localy in iter(loader):
        localx = Variable(localx.float())
        localy = Variable(localy.long())
        output = model2.forward(localx)
        loss = criterion(output, localy)
        model2.zero_grad()
        loss.backward()
        optimizer.step()
        localaccum.append(loss.data[0])
    accum.append(np.mean(localaccum))
plt.plot(accum);                      
```


    MLP(
      (fc_initial): Linear(in_features=2, out_features=2)
      (fc_mid): ModuleList(
        (0): Linear(in_features=2, out_features=2)
      )
      (fc_final): Linear(in_features=2, out_features=2)
    )



![png](MLP_Classification_files/MLP_Classification_14_1.png)


The out put from the foward pass is run on the entire test set. Since pytorch tracks layers upto but before the loss, this handily gives us the softmax output, which we can then use `np.argmax` on.



```python
testoutput = model2.forward(Variable(torch.from_numpy(X_test).float()))
testoutput
```





    Variable containing:
    -1.0148  0.4436
     0.6828 -0.4708
    -0.8730  0.3666
     2.2660 -1.2822
    -1.0148  0.4430
    -0.5756  0.2068
    -1.1663  0.5271
    -1.4427  0.6828
     1.1829 -0.7268
    -1.1666  0.5313
    -1.9428  0.9676
     1.9172 -1.1055
    -0.9402  0.4020
    -2.2079  1.1137
    -1.9250  0.9513
     2.6874 -1.4843
    -2.1592  1.0845
    -0.0659 -0.0744
    -0.8134  0.3352
    -0.5401  0.1822
    -0.1753 -0.0170
     0.2418 -0.2399
    -0.0554 -0.0756
    -1.6703  0.8123
     1.7919 -1.0436
    -0.6726  0.2553
     2.2303 -1.2600
     2.4532 -1.3706
    -1.2964  0.6051
     2.2575 -1.2733
     0.5640 -0.4038
     0.4282 -0.3426
    -1.4461  0.6873
    -0.5738  0.2030
     1.0371 -0.6625
    -1.9696  0.9806
     2.6469 -1.4642
    -1.4156  0.6717
     1.0648 -0.6705
    -2.0460  1.0231
     1.6534 -0.9757
     0.9866 -0.6336
     1.4577 -0.8756
    -1.9258  0.9526
    -1.7462  0.8521
    -0.3215  0.0669
     0.1335 -0.1768
     2.0103 -1.1513
     2.6172 -1.4519
    -1.9312  0.9587
     2.4479 -1.3689
    -1.7184  0.8358
    -1.6063  0.7779
    -0.6022  0.2229
    -1.9381  0.9587
     1.4837 -0.8872
    -2.0339  1.0163
     1.3415 -0.8165
    -1.6166  0.7793
    -0.7364  0.2997
     0.6362 -0.4505
     1.0451 -0.6637
    -0.9526  0.4054
     0.8715 -0.5710
     1.0164 -0.6470
    -1.6290  0.7830
     1.2519 -0.7696
     2.7552 -1.5166
     1.4722 -0.8835
    -0.1235 -0.0354
    -2.4374  1.2417
    -1.7966  0.8823
     1.2768 -0.7818
    -1.5325  0.7338
     0.6564 -0.4589
    -1.4223  0.6772
     0.1141 -0.1708
     2.8536 -1.5634
     1.4749 -0.8840
    -2.1556  1.0811
    -1.1014  0.4953
     2.4111 -1.3503
     0.3696 -0.2998
     1.9218 -1.1090
    -2.1860  1.1011
    -1.9611  0.9743
    -2.0206  1.0089
     1.4806 -0.8903
    -1.7845  0.8699
     1.5550 -0.9203
    -0.3207  0.0674
     1.1137 -0.6941
    -0.1694 -0.0114
    -1.1096  0.4967
    -1.9456  0.9632
    -0.6612  0.2450
    -1.0257  0.4481
     1.5413 -0.9140
     0.7135 -0.4903
    -1.4325  0.6753
     1.2146 -0.7496
    -0.8555  0.3586
     0.2657 -0.2511
    -1.6274  0.7868
    -0.6895  0.2640
    -2.1811  1.0958
    -1.8038  0.8823
     2.9968 -1.6302
     2.5411 -1.4126
    -0.7236  0.2802
    -1.6127  0.7783
    -0.0336 -0.0948
     1.6756 -0.9833
     2.2168 -1.2508
    -2.2032  1.1093
    -0.8199  0.3381
    -1.1334  0.5141
    -1.8456  0.9059
     0.6492 -0.4560
    -1.9217  0.9508
     0.2940 -0.2648
    -0.7575  0.3048
     2.5110 -1.3998
     1.4075 -0.8507
    -1.5510  0.7399
    -1.6721  0.8072
    -1.2441  0.5735
     2.2027 -1.2485
    -0.3996  0.1010
    -0.7868  0.3236
     2.5064 -1.3985
     2.7387 -1.5076
     2.2959 -1.2936
     1.4818 -0.8827
     2.4564 -1.3724
     2.2689 -1.2816
     2.4745 -1.3822
    -0.5005  0.1688
     1.4225 -0.8581
    -2.1107  1.0574
    -0.5257  0.1748
    -2.1880  1.1014
     0.7326 -0.5023
     2.4584 -1.3753
     1.5118 -0.8971
    -1.2930  0.5997
    -1.1720  0.5367
    -0.9360  0.3990
     1.3598 -0.8280
    -1.7808  0.8707
    -1.4554  0.6842
    -1.8742  0.9235
     2.4121 -1.3518
     1.2831 -0.7767
     1.2989 -0.7906
     1.6237 -0.9552
     1.4837 -0.8896
    -1.6923  0.8255
     2.2927 -1.2902
     1.1459 -0.7171
    [torch.FloatTensor of size 160x2]





```python
y_pred = testoutput.data.numpy().argmax(axis=1)
y_pred
```





    array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0,
           1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1,
           0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0,
           1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0,
           1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1,
           1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1,
           0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0])



You can write your own but we import some metrics from sklearn



```python
from sklearn.metrics import confusion_matrix, accuracy_score
confusion_matrix(y_test, y_pred)
```





    array([[66, 16],
           [11, 67]])





```python
accuracy_score(y_test, y_pred)
```





    0.83125000000000004



We can wrap this machinery in a function, and pass this function to `points_plot` to predict on a grid and thus give us a boundary viz



```python
def make_pred(X_set):
    output = model2.forward(Variable(torch.from_numpy(X_set).float()))
    return output.data.numpy().argmax(axis=1)
```




```python
ax = plt.gca()
points_plot(ax, X_train, X_test, y_train, y_test, make_pred);
```



![png](MLP_Classification_files/MLP_Classification_23_0.png)


## Making a `scikit-learn` like interface

Since we want to run many experiments, we'll go ahead and wrap our fitting process in a sklearn style interface. Another example of such an interface is [here](https://github.com/vinhkhuc/PyTorch-Mini-Tutorials/blob/master/3_neural_net.py)



```python
from tqdm import tnrange, tqdm_notebook
class MLPClassifier:
    
    def __init__(self, input_dim, hidden_dim, 
                 output_dim, nonlinearity = fn.tanh, 
                 additional_hidden_wide=0):
        self._pytorch_model = MLP(input_dim, hidden_dim, output_dim, nonlinearity, additional_hidden_wide)
        self._criterion = nn.CrossEntropyLoss(size_average=True)
        self._fit_params = dict(lr=0.1, epochs=200, batch_size=64)
        self._optim = torch.optim.SGD(self._pytorch_model.parameters(), lr = self._fit_params['lr'] )
        
    def __repr__(self):
        num=0
        for k, p in self._pytorch_model.named_parameters():
            numlist = list(p.data.numpy().shape)
            if len(numlist)==2:
                num += numlist[0]*numlist[1]
            else:
                num+= numlist[0]
        return repr(self._pytorch_model)+"\n"+repr(self._fit_params)+"\nNum Params: {}".format(num)
    
    def set_fit_params(self, *, lr=0.1, epochs=200, batch_size=64):
        self._fit_params['batch_size'] = batch_size
        self._fit_params['epochs'] = epochs
        self._fit_params['lr'] = lr
        self._optim = torch.optim.SGD(self._pytorch_model.parameters(), lr = self._fit_params['lr'] )
        
    def fit(self, X_train, y_train):
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
        loader = torch.utils.data.DataLoader(dataset, batch_size=self._fit_params['batch_size'], shuffle=True)
        self._accum=[]
        for k in tnrange(self._fit_params['epochs']):
            localaccum = []
            for localx, localy in iter(loader):
                localx = Variable(localx.float())
                localy = Variable(localy.long())
                output = self._pytorch_model.forward(localx)
                loss = self._criterion(output, localy)
                self._pytorch_model.zero_grad()
                loss.backward()
                self._optim.step()
                localaccum.append(loss.data[0])
            self._accum.append(np.mean(localaccum))
        
    def plot_loss(self):
        plt.plot(self._accum, label="{}".format(self))
        plt.legend()
        plt.show()
        
    def plot_boundary(self, X_train, X_test, y_train, y_test):
        points_plot(plt.gca(), X_train, X_test, y_train, y_test, self.predict);
        plt.show()
        
    def predict(self, X_test):
        output = self._pytorch_model.forward(Variable(torch.from_numpy(X_test).float()))
        return output.data.numpy().argmax(axis=1)
        
```


Some points about this:

- we provide the ability to change the fitting parameters
- by implementing a `__repr__` we let an instance of this class print something useful. Specifically we created a count of the number of parameters so that we can get a comparison of data size to parameter size.

## Testing on a wildly overfit model



```python
clf = MLPClassifier(input_dim=2, hidden_dim=1000, output_dim=2, nonlinearity=fn.tanh, additional_hidden_wide=1)
clf.set_fit_params(epochs=1000)
print(clf)
clf.fit(X_train,y_train)
```


    MLP(
      (fc_initial): Linear(in_features=2, out_features=1000)
      (fc_mid): ModuleList(
        (0): Linear(in_features=1000, out_features=1000)
      )
      (fc_final): Linear(in_features=1000, out_features=2)
    )
    {'lr': 0.1, 'epochs': 1000, 'batch_size': 64}
    Num Params: 1006002




    




```python
clf.plot_loss()
```



![png](MLP_Classification_files/MLP_Classification_29_0.png)




```python
ypred = clf.predict(X_test)
#training and test accuracy
accuracy_score(y_train, clf.predict(X_train)), accuracy_score(y_test, ypred)
```





    (0.91666666666666663, 0.86250000000000004)





```python
clf.plot_boundary(X_train, X_test, y_train, y_test)
```



![png](MLP_Classification_files/MLP_Classification_31_0.png)


## Experimentation Space

Here is space for you to play. You might want to collect accuracies on the traing and test set and plot on a grid of these parameters or some other visualization. Notice how you might want to adjust number of epochs for convergence.



```python
for additional in [1, 2, 3]:
    for hdim in [2, 10, 20, 100, 1000]:
        print('====================')
        print('Additional', additional, "hidden", hdim)
        clf = MLPClassifier(input_dim=2, hidden_dim=hdim, output_dim=2, nonlinearity=fn.tanh, additional_hidden_wide=additional)
        if additional > 2 and hdim > 50:
            clf.set_fit_params(epochs=1000)
        else:
            clf.set_fit_params(epochs=500)
        print(clf)
        clf.fit(X_train,y_train)
        clf.plot_loss()
        clf.plot_boundary(X_train, X_test, y_train, y_test)
        print("Train acc", accuracy_score(y_train, clf.predict(X_train)))
        print("Test acc", accuracy_score(y_test, clf.predict(X_test)))

```


    ====================
    Additional 1 hidden 2
    MLP(
      (fc_initial): Linear(in_features=2, out_features=2)
      (fc_mid): ModuleList(
        (0): Linear(in_features=2, out_features=2)
      )
      (fc_final): Linear(in_features=2, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 18




    



![png](MLP_Classification_files/MLP_Classification_33_3.png)



![png](MLP_Classification_files/MLP_Classification_33_4.png)


    Train acc 0.833333333333
    Test acc 0.85
    ====================
    Additional 1 hidden 10
    MLP(
      (fc_initial): Linear(in_features=2, out_features=10)
      (fc_mid): ModuleList(
        (0): Linear(in_features=10, out_features=10)
      )
      (fc_final): Linear(in_features=10, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 162




    



![png](MLP_Classification_files/MLP_Classification_33_8.png)



![png](MLP_Classification_files/MLP_Classification_33_9.png)


    Train acc 0.9
    Test acc 0.88125
    ====================
    Additional 1 hidden 20
    MLP(
      (fc_initial): Linear(in_features=2, out_features=20)
      (fc_mid): ModuleList(
        (0): Linear(in_features=20, out_features=20)
      )
      (fc_final): Linear(in_features=20, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 522




    



![png](MLP_Classification_files/MLP_Classification_33_13.png)



![png](MLP_Classification_files/MLP_Classification_33_14.png)


    Train acc 0.883333333333
    Test acc 0.8875
    ====================
    Additional 1 hidden 100
    MLP(
      (fc_initial): Linear(in_features=2, out_features=100)
      (fc_mid): ModuleList(
        (0): Linear(in_features=100, out_features=100)
      )
      (fc_final): Linear(in_features=100, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 10602




    



![png](MLP_Classification_files/MLP_Classification_33_18.png)



![png](MLP_Classification_files/MLP_Classification_33_19.png)


    Train acc 0.908333333333
    Test acc 0.88125
    ====================
    Additional 1 hidden 1000
    MLP(
      (fc_initial): Linear(in_features=2, out_features=1000)
      (fc_mid): ModuleList(
        (0): Linear(in_features=1000, out_features=1000)
      )
      (fc_final): Linear(in_features=1000, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 1006002




    



![png](MLP_Classification_files/MLP_Classification_33_23.png)



![png](MLP_Classification_files/MLP_Classification_33_24.png)


    Train acc 0.9125
    Test acc 0.88125
    ====================
    Additional 2 hidden 2
    MLP(
      (fc_initial): Linear(in_features=2, out_features=2)
      (fc_mid): ModuleList(
        (0): Linear(in_features=2, out_features=2)
        (1): Linear(in_features=2, out_features=2)
      )
      (fc_final): Linear(in_features=2, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 24




    



![png](MLP_Classification_files/MLP_Classification_33_28.png)



![png](MLP_Classification_files/MLP_Classification_33_29.png)


    Train acc 0.8875
    Test acc 0.875
    ====================
    Additional 2 hidden 10
    MLP(
      (fc_initial): Linear(in_features=2, out_features=10)
      (fc_mid): ModuleList(
        (0): Linear(in_features=10, out_features=10)
        (1): Linear(in_features=10, out_features=10)
      )
      (fc_final): Linear(in_features=10, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 272




    



![png](MLP_Classification_files/MLP_Classification_33_33.png)



![png](MLP_Classification_files/MLP_Classification_33_34.png)


    Train acc 0.9
    Test acc 0.88125
    ====================
    Additional 2 hidden 20
    MLP(
      (fc_initial): Linear(in_features=2, out_features=20)
      (fc_mid): ModuleList(
        (0): Linear(in_features=20, out_features=20)
        (1): Linear(in_features=20, out_features=20)
      )
      (fc_final): Linear(in_features=20, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 942




    



![png](MLP_Classification_files/MLP_Classification_33_38.png)



![png](MLP_Classification_files/MLP_Classification_33_39.png)


    Train acc 0.9
    Test acc 0.875
    ====================
    Additional 2 hidden 100
    MLP(
      (fc_initial): Linear(in_features=2, out_features=100)
      (fc_mid): ModuleList(
        (0): Linear(in_features=100, out_features=100)
        (1): Linear(in_features=100, out_features=100)
      )
      (fc_final): Linear(in_features=100, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 20702




    



![png](MLP_Classification_files/MLP_Classification_33_43.png)



![png](MLP_Classification_files/MLP_Classification_33_44.png)


    Train acc 0.908333333333
    Test acc 0.8875
    ====================
    Additional 2 hidden 1000
    MLP(
      (fc_initial): Linear(in_features=2, out_features=1000)
      (fc_mid): ModuleList(
        (0): Linear(in_features=1000, out_features=1000)
        (1): Linear(in_features=1000, out_features=1000)
      )
      (fc_final): Linear(in_features=1000, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 2007002




    



![png](MLP_Classification_files/MLP_Classification_33_48.png)



![png](MLP_Classification_files/MLP_Classification_33_49.png)


    Train acc 0.875
    Test acc 0.80625
    ====================
    Additional 3 hidden 2
    MLP(
      (fc_initial): Linear(in_features=2, out_features=2)
      (fc_mid): ModuleList(
        (0): Linear(in_features=2, out_features=2)
        (1): Linear(in_features=2, out_features=2)
        (2): Linear(in_features=2, out_features=2)
      )
      (fc_final): Linear(in_features=2, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 30




    



![png](MLP_Classification_files/MLP_Classification_33_53.png)



![png](MLP_Classification_files/MLP_Classification_33_54.png)


    Train acc 0.895833333333
    Test acc 0.8875
    ====================
    Additional 3 hidden 10
    MLP(
      (fc_initial): Linear(in_features=2, out_features=10)
      (fc_mid): ModuleList(
        (0): Linear(in_features=10, out_features=10)
        (1): Linear(in_features=10, out_features=10)
        (2): Linear(in_features=10, out_features=10)
      )
      (fc_final): Linear(in_features=10, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 382




    



![png](MLP_Classification_files/MLP_Classification_33_58.png)



![png](MLP_Classification_files/MLP_Classification_33_59.png)


    Train acc 0.891666666667
    Test acc 0.8875
    ====================
    Additional 3 hidden 20
    MLP(
      (fc_initial): Linear(in_features=2, out_features=20)
      (fc_mid): ModuleList(
        (0): Linear(in_features=20, out_features=20)
        (1): Linear(in_features=20, out_features=20)
        (2): Linear(in_features=20, out_features=20)
      )
      (fc_final): Linear(in_features=20, out_features=2)
    )
    {'lr': 0.1, 'epochs': 500, 'batch_size': 64}
    Num Params: 1362




    



![png](MLP_Classification_files/MLP_Classification_33_63.png)



![png](MLP_Classification_files/MLP_Classification_33_64.png)


    Train acc 0.908333333333
    Test acc 0.86875
    ====================
    Additional 3 hidden 100
    MLP(
      (fc_initial): Linear(in_features=2, out_features=100)
      (fc_mid): ModuleList(
        (0): Linear(in_features=100, out_features=100)
        (1): Linear(in_features=100, out_features=100)
        (2): Linear(in_features=100, out_features=100)
      )
      (fc_final): Linear(in_features=100, out_features=2)
    )
    {'lr': 0.1, 'epochs': 1000, 'batch_size': 64}
    Num Params: 30802




    



![png](MLP_Classification_files/MLP_Classification_33_68.png)



![png](MLP_Classification_files/MLP_Classification_33_69.png)


    Train acc 0.929166666667
    Test acc 0.88125
    ====================
    Additional 3 hidden 1000
    MLP(
      (fc_initial): Linear(in_features=2, out_features=1000)
      (fc_mid): ModuleList(
        (0): Linear(in_features=1000, out_features=1000)
        (1): Linear(in_features=1000, out_features=1000)
        (2): Linear(in_features=1000, out_features=1000)
      )
      (fc_final): Linear(in_features=1000, out_features=2)
    )
    {'lr': 0.1, 'epochs': 1000, 'batch_size': 64}
    Num Params: 3008002




    



![png](MLP_Classification_files/MLP_Classification_33_73.png)



![png](MLP_Classification_files/MLP_Classification_33_74.png)


    Train acc 0.958333333333
    Test acc 0.875

