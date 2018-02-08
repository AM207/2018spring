---
title: Gradient Descent and SGD
shorttitle: gradientdescent
notebook: gradientdescent.ipynb
noline: 1
summary: ""
keywords: ['optimization', 'gradient descent', 'sgd', 'minibatch sgd', 'linear regression']
layout: wiki
---
{% assign links = site.data.wikilinks %}

## Contents
{:.no_toc}
*  
{: toc}


## Contents
{:.no_toc}
* 
{: toc}



```python
%matplotlib inline
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy import stats 

from sklearn.datasets.samples_generator import make_regression 
```


A lot of the animations here were adapted from: http://tillbergmann.com/blog/python-gradient-descent.html

A great discussion (and where momentum image was stolen from) is at http://sebastianruder.com/optimizing-gradient-descent/

Gradient descent is one of the most popular algorithms to perform optimization and by far the most common way to optimize neural networks. Gradient descent is a way to minimize an objective function $J_{\theta}$ parameterized by a model's parameters $\theta \in \mathbb{R}^d$ by updating the parameters in the opposite direction of the gradient of the objective function $\nabla_J J(\theta)$ w.r.t. to the parameters. The learning rate $\eta$ determines the size of the steps we take to reach a (local) minimum. In other words, we follow the direction of the slope of the surface created by the objective function downhill until we reach a valley.

There are three variants of gradient descent, which differ in how much data we use to compute the gradient of the objective function. Depending on the amount of data, we make a trade-off between the accuracy of the parameter update and the time it takes to perform an update.

## Example: Linear regression

Let's see briefly how gradient descent can be useful to us in least squares regression. Let's asssume we have an output variable $y$ which we think depends linearly on the input vector $x$. We approximate $y$ by

$$f_\theta (x) =\theta^T x$$

The cost function for our linear least squares regression will then be

$$J(\theta) = \frac{1}{2} \sum_{i=1}^m (f_\theta (x^{(i)}-y^{(i)})^2$$


We create a regression problem using sklearn's `make_regression` function:



```python
#code adapted from http://tillbergmann.com/blog/python-gradient-descent.html
x, y = make_regression(n_samples = 100, 
                       n_features=1, 
                       n_informative=1, 
                       noise=20,
                       random_state=2017)
```




```python
x = x.flatten()
```




```python
slope, intercept, _,_,_ = stats.linregress(x,y)
best_fit = np.vectorize(lambda x: x * slope + intercept)
```




```python
plt.plot(x,y, 'o', alpha=0.5)
grid = np.arange(-3,3,0.1)
plt.plot(grid,best_fit(grid), '.')
```





    [<matplotlib.lines.Line2D at 0x111db8400>]




![png](gradientdescent_files/gradientdescent_8_1.png)


## Batch gradient descent

Assume that we have a vector of paramters $\theta$ and a cost function $J(\theta)$ which is simply the variable we want to minimize (our objective function). Typically, we will find that the objective function has the form:

$$J(\theta) =\sum_{i=1}^m J_i(\theta)$$

where $J_i$ is associated with the i-th observation in our data set. The batch gradient descent algorithm, starts with some initial feasible  $\theta$ (which we can either fix or assign randomly) and then repeatedly performs the update:

$$\theta := \theta - \eta \nabla_{\theta} J(\theta) = \theta -\eta \sum_{i=1}^m \nabla J_i(\theta)$$

where $\eta$ is a constant controlling step-size and is called the learning rate. Note that in order to make a single update, we need to calculate the gradient using the entire dataset. This can be very inefficient for large datasets.

In code, batch gradient descent looks like this:

```python
for i in range(n_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad`
```
  
For a given number of epochs $n_{epochs}$, we first evaluate the gradient vector of the loss function using **ALL** examples in the data set, and then we update the parameters with a given learning rate. This is where Theano and automatic differentiation come in handy, and you will learn about them in lab.

Batch gradient descent is guaranteed to converge to the global minimum for convex error surfaces and to a local minimum for non-convex surfaces.

In the linear example it's easy to see that our update step then takes the form:

$$\theta_j := \theta_j + \alpha \sum_{i=1}^m (y^{(i)}-f_\theta (x^{(i)})) x_j^{(i)}$$
for every $j$ (note $\theta_j$ is simply the j-th component of the $\theta$ vector).




```python
def gradient_descent(x, y, theta_init, step=0.001, maxsteps=0, precision=0.001, ):
    costs = []
    m = y.size # number of data points
    theta = theta_init
    history = [] # to store all thetas
    preds = []
    counter = 0
    oldcost = 0
    pred = np.dot(x, theta)
    error = pred - y 
    currentcost = np.sum(error ** 2) / (2 * m)
    preds.append(pred)
    costs.append(currentcost)
    history.append(theta)
    counter+=1
    while abs(currentcost - oldcost) > precision:
        oldcost=currentcost
        gradient = x.T.dot(error)/m 
        theta = theta - step * gradient  # update
        history.append(theta)
        
        pred = np.dot(x, theta)
        error = pred - y 
        currentcost = np.sum(error ** 2) / (2 * m)
        costs.append(currentcost)
        
        if counter % 25 == 0: preds.append(pred)
        counter+=1
        if maxsteps:
            if counter == maxsteps:
                break
        
    return history, costs, preds, counter
```




```python
np.random.rand(2)
```





    array([ 0.64739517,  0.92593885])





```python
xaug = np.c_[np.ones(x.shape[0]), x]
theta_i = [-15, 40] + np.random.rand(2)
history, cost, preds, iters = gradient_descent(xaug, y, theta_i, step=0.1)
theta = history[-1]
```




```python
print("Gradient Descent: {:.2f}, {:.2f} {:d}".format(theta[0], theta[1], iters))
print("Least Squares: {:.2f}, {:.2f}".format(intercept, slope))
```


    Gradient Descent: -3.73, 82.80 73
    Least Squares: -3.71, 82.90




```python
theta
```





    array([ -3.72552265,  82.79705269])



One can plot the reduction of cost:



```python
plt.plot(range(len(cost)), cost);
```



![png](gradientdescent_files/gradientdescent_16_0.png)


The following animation shows how the regression line forms:



```python
from JSAnimation import IPython_display
```




```python
def init():
    line.set_data([], [])
    return line,

def animate(i):
    ys = preds[i]
    line.set_data(xaug[:, 1], ys)
    return line,



fig = plt.figure(figsize=(10,6))
ax = plt.axes(xlim=(-3, 2.5), ylim=(-170, 170))
ax.plot(xaug[:,1],y, 'o')
line, = ax.plot([], [], lw=2)
plt.plot(xaug[:,1], best_fit(xaug[:,1]), 'k-', color = "r")

anim = animation.FuncAnimation(fig, animate, init_func=init,
                        frames=len(preds), interval=100)
anim.save('images/gdline.mp4')
anim
```






<script language="javascript">
  /* Define the Animation class */
  function Animation(frames, img_id, slider_id, interval, loop_select_id){
    this.img_id = img_id;
    this.slider_id = slider_id;
    this.loop_select_id = loop_select_id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.frames = new Array(frames.length);

    for (var i=0; i<frames.length; i++)
    {
     this.frames[i] = new Image();
     this.frames[i].src = frames[i];
    }
    document.getElementById(this.slider_id).max = this.frames.length - 1;
    this.set_frame(this.current_frame);
  }

  Animation.prototype.get_loop_state = function(){
    var button_group = document[this.loop_select_id].state;
    for (var i = 0; i < button_group.length; i++) {
        var button = button_group[i];
        if (button.checked) {
            return button.value;
        }
    }
    return undefined;
  }

  Animation.prototype.set_frame = function(frame){
    this.current_frame = frame;
    document.getElementById(this.img_id).src = this.frames[this.current_frame].src;
    document.getElementById(this.slider_id).value = this.current_frame;
  }

  Animation.prototype.next_frame = function()
  {
    this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));
  }

  Animation.prototype.previous_frame = function()
  {
    this.set_frame(Math.max(0, this.current_frame - 1));
  }

  Animation.prototype.first_frame = function()
  {
    this.set_frame(0);
  }

  Animation.prototype.last_frame = function()
  {
    this.set_frame(this.frames.length - 1);
  }

  Animation.prototype.slower = function()
  {
    this.interval /= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.faster = function()
  {
    this.interval *= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.anim_step_forward = function()
  {
    this.current_frame += 1;
    if(this.current_frame < this.frames.length){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.first_frame();
      }else if(loop_state == "reflect"){
        this.last_frame();
        this.reverse_animation();
      }else{
        this.pause_animation();
        this.last_frame();
      }
    }
  }

  Animation.prototype.anim_step_reverse = function()
  {
    this.current_frame -= 1;
    if(this.current_frame >= 0){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.last_frame();
      }else if(loop_state == "reflect"){
        this.first_frame();
        this.play_animation();
      }else{
        this.pause_animation();
        this.first_frame();
      }
    }
  }

  Animation.prototype.pause_animation = function()
  {
    this.direction = 0;
    if (this.timer){
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  Animation.prototype.play_animation = function()
  {
    this.pause_animation();
    this.direction = 1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function(){t.anim_step_forward();}, this.interval);
  }

  Animation.prototype.reverse_animation = function()
  {
    this.pause_animation();
    this.direction = -1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function(){t.anim_step_reverse();}, this.interval);
  }
</script>

<div class="animation" align="center">
    <img id="_anim_imgCBBRMWBTNHRAGPIA">
    <br>
    <input id="_anim_sliderCBBRMWBTNHRAGPIA" type="range" style="width:350px" name="points" min="0" max="1" step="1" value="0" onchange="animCBBRMWBTNHRAGPIA.set_frame(parseInt(this.value));"></input>
    <br>
    <button onclick="animCBBRMWBTNHRAGPIA.slower()">&#8211;</button>
    <button onclick="animCBBRMWBTNHRAGPIA.first_frame()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAgaeZk4EQAAASlJREFUKM/dkj9LQnEUhp9zr3bpj1uBcKGiJWxzLWivKAIRjIhcCqcgqJbKRagPICiVSVEuNTu0tLYGUg4tkRGUdxLJ0u79Ndxr5FfwTO/L+xzO4XCgO+v2T70AFU+/A/Dhmlzg6Pr0DKAMwOH4zQxAAbAkv2xNeF2RoQUVc1ytgttXUbWVdN1dOPE8pz4j4APQsdFtKA0WY6vpKjqvVciHnvZTS6Ja4HgggJLs7MHxl9nCh8NYcO+iGG0agiaC4h9oa6Vsw2yiK+QHSZT934YoEQABNBcTNDszsrhm1m1B+bFS86PT6QFppx6oeSaeOwlMXRp1h4aK13Y2kuHhUo9ykPboPvFjeEvsrhTMt3ylHyB0r8KZyYdCrbfj4OveoHMANjuyx+76rV+/blxKMZUnLgAAAABJRU5ErkJggg=="></button>
    <button onclick="animCBBRMWBTNHRAGPIA.previous_frame()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAgyTCyQ6wAAANRJREFUKM9jYBjO4AiUfgzFGGAp4+yayUvX6jMwMDCsYmBgOCS4OAOrSYmMgcc8/pd5Q3irC+Neh/1AlmeBMVgZmP8yMLD8/c/cqv9r90whzv/MX7Eq/MfAwMDIwCuZdfSV8U8WDgZGRmYGrAoZGRgY/jO8b3sj/J2F6T8j4z80pzEhmIwMjAxsSbqqlkeZGP//Z8SlkJnhPwMjwx/Guoe1NhmRwk+YGH5jV8jOwMPHzcDBysAwh8FrxQwtPU99HrwBXsnAwMDAsJiBgYGBoZ1xmKYqALHhMpn1o7igAAAAAElFTkSuQmCC"></button>
    <button onclick="animCBBRMWBTNHRAGPIA.reverse_animation()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAgmVvZElgAAAVFJREFUKM+t0k8ow3EYx/H3s/2aLDUSZctFkgsHEi1XLi5ukpPSWsuJklwclsPSsDKFi7MSJ0I5qF2GHO2m0FY7+BdNv7Y9DpuxDSt5vsfvq+fT9/k+8D8VBxIAWH6H0ead4Qb5BRwCENoceZi5Stl/6BgCBmtWhjzxg4mUQ02rAhil7JgB9tze7aTLxFAKsUUd14B9ZzCyFUk401gQyQJaDNcBHwv7t7ETd0ZVQFEEzcNCdE/1wtj15imGWlEB8qkf2QaAWjbG/bPSamIDyX65/iwDIFx7tWjUvWCoSo5oGbYATN7PORt7W9IZEQXJH8ohuN7C0VVX91KNqYhq4a1lEGJI0j892tazXCWQRUpwAbYDcHczPxXuajq3mbnhfANz5eOJxsuNvs7+jud0UcuyL3QAkuEMx4rnIvBYq1JhEwPAUb3fG7x8tVdc292/7Po7f2VqA+Yz7ZwAAAAASUVORK5CYII="></button>
    <button onclick="animCBBRMWBTNHRAGPIA.pause_animation()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAkR91DQ2AAAAKtJREFUKM9jYCANTEVib2K4jcRbzQihGWEC00JuNjN8Z2Q0Zo3VYWA4lL005venH9+c3ZK5IfIsMIXMBtc12Bj+MMgxMDAwMPzWe2TBzPCf4SLcZCYY4/9/RgZGBiaYFf8gljFhKiQERhUOeoX/Gf8y/GX4y/APmlj+Mfxj+MfwH64Qnnq0zr9fyfLrPzP3eQYGBobvk5x4GX4xMIij23gdib0cRWYHiVmAAQDK5ircshCbHQAAAABJRU5ErkJggg=="></button>
    <button onclick="animCBBRMWBTNHRAGPIA.play_animation()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAkEmo00MwAAAS9JREFUKM+tkj1IQmEUhp9j94LQj0FD4RRBLdLQ3ftb26PRcCiQIIiIDFwKC0OhaAiam5wVDBpqCKohQojMLYzaAiUatOtpuQrKVQl64fu+4Xt4OLwc+Fs+nNM16jsPAWS6gZXggoZfXmfhog3hcZ6aTXF87Sp68OmH4/YggAo8bmfyyeh6Z1AAKPVldyO1+Iz2uILq3AriJSe3l+H7aj+cuRnrTsVDxSxay+VYbMDnCtZxxQOU9G4nlU9E1HQBxRkCQMRGRnIbpxMARkvxCIoAorYMMrq0mJ0qu4COUW3xyVDqJC4P+86P0ewDQbQqgevhlc2C8ETApXAEFLzvwa3EXG9BoIE1GQUbv1h7k4fTXxBu6cKgUbX5M3ZzNC+a7rQ936HV56SlRpcle+Mf8wvgJ16zo/4BtQAAAABJRU5ErkJggg=="></button>
    <button onclick="animCBBRMWBTNHRAGPIA.next_frame()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAkd/uac8wAAAMhJREFUKM9jYBie4DEUQ8B+fEq3+3UrMzAwMFxjYGBgYJizYubaOUxYFUaXh/6vWfRfEMIL/+//P5gZJoei4/f/7wxnY1PeNUXdE2RgYGZgYoCrY2BBVsjKwMDAwvCS4f3SG/dXxm5gYESSQ1HIwvCPgZmB8f8Pxv+Kxxb/YfiPJIdi9T8GJgaG/38ZFd4Fx0xUYsZt4h8GBgb2D2bLy7KnMTAwMEIxFoVCXIYr1IoDnkF4XAysqNIwUMDAwMDAsADKS2NkGL4AAIARMlfNIfZMAAAAAElFTkSuQmCC"></button>
    <button onclick="animCBBRMWBTNHRAGPIA.last_frame()"><img class="anim_icon" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABQAAAAUCAQAAAAngNWGAAAAAXNSR0IArs4c6QAAAAJiS0dEAP+Hj8y/AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAB3RJTUUH3QURCAknOOpFQQAAAS9JREFUKM/dkrEvQ3EQxz/33mtoQxiYpANbLU26NAabSCcSUouGBVNDjYQaOiDpIEiKjURIw2Kx04hEYmkHEpGoJpSISaXq9Wd4P03/ht5y98197/u9XA4aK4rAWw3lgWddZ3S+/G9mEovtAB8AHE4pgTQAx8PbJweRmsq6GimmNpxaNYXVzMNNCI6A2figimwCGACK786zuWgh3qcsKf/w0pM4X0m/doNVFVzVGlEQsdRj193VxEWpH0RsdRu+zi3tVMqCAsDShoiYqiSV4OouVDFEqS9Pbiyg7vV62lpQ2BJ4Gg0meg0MbNpkYG/e+540NNFyrE1a8qHk5BaAjfnrzUaHfAWImVrLIXbgnx4/9X06s35cweWsVACa3a24PVp0X+rPv1aHFnSONdiL8Qci0lzwpOM5sQAAAABJRU5ErkJggg=="></button>
    <button onclick="animCBBRMWBTNHRAGPIA.faster()">+</button>
  <form action="#n" name="_anim_loop_selectCBBRMWBTNHRAGPIA" class="anim_control">
    <input type="radio" name="state" value="once" > Once </input>
    <input type="radio" name="state" value="loop" checked> Loop </input>
    <input type="radio" name="state" value="reflect" > Reflect </input>
  </form>
</div>


<script language="javascript">
  /* Instantiate the Animation class. */
  /* The IDs given should match those used in the template above. */
  (function() {
    var img_id = "_anim_imgCBBRMWBTNHRAGPIA";
    var slider_id = "_anim_sliderCBBRMWBTNHRAGPIA";
    var loop_select_id = "_anim_loop_selectCBBRMWBTNHRAGPIA";
    var frames = new Array(0);
    
  frames[0] = "data:image/png;base64,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"
  frames[1] = "data:image/png;base64,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JC0tTdKRn5nbbrtNd9xxR5hXhGDjn4JBMHz4cA0fPtzt11599VV9//33GjNmjD766KMQryx+eXom33zzje666y5NnDhRffv2DcPK4pu3nxWEX3p6uiorKx3j+vp6wjPgwc6dOzV27FiNHDlSl112WbiXgyCjBjpEXnzxRb377ruSjvymyn/eCb/vvvtOt99+u2bOnKnzzjsv3MsBIk6fPn20fPlySdLatWvVvXv3MK8IiEylpaW64YYbNGHCBF111VXhXg5CgK2EELnyyit1zz33aMGCBaqrq9P06dPDvaS4N3PmTNXU1OjRRx+VdGS37fnnnw/zqoDIMXjwYH366acaMWKE7HY7/9wCPHjhhRdUVlam5557Ts8995ykIy97tmzZMswrQ7BY7Ha7PdyLAAAAAKIFJRwAAACACQRoAAAAwAQCNAAAAGACARoAAAAwgQANAAAAmECABgAAAEwgQAMAAAAmEKABAAAAEwjQAAAAgAkEaAAAAMAEAjQAAABgAgEaAAAAMIEADQAAAJhAgAYAAABMIEADAAAAJhCgAQAAABMI0AAAAIAJBGgAAADABAI0AAAAYAIBGgAAADCBAA0AAACYQIAGAAAATCBAAwAAACYQoAEAAAATCNAAAACACQRoAAAAwAQCNAAAAGACARoAAAAwgQANAAAAmECABgAAAEwgQAMAAAAmEKABAAAAEwjQAAAAgAkEaAAAAMAEAjQAAABgAgEaAAAAMIEADQAAAJhAgAYAAABMIEADAAAAJvw/hBshZjPF298AAAAASUVORK5CYII="
  frames[2] = "data:image/png;base64,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"


    /* set a timeout to make sure all the above elements are created before
       the object is initialized. */
    setTimeout(function() {
        animCBBRMWBTNHRAGPIA = new Animation(frames, img_id, slider_id, 100, loop_select_id);
    }, 0);
  })()
</script>




Remember that the linear regression cost function is convex, and more precisely quadratic. We can see the path that gradient descent takes in arriving at the optimum:



```python
from mpl_toolkits.mplot3d import Axes3D

def error(X, Y, THETA):
    return np.sum((X.dot(THETA) - Y)**2)/(2*Y.size)

def make_3d_plot(xfinal, yfinal, zfinal, hist, cost, xaug, y):
    ms = np.linspace(xfinal - 20 , xfinal + 20, 20)
    bs = np.linspace(yfinal - 40 , yfinal + 40, 40)
    M, B = np.meshgrid(ms, bs)
    zs = np.array([error(xaug, y, theta) 
                   for theta in zip(np.ravel(M), np.ravel(B))])
    Z = zs.reshape(M.shape)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.1)
    ax.contour(M, B, Z, 20, color='b', alpha=0.5, offset=0, stride=30)
    ax.set_xlabel('Intercept')
    ax.set_ylabel('Slope')
    ax.set_zlabel('Cost')
    ax.view_init(elev=30., azim=30)
    ax.plot([xfinal], [yfinal], [zfinal] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7);
    ax.plot([t[0] for t in hist], [t[1] for t in hist], cost , markerfacecolor='b', markeredgecolor='b', marker='.', markersize=5);
    ax.plot([t[0] for t in hist], [t[1] for t in hist], 0 , alpha=0.5, markerfacecolor='r', markeredgecolor='r', marker='.', markersize=5)
    
def gd_plot(xaug, y, theta, cost, hist):
    make_3d_plot(theta[0], theta[1], cost[-1], hist, cost, xaug, y)
```




```python
gd_plot(xaug, y, theta, cost, history)
```



![png](gradientdescent_files/gradientdescent_22_0.png)




```python
from mpl_toolkits.mplot3d import Axes3D

def error(X, Y, THETA):
    return np.sum((X.dot(THETA) - Y)**2)/(2*Y.size)

ms = np.linspace(theta[0] - 20 , theta[0] + 20, 20)
bs = np.linspace(theta[1] - 40 , theta[1] + 40, 40)

M, B = np.meshgrid(ms, bs)

zs = np.array([error(xaug, y, theta) 
               for theta in zip(np.ravel(M), np.ravel(B))])
Z = zs.reshape(M.shape)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.2)
ax.contour(M, B, Z, 20, color='b', alpha=0.5, offset=0, stride=30)


ax.set_xlabel('Intercept')
ax.set_ylabel('Slope')
ax.set_zlabel('Cost')
ax.view_init(elev=30., azim=30)
ax.plot([theta[0]], [theta[1]], [cost[-1]] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7);
#ax.plot([history[0][0]], [history[0][1]], [cost[0]] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7);


ax.plot([t[0] for t in history], [t[1] for t in history], cost , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2);
ax.plot([t[0] for t in history], [t[1] for t in history], 0 , markerfacecolor='r', markeredgecolor='r', marker='.', markersize=2);
```



![png](gradientdescent_files/gradientdescent_23_0.png)


## Stochastic gradient descent

As noted, the gradient descent algorithm makes intuitive sense as it always proceeds in the direction of steepest descent (the gradient of $J$) and guarantees that we find a local minimum (global under certain assumptions on $J$). When we have very large data sets, however, the calculation of $\nabla (J(\theta))$ can be costly as we must process every data point before making a single step (hence the name "batch"). An alternative approach, the stochastic gradient descent method, is to update $\theta$ sequentially with every observation. The updates then take the form:

$$\theta := \theta - \alpha \nabla_{\theta} J_i(\theta)$$

This stochastic gradient approach allows us to start making progress on the minimization problem right away. It is computationally cheaper, but it results in a larger variance of the loss function in comparison with batch gradient descent. 

Generally, the stochastic gradient descent method will get close to the optimal $\theta$ much faster than the batch method, but will never fully converge to the local (or global) minimum. Thus the stochastic gradient descent method is useful when we want a quick and dirty approximation for the solution to our optimization problem. A full recipe for stochastic gradient descent follows:

- Initialize the parameter vector $\theta$ and set the learning rate $\alpha$
- Repeat until an acceptable approximation to the minimum is obtained:
    - Randomly reshuffle the instances in the training data.
    - For $i=1,2,...m$ do: $\theta := \theta - \alpha \nabla_\theta J_i(\theta)$
    
The reshuffling of the data is done to avoid a bias in the optimization algorithm by providing the data examples in a particular order. In code, the algorithm should look something like this:

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

For a given epoch, we first reshuffle the data, and then for a single example, we evaluate the gradient of the loss function and then update the params with the chosen learning rate.

The update for linear regression is:

$$\theta_j := \theta_j + \alpha (y^{(i)}-f_\theta (x^{(i)})) x_j^{(i)}$$




```python
def sgd(x, y, theta_init, step=0.001, maxsteps=0, precision=0.001, ):
    costs = []
    currentcosts = []
    m = y.size # number of data points
    oldtheta = 0
    theta = theta_init
    history = [] # to store all thetas
    preds = []
    grads = []
    xs = []
    ys = []
    counter = 0
    oldcost = 0
    epoch = 0
    i = 0 #index
    xs.append(x[i,:])
    ys.append([y[i]])
    pred = np.dot(x[i,:], theta)
    error = pred - y[i]
    gradient = x[i,:].T*error
    grads.append(gradient)
    currentcost = np.sum(error ** 2) / 2
    print("Init", gradient, x[i,:],y[i])
    print ("Init2", currentcost, theta)
    currentcosts.append(currentcost)
    counter+=1
    preds.append(pred)
    costsum = currentcost
    costs.append(costsum/counter)
    history.append(theta)
    print("start",counter, costs, oldcost)
    while 1:
        #while abs(costs[counter-1] - oldcost) > precision:
        #while np.linalg.norm(theta - oldtheta) > precision:
        gradient = x[i,:].T*error
        grads.append(gradient)
        oldtheta = theta
        theta = theta - step * gradient  # update
        history.append(theta)
        i += 1
        if i == m:#reached one past the end.
            #break
            epoch +=1
            neworder = np.random.permutation(m)
            x = x[neworder]
            y = y[neworder]
            i = 0
        xs.append(x[i,:])
        ys.append(y[i])
        pred = np.dot(x[i,:], theta)
        error = pred - y[i]
        currentcost = np.sum(error ** 2) / 2
        currentcosts.append(currentcost)
        
        #print("e/cc",error, currentcost)
        if counter % 25 == 0: preds.append(pred)
        counter+=1
        costsum += currentcost
        oldcost = costs[counter-2]
        costs.append(costsum/counter)
        #print(counter, costs, oldcost)
        if maxsteps:
            #print("in maxsteps")
            if counter == maxsteps:
                break
        
    return history, costs, preds, grads, counter, epoch, xs, ys, currentcosts
```




```python
history2, cost2, preds2, grads2, iters2, epoch2, x2, y2, cc2 = sgd(xaug, y, theta_i, maxsteps=5000, step=0.01)

```


    Init [-24.75520774  -0.79844029] [ 1.          0.03225343] 11.5348518902
    Init2 306.410155183 [-14.53764872  40.84195039]
    start 1 [306.41015518349798] 0




```python
print(iters2, history2[-1], epoch2, grads2[-1])
```


    5000 [ -3.15191155  82.514033  ] 49 [-41.75304983  22.05572989]




```python
plt.plot(range(len(cost2[-10000:])), cost2[-10000:], alpha=0.4);
```



![png](gradientdescent_files/gradientdescent_28_0.png)




```python
gd_plot(xaug, y, theta, cost2, history2)
```



![png](gradientdescent_files/gradientdescent_29_0.png)




```python
plt.plot([t[0] for t in history2], [t[1] for t in history2],'o-', alpha=0.1)
```





    [<matplotlib.lines.Line2D at 0x11e3f11d0>]




![png](gradientdescent_files/gradientdescent_30_1.png)


#### Animating SGD

Here is some code to make an animation of SGD. It shows how the risk surfaces being minimized change, and how the minimum desired is approached.



```python
def error2(X, Y, THETA):
    #print("XYT", THETA, np.sum((X.dot(THETA) - Y)**2))
    return np.sum((X.dot(THETA) - Y)**2)/(2*Y.size)


def make_3d_plot2(num, it, xfinal, yfinal, zfinal, hist, cost, xaug, y):
    ms = np.linspace(xfinal - 20 , xfinal + 20, 20)
    bs = np.linspace(yfinal - 50 , yfinal + 50, 40)
    M, B = np.meshgrid(ms, bs)
    zs = np.array([error2(xaug, y, theta) 
                   for theta in zip(np.ravel(M), np.ravel(B))])
    Z = zs.reshape(M.shape)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    ax.plot_surface(M, B, Z, rstride=1, cstride=1, color='b', alpha=0.1)
    ax.contour(M, B, Z, 20, color='b', alpha=0.5, offset=0, stride=30)
    ax.set_xlabel('Intercept')
    ax.set_ylabel('Slope')
    ax.set_zlabel('Cost')
    ax.view_init(elev=30., azim=30)
    #print("hist", xaug, y, hist, cost)
    ax.plot([xfinal], [yfinal], [zfinal] , markerfacecolor='r', markeredgecolor='r', marker='o', markersize=7);
    #ax.plot([t[0] for t in hist], [t[1] for t in hist], cost , markerfacecolor='b', markeredgecolor='b', marker='.', markersize=5);
    ax.plot([t[0] for t in hist], [t[1] for t in hist], 0 , alpha=0.5, markerfacecolor='r', markeredgecolor='r', marker='.', markersize=5)
    ax.set_zlim([0, 3000])
    plt.title("Iteration {}".format(it))
    plt.savefig("images/3danim{0:03d}.png".format(num))
    plt.close()
```




```python
print("fthetas",theta[0], theta[1], "len", len(history2))
ST = list(range(0, 750, 10)) + list(range(750, 5000, 250))
len(ST)
```


    fthetas -3.72552265122 82.7970526876 len 5000





    92





```python
for i in range(len(ST)):
    #print(history2[i*ST[i]], cc2[i*ST[i]])
    make_3d_plot2(i, ST[i], theta[0], theta[1], cost2[-1], [history2[ST[i]]], [cc2[ST[i]]], np.array([x2[ST[i]]]), np.array([y2[ST[i]]]))
```


Using Imagemagick we can produce a gif animation:
(`convert -delay 20 -loop 1 3danim*.png animsgd.gif`)

(I set this animation to repeat just once. (`loop 1`). Reload this cell to see it again. On the web page right clicking the image might allow for an option to loop again)

![](images/animsgd.gif)

## Mini-batch gradient descent
What if instead of single example from the dataset, we use a batch of data examples witha given size every time we calculate the gradient:

$$\theta = \theta - \eta \nabla_{\theta} J(\theta; x^{(i:i+n)}; y^{(i:i+n)})$$

This is what mini-batch gradient descent is about. Using mini-batches has the advantage that the variance in the loss function is reduced, while the computational burden is still reasonable, since we do not use the full dataset. The size of the mini-batches becomes another hyper-parameter of the problem. In standard implementations it ranges from 50 to 256. In code, mini-batch gradient descent looks like this:

```python
for i in range(mb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

The difference with SGD is that for each update we use a batch of 50 examples to estimate the gradient.

## Variations on a theme

### Momentum

Often, the cost function has ravines near local optima, ie. areas where the shape of the function is significantly steeper in certain dimensions than in others. This migh result in a slow convergence to the optimum, since standard gradient descent will keep oscillating about these ravines. In the figures below, the left panel shows convergence without momentum, and the right panel shows the effect of adding momentum:

<table><tr><td><img src="http://sebastianruder.com/content/images/2015/12/without_momentum.gif", width=300, height=300></td><td><img src="http://sebastianruder.com/content/images/2015/12/with_momentum.gif", width=300, height=300></td></tr></table>

One way to overcome this problem is by using the concept of momentum, which is borrowed from physics. At each iteration, we remember the update $v = \Delta \theta$ and use this *velocity* vector (which as the same dimension as $\theta$) in the next update, which is constructed as a combination of the cost gradient and the previous update:

$$v_t = \gamma v_{t-1} +  \eta \nabla_{\theta} J(\theta)$$
$$\theta = \theta - v_t$$

The effect of this is the following: the momentum terms increases for dimensions whose gradients point in the same direction, and reduces the importance of dimensions whose gradients change direction. This avoids oscillations and improves the chances of rapid convergence. The concept is analog to the  a rock rolling down a hill: the gravitational field (cost function) accelerates the particule (weights vector), which accumulates momentum, becomes faster and faster and tends to keep travelling in the same direction. A commonly used value for the momentum parameter is $\gamma = 0.5$.
