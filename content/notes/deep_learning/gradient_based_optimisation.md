---
title: Gradient-Based Optimisation
linktitle: Gradient-Based Algorithms
toc: true
type: book
date: "2019-05-05T00:00:00+01:00"
draft: false
url_code: "https://github.com/andneo/andneo_code/tree/master/deep_learning/optimisation_algorithms"
menu:
  deep_learning:
    parent: Optimising Functions
    weight: 3

# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 4
---

Optimisation is a ubiquitous process, people do it everyday. A classic example would be: _How do I get somewhere as (**fast, cheap, environmentally friendly**) as possible?_  Nature optimises too, a big part of what I did in my PhD was finding out how systems of particles optimise their potential energy. 
Needless to say, optimisation is an important process, and quite a bit of effort has been dedicated to developing algorithms that do it well.
Optimisation algorithms form a key component of neural networks, as we will see. 
The first section of these notes introduce gradient-based optimisation algorithms, and how to implement them in Python. 

Before getting stuck into optimisation algorithms, we should first introduce some notation.
First, and maybe most important, is the {{< hl >}} objective function {{< /hl >}}.
This function should provide some quantitative description of our process of interest (think of it as a measure of success).
We denote the objective function as $f(\\textbf{x})$, where $\\textbf{x}$ is an $n$-dimensional vector of parameters.

Consider, for now, the following arbitrary function where $\\textbf{x}=(x,y)^{T}$:
$$
z = f(x,y) = x^2 - y^2
$$
I have plotted this function for $x,y\in[-2,2]$ below, and we can see (from the 3D representation on the left) it has a [saddle](https://en.wikipedia.org/wiki/Saddle_point) shape.
When $\textbf{x}=(0,0)^{T}$, $f(x,y)$ is at a [minimum](https://en.wikipedia.org/wiki/Maxima_and_minima) in the xz-plane and at a [maximum](https://en.wikipedia.org/wiki/Maxima_and_minima) in the yz-plane.
{{< figure library="true" src="deep_learning/images/gradient.png" title="" >}}

On the right I've shown a 2D [contour plot](https://www.statisticshowto.com/contour-plots/) of the same function, as well as the *_gradient_*.
The gradient for this function is given by:
\begin{align}
    \textbf{g}(\textbf{x}) \equiv \nabla f(x,y) &= \Big(\dfrac{\partial f}{\partial x}, \dfrac{\partial f}{\partial y}\Big)^{T} \\\\
    &= \Big(2x, -2y\Big)^{T}
\end{align}
The gradient defines a [vector field](https://en.wikipedia.org/wiki/Vector_field), which points in the [direction of maximum rate of increase](http://mathonline.wikidot.com/the-maximum-rate-of-change-at-a-point-on-a-function-of-sever) for its corresponding function.
This is an important property of the gradient which we are going to exploit.

## Optimisation Problems
We are interested in solving the general optimisation problem:
$$
    \\min_{\\textbf{x}\\in\\mathbb{R}^{n}}f(\\textbf{x})
$$
In other words, we want to find a set of coordinates which correspond to a [minimum](https://en.wikipedia.org/wiki/Maxima_and_minima) of the function $f(\\textbf{x})$.
We are going to do this using an optimisation algorithm.
There are different algorithms to choose from, but they all follow the same general process: iteratively update an initial guess for $\\textbf{x}$ until some termination condition is satisified.

The difference between different algorithms comes from how we move from $\\textbf{x}\_{i}$, at iteration $i$ of our algorithm, to $\\textbf{x}\_{i+1}$.
We want to write some Python code that can optimise arbitrary functions using different optimisation algorithms.
The first step is to define a ``` Class ```:
```python
class Optimise:
    def __init__(self, X, function, gradient, err, method):
    #   Initialise input parameters for the optimisation algorithms
        self.X      = X         # Initial coordinates.
        self.f      = function  # Function to be optimised.
        self.g      = gradient  # Gradient of the function.
        self.err    = err       # Threshold convergence value
        self.method = method    # ID of the line search method
```
This ``` Class ``` has three key attributes:
1. ``` X ```: The initial coordinates from which to optimise from, $\textbf{x}\_{i=0}$.
2. ``` f ```: The objective function to be optimised, $f(\textbf{x})$.
3. ``` g ```: The gradient of the objective function, $\textbf{g}(\textbf{x})$.

There are two more parameters, ``` err ``` and ``` method ```.
We use the ``` err ``` parameter to define our termination condition, and the ``` method ``` variable to decide which method to use: 
```python
#   Initialise parameters describing the convergence of the optimisation algorithms
    self.nsteps = 0
    self.path   = []
    self.steps  = []
#   Perform local optimisation.
    if(method==1):
        self.steepest_descent()
    elif(method==2):
        self.conjugate_gradient()
#   Extract the coordinates of the local minimum and the path taken to it.
    self.minimum = self.path[-1]
    self.path    = np.array(self.path).T
```
We have also created some new variables which we are going to use to monitor the path the algorithms take to the minimum, and their convergence properties.
## Line Search Strategies
We are going to focus on *line search strategies* -- algorithms where we select a direction $\\textbf{d}\_{i}$ from $\\textbf{x}\_{i}$, and search along that direction to find a new set of coordinates $\\textbf{x}\_{i+1}$ where $f(\\textbf{x}\_{i}) < f(\\textbf{x}\_{i+1})$ (i.e., a descent direction). We write this as:
$$
    \\textbf{x}\_{i+1} = \\textbf{x}\_{i} + \\alpha\_{i}\\textbf{d}\_{i}, \\quad i=0, 1, 2, ...
$$
Therefore, we have two tasks at each iteration:
1. Determine $\\textbf{d}\_{i}$
2. Determine $\\alpha\_{i}$ (the step length).

### Steepest Descent
Using Taylor's theorem we know that for any $\textbf{d}\_{i}$ and $\alpha\_{i}$ we have:
$$
    f(\textbf{x}\_{i}+\alpha\_{i}\textbf{d}\_{i}) = f(\textbf{x}\_{i}) + \alpha\_{i}\textbf{d}\_{i}^{T}\textbf{g}(\textbf{x}\_{i}) + \mathcal{O}(\alpha^{2}_{i})
$$
If we look at this second term a bit more closely we can define a condition for $\textbf{d}\_{i}$ to be a descent direction:
$$
    \textbf{d}\_{i}^{T}\textbf{g}(\textbf{x}\_{i}) = \|\|\textbf{d}\_{i}^{T}\|\|\cdot\|\|\textbf{g}(\textbf{x}\_{i})\|\|\cos\theta\_{i} < 0
$$
where $\theta\_{i}$ is the angle between $\textbf{d}\_{i}$ and $\textbf{g}\_{i}$. 
It should be easy to see that the steepest descent direction of a function is given by $\textbf{d}\_{i}=-\textbf{g}\_{i}$ (i.e., when $\cos\theta\_{i}=-1$).

This is the most obvious choice for a search direction, and forms the basis of our entry level optimisation algorithm:
```python
def steepest_descent(self):
#   Define the initial coordinates for iteration i=0
    x0 = self.X; xi = x0
#   Compute the gradient and the square of its magnitude at i=0
    gi = np.array(self.g(*xi)); gd = np.dot(gi,gi)
#   Add the initial coordinates the path to the local minimum.
    self.path.append(xi)
#   Calculate the square of the convergence threshold.
    errsq = self.err**2
#   Iteratively update the coordinates using the Steepest Descent algorithm
#   until the convergence criterion is met.
    while gd > errsq:
#   Determine the step size for this iteration using the backtracking algorithm.
        a = self.backtrack(xi=xi,gi=gi,di=-gi, a0=1)
#   Update the coordinates
        xi = xi - a*gi
#   Calculate the gradient and the square of its magnitude at the new coordinates
        gi = np.array(self.g(*xi)); gd = np.dot(gi,gi)
#   Update parameters describing the convergence of the optimisation algorithm.
        self.path.append(xi); self.nsteps += 1; self.steps.append(a)
```
One issue with this method is that it becomes very inefficient near the minimum where the magnitude of the gradient is very small.
### Conjugate Gradient
The conjugate gradient method has proved to be quite a successful optimisation algorithm and doesn't suffer from the same inefficiencies near the minimum as the steepest descent method.
The principle is exactly the same as the steepest descent method, but now we replace $\textbf{d}\_{i}=-\textbf{g}\_{i}$ with:
$$
\textbf{d}\_{i} = 
\begin{cases}
    -\textbf{g}(\textbf{x}\_{i}) & \text{if } i=0 \\\\
    -\textbf{g}(\textbf{x}\_{i})+\dfrac{\textbf{g}(\textbf{x}\_{i})^{T}\textbf{g}(\textbf{x}\_{i})}{\textbf{g}(\textbf{x}\_{i-1})^{T}\textbf{g}(\textbf{x}\_{i-1})}\textbf{d}\_{i-1} & \text{otherwise}
\end{cases}
$$
We can see that we now have some history in our search direction.
```python
def conjugate_gradient(self):
#   Define the initial coordinates for iteration i=0  
    x0 = self.X; xi = x0
#   Compute the gradient and the square of its magnitude at i=0
    gi = np.array(self.g(*xi)); gd = np.dot(gi,gi)
#   Compute the search direction, taking it to be equal to the negative of the 
#   gradient at iteration i=0.
    di = -gi
#   Add the initial coordinates the path to the local minimum.
    self.path.append(xi)
#   Calculate the square of the convergence threshold.
    errsq = self.err**2
#   Iteratively update the coordinates using the Conjugate Gradient algorithm
#   until the convergence criterion is met.    
    while gd > errsq:
#   Determine the step size for this iteration using the backtracking algorithm.
        a = self.backtrack(xi=xi,gi=gi,di=di,a0=1)
#   Update the coordinates
        xi = xi + a*di
#   Save the old gradient and search direction, which will be used to calculate 
#   the search direction for the next iteration.
        gj  = gi; gdj = gd; dj  = di
#   Calculate the gradient and the square of its magnitude at the new coordinates
        gi = np.array(self.g(*xi)); gd = np.dot(gi,gi)
#   Calculate the search direction for the next iteration.
        di = dj*np.dot(gi,(gi-gj))/gdj - gi
#   Update parameters describing the convergence of the optimisation algorithm.
        self.path.append(xi); self.nsteps += 1; self.steps.append(a)
```
The layout of the two optimisation algorithms are identical, the only difference with the conjugate gradient method is that we need to save the gradient and search direction at each iteration to use in the next one.

### Deciding the Step Length
Now that we know how to extract a search direction which points towards the minimum, we need to figure out how far we want to travel in that direction.

You may have noticed in the code snippets above for the steepest descent and conjugate gradient methods a third method is called: ``` backtrack(xi=xi,gi=gi,di=di,a0=1) ```.
This is the method we use to determine our step size $\alpha\_{i}$. 
#### Backtracking
When computing $\alpha\_{i}$ we face a tradeoff: We want $\alpha\_{i}$ to make a substantial decrease in $f(\textbf{x}\_{i})$, but we also don't want to waste a lot of time choosing its value.
In practise we try out a number of candidate $\alpha\_{i}$ values, and choose the first one which satisfies some conditions.
We are going to use the *sufficient decrease* condition:
$$
    f(\textbf{x}\_{i}+\alpha\_{i}\textbf{d}\_{i}) \leq f(\textbf{x}\_{i}) + c\_{1}\alpha\_{i}\textbf{g}(\textbf{x}\_{i})^{T}\textbf{d}\_{i}, \quad c\_{1}\in(0,1)
$$
One issue with this inequality is that is satisfied for all sufficiently small values $\alpha\_{i}$.
This issue can be avoided though by choosing the candidate $\alpha\_{i}$ values appropriately.
We do this using the *backtracking* approach:
```python
def backtrack(self, xi, gi, di, a0, c1=0.5, tau=0.5):
#   Calculate the value of the function at the coordinates for the 
#   current iteration of the optimisation algorithm.
    fi = self.f(*xi)
#   Calculate the dot product of the gradient and the search direction,
#   to be used to evaluate the Armijo condition.  
    gi = np.dot(gi, di)
    ai = a0
#   While the step size does not provide a sufficient decrease in the function f(X),
#   adjust the step size using the contraction factor tau.
    while( self.f( *(xi+ai*di) ) > (fi + c1*ai*gi) ):
        ai *= tau

    return ai
```
We select an initial guess for $\alpha^{0}\_{i}$ (which we just set to 1 by default) and iteratively decrease its size by the factor $\tau\in(0,1)$ until the sufficient decrease condition is satisfied.
## Testing our Optimisation Algorithms
Now that we've built our optimiser we are ready to test it out.
There are a set of canonical [test functions](https://en.wikipedia.org/wiki/Test_functions_for_optimization) for optimisation algorithms, which we'll use to be absolutely sure our code works.
### Beale function
We'll start of with the 2D Beale function:
$$ f(x,y) = (1.5-x+xy)^{2} + (2.25-x+xy^{2})^{2} + (2.625-x+xy^{3})^{2} $$
which has the minimum $f(3,0.5) = 0$.

Fist we need to define our function in Python, which we can do really easily using a [Lambda expression](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions):
```python
f = lambda x,y: (1.5-x+x*y)**2 + (2.25-x+x*y**2)**2 + (2.625-x+x*y**3)**2
```
Similarly, we need to define our gradient vector:
$$
    \textbf{g}(x,y) = \Bigg(\dfrac{\partial f(x,y)}{\partial x}, \dfrac{\partial f(x,y)}{\partial y}\Bigg)^{T}
$$
where we have: 
\begin{align}
    \dfrac{\partial f(x,y)}{\partial x} = 2\big[(1.5-x&+xy)(y-1) + (2.25-x+xy^{2})(y^{2}-1) \\\\
    &+ (2.625-x+xy^{3})(y^{3}-1)\big]
\end{align}

\begin{align}
    \dfrac{\partial f(x,y)}{\partial y} = 2\big[(1.5&-x+xy)x + (2.25-x+xy^{2})(2xy) \\\\
    &+ (2.625-x+xy^{3})(3xy^{2})\big]
\end{align}
Again, we can do this using a [Lambda expression](https://docs.python.org/3/tutorial/controlflow.html#lambda-expressions) that returns a list:
```python
g = lambda x,y: [2*( (1.5-x+x*y)*(y-1) + (2.25-x+x*y**2)*(y**2-1) + (2.625-x+x*y**3)*(y**3-1)   ),
                 2*( (1.5-x+x*y)*x     + (2.25-x+x*y**2)*(2*x*y)  + (2.625-x+x*y**3)*(3*x*y**2) )]
```
Now we need to import our ```Optimise Class``` (which I have saved in a file called optimiser.py) and call it using the different ```method``` IDs:
```python
import optimiser
# Optimise using the Steepest Descent method
sd = optimiser.Optimise(X=[3.,4.],function=f,gradient=g,err=1.e-9,method=1)
print(sd.minimum, sd.nsteps)
# Optimise using the Conjugate Gradient algorithm
cg = optimiser.Optimise(X=[3.,4.],function=f,gradient=g,err=1.e-9,method=2)
print(cg.minimum, cg.nsteps)
```
Now if we run this we should get something like this:
```console
aneo@computer:~$ python test_function.py
[3.  0.5] 1118
[3.  0.5] 50
```
## Comparing the different algorithms 
{{< video src="/media/deep_learning/videos/linesearch.mp4" type="video/mp4" controls="yes" >}}

{{< list_children >}}