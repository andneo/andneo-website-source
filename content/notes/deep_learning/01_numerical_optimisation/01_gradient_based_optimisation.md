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
summary: A brief overview on the theory underpinning gradient-based optimisation algorithms (steepest descent and conjugate gradients) and their implemention.
# Prev/next pager order (if `docs_section_pager` enabled in `params.toml`)
weight: 4
---

##  Introduction
Optimisation is a ubiquitous process, people do it everyday. A classic example would be: _How do I get somewhere as_ (_**fast, cheap, environmentally friendly, ...**_) _as possible?_  Nature optimises too, a big part of what I did in my PhD was finding out how systems of particles optimise their potential energy. 
Needless to say, optimisation is an important process, and quite a bit of effort has been dedicated to developing algorithms that do it well.
As we will see, optimisation algorithms form a key component of neural networks. 
The first section of these notes introduce some of the theory underpinning these optimisation algorithms, and how to implement them in Python. 

##  Some Background Maths
Before getting stuck into optimisation algorithms, we should first introduce some of the maths needed to understand the algorithms and the notation we'll use.
First, and maybe most important, is the {{< hl >}} objective function {{< /hl >}}.
This function should provide some quantitative description of our process of interest (think of it as a measure of success).

For instance, in our above example of getting to some destination the objective function could be some mathematical expression which involves the time or cost of the journey.
We denote the objective function as $f(\textbf{x})$, where $\textbf{x}$ is an $n$-dimensional vector of parameters.
So in our journey example, $\textbf{x}$ would be a 2d vector where the first parameter ($x_{0}$) would be time and the second parameter ($x_{1}$) would be cost.
We could therefore write $\textbf{x}=(x_{0},x_{1})^{T}$ (where the $^{T}$ means we are dealing with a [column vector](https://en.wikipedia.org/wiki/Row_and_column_vectors)).
Our objective function will then involve some combination of these parameters, for instance this could be:
$$
f(\textbf{x}) = x_{0}^2 + x_{1}^2
$$
I have plotted this function for $x_{0},x_{1}\in[-10,10]$ below, and we can see (from the 3D representation on the left) it has a bowl-like shape (actually we would call this a [paraboloid](https://en.wikipedia.org/wiki/Paraboloid)).
We can also see that when $\textbf{x}=(0,0)^{T}$, $f(x_{0},x_{1})=0$. We would say that this point corresponds to the [minimum](https://en.wikipedia.org/wiki/Maxima_and_minima) of the function $f(\textbf{x})$.
{{< figure library="true" src="/media/deep_learning/images/gradient.svg" title="" >}}

On the right I've shown a 2D [contour plot](https://www.statisticshowto.com/contour-plots/) of the same function, as well as the [*_gradient_*](https://en.wikipedia.org/wiki/Gradient).
The gradient for this function is given by:
\begin{align}
    \textbf{g}(\textbf{x}) \equiv \nabla f(x\_{0},x\_{1}) &= \Big(\dfrac{\partial f}{\partial x\_{0}}, \dfrac{\partial f}{\partial x\_{1}}\Big)^{T} = \Big(2x\_{0}, 2x\_{1}\Big)^{T}
\end{align}
The gradient defines a [vector field](https://en.wikipedia.org/wiki/Vector_field), which points in the [direction of maximum rate of increase](http://mathonline.wikidot.com/the-maximum-rate-of-change-at-a-point-on-a-function-of-sever) for its corresponding function.
This is an important property of the gradient which we are going to exploit.
If none of this is familiar to you I would suggest going through a course like [this one](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/syllabus/).

## Optimisation Problems
We are interested in solving the general optimisation problem:
$$
    \min_{\textbf{x}\in\mathbb{R}^{n}}f(\textbf{x})
$$
In other words, we want to find a set of coordinates which correspond to a [minimum](https://en.wikipedia.org/wiki/Maxima_and_minima) of the function $f(\textbf{x})$.
We are going to do this using an optimisation algorithm.
There are different algorithms to choose from, but they all follow the same general process: iteratively update an initial guess for $\textbf{x}$ until some termination condition is satisified.

The difference between different algorithms comes from how we move from $\textbf{x}\_{i}$, at iteration $i$ of our algorithm, to $\textbf{x}\_{i+1}$.
We want to write some Python code that can optimise arbitrary functions using different optimisation algorithms.
The first step is to define a ``` Class ```:
```python
class Optimise:
    def __init__(self, X, function, gradient, err, method):
    #   Initialise input parameters for the optimisation algorithms
        self.X      = X         # Coordinates of the function.
        self.f      = function  # Function to be optimised.
        self.g      = gradient  # Gradient of the function.
        self.err    = err       # Threshold convergence value
        self.method = method    # ID of the line search method
```
This ``` Class ``` has three key attributes:
1. ``` X ```: The coordinates which we are to optimise, $\textbf{x}$.
2. ``` f ```: The objective function to be optimised, $f(\textbf{x})$.
3. ``` g ```: The gradient of the objective function, $\textbf{g}(\textbf{x})$.

There are two more parameters, ``` err ``` and ``` method ```.
We use the ``` err ``` parameter to define our termination condition, and the ``` method ``` variable to decide which method to use. For now we'll just include the steepest descent algorithm. 
```python
#   Initialise parameters describing the convergence of the optimisation algorithms
    self.nsteps = 0
    self.path   = []
    self.steps  = []
#   Perform local optimisation.
    if(method==1):
        self.steepest_descent()
#   Extract the coordinates of the local minimum and the path taken to it.
    self.minimum = self.path[-1]
    self.path    = np.array(self.path).T
```
We have also created some new variables which we are going to use to monitor the path the algorithms take to the minimum, and their convergence properties.
## Line Search Strategies
We are going to focus on *line search strategies* -- algorithms where we select a direction $\textbf{d}\_{i}$ from $\textbf{x}\_{i}$, and search along that direction to find a new set of coordinates $\textbf{x}\_{i+1}$ where $f(\textbf{x}\_{i+1}) < f(\textbf{x}\_{i})$ (i.e., a descent direction). We write this as:
$$
    \textbf{x}\_{i+1} = \textbf{x}\_{i} + \alpha\_{i}\textbf{d}\_{i}, \quad i=0, 1, 2, ...
$$
Therefore, we have two tasks at each iteration:
1. Determine $\textbf{d}\_{i}$
2. Determine $\alpha\_{i}$ (the step length).

### Gradient Descent
An obvious choice for $\textbf{d}\_{i}$ becomes apparent if we look back at the 2D contour plot of $f(\textbf{x})$ above, specifically focusing on the gradient field. From this it should be easy to see that a descent direction of any function is given by $\textbf{d}\_{i}=-\textbf{g}\_{i}$. 
This is therefore the most obvious choice for a search direction, and forms the basis of our entry level optimisation algorithm:
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

You'll see in the code snippet we call a function ```backtrack()```, this function chooses an appropriate step length $\alpha\_{i}$ which we'll go over in the next [section](./#deciding-the-step-length).

Let's try and think about what moving along $-\textbf{g}\_{i}$ actually represents a little more intuitively (this is mainly an exercise which will help us a little later on when we discuss [quasi-Newton methods](/notes/deep_learning/01_numerical_optimisation/02_quasi_newton_methods)). Assuming we know the value of our function $f$ at some point $\textbf{x}\_{i}$, [Taylor's theorem](https://en.wikipedia.org/wiki/Taylor%27s_theorem) tells us that the best linear approximation to our function at a new point  $\textbf{x}\_{i+1}=\textbf{x}\_{i}+\Delta\textbf{x}\_{i}$ is
$$
    f(\textbf{x}\_{i+1}) = f(\textbf{x}\_{i}) + \Delta\textbf{x}\_{i}^{T}\textbf{g}(\textbf{x}\_{i})
$$
where $\Delta\textbf{x}\_{i}$ is some displacement vector (essentially it is $\alpha\_{i}\textbf{d}\_{i}$). We can immediately see how this equation is going to be useful for our optimisation algorithm. We want to choose $\Delta\textbf{x}\_{i}$ such that $f(\textbf{x}\_{i+1}) < f(\textbf{x}\_{i})$, which means that for $\Delta\textbf{x}\_{i}$ to represent a descent direction we must have
$$
    \Delta\textbf{x}\_{i}^{T}\textbf{g}(\textbf{x}\_{i}) = \|\|\Delta\textbf{x}\_{i}\|\|~\|\|\textbf{g}(\textbf{x}\_{i})\|\|\cos\theta\_{i} < 0
$$
where $\theta\_{i}$ is the angle between $\Delta\textbf{x}\_{i}$ and $\textbf{g}(\textbf{x}\_{i})$, and it should now be clear that the choice for $\Delta\textbf{x}\_{i}$ which will give us the most rapid decrease in this linear approximation to our function is $\Delta\textbf{x}\_{i}=-\textbf{g}(\textbf{x}\_{i})$ since $\cos\theta\_{i}=-1$ in this scenario.
{{< load-plotly >}}
{{< plotly json="/media/deep_learning/plotly/test_file.json" height="600px" >}}
<!-- We can show that this is actually the case. To do this, our first step is to use Taylor's theorem to approximate our function $f(\textbf{x})$ any $\textbf{d}\_{i}$ and $\alpha\_{i}$ we have:
$$
    f(\textbf{x}\_{i}+\alpha\_{i}\textbf{d}\_{i}) = f(\textbf{x}\_{i}) + \alpha\_{i}\textbf{d}\_{i}^{T}\textbf{g}(\textbf{x}\_{i}) + \frac{1}{2}\alpha^{2}\_{i}\textbf{d}\_{i}^{T}\textbf{H}(\textbf{x}\_{i})\textbf{d}\_{i}
$$
> $\textbf{H}(\textbf{x})\equiv\nabla^{2}f(\textbf{x})$ is the [Hessian matrix](https://en.wikipedia.org/wiki/Hessian_matrix) for our function. For example, the Hessian in our journey problem above would be:
> 
> 
> $$
    \textbf{H}(x_{0},x_{1}) =
    \begin{bmatrix} 
    \frac{\partial^{2}f}{\partial x^{2}_{0}} & \frac{\partial^{2}f}{\partial x\_{0}x\_{1}} \\\\
    \frac{\partial^{2}f}{\partial x\_{1}x\_{0}} & \frac{\partial^{2}f}{\partial x^{2}\_{1}}
    \end{bmatrix}
> $$
> 
The first two terms in the expansion give us a first order Taylor approximation to our function, while the above equation is a second order approximation. If $\textbf{d}\_{i}$ does actually represent a descent direction then our first order approximation says $f(\textbf{x}\_{i}+\alpha\_{i}\textbf{d}\_{i}) < f(\textbf{x}\_{i})$. That also means $f(\textbf{x}\_{i}) + \alpha\_{i}\textbf{d}\_{i}^{T}\textbf{g}(\textbf{x}\_{i})<f(\textbf{x}\_{i})$. Using this, we can define a condition for $\textbf{d}\_{i}$ to be a descent direction:
$$
    \alpha\_{i}\textbf{d}\_{i}^{T}\textbf{g}(\textbf{x}\_{i}) < 0
$$
Since $ \textbf{d}\_{i}^{T}\textbf{g}(\textbf{x}\_{i}) = \|\|\textbf{d}\_{i}\|\|~\|\|\textbf{g}(\textbf{x}\_{i})\|\|\cos\theta\_{i} < 0$ (where $\theta\_{i}$ is the angle between $\textbf{d}\_{i}$ and $\textbf{g}\_{i}$) it should be easy to see that the steepest descent direction of a function is given by $\textbf{d}\_{i}=-\textbf{g}\_{i}$ (i.e., when $\cos\theta\_{i}=-1$). -->


One issue with this method is that it becomes very inefficient near the minimum where the magnitude of the gradient is very small. This becomes a lot clearer if try and think about what moving along $-\textbf{g}(\textbf{x}_k)$ actually represents a little more intuitively.


<!-- ### Conjugate Gradient
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
The layout of the two optimisation algorithms are identical, the only difference with the conjugate gradient method is that we need to save the gradient and search direction at each iteration to use in the next one. -->

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