#!/usr/bin/env python
# coding: utf-8

# # 5. Iterative Computation: Exercise
# 
# Name: 

# $$ % Latex macros
# \newcommand{\mat}[1]{\begin{pmatrix} #1 \end{pmatrix}}
# \newcommand{\p}[2]{\frac{\partial #1}{\partial #2}}
# \newcommand{\b}[1]{\boldsymbol{#1}}
# \newcommand{\w}{\boldsymbol{w}}
# \newcommand{\x}{\boldsymbol{x}}
# \newcommand{\y}{\boldsymbol{y}}
# $$

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Newton's method in n dimension
# Newton's method can be generalized for $n$ dimensional vector $x \in \Re^n$ and $n$ dimensional function $f(x)={\bf0} \in \Re^n$ as
# $$ x_{k+1} = x_k - J(x_k)^{-1}f(x_k) $$
# where $J(x)$ is the *Jacobian matrix*
# $$ J(x) = \mat{\p{f_1}{x_1} & \cdots & \p{f_1}{x_n}\\
#     \vdots & & \vdots\\
#     \p{f_n}{x_1} & \cdots & \p{f_n}{x_n}} $$

# 1) Define a function that computes
# $$ f(x) = 
#     \left(\begin{array}{c} a_0 + a_1 x_1^2 + a_2 x_2^2\\
#     b_0 + b_1 x_1 + b_2 x_2\end{array}\right)
# $$
# and its Jacobian.

# In[2]:


def f(x, a, b, deriv=True):
    """y[0] = a[0] + a[1]*x[0]**2 + a[2]*x[1]**2\    y[1] = b[0] + b[1]*x[0] + b[2]*x[1]
    also return the Jacobian if derive==True"""
    y0 = 
    y1 = 
    if deriv:
        J = 
        
        return np.array([y0, y1]), np.array(J)
    else:
        return np.array([y0, y1])


# In[ ]:


a = [-1, 1, 1]
b = [-1, 1, 2]


# In[ ]:


f([1,1],a,b)


# 2) Consider the case of $a = [-1, 1, 1]$ and $b = [-1, 1, 2]$ and visualize parabollic and linear surfaces.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


x = np.linspace(-2, 2, 25)
y = np.linspace(-2, 2, 25)
X, Y = np.meshgrid(x, y)
XY = np.array([X,Y])  # (2,25,25) array
Z = 
ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
ax.plot_surface(X, Y, Z[0])


# 3) Implement Newton's method for vectors.

# In[ ]:


def newton(f, x0, *args, target=1e-6, maxstep=20):
    """Newton's method. 
        f: should also return Jacobian matrix
        x0: initial guess
        *args: parameter for f(x,*args)
        target: accuracy target"""
    n = len(x0)  # dimension
    x = np.zeros((maxstep+1, n))
    y = np.zeros((maxstep, n))
    x[0] = x0
    for i in range(maxstep):
        y[i], J = f(x[i], *args)
        if   < target:
            break  # converged!
        x[i+1] = 
    else:
        print('did not coverge in', maxstep, 'steps.')
    return x[:i+1], y[:i+1]


# 4) Test how it works from different initial guesses.

# In[ ]:


newton(f, [0,1], a, b)


# In[ ]:


newton(f, [1,1], a, b)


# In[ ]:





# In[ ]:





# ## 2. Bifurcation and Chaos
# A value of $x_k$ that stays unchanged after applying a map $f$ to it (i.e. $x_k = f(x_k) = x_{k+1}$) is called a "fixed point" of $f$. 
# 
# Let us consider the logistic map
# $$ x_{k+1} = a x_k(1 - x_k) $$

# 1) Plot $x_{k+1}=ax_k(1-x_k)$ along with $x_{k+1}=x_k$ for $a=0.5, 2, 3.3$.
# 
# What are the fixed points of these maps?

# In[ ]:





# 2) A fixed point is said to be "stable" when nearby values of $x_k$ also converge to the fixed point after applying $f$ many times; it's said to be "unstable" when nearby values of $x_k$ diverge from it. 
# 
# Draw "cobweb plots" on top of each of the previous plots to visualize trajectories. 
# Try several different initial values of $x_k$.
# 
# Are the fixed points you found stable or unstable?
# 
# How is the stability related to the slope (derivative) of $f(x_k)=ax_k(1-x_k)$ at the fixed point?

# 3: optional) A *bifurcation diagram* is a plot of trajectories versus a parameter.  
# draw the bifurcation diagram for parameter $a$ $(1 \le a \le 4)$, like below:  
# ![bifurcation](figures/bifurcation.png)
# 
# Hint:
# * Use the `logistic()` and `iterate()` functions from the previous lecture.
# * For each value of $a$, show the trajectory (i.e., the values that $x_k$ took over some iterations) of the map after an initial transient. 
# * Since $x_k$ is 1D, you can plot the trajectory on the y axis. For example, take 200 points in $1 \le a \le 4$, run 1000 step iterations for each $a$, and plot $x$ after skipping first 100 steps.

# In[ ]:





# In[ ]:





# ## 3. Recursive call and fractal
# 
# Draw the Sherpinski gasket as below.
# 
# ![shelpinski](figures/shelpinski.png)

# In[ ]:





# In[ ]:




