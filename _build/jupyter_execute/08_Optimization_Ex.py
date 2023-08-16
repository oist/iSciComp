#!/usr/bin/env python
# coding: utf-8

# # 8. Optimization: Exercise
# 
# Name: 
# 
# Date: 
# 
# (Please submit this .ipynb file with your name and its PDF copy.)

# ## 1. Try with you own function 
# 1) Define a function of your interest (with two or more inputs) to be minimized or maximized.  
# It can be an explicit mathematical form, or given implictly as a result of simulation.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# This is just an example. Please define your own function.
def fun(x, a=1.):
    """sine in quardatic valley"""
    x = np.array(x)
    y = (x[0]**2+x[1]**2) + a*np.sin(x[0])
    return y


# 2) Visualize the function, e.g., by surface plot or contour plot.

# In[3]:


w = 5
N = 10
x = np.linspace(-w,w,N)
x0, x1 = np.meshgrid(x, x)
#x2 = np.array(x2).transpose(2,1,0)  # (x0,x1) in last dimention
y = fun((x0, x1), 10)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.plot_surface(x0, x1, y, cmap='viridis')
plt.xlabel("x0"); plt.ylabel("x1"); 


# In[ ]:





# In[ ]:





# 3) Mixmize or minimize the function using two or more optimization algorithms, e.g.
# 
# * Gradient ascent/descent
# * Newton-Raphson method
# * Evolutionary algorithm
# * scpy.optimize
# 
# and compare the results with different starting points and parameters. 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# Option) Set equality or inequality constraints and apply an algorithm for constrained optimization.

# In[ ]:





# In[ ]:




