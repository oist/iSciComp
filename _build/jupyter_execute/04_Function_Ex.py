#!/usr/bin/env python
# coding: utf-8

# # 4. Functions and Classes: Exercise
# 
# Name: 

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1. Functions
# Define the following functions and show some sample outputs.  
# 1) Factorial of n: $1 \times 2 \times \cdots \times n$.

# In[2]:


def factorial(n):
    """factorial of n"""


# In[3]:


for n in range(1, 10):
    print(factorial(n))


# In[ ]:





# 2) For a circle of radius r (default r=1), given x coordinate, return possible y coordinates (two, one, or None).

# In[4]:


def circley(x, r=1):
    


# 3) Draw a star-like shape with n vertices, every m-th vertices connected, with default of n=5 and m=2.

# In[ ]:


def star(n=5, m=2):
    


# In[ ]:





# 4) Any function of your interest

# In[ ]:





# ### 2. Classes
# 1) Define the `Vector` class with the following methods and test that they work correctly.  
# * `norm`, `normalize`: as in the previous class (use L^p norm, with default p=2).
# * `scale(s)`: multiply each component by scalar s.
# * `dot(v)`: a dot product with another vector v.

# In[ ]:


class Vector:
    """A class for vector calculation."""
    


# In[ ]:


x = Vector([0, 1, 2])
x.vector


# In[ ]:


x.scale(3)
x.vector


# In[ ]:


y = Vector([1, 2, 3])
x.dot(y)


# In[ ]:





# 2) Save the class Vector as a module `vector.py`.

# 3) Import the module and test how it works.

# In[ ]:


import vector


# In[ ]:


import importlib


# In[ ]:


importlib.reload(vector) # This is needed after updating a module


# In[ ]:


x = vector.Vector([0, 1, 2])
x.vector


# In[ ]:





# In[ ]:





# In[ ]:




