#!/usr/bin/env python
# coding: utf-8

# # 2. Visualization: Exercise
# 
# Name: 
# 
# Date: 
# 
# (Please submit this .ipynb file with your name and its PDF copy.)

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1) Plotting curves
# a) Draw a spiral.

# In[ ]:





# b) Draw a "$\infty$" shape.

# In[ ]:





# c) Draw a "flower-like" shape.

# In[ ]:





# ## 2) Scatter plots
# 
# Let us take the famous *iris* data set.  
# First four columns are:
# * SepalLength, SepalWidth, PetalLength, PetalWidth  
# 
# The last column is the flower type:
# * 1:Setosa, 2:Versicolor, 3:Virginica

# In[2]:


get_ipython().system('head data/iris.txt')


# First, we'll read the data set from a text file.

# In[3]:


X = np.loadtxt('data/iris.txt', delimiter=',')
print(X.shape, X)


# a) Make a scatter plot of the first two columns, with a distinct marker color for each flower type.

# In[ ]:





# b) Create a matrix of pair-wise scatter plots like this:  
# ![pairs](figures/pairs.png)
# You can use `plt.tight_layout()` to adjust the space between subplots.

# In[ ]:





# c) Make a `quiver` plot, representing sepal data by position, petal data by arrows, and flower type by arrow color.

# In[ ]:





# d) Make a 3D scatter plot of the sepal and petal data, with the 4th column represented by marker size.

# In[ ]:





# ## 3) Surface plots
# a) Draw a wavy surface (not just a sine curve extended in the 3rd dimension).

# In[ ]:





# b) Draw the surface of a (half) cylinder.  
# Note that the mesh grid does not need to be square.

# A half cylinder (0 <= theta <= pi), using a square mesh grid:

# In[ ]:





# A full cylinder (0 <= theta < 2pi), using a cylindrical mesh grid:

# In[ ]:





# c) Draw the surface of a sphere.

# In[ ]:





# In[ ]:





# In[ ]:




