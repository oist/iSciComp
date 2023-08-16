#!/usr/bin/env python
# coding: utf-8

# # 2. Visualization: Exercise Solutions

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1) Plotting curves
# a) Draw a spiral.

# In[2]:


t = np.arange(0, 50, 0.1)
x = t*np.cos(t)
y = t*np.sin(t)
plt.plot(x, y)
plt.axis('square')


# b) Draw a "$\infty$" shape.

# In[3]:


t = np.arange(0,10,0.1)
x = np.sin(t)
y = np.sin(2*t)
plt.plot(x,y)


# c) Draw a "flower-like" shape.

# In[4]:


t = np.linspace(0, 2*np.pi, 200)
x = np.cos(t) + np.sin(4*t)
y = np.sin(t) + np.cos(4*t)
plt.plot(x, y)
plt.axis('equal')


# ## 2) Scatter plots
# 
# Let us take the famous *iris* data set.  
# First four columns are:
# * SepalLength, SepalWidth, PetalLength, PetalWidth  
# 
# The last column is the flower type:
# * 1:Setosa, 2:Versicolor, 3:Virginica

# In[5]:


get_ipython().system('head data/iris.txt')


# First, we'll read the data set from a text file.

# In[6]:


X = np.loadtxt('data/iris.txt', delimiter=',')
print(X.shape, X)


# a) Make a scatter plot of the first two columns, with a distinct marker color for each flower type.

# In[7]:


plt.scatter(X[:,0], X[:,1], c=X[:,-1])


# b) Create a matrix of pair-wise scatter plots like this:  
# ![pairs](figures/pairs.png)

# In[8]:


plt.figure(figsize=(6, 6))  # a bit larger area
d = 4  # data dimension
for i in range(1, d):  # rows: X1 to Xd
    for j in range(d-1):  # columns: X0 to Xd-1
        if j < i:
            plt.subplot(d-1, d-1, (i-1)*(d-1) + j+1)
            plt.scatter(X[:,i], X[:,j], c=X[:,-1])
            plt.ylabel('X{0}'.format(i))
            plt.xlabel('X{0}'.format(j))
plt.tight_layout()  # adjust the space between subplots


# c) Make a `quiver` plot, representing sepal data by position, petal data by arrows, and flower type by arrow color.

# In[9]:


plt.quiver(X[:,0], X[:,1], X[:,2], X[:,3], X[:,-1])


# In[10]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# d) Make a 3D scatter plot of the sepal and petal data, with the 4th column represented by marker size.

# In[11]:


fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X[:,0], X[:,1], X[:,2], s=20*X[:,3], c=X[:,-1])


# ## 3) Surface plots
# a) Draw a wavy surface (not just a sine curve extended in the 3rd dimension).

# In[12]:


x = np.linspace(-10, 10, 50)
y = np.linspace(-10, 10, 50)
X, Y = np.meshgrid(x, y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)/R
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')


# b) Draw the surface of a (half) cylinder.  
# Note that the mesh grid does not need to be square.

# A half cylinder (0 <= theta <= pi), using a square mesh grid:

# In[13]:


r = 1
x = np.linspace(-r, r, 50)
y = np.linspace(-r, r, 50)
X, Y = np.meshgrid(x, y)
Z = np.sqrt(r**2 - Y**2)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)


# A full cylinder (0 <= theta < 2pi), using a cylindrical mesh grid:

# In[14]:


r = 1
x = np.linspace(-r, r, 50)
th = np.linspace(0, 2*np.pi, 50)
X, Th = np.meshgrid(x, th)
Y = r*np.cos(Th)
Z = r*np.sin(Th)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)


# c) Draw the surface of a sphere.

# In[15]:


r = 1
th = np.linspace(-np.pi/2, np.pi/2, 50)  # latitude
ph = np.linspace(-np.pi, np.pi, 50)  # longitude
Th, Ph = np.meshgrid(th, ph)
X = r*np.cos(Th)*np.cos(Ph)
Y = r*np.cos(Th)*np.sin(Ph)
Z = r*np.sin(Th)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)
ax.set_box_aspect((1,1,1))


# In[ ]:





# In[ ]:




