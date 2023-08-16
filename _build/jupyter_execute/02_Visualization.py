#!/usr/bin/env python
# coding: utf-8

# # 2. Visualization

# Visualizatin is vital in data analysis and scientific computing, to
# 
# * get intuitive understanding
# * come up with a new hypothesis
# * detect a bug or data anormaly
# 
# That is why we'll cover this topic at the beginning of this course.

# # Matplotlib
# Matplotlib is the standard graphics package for Python. 
# It mimics many graphics functions of MATLAB.  
# The Matplotlib gallery (http://matplotlib.org/stable/gallery) illustrates variety of plots.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# Usually matplotlib opens a window for a new plot.
# 
# A nice feature of Jupyter notebook is that you can embed the figures produced by your program within the notebook by the following *magic* command  
# `%matplotlib inline`  
# (It may be a default setting in recent jupyter notebook).

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Plotting functions
# The standard way is to prepare an array for x values, compute y values, and call `plot( )` function.

# In[3]:


# make an array from 0 to 10, the default is 50 points
x = np.linspace(0, 10)


# In[4]:


# comupute a function for each point
y = x*np.sin(x)
# plot the points
plt.plot(y)  # x is the index of y


# There are multiple ways to pass variables to plot():
# * `plot(y)`: x is assumed as the indices of y 
# * `plot(x, y)`: specify both x and y values
# * `plot(x1, y1, x2, y2,...)`: multiple lines
# * `plot(x, Y)`: lines for columns of matrix Y

# In[5]:


# specify both x and y values
plt.plot(x, y)


# In[6]:


# take another function of x
y2 = x*np.cos(x)


# In[7]:


# plot two lines
plt.plot(x, y, x, y2)


# In[8]:


# phase plot
plt.plot(y, y2);
# you can supress <matplotlib...> output by ;


# In[9]:


# plot multiple lines by a matrix
Y = np.array([y, y2])  # stack data in two rows
plt.plot(x, Y.T); # transpose to give data in two columns


# In[10]:


# plot multiple lines by a matrix
Y = np.array([np.sin(k*x) for k in range(4)])
plt.plot(x, Y.T);


# In[ ]:





# ## Options for plotting
# Line styles can be specified by
# * `color=` (or `c=` )  for color by code, name, RGB or RGBA  
#     - code: 'r', 'g', 'b', 'y', 'c', 'm', 'k', 'w'
# 
# 
# * `marker=` for marker style
#     - code: '.', 'o', '+', '*', '^', ...
# 
# 
# * `linestyle=` (or `ls=` ) for line style
#     - code: '-', '--', ':', '-.', ...
#     
#     
# * `linewidth=` (or `lw=` ) for line width
# * a string of color, marker and line sytel codes, e.g. 'ro:'

# In[11]:


# using code string
plt.plot(x, y, 'm:', x, y2, 'c*');  # magenta dash-dot, cyan circle


# In[12]:


# using keyword=value
plt.plot(x, y, c=[0.2,0.5,0.8,0.5], marker='o', markersize=10, ls='-.', lw=2);


# In[ ]:





# It's a good practice to add axis lables and plot title.

# In[13]:


plt.plot(x, y)
plt.title('oscillation')
plt.xlabel('time ($\mu s$)')
plt.ylabel('amplitude')


# It is also nice to add a legend box.

# In[14]:


ax = plt.plot(x, y, x, y2)
plt.legend(('x sin x','x cos x'))


# You can control axis ranges and scaling.

# In[15]:


plt.plot(x, y)
plt.xlim(-1, 5)
plt.ylim(-4, 4)


# In[16]:


plt.plot(y, y2)
plt.axis('equal')  # equal scaling for x and y


# In[17]:


plt.plot(y, y2)
plt.axis('square')  # in square plot area


# You can create a fiure of your preferred size by `plt.figure()` function

# In[18]:


fig = plt.figure(figsize=(6, 6))
plt.plot(y, y2)


# In[ ]:





# ## Bar plot and histogram

# In[19]:


i = np.arange(10)
j = i**2
plt.bar(i, j)


# `np.random.randn()` gives random numbers from the normal distribution

# In[20]:


z = np.random.randn(100)
plt.hist(z)


# In[ ]:





# ## Subplot and axes
# You can create multiple *axes* in a figure by subplot(rows, columns, index).   
# It uses a MATLAB legacy for index starting from 1.  

# In[21]:


plt.subplot(2, 2, 1)
plt.plot(x, y)
plt.subplot(2, 2, 2)
plt.plot(y, x)
plt.subplot(2, 2, 3)
plt.plot(x, y2)
plt.subplot(2, 2, 4)
plt.plot(y, y2)


# In[ ]:





# ## Figure and axes
# When you make a plot, matplotlib creates a *figure* object with an *axes* object.  
# You can use `gcf()` and `gca()` to identify them and `getp()` and `setp()` to access their parameters.

# In[22]:


plt.plot(x, y)
fig = plt.gcf()  # get current figure
plt.getp(fig)  # show all parameters


# In[23]:


plt.plot(x, y)
fig = plt.gcf()
plt.setp(fig, size_inches=(8,4), facecolor=[0.5,0.5,0.5])


# In[24]:


plt.plot(x, y)
ax = plt.gca()  # get current axes
plt.getp(ax)
plt.setp(ax, facecolor='y')  # change parameters


# In[ ]:





# ## Visualizing data in 2D
# A standard way for visualizing pariwise data is a scatter plot.

# In[25]:


n = 100
x = np.random.uniform(-1, 1, n)  # n points in [-1,1]
y = 2*x + np.random.randn(n)  # scale and add noise
plt.plot(x, y, 'o')


# By `scatterplot( )` you can specify the size and the color of each point to visualize higher dimension information.

# In[26]:


z = x**2 + y**2
c = y - 2*x
# z for size, c for color
plt.scatter(x, y, z, c)
plt.colorbar()


# In[ ]:





# ## Visualizing a matrix or a function in 2D space

# meshgrid() is for preparing x and y values in a grid.

# In[27]:


x = np.linspace(-4, 4, 9)
y = np.linspace(-3, 3, 7)
print(x, y)
X, Y = np.meshgrid(x, y)
print(X, Y)


# We can use imshow() to visualize a matrix as an image.

# In[28]:


Z = X**2 * Y
print(Z)
plt.imshow(Z)


# In[29]:


# some more options
plt.imshow(Z, origin='lower', extent=(-4.5, 4.5, -3.5, 3.5))
plt.colorbar()


# ### color maps
# `imshow( )` maps a scalar Z value to color by a colormap. The standard color map *viridis* is friedly to color blindness and monochrome printing. You can also choose other color maps.

# In[30]:


plt.imshow(Z, cmap='jet')


# ### contour plot

# In[31]:


x = np.linspace(-4, 4, 25)
y = np.linspace(-4, 4, 25)
X, Y = np.meshgrid(x, y)
Z = X**2 + 2*Y**2
plt.contour(X, Y, Z)
plt.axis('square')


# ### vector field by `quiver( )`

# In[32]:


x = np.linspace(-3, 3, 15)
y = np.linspace(-3, 3, 15)
X, Y = np.meshgrid(x, y)
# Van del Pol model
k = 1  # paramter
U = Y  # dx/dt
V = k*(1 - X**2)*Y - X  # dy/dt
plt.quiver(X, Y, U, V)


# In[ ]:





# ## 3D Visualization
# You can create a 3D axis by `projection='3d'` option.

# ### lines and points in 3D

# In[33]:


# spiral data
x = np.linspace(0, 20, 100)
y = x*np.sin(x)
z = x*np.cos(x)


# In[34]:


# create a figure and 3D axes
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot(x, y, z)


# In[35]:


ax = plt.figure().add_subplot(projection='3d')
# scatter plot with x value mapped to color
ax.scatter(x, y, z, c=x)


# In[ ]:





# ### surface plot

# In[36]:


x = np.linspace(-5, 5, 25)
y = np.linspace(-5, 5, 25)
X, Y = np.meshgrid(x, y)
Z = X*Y


# In[37]:


ax = plt.figure().add_subplot(projection='3d')
ax.plot_surface(X, Y, Z)


# You may want to use `notebook` option of the magic command to rotate the figure to pick the best viewpoint.
# ```
# %matplotlib notebook
# ```
# (You may have to run the magic command twice to take effect.)

# In[38]:


get_ipython().run_line_magic('matplotlib', 'notebook')


# You can color the surface by the height.

# In[39]:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# map Z value to 'viridis' colormap
ax.plot_surface(X, Y, Z, cmap='viridis');


# In[ ]:





# ### surface by wire frame

# In[40]:


# make a 3D axis in one line
ax = plt.figure().add_subplot(111, projection='3d')
# wireframe plot
ax.plot_wireframe(X, Y, Z)


# In[ ]:





# ### 3D vector field by `quiver( )`

# In[41]:


## Lorenz equation
x = np.linspace(-10, 10, 9)
y = np.linspace(-20, 20, 9)
z = np.linspace(-20, 20, 5)
X, Y, Z = np.meshgrid(x, y, z)
#print(X)
# parameters
p, r, b = 10, 28, 8/3
U = Y  # dx/dt
V = -X # dy/dt
W = - Z   # dz/dt
ax = plt.figure().add_subplot(111, projection='3d')
ax.quiver(X, Y, Z, U, V, W, length=0.1)


# For more advanced 3D visualization, you may want to use a specialized library like `mayavi` 
# https://docs.enthought.com/mayavi/mayavi/

# In[ ]:





# ## Animation
# It is often helpful to visualize the result while it is computed, rather than after all computation.  
# The simplest way is to repeat computing, drawing, and a short pause.

# It does not work well with embedded figures, so open a figure window.

# In[42]:


# list options of %matplotlib
get_ipython().run_line_magic('matplotlib', '-l')


# In[43]:


get_ipython().run_line_magic('matplotlib', 'tk')


# You may need to restart the kernel to chage the setting.

# In[44]:


import numpy as np
import matplotlib.pyplot as plt


# In[45]:


# set parameter
v = 0.2
# prepare x axis
x = np.linspace(0, 10)
for t in range(50):
    # compute the new result
    y = np.sin(x - v*t)
    # clear previous plot
    plt.cla()
    # draw a new plot
    plt.plot(x, y)
    # puase 0.01 sec
    plt.pause(0.01)


# You can also use `animation` package to store the frames and play them back.

# In[46]:


from matplotlib import animation


# In[47]:


fig = plt.figure()
frames = []  # prepare frame

# traveling wave velocity
v = 0.2  # velocity
# prepare x axis
x = np.linspace(0, 10)
for t in range(50):
    # compute the new result
    y = np.sin(x - v*t)
    # draw a new plot
    artists = plt.plot(x, y)
    # append to the frame
    frames.append(artists)

anim = animation.ArtistAnimation(fig, frames, interval = 10)


# You can save the movie in a gif file.

# In[48]:


anim.save("anim.gif", writer='pillow')


# In[ ]:




