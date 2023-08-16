#!/usr/bin/env python
# coding: utf-8

# # 4. Functions and Classes

# Let us learn how to define your own *functions*, and further organize them into a *class* for neatness and extensibility.
# 
# References: Python Tutorial (https://docs.python.org/3/tutorial/)
# * Section 4.7-4.8: Functions
# * Chapter 6: Modules
# * Chapter 9: Classes

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Defining functions
# If you find yourself running the same codes again and again with different inputs, it is time to define them as a *function*.
# 
# Here is a simple example:

# In[2]:


def square(x):
    """Compute x*x"""
    # result returned
    return x*x


# In[3]:


square(3)


# In[4]:


a = np.array([1, 2, 3])
# input `x` can be anything for which `x*x` is valid
square(a)


# The line encosed by """ """ is called a *Docstring*, which is shown by `help( )` command.

# In[5]:


help(square)


# In[6]:


get_ipython().run_line_magic('pinfo', 'square')


# A function does not need to return anything.

# In[7]:


def print_square(x):
    """Print x*x"""
    print(x*x)
# the end of indentation is the end of definition
print_square(a)    


# A function can return multiple values.

# In[8]:


def square_cube(x):
    """Compute x**2 and x**3"""
    # return multiple values separated by comma
    return x**2, x**3
# results can be assigned to variables separated by comma
b, c = square_cube(a)
print(b, c)


# In[9]:


square_cube(3)


# In[ ]:





# ### Arguments and local variables
# A function can take single, multiple, or no arguments (inputs).  
# An argumet can be required, or optional with a default value.  
# An argument can be specified by the position, or a keyword.

# In[10]:


def norm(x, p=2):
    """Give the L^p norm of a vector."""
    y = abs(x) ** p
    return np.sum(y) ** (1/p)


# In[11]:


a = np.array([1, 2, -2])
norm(a)  # default p=2


# In[12]:


norm(a, 1)  # specify by position


# In[13]:


norm(p=1, x=a)  # specify by the keywords, in any oder


# In[ ]:





# ### Local and global variables
# Arguments and variables assigned in a function are registered in a local *namespace*.

# In[14]:


y = 0  # global variable
norm(a)  # this uses `y` as local variable, y=[1, 4, 9]
print(y)  # the global variable `y` is not affected


# Any *global* variables can be referenced within a function.

# In[15]:


a = 1  # global variable
def add_a(x):
    """Add x and a."""
    return a + x
print(add_a(1))  # 1 + 1
a = 2
print(add_a(1))  # 1 + 2


# To modify a global variable from inside a function, it have to be declaired as `global`.

# In[16]:


a = 1
def addto_a(x):
    """Add x into a."""
    global a
    a = a + x  # add x to a
addto_a(1)  # a = a + 1
print(a)
addto_a(1)  # a = a + 1
print(a)


# You can modify an argument in a function.

# In[17]:


def double(x):
    """Double x"""
    x = 2 * x
    return x
double(1)


# In[ ]:





# ## Scripts, modules, and packages
# Before Jupyter (iPython) notebook was created, to reuse any code, you had to store it in a text file, with `.py` extension by convention. This is called a *script*.

# In[18]:


get_ipython().run_line_magic('cat', 'haisai.py')


# The standard way of running a script is to type in a terminal:
# ```
# $ python haisai.py
# ```
# In a Jupyter notebook, you can use `%run` magic command.

# In[19]:


get_ipython().run_line_magic('run', 'haisai.py')


# You can edit a python script by any text editor. 
# 
# In Jupyter notebook's `Files` window, you can make a new script as a Text file by `New` menu, or edit an existing script by clicking the file name.

# In[ ]:





# A script with function definitions is called a *module*.  

# In[ ]:


get_ipython().run_line_magic('cat', 'lp.py')


# You can import a module and use its function by `module.function()`.

# In[ ]:


import lp


# In[ ]:


help(lp)


# In[ ]:


a = np.array([-3, 4])
lp.norm(a)


# In[ ]:


lp.normalize(a, 1)


# Caution: Python reads in a module only upon the first `import`, as popular modules like `numpy` are imorted in many modules. If you modify your module, you need to restart your kernel or call `importlib.reload()`.

# In[ ]:


import importlib
importlib.reload(lp)


# A collection of modules are put in a directory as a *package*.  

# In[ ]:


# see how numpy is organized
get_ipython().run_line_magic('ls', '$CONDA_PREFIX/lib/python3.9/site-packages/numpy')


# In[ ]:





# ## Object Oriented Programming
# Object Oriented Programming has been advocated since 1980's in order to avoid naming coflicts and to allow incremental software development by promoting modularity.
# 
# Examples are: SmallTalk, Objective C, C++, Java,... and Python!  
# 
# Major features of OOP is:
# * define data structure and functions together as a *Class*
# * an *instance* of a class is created as an *object*
# * the data (attributes) and functions (methods) are referenced as `instance.attribute` and `instance.method()`.
# * a new class can be created as a *subclass* of existing classes to inherit their attributes and methods.

# ## Defining a basic class
# Definition of a class starts with  
# ```class ClassName(BaseClass):```  
# and include
# * definition of attributes
# * `__init__()` method called when a new instance is created
# * definition of other methods
# 
# The first argument of a method specifies the instance, which is named `self` by convention.

# In[ ]:


class Vector:
    """A class for vector calculation."""
    default_p = 2
    
    def __init__(self, arr):  # make a new instance
        self.vector = np.array(arr)     # array is registered as a vector
    
    def norm(self, p=None):
        """Give the L^p norm of a vector."""
        if p == None:
            p = self.default_p
        y = abs(self.vector) ** p
        return np.sum(y) ** (1/p)
    
    def normalize(self):
        """normalize the vector"""
        u = self.vector/self.norm()
        self.vector = u
    


# A new instance is created by calling the class like a function.

# In[ ]:


x = Vector([0, 1, 2])


# Attributes and methods are referenced by `.`

# In[ ]:


x.vector


# In[ ]:


x.norm()


# In[ ]:


x.norm(3)


# In[ ]:


x.default_p = 1


# In[ ]:


x.norm()


# In[ ]:


x.normalize()
x.vector


# In[ ]:


# another instance
y = Vector([0, 1, 2, 3])


# In[ ]:


y.norm()


# A subclass can inherit attributes and methods of base class.

# In[ ]:


class Vector2(Vector):
    """For more vector calculation."""
    
    def double(self):
        u = 2*self.vector
        self.vector = u
        


# In[ ]:


z = Vector2([1, 2, 3])
z.vector


# In[ ]:


z.double()
z.vector


# In[ ]:





# In[ ]:




