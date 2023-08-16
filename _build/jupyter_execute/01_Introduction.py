#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction to Python

# Python is a programming language developed in 1990s by Guido van Rossum.  
# Its major features are:
# - consice -- (relatively) easy to read
# - extensible -- (so-called) object oriented
# - free! -- unlike Matlab
# 
# It was originally used for "scripting" sequences of processing.  
# Now it is widely used for scientific computing as well.

# ## Installing Python

# Most Linux and Mac machines usually have Python pre-installed.  
# To install and setup a variety of packages, it is the best to install a curated distribution, such as:  
# * Anaconda: http://anaconda.com

# ## Starting Python

# From a terminal, type
# ```
# $ python  
# ```
# to start a python interpreter.  

# ## Python as a calculator

# At the python prompt `>>>`, try typing numbers and operators, like
# ```
# >>> 1+1
# ```

# In[1]:


1+1


# In[2]:


2**8


# In[3]:


# Uncomment (remove #) the line below
# exp(2)


# The plain Python does not include math functions. You need to import numpy.

# In[ ]:





# ## Jupyter Notebook
# For building a program step-by-step with notes and results attached, it is highly recommended to use a notebook interface, such as Jupyter Notebook (https://jupyter.org), which is included in Anaconda and other popular distributions.
# 
# To start Jupyter Notebook type in the terminal
# ```
# $ jupyter notebook
# ```
# which should open a web page showing your working directory.
# 
# You can create a new notebook from the New menu on the upper right corner, or open an existing .ipynb file like this.
# 

# ### Working with the notebook
# A notebook is made of "cells."  
# You can make a new cell by "+" button on the Toolbar, or by typing ESC A (above) or ESC B (below).  
# You can make a cell as Markdown (documentation) by ESC M, as Code by ESC Y, or simply by the Toolbar menu.  
# You can delete a cell by ESC DD, or the Cut button on the Toolbar.  

# In[ ]:





# ### Markdown cell
# 
# Markdown is a simple text formatting tool, with
# 
# `#, ##, ###,...` for headings
# 
# `*, +, -,...` for bullets
# * item 1
# * item 2
# 
# `$  $` for Latex equations like $\sum_{i=1}^n \alpha_i$ in line
# 
# `$$  $$` for equations centered in a separate line
# $$\sum_{i=1}^n \alpha_i$$
# 
# and two spaces at the end of the line  
# for a line break.
# 
# See https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet for details.
# 
# You can format a Markdown cell by Shift+Return, and go back to Edit mode by Return

# ### Code cell
# You can type Control+Return to run the cell or Shift+Return to run and move to the next cell.   
# You can also use the triangle button or "Cell" menu to run cells.

# In[4]:


2*3


# ## Integer and floating-point numbers
# A number can be an *integer* or *floating-point*, which sometimes needs distinction.

# In[5]:


type(1)


# In[6]:


type(1.5)


# In Python 3, division of integers can produce a float.  
# In Python 2, it was truncated to an integer.

# In[7]:


3 / 2  # 1.5 by Python 3; 1 by Python 2


# You can perform integer division by `//` and get the remainder by `%`.

# In[8]:


5 // 2


# In[9]:


5 % 2


# To make an integer as a floating point number, you can add `.`

# In[10]:


type(1.)


# In[ ]:





# ## Variables
# You can assing a number or result of computation to a variable.  

# In[11]:


a = 1


# In[12]:


a


# In[13]:


b = a + a
b


# Multiple variables can be assigned at once.

# In[14]:


a, b = 1, 2
print(a, b)


# In[ ]:





# ## Print function
# You can check the content of a variable or an expression by typing it at the bottom of a cell.
# 
# You can use `print` function to check the variables anywhere in a cell, multiple items at onece.

# In[15]:


a = 2
print('a =', a)
b = 3
print('a =', a, '; b =', b)


# You can use `.format()` for elaborate formatting with `:d` for integers, `:f` for floating point numbers, and `:s` for text strings.

# In[16]:


c = a/b
print('{0:2d} devided by {1:3d} is about {2:.4f}.'.format(a, b, c))


# In[ ]:





# ## Lists

# You can create a list by surrounding items by [ ].

# In[17]:


b = [1, 2, 3, 4]
b


# An item can be referenced by [ ], with index starting from 0. 

# In[18]:


b[1]  # 2nd item


# In[19]:


b[-1]  # last item


# A colon can be used for indexing a part of list.

# In[20]:


b[1:3]  # 2nd to 3rd


# In[21]:


b[:3]  # first to third


# In[22]:


b[1:]  # 2nd to last


# In[23]:


b[1::2]  # from 1st, step by 2


# In[24]:


b[::-1]  # all in reverse order


# In[ ]:





# For lists, + means concatenation

# In[25]:


b + b


# You can create a nested list, like a matrix

# In[26]:


A = [[1,2,3],[4,5,6]]
A


# An item in a nested list can be picked by [ ][ ], but not [ , ]

# In[27]:


A[1]


# In[28]:


A[1][2]


# In[29]:


A[1,2]  # this causes an error for a list


# A list can contain different types of itmes with different lengths.

# In[ ]:


a = [1, 2, 3.14, 'apple', "orange", [1, 2]]
a


# When you assign a list to another list, only the pointer is copied.

# In[ ]:


a = [1, 2, 3]
b = a
b[1] = 4
a


# When you want to copy the content, use [:]

# In[ ]:


a = [1, 2, 3]
b = a[:]
b[1] = 4
a


# In[ ]:





# ## Dictionary
# When you store data as a list, you have to remember what you stored 1st, 2nd, ...
# 
# A dictionary allows you to access the value by name with `key:value` pairs.

# In[ ]:


# Postal codes in our neighborhood
postal = {'onna':9040411, 'tancha':9040412, 'fuchaku':9040413, 'oist':9040495}


# You can check the value for a key by `[ ]`.

# In[ ]:


postal['oist']


# In[ ]:





# ## `if` branch
# Branching by `if` statement looks like this. In Python, indentation specifes where a block of code starts and ends.

# In[ ]:


x = 1


# In[ ]:


if x>0:
    y = x
else:
    y = 0
y


# There is a shorthand notaion with the condition in the middle:

# In[ ]:


x if x>0 else 0


# In[ ]:





# ## `for` loop
# A common way of `for` loop is by `range()` function.  
# Don't forget a collon and indentation.

# In[ ]:


j = 0
for i in range(5):
    j = j + i
    print(i, j)


# You can specify start, end and interval.

# In[ ]:


for i in range(3,9,2):
    print(i)


# `for` loop can also be over a list.

# In[ ]:


a = [1, 2, 3]
for x in a:
    print(x**2)


# In[ ]:


s = "hello"
for c in s:  # characters in a string
    print(c)


# In[ ]:


s = ["hello", "goodby"]
for c in s:  # strings in a list
    print(c)


# `enumerate()` function gives pairs of index and content of a list.

# In[ ]:


for i, c in enumerate(s):
    print(i, c)


# You can also apply a for loop for a dictionary.

# In[ ]:


for k in postal: # get the key
    print('%8s:'%k, postal[k])


# In[ ]:


# get key-value pair
for (k, v) in postal.items():
    print('{0:8s}: {1}'.format(k, v))


# In[ ]:





# ## List 'comprehension'
# There is a quick way of constructing a list from a for loop.

# In[ ]:


y = [x**2 for x in range(5)]
y


# In[ ]:





# ## Numpy arrays
# For most computation, you need to import `numpy` package by the following convention:

# In[ ]:


import numpy as np


# Numpy `ndarray` is specialized for storing numbers of the same type.

# In[ ]:


b = np.array([1, 2, 3])
b


# In[ ]:


type(b)


# In[ ]:


type(b[0])


# Like a list, the index starts from zero

# In[ ]:


b[1]


# Operators work component-wise.

# In[ ]:


b + b


# In[ ]:


b * b


# In[ ]:


b + 1  # broadcast


# `arange()` gives an evenly spaced array.

# In[ ]:


np.arange(10)


# In[ ]:


np.arange(0, 10, 0.5)


# `linspace()` gives an array *including* the last point.

# In[ ]:


np.linspace(0, 10)


# In[ ]:


np.linspace(0, 10, num=11)


# ## Nested array
# You can make a matrix as a nested array.

# In[ ]:


A = np.array([[1,2],[3,4]])
A


# Components can be accessed by [ , ]

# In[ ]:


A[1][1]


# In[ ]:


A[1,0]  # this if fine for a numpy ndarray, not for a regular list


# Take the first row

# In[ ]:


A[0]


# In[ ]:


A[0,:]


# Take the second column

# In[ ]:


A[:,1]


# Component-wise arithmetics

# In[ ]:


A + A


# In[ ]:


A * A


# A matrix product is inner products of rows and columns, such as
# $$\newcommand{\mat}[1]{\begin{pmatrix} #1 \end{pmatrix}}$$
# $$ \mat{ a & b\\ c & d}\mat{ v & x\\ w & y} = \mat{ av+bw & ax+by\\ cv+dw & cx+dy}. $$
# From Python 3.5, `@` symbol does the matrix product.

# In[ ]:


# matrix product
A @ A  # it should give [[1*1+2*3, 1*2+2*4], [3*1+4*3, 3*2+4*4]]


# In[ ]:





# ### Common matrices

# In[ ]:


np.zeros([2,3])


# In[ ]:


np.eye(4)


# In[ ]:


np.empty([3,2])  # the contents are not specified


# In[ ]:


np.empty([3,2], dtype=int)   # to specify the data type


# In[ ]:





# ## Magic functions
# In Jupyter notebook (or ipython), many *magic* functions preceded by `%` are available for working with the file system, etc.

# In[ ]:


# present working directory
get_ipython().run_line_magic('pwd', '')


# You can use `%quickref` to see the list of magic functions

# In[ ]:


get_ipython().run_line_magic('quickref', '')


# Or `%magic` for the full documentation.

# In[ ]:


get_ipython().run_line_magic('magic', '')


# You can also use `!` to run an OS command or a program.

# In[ ]:


get_ipython().system('pwd')


# In[ ]:


get_ipython().system('hostname')


# In[ ]:





# ## Saving and loading data
# You can work with files by `open()`, `write()` and `read()` functions.

# In[ ]:


with open('haisai.txt', 'w') as f:
    f.write('Haisai!\n')
    f.write('Mensore!\n')
# f is closed when the `with` block is finished


# In[ ]:


get_ipython().run_line_magic('cat', 'haisai.txt')


# In[ ]:


with open('haisai.txt', 'r') as f:
    s = f.read()
print(s)


# In[ ]:





# A common way of writing/reading a data file is to use `savetxt()` and `loadtxt()` functions of `numpy`.

# In[ ]:


X = [ [i, i**2] for i in range(5)]
X


# In[ ]:


np.savetxt("square.txt", X)  # by default, delimited by a space


# In[ ]:


get_ipython().run_line_magic('cat', 'square.txt')


# Another common format is *CSV*, comma-separated value.

# In[ ]:


np.savetxt("square.csv", X, delimiter=",", fmt="%1d, %.5f")
get_ipython().run_line_magic('ls', 's*')


# In[ ]:


get_ipython().run_line_magic('cat', 'square.csv')


# In[ ]:


Y = np.loadtxt("square.txt")
Y


# In[ ]:


Y = np.loadtxt("square.csv", delimiter=",")
Y


# In[ ]:





# ## Getting help
# Python offers several ways of getting help.

# In[ ]:


help()


# In[ ]:


help(np.savetxt)


# In Jupyter notebook, you can use `?` for quick help.

# In[ ]:


get_ipython().run_line_magic('psearch', 'np.*txt')


# In[ ]:


get_ipython().run_line_magic('pinfo', 'np.loadtxt')


# In[ ]:


get_ipython().run_line_magic('pinfo2', 'np.loadtxt')


# You can use 'Help' menu to jump to a variety of documentations.

# In[ ]:




