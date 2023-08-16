#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction to Python: Exercise Solutions

# In[1]:


import numpy as np


# ##  Lists and loops
# Create the following lists, possibly in multiple ways.
# 
# 1) odd positive integers below 20.

# In[2]:


N = 20
j = 0
y = []
x = [x + 1 for x in range(N)]
for i in range (N): 
    if x[i]%2!=0: 
        y.append(x[i])

print (y)


# In[3]:


odd = []
for i in range(20):
    if i % 2 == 1:
        odd = odd + [i]
print(odd)


# In[4]:


A = []
for i in range(0,10):
    A = A + [2*i+1]
print(A)


# In[5]:


odd = [i for i in range (1,21,2)]
print(odd)


# In[6]:


print(list(range(1,21,2)))


# In[ ]:





# 2) sums from 1 to n, for up to n=10.

# In[7]:


n = 10 
a = [a+1 for a in range (n)]
for i in range (1,n):
    a[i] = a[i-1] + a[i]
print(a)


# In[8]:


sums = []
sum_all = 0
for i in range(1, 11):
    sum_all += i
    sums.append(sum_all)
print(sums)


# In[9]:


k=[sum(list(range(1,i))) for i in range(2,12)]
k


# In[ ]:





# 3) prime numbers below n=20.

# In[10]:


n = 20
A = []
for i in range(2,n):
    count = 0
    for j in range(2,i):
        if(i%j==0):
            count = 1        
    if(count==0):
        A.append(i)
print(A)


# In[11]:


n = 20
y = []
for i in range(2,n):
    for j in range (2,i):
        if i%j == 0:
            break
    else: # if the for loop doesn't break
        y.append(i)
print(y)


# In[12]:


n = 20
list_d = list()
for i in range(2,n):
    if all(i%j!=0 for j in range(2,i)):
        list_d.append(i)
print(list_d)


# In[13]:


import sympy
list(sympy.primerange(0, 20))


# In[ ]:





# 4) n=10 random numbers between 0 and k=5.  
# (use np.random.randint())

# In[14]:


n = 10
k = 5
r = []
for i in range(n):
    r.append(np.random.randint(k))
r


# In[15]:


n = 10
k = 5
[ np.random.randint(k) for i in range(n)]


# In[16]:


np.random.randint(0, 5, 10)


# In[17]:


np.random.randint(5, size=10)


# In[ ]:





# 5) from two lists of the same length, make a list with the larger of the items at the same position.

# In[18]:


n = 10 
min = 0
max = 5
x = np.random.randint(min, max, n)
y = np.random.randint(min, max, n)
print(x, y)
z = np.zeros(n)
for i in range(n):
    if x[i] > y[i]:
        z[i] = x[i]
    else:
        z[i] = y[i] 
print(z)


# In[19]:


n = 10
x = np.arange(n) # 0 to n-1
y = np.arange(n,0,-1)  # n to 1
z = [ np.max((x[i], y[i])) for i in range(n)]
np.array(z)


# In[20]:


np.max([x,y], axis=0)


# In[ ]:





# 6) from a random list with 7 items, find the median 

# In[21]:


n = 7
a = np.random.randint(0, 10, n)
print(a)
sorted_a = sorted(a)
print(sorted_a)
i = int((len(a))/2)
sorted_a[i]


# In[22]:


a = np.random.randint(0, 10, n)
print(a)
a.sort()
print(a)
a[int((len(a))/2)]


# In[23]:


np.median(a)


# In[ ]:





# ## Arrays and matrices

# 1) an m-by-n matrix with random integers from 0 to k.

# In[24]:


# for example
m = 3
n = 4
k = 10
matrix = np.zeros((m,n))
for i in range(m):
    for j in range(n):
        matrix[i,j] = np.random.randint(k)
print(matrix)


# In[25]:


m, n, k = 3, 4, 10
A = [[np.random.randint(0,k) for j in range(n)] for i in range(m)]
np.array(A)


# In[26]:


m = 3
n = 4
k = 10
np.random.randint(0, k, (m,n))


# In[ ]:





# 2) from a matrix, make a sub matrix of items in odd rows and even columns.

# In[27]:


m = 3
n = 4
k = 10
A = np.random.randint(0, k, (m,n))
print(A)
B = []
for i in range(0, m, 2):
    b = []
    for j in range(1, n, 2):
        b.append(A[i,j])
    B.append(b)
print(np.array(B))


# In[28]:


m, n, k = 3, 4, 10
A = np.random.randint(0, k, (m,n))
print(A)
B = A[::2, 1::2]
print(B)


# In[ ]:





# 3) make a m-by-n matrix with (i,j) component as i/j.

# In[29]:


m = 3
n = 4
A = np.zeros((m,n))
for i in range(m):
    for j in range(n):
        A[i,j] = (i+1)/(j+1)
print(A)


# In[30]:


A = [[(i+1)/(j+1) for j in range(n)] for i in range(m)]
np.array(A)


# In[ ]:





# 4) for a n-by-n matrix $A$, compute the k-th power $A^k$.

# In[31]:


A = np.array([[2,0],[0,3]])
print(A)
k = 3
B = A
if k>1:
    for i in range(k-1):
        B = B @ A
print(B)


# In[32]:


n = 3
A = np.diag(np.random.randn(n))
print(A)
k = 2
B = np.eye(len(A))
for i in range(k):
    B = B @ A
print(B)


# In[ ]:




