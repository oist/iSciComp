#!/usr/bin/env python
# coding: utf-8

# # 3. Vectors and Matrices: Exercise
# 
# Name: Kenji Doya
# 
# Date: Oct. 7, 2022

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


# ## 1) Determinant and eigenvalues
# 1) For a 2x2 matrix
# $A = \left(\begin{array}{cc} a & b\\ c & d \end{array}\right)$,
# let us verify that $\det A = ad - bc$ in the case graphically shown below ($a, b, c, d$ are positive).

# In[2]:


A = np.array([[4, 1], [2, 3]])
plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0])
plt.plot([0, A[0,0]+A[0,1], A[0,0]+A[0,1], 0, 0], 
         [0, 0, A[1,0]+A[1,1], A[1,0]+A[1,1], 0])
plt.plot([A[0,0], A[0,0]+A[0,1], A[0,0]+A[0,1], A[0,0], A[0,0]], 
         [0, 0, A[1,0], A[1,0], 0])
plt.plot([0, A[0,1], A[0,1], 0, 0], 
         [A[1,1], A[1,1], A[1,0]+A[1,1], A[1,0]+A[1,1], A[1,1]])
plt.plot([0, A[0,0], A[0,0]+A[0,1], A[0,1], 0], 
         [0, A[1,0], A[1,0]+A[1,1], A[1,1], 0])
plt.axis('equal')
plt.text(A[0,0], A[1,0], '(a,c)')
plt.text(A[0,1], A[1,1], '(b,d)')
plt.text(A[0,0]+A[0,1], A[1,0]+A[1,1], '(a+b,c+d)');


# A unit square is transformed into a parallelogram. Its area $S$ can be derived as follows:  
# Large rectangle: $ S_1 = (a+b)(c+d) = ac+ad+bc+bd $  
# Small rectangle: $ S_2 = bc $  
# Bottom/top triangle: $ S_3 = ac/2 $  
# Left/right triangle: $ S_4 = bd/2 $  
# Parallelogram: 
# $$ S = S_1 - 2S_2 - 2S_3 - 2S_4 = ad - bc$$ 

# 2) The determinant equals the product of all eigenvalues. Verify this numerically for multiple cases and explain intuitively why that should hold.

# In[3]:


#A = np.array([[1, 2], [3, 4]])
m = 4
A = np.random.randn(m,m)
print(A)
lam, V = np.linalg.eig(A)
print('eigenvalues = ', lam)
print('product = ', np.product(lam))
det = np.linalg.det(A)
print('detrminant = ', det)


# The determinant represents how much the volume in the original space is expanded or shrunk.
# 
# The eigenvalues represent how much a segment in the direction of eigen vector is scaled in length.
# 
# Therefore, the producs of all eigenvalues should equal to the determinant.

# ## 2) Eigenvalues and matrix product
# 1) Make a random (or hand-designed) $m\times m$ matrix $A$. Compute its eigenvalues and eigenvectors. From a random (or your preferred) initial point $\b{x}$, compute $A\b{x}, A^2\b{x}, A^3\b{x},...$ and visualize the points. Then characterize the behavior of the points with respect the eigenvalues and eigenvectors.

# In[4]:


m = 4
A = np.random.randn(m,m)
print('A = ', A)
L, V = np.linalg.eig(A)
print('eigenvalues = ', L)
#print('eigenvectors =\n', V)


# In[5]:


# take a point and see how it moves
K = 20  # steps
x = np.zeros((m, K))
x[:,0] = np.random.randn(m) # random initial state
for k in range(K-1):
    x[:,k+1] = A @ x[:,k]  # x_{k+1} = A x_k
# plot the trajectory
plt.plot( x.T, 'o-')
plt.xlabel("k"); plt.ylabel("$x_i$");


# In[6]:


plt.plot( x[0,:], x[1,:])


# 2) Do the above with several different matrices

# In[7]:


A = np.random.randn(m,m)
print('A = ', A)
L, V = np.linalg.eig(A)
print('eigenvalues = ', L)
for k in range(K-1):
    x[:,k+1] = A @ x[:,k]  # x_{k+1} = A x_k
# plot the trajectory
plt.plot( x.T, 'o-')
plt.xlabel("k"); plt.ylabel("$x_i$");


# In[ ]:





# ## 3) Principal component analysis
# Read in the "digits" dataset, originally from `sklearn`.

# In[8]:


data = np.loadtxt("data/digits_data.txt")
target = np.loadtxt("data/digits_target.txt", dtype='int64')
m, n = data.shape
print(m, n)


# The first ten samples look like these:

# In[9]:


plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(1,10,i+1)
    plt.imshow(data[i].reshape((8,8)))
    plt.title(target[i])
    plt.axis('off')


# 1) Compute the principal component vectors from all the digits and plot the eigenvalues from the largest to smallest.

# In[10]:


# subtract the mean
Xm = np.mean(data, axis=0)
X = data - Xm
#C = np.cov(X, rowvar=False)
C = (X.T @ X)/(m-1)
lam, V = np.linalg.eig(C)
# columns of V are eigenvectors
# it is not guaranteed that the eigenvalues are sorted, so sort them
ind = np.argsort(-lam)  # indices for sorting, descending order
L = lam[ind]
V = V[:,ind]
print('L, V = ', L, V)
plt.plot(L);


# In[11]:


# use SVD
U, S, Vt = np.linalg.svd(X, full_matrices=False)
# columns of V, or rows of Vt are eigenvectors
L = S**2/(m-1)  # eigenvalues
print('L, Vt = ', L, Vt)
plt.plot(L);


# In[ ]:





# 2) Visualize the principal vectors as images.

# In[12]:


plt.figure(figsize=(8,8))
for i in range(n):
    plt.subplot(8,8,i+1)
    plt.imshow(V[:,i].reshape((8,8)))
    #plt.imshow(Vt[i].reshape((8,8)))
    plt.axis('off')


# 3) Scatterplot the digits in the first two or three principal component space, with different colors/markers for digits.

# In[13]:


# columns of V are eigenvectors
Z = X @ V
plt.scatter(Z[:,0], Z[:,1], c=target, marker='.')
plt.setp(plt.gca(), xlabel='PC1', ylabel='PC2')
plt.axis('square');


# In[14]:


plt.figure(figsize=(8,8))
plt.scatter(Z[:,0], Z[:,1], c=target, marker='.')
# add labels to some points
for i in range(100):
    plt.text(Z[i,0], Z[i,1], str(target[i]))
plt.setp(plt.gca(), xlabel='PC1', ylabel='PC2');


# In[15]:


# In 3D
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='3d')
ax.scatter(Z[:,0], Z[:,1], Z[:,2], c=target, marker='o')
# add labels to some points
for i in range(200):
    ax.text(Z[i,0], Z[i,1], Z[i,2], str(target[i]))
plt.setp(plt.gca(), xlabel='PC1', ylabel='PC2', zlabel='PC3');


# 4) Take a sample digit, decompose it into principal components, and reconstruct the digit from the first $m$ components. See how the quality of reproduction depends on $m$.

# In[16]:


K = 8  # PCs to be considered
i = np.random.randint(m) # pick a random sample
plt.figure(figsize=(10,4))
plt.subplot(1,K,1)
plt.imshow(data[i].reshape((8,8))) # original
plt.title(target[i])
plt.axis('off')
for k in range(1,K):  # number of PCs
    Xrec = Xm + V[:,:k] @ Z[i,:k] 
    plt.subplot(1,K,k+1)
    plt.imshow(Xrec.reshape((8,8))) # reconstructed
    plt.title(k)
    plt.axis('off')


# In[ ]:




