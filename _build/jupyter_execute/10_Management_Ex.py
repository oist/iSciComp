#!/usr/bin/env python
# coding: utf-8

# # 10. Software Management: Exercise
# 
# Name: 

# ## GitHub
# 
# Let us try collaborative software development by OIST GitHub
# https://github.com/oist
# 
# For account setup, see https://groups.oist.jp/it/github-oist
# 
# * Create your GitHub account if you haven't
# 
# * Submit the request form: https://oist.service-now.com/sp?id=sc_cat_item&sys_id=2899d8dcdbad2f40d7c7e5951b961985&sysparm_category=332e888edb47df004a187b088c9619be
# 
# Our course repository is https://github.com/oist/ComputationalMethods2022
# 

# ### SSH key
# For using a private repository, you need to setup a SSH key for secure connection.
# 
# If you have not set up a SSH key on your computer, make a new one by
# ```
# mkdir ~/.ssh
# cd ~/.ssh
# ssh-keygen -t rsa
# ```
# You shuld set a passphrase, which you will be asked when using the key.
# 
# On GitHub, from your account menu on the top right corner, select *Settings* and then *SSH and CPG keys* on the left side bar.
# 
# Press *New SSH Key*, paste the content of `.ssh/rsa.pub`, and press *Add SSH*.
# 
# You may also need to press *Configure SSO* to link it with your OIST login.
# 
# If SSH doesn't work, an alternative way is to use *https* after generating a *personal access token* from *Settings/Developer settings*.

# ### Cloning a repository
# 
# If you just use a copy of a stable software, and not going to contribute your changes, just downloading a zip file is fine.
# 
# But if you would congribute to joint development, or catch up with updates, `git clone` is the better way.

# ### Cloning ComputationalMethods repository
# 
# To download a copy of the repository, run
# 
# ```git clone git@github.com:oist/ComputationalMethods2022.git```
# 
# You are asked to input the passphrase you set in creating your SSH Key.
# 
# This should create a folder `ComputationalMethods2022`.

# In[1]:


get_ipython().run_line_magic('pwd', '')


# In[2]:


get_ipython().system('git clone git@github.com:oist/ComputationalMethods2022.git')


# In[ ]:


get_ipython().run_line_magic('ls', '')


# Move into the folder and test `odesim.py` program.

# In[ ]:


get_ipython().run_line_magic('cd', 'ComputationalMethods2022')


# In[ ]:


get_ipython().run_line_magic('ls', '')


# From the console you can run interactively after reading the module as:
# 
# `python -i odesim.py`
# 
# `sim = odesim('first')`
# 
# `sim.run()`

# In[ ]:


from odesim import *


# In[ ]:


sim = odesim('first')


# In[ ]:


sim.run()


# In[ ]:





# ### Your branch
# 
# Now make your own branch, check it out, and add your own ODE module.

# In[ ]:


get_ipython().system('git branch myname')
get_ipython().system('git checkout myname')


# Make a copy of a dynamics file `first.py` or `second.py`, implement your own ODE, and save with a new name, e.g. `vdp.py`.
# 
# Run odesim and confirm that your ODE runs appropriately.
# 
# Then you can add and commit your change.

# In[ ]:


get_ipython().system('git status')


# In[ ]:


get_ipython().system('git add vdp.py')


# In[ ]:


get_ipython().system('git commit -m "adding my model vdp.py"')


# In[ ]:


get_ipython().system('git log --graph --oneline --all')


# Now push your branch to GitHub repository by, e.g.
# 
# `git push origin myname` 

# In[ ]:


get_ipython().system('git push origin myname')


# Check the status on GitHub:
# https://github.com/oist/ComputationalMethods2022
# 
# and make a pull request for the repository administrator to check your updates.
# 
# The administrator may reply back with a comment for revision or merge your change to the main branch.

# In[ ]:





# ### Pulling updates
# While you are working on your local code, the codes on the origial repository may be updated. You may also want to check the branches other people have created.
# 
# You can use `git pull` to reflect the changes in the GitHub to your local repository.
# 
# You can use `git branch` to see what branches are there and `git checkout` to try with the codes in other branches.

# In[ ]:


get_ipython().system('git pull')


# In[ ]:


get_ipython().system('git branch')


# In[ ]:





# Optional) In addition to adding a new module, you are welcome to improve the main program `odesim.py` itself. For example,
# 
# * add other visualization like a phese plot.
# 
# * fix any bugs or improve error handling.
# 
# * add documentation.
# 
# * ...

# In[ ]:




