#!/usr/bin/env python
# coding: utf-8

# In[6]:


import math


# In[1]:


emi ={}
emi['E'] = {}
emi['5'] = {}
emi['I'] = {}


emi['E']['A'] = .25
emi['E']['C'] = .25
emi['E']['G'] = .25
emi['E']['T'] = .25

emi['5']['A'] = .05
emi['5']['C'] = 0.0
emi['5']['G'] = .95
emi['5']['T'] = 0.0

emi['I']['A'] = .4
emi['I']['C'] = .1
emi['I']['G'] = .1
emi['I']['T'] = .4

P ={}

P['E'] = [0.9, 0.1]
P['5'] = [0, 1]
P['I'] = [0.9, 0.1]


# In[8]:


ipseq = "CTTCATGTGAAAGCAGACGTAAGTCA"


# In[11]:


def finprob(statepath):
    ipseqlist = list(ipseq)
    statepathlist = list(statepath)
    probab = 1
    for i in range(len(ipseqlist)):
        ip = ipseqlist[i]
        state = statepathlist[i]
#         print(ip,state)
        if(i==0):
            probab*=emi[state][ip] 
        else:
            probab*=emi[state][ip]
            if statepathlist[i-1]==state:
                probab*=P[state][0]
            else:
                probab*=P[state][1]
    probab*=0.1
    return probab


# In[15]:


statepath = "EEEEEEEEEEEEEEEEEE5IIIIIII$"
print(math.log(finprob(statepath)))


statepath = "EEEEEE5IIIIIIIIIIIIIIIIIIII$"
print(math.log(finprob(statepath)))

statepath = "EEEEEEEE5IIIIIIIIIIIIIIIIII$"
print(math.log(finprob(statepath)))

statepath = "EEEEEEEEEEEE5IIIIIIIIIIIIII$"
print(math.log(finprob(statepath)))

statepath = "EEEEEEEEEEEEEEE5IIIIIIIIIII$"
print(math.log(finprob(statepath)))

statepath = "EEEEEEEEEEEEEEEEEEEEEE5IIII$"
print(math.log(finprob(statepath)))


# In[ ]:




