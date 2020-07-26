#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# This is the library for the Reservoir Computing got it by: https://github.com/cknd/pyESN
from pyESN import ESN 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('D:\\DATASET\\2013\\VYSAPURAM-Wind_data2013.csv')
data=data['80mWS']

data=np.asarray(data)

dataa=data
#data=np.insert(data,0,0)
#data=data[:2000]
print((data))


# In[3]:


x1=[1,2,3]
x1=np.asarray(x1)
x2=[3,4,5]
x3=[]
x3=np.asarray(x3)
x2=np.asarray(x2)
x3=np.concatenate((x3,x2),axis=0)
print(x2)


# In[4]:


def anom(d):
    
    d=d[12958:17134]
    #print(len(d))
    days=(math.ceil(len(d)/144))
    daily=[]
    #daily.append([1,2,3])
    #daily.append([2,4])
    #print(daily)
    c=0
    for i in range(0,4176,144):
        daily.append(d[i:i+144])
    avg=np.zeros(144)
    for i in range(len(daily)):
        avg=avg+daily[i]
    avg=avg/(len(daily))
    #print((daily[28]))
    for i in range(len(daily)):
        daily[i]=daily[i]-avg
    anomaly=[]
    anomaly=np.asarray(anomaly)
    for i in range(len(daily)):
        anomaly=np.concatenate((anomaly,daily[i]),axis=0)
    return anomaly


# In[5]:


data=anom(data)
print(data)


# In[6]:


def MSE(yhat, y):
    return np.sqrt(np.mean((yhat.flatten() - y)**2))


# In[7]:


trainlen = int(0.7*len(data))
test= len(data)-trainlen
print(test)


# In[8]:


sparsity   = 0.4
rand_seed  = 25
noise = 0.005

radius_set = [0.1,0.2,0.3,0.4,0.5,0.6, 0.7,0.8, 0.9, 1, 1.1, 1.2,1.3,1.4]
reservoir_set = [ 10,20,30,50,70,100,150,200,250,300,350 ]
k_set = [0.2,0.3,0.4,0.5,0.6]

k_size = len(k_set)
radius_set_size  = len(radius_set)
reservoir_set_size = len(reservoir_set)
loss = np.zeros([k_size,radius_set_size, reservoir_set_size])


for v in range(k_size):
    futureTotal=test-(test%144)
    sparsity=k_set[v]
    for l in range(radius_set_size):
        rho = radius_set[l]
        for j in range(reservoir_set_size):
            n_reservoir = reservoir_set[j]
            future = 144
            pred_tot=np.zeros(futureTotal)

            esn = ESN(n_inputs = 1,
              n_outputs = 1, 
              n_reservoir = n_reservoir,
              sparsity=sparsity,
              random_state=rand_seed,
              spectral_radius = rho,
              noise=noise)

            for i in range(0,futureTotal,future):
                pred_training = esn.fit(np.ones(trainlen),data[i:trainlen+i])
                prediction = esn.predict(np.ones(future))
                pred_tot[i:i+future] = prediction[:,0]
            
            loss[v, l, j] = MSE(pred_tot, data[trainlen:trainlen+futureTotal])        
            print('sparsity = ', k_set[v],'rho = ', radius_set[l], ', reservoir_sixe = ', reservoir_set[j], ', MSE = ', loss[v][l][j] )


# In[11]:


print(np.min(loss))


# In[9]:


parameters=[]
for i in range(k_size):
    minloss=np.min(loss[i])
    index_min=np.where(loss[i]==minloss)
    spec=index_min[0]
    reserv=index_min[1]
    parameters.append([radius_set[spec[0]],reservoir_set[reserv[0]]])
print(parameters)


# In[101]:


#parameters=[[0.9, 50], [1, 50], [0.7, 100]]


# In[40]:


sparsity   = 0.4
rand_seed  = 25
noise = 0.005

radius_set = [0.5, 0.7, 0.9, 1,  1.1, 1.2]
reservoir_set = [ 10,20,50,70,100,150,200,250 ]
k_set = [3,6,12,36,72,144]

k_size = len(k_set)
radius_set_size  = len(radius_set)
reservoir_set_size = len(reservoir_set)
loss_k = np.zeros(k_size)
p=[]

for i in range(k_size):
    futureTotal=test-(test%k_set[i])
    future=k_set[i]
    spectral_radius=parameters[i][0]
    n_reservoir=parameters[i][1]
    esn = ESN(n_inputs = 1,
          n_outputs = 1, 
          n_reservoir = n_reservoir,
          sparsity=sparsity,
          random_state=rand_seed,
          spectral_radius = spectral_radius,
          noise=noise)

    pred_tot=np.zeros(futureTotal)

    for j in range(0,futureTotal,future):
        pred_training = esn.fit(np.ones(trainlen),data[j:trainlen+j])
        prediction = esn.predict(np.ones(future))
        pred_tot[j:j+future] = prediction[:,0]
    
    
    p.append(pred_tot)
    
    loss_k[i] = MSE(pred_tot, data[trainlen:trainlen+futureTotal])


# In[41]:


print(loss_k)


# In[42]:


plt.figure(figsize=(16,8))
#plt.plot(range(1000,trainlen+futureTotal),data[1000:trainlen+futureTotal],'b',label="Data", alpha=0.3)
#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
cc=['half hourly','hourly','two hourly','half day','1 day']
xx=['b','g','r','c','m','y']

plt.plot(k_set,loss_k,xx[3],  alpha=0.8, label='')


plt.title(r'Variation of MSE different time periods', fontsize=25)
plt.xlabel(r'Time periods', fontsize=20,labelpad=10)
plt.ylabel(r'MSE', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
sns.despine()


# In[43]:


print(loss_k)


# In[44]:


print(p[1])


# In[62]:


temp=data
range_1=3000-trainlen
range_2=4000-trainlen

plt.figure(figsize=(16,8))
plt.plot(range(trainlen+range_1,trainlen+range_2),temp[trainlen+range_1:trainlen+range_2],'k',label="Actual", alpha=0.8)
#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
cc=['half hourly','hourly','two hourly','six hourly','half day','1 day']
xx=['b','g','r','c','m','y']
for i in range(1):
    c=1
    vv=p[c]
    
    plt.plot(range(trainlen+range_1,trainlen+range_2),vv[range_1:range_2],xx[c],  alpha=0.8, label=cc[c])



plt.title(r'Anomalies Prediction by ESN network for 1 day', fontsize=25)
plt.xlabel(r'Time (1 unit = 10mins)', fontsize=20,labelpad=10)
plt.ylabel(r'80mWS', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
plt.savefig('24 hourly.png')
sns.despine()


# In[66]:




no_of_days=int(futureTotal/144)
mse_perday_intervals=[]

for i in range(k_size):
    
    count=0
    mse_= np.zeros(no_of_days)
    pp=p[i]
    print(len(pp))
    for j in range(0,futureTotal,144):
        if count<no_of_days:
            mse_[count]= MSE(pp[j:j+144], data[trainlen+j:trainlen+j+144])
            count+=1
    mse_perday_intervals.append(mse_)


# In[64]:


print(mse_perday_intervals)


# In[65]:


plt.figure(figsize=(16,8))
#plt.plot(range(1000,trainlen+futureTotal),data[1000:trainlen+futureTotal],'b',label="Data", alpha=0.3)
#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
cc=['half hourly','hourly','two hourly','six hourly','half day','1 day']
xx=['b','g','r','c','m','y']
for i in range(k_size):
    aa=mse_perday_intervals[i]
    plt.plot(range(0,no_of_days),aa,xx[i],  alpha=0.8, label=cc[i])


plt.title(r'Variation of MSE per day with different time intervals', fontsize=25)
plt.xlabel(r'Time (1 unit = 1 day)', fontsize=20,labelpad=10)
plt.ylabel(r'MSE', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
sns.despine()


# In[ ]:





# In[ ]:





# In[ ]:




