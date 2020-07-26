#!/usr/bin/env python
# coding: utf-8

# In[155]:


import numpy as np
import math
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# This is the library for the Reservoir Computing got it by: https://github.com/cknd/pyESN
from pyESN import ESN 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[156]:


data = pd.read_csv('D:\\DATASET\\2013\\VYSAPURAM-Wind_data2013.csv')
data=data['80mWS']

data=np.asarray(data)

dataa=data
#data=np.insert(data,0,0)
#data=data[:2000]
print((data))


# In[157]:


x1=[1,2,3]
x1=np.asarray(x1)
x2=[3,4,5]
x3=[]
x3=np.asarray(x3)
x2=np.asarray(x2)
x3=np.concatenate((x3,x2),axis=0)
print(np.log(x1))


# In[158]:


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
    avg=np.zeros([144])
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


# In[159]:


data=anom(data)
#data=np.log(data)
print(data)


# In[7]:


def MSE(yhat, y):
    return np.sqrt(np.mean((yhat.flatten() - y)**2))


# In[8]:


trainlen = int(0.7*len(data))
test= len(data)-trainlen
print(test)


# In[38]:


sparsity   = 0.4
rand_seed  = 25
noise = 0.005

radius_set = [0.5, 0.7, 0.9, 1,  1.1, 1.2]
reservoir_set = [ 10,20,50,70,100,150,200,250 ]
k_set = [3,6,12,36,72,144]

k_size = len(k_set)
radius_set_size  = len(radius_set)
reservoir_set_size = len(reservoir_set)
loss = np.zeros([k_size,radius_set_size, reservoir_set_size])


for v in range(k_size):
    futureTotal=test-(test%k_set[v])
    for l in range(radius_set_size):
        rho = radius_set[l]
        for j in range(reservoir_set_size):
            n_reservoir = reservoir_set[j]
            future = k_set[v]
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
            
            loss[v, l, j] = MSE(pred_tot, data[trainlen-1:trainlen+futureTotal-1])        
            print('window = ', k_set[v],'rho = ', radius_set[l], ', reservoir_sixe = ', reservoir_set[j], ', MSE = ', loss[v][l][j] )


# In[39]:


parameters=[]
for i in range(k_size):
    minloss=np.min(loss[i])
    index_min=np.where(loss[i]==minloss)
    spec=index_min[0]
    reserv=index_min[1]
    parameters.append([radius_set[spec[0]],reservoir_set[reserv[0]]])
print(parameters)


# In[1]:


parameters=[[1.1, 150], [1.1, 150], [0.7, 250], [0.9, 300], [0.5, 70], [0.7, 200]]


# In[95]:


sparsity   = 0.4
rand_seed  = 25
noise = 0.005
R2=[]
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
    R2.append((r2_score(pred_tot,data[trainlen-k_set[i]:trainlen+futureTotal-k_set[i]])))
    loss_k[i] = MSE(pred_tot, data[trainlen-k_set[i]:trainlen+futureTotal-k_set[i]])


# In[96]:


print(loss_k)
print(R2)


# In[97]:


plt.figure(figsize=(16,8))
#plt.plot(range(1000,trainlen+futureTotal),data[1000:trainlen+futureTotal],'b',label="Data", alpha=0.3)
#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
cc=['half hourly','hourly','two hourly','six hourly','half day','1 day']
xx=['b','g','r','c','m','y']

plt.plot(cc,loss_k,xx[3],  alpha=0.8, label='')


plt.title(r'Variation of MSE different time periods', fontsize=25)
plt.xlabel(r'Time periods', fontsize=20,labelpad=10)
plt.ylabel(r'RMSE', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
sns.despine()


# In[98]:


print(loss_k)


# In[99]:


print(p[1])


# In[110]:


temp=data
range_1=3000-trainlen
range_2=3144-trainlen
v=[]
for i in range(0,144):
    v.append(i/6)

plt.figure(figsize=(16,8))

#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
cc=['half hourly','hourly','two hourly','six hourly','half day','1 day']
xx=['b','g','r','c','m','y']
for i in range(1):
    c=3
    vv=p[c]
    plt.plot(v,temp[trainlen-k_set[c]+range_1:trainlen-k_set[c]+range_2],'k',label="Actual", alpha=0.8)
    plt.plot(v,vv[range_1:range_2],xx[c],  alpha=0.8, label=cc[c])



plt.title(r'Anomalies Prediction by ESN network for 1 day', fontsize=25)
plt.xlabel(r'Time (1 unit = 1 hour)', fontsize=20,labelpad=10)
plt.ylabel(r'80mWS Anomaly', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
plt.savefig('24 hourly.png')
sns.despine()


# In[76]:


cc=['half hourly','hourly','two hourly','six hourly','half day','1 day']
for i in range(k_size):
    futureTotal_=test-(test%k_set[i])
    plt.figure(figsize=(12,5))
    plt.title(cc[i], fontsize=25)
    
    plt.scatter(data[trainlen:trainlen+futureTotal_],p[i],alpha=0.8)
    plt.xlabel('Actual values')
    plt.ylabel('Predicted values')
    plt.text(2.5, -6, 'R-squared = %0.2f' % R2[i])
    plt.show()


# In[107]:


print(R2)


# In[20]:




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


# In[21]:


print(mse_perday_intervals)


# In[22]:


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




