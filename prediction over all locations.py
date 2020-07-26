#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.metrics import r2_score
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
import random
warnings.filterwarnings('ignore')

# This is the library for the Reservoir Computing got it by: https://github.com/cknd/pyESN
from pyESN import ESN 
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


def MSE(yhat, y):
    return np.sqrt(np.mean((yhat.flatten() - y)**2))


# In[ ]:





# In[3]:


k_set=[3,6,12,36,72,144]
k_size=len(k_set)
#distance_from_centre=[335,462,800,380,335,367,670,740,640,476,236,350,380,202,330,740,405,735,369,850,870,415,430]
#print(min(distance_from_centre))
#distance=[300,350,400,450,500,550,600,650,700,750,800,850,900]
#N=[2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
#no_of_location=len(distance_from_centre)


# In[4]:


def loc(distancee):
    c=[]
    for i in range(len(distance_from_centre)):
        if distance_from_centre[i]<=distancee:
            c.append(i)
    return c


# In[5]:



actual=pd.read_csv('D:\\DATASET\\2013\80mWS\\results\\actual.csv')
actual=np.asarray(actual)
actual=actual.reshape(23,4176)
print(actual)


# In[6]:


def appn_data(n,dist):
    yy=np.zeros([len(actual[0])])
    for zz in range(n):
        yy=((yy+actual[dist[zz]]))
    return yy


# In[7]:


#parameters=[[0.5, 10], [0.5, 70], [0.5, 70], [0.7, 200], [0.7, 200], [0.4, 70]]


# In[9]:


trainlen=int(0.7*len(actual[0]))
test=len(actual[0])-trainlen
print(test)


# In[10]:


sparsity   = 0.4
rand_seed  = 25
noise = 0.005

radius_set = [0.5, 0.7, 0.9, 1,  1.1, 1.2]
reservoir_set = [ 10,20,50,70,100,150,200,250,300 ]
k_set = [3,6,12,36,72,144]
distance=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
k_size = len(k_set)
radius_set_size  = len(radius_set)
reservoir_set_size = len(reservoir_set)
loss = np.zeros([k_size,radius_set_size, reservoir_set_size])
data=appn_data(23,distance)
print(data.shape)
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
            
            loss[v, l, j] = MSE(pred_tot, data[trainlen-future:trainlen+futureTotal-future])        
            print('window = ', k_set[v],'rho = ', radius_set[l], ', reservoir_sixe = ', reservoir_set[j], ', MSE = ', loss[v][l][j] )


# In[11]:


parameters=[]
for i in range(k_size):
    minloss=np.min(loss[i])
    index_min=np.where(loss[i]==minloss)
    spec=index_min[0]
    reserv=index_min[1]
    parameters.append([radius_set[spec[0]],reservoir_set[reserv[0]]])
print(parameters)


# In[12]:


parameters=[[1.1, 150], [1.1, 150], [0.7, 250], [0.9, 300], [0.5, 70], [0.7, 200]]


# In[26]:


N=[23]
MSE_hourly=np.zeros([len(N)])
MSE_half_hourly=np.zeros([len(N)])
MSE_two_hourly=np.zeros([len(N)])
MSE_six_hourly=np.zeros([len(N)])
MSE_half_day=np.zeros([len(N)])
MSE_one_day=np.zeros([len(N)])
sparsity=0.4
rand_seed=25
noise=0.005
distance=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
coor_half_hourly=[]
coor_hourly=[]
coor_two_hourly=[]
coor_six_hourly=[]
coor_half_day=[]
coor_one_day=[]
p_half_hourly=[]
r2_half_hourly=[]
p_hourly=[]
r2_hourly=[]
p_two_hourly=[]
r2_two_hourly=[]
p_six_hourly=[]
r2_six_hourly=[]
p_half_day=[]
r2_half_day=[]
p_one_day=[]
r2_one_day=[]

for q in range(k_size):
    trainlen=int(0.7*len(actual[0]))
    futureTotal=test-(test%k_set[q])
    spectral_radius=parameters[q][0]
    n_reservoir=parameters[q][1]
    future=k_set[q]
    
    xx=distance
    for j in range(len(N)):
        #x=np.zeros([futureTotal])
        #x=np.asarray(x)
        y=np.zeros([len(actual[0])])
        #y=np.asarray(y)

        if(N[j]<=len(xx)):
            
            y=appn_data(N[j],distance)


            esn = ESN(n_inputs = 1,
                  n_outputs = 1, 
                  n_reservoir = n_reservoir,
                  sparsity=sparsity,
                  random_state=rand_seed,
                  spectral_radius = spectral_radius,
                  noise=noise)

            pred_tot=np.zeros(futureTotal)

            for jj in range(0,futureTotal,future):
                pred_training = esn.fit(np.ones(trainlen),y[jj:trainlen+jj])
                prediction = esn.predict(np.ones(future))
                pred_tot[jj:jj+future] = prediction[:,0]
            
            
            
            
            if q==0:
                p_half_hourly.append(pred_tot)
                MSE_half_hourly[j]=MSE(pred_tot,y[trainlen-future:trainlen-future+futureTotal])
                r2_half_hourly.append(r2_score(pred_tot,y[trainlen-future:trainlen-future+futureTotal]))
                coor_half_hourly.append(np.corrcoef(pred_tot,y[trainlen-future:trainlen-future+futureTotal])[0,1])
                
            if q==1:
                p_hourly.append(pred_tot)
                MSE_hourly[j]=MSE(pred_tot,y[trainlen-future:trainlen-future+futureTotal])
                r2_hourly.append(r2_score(pred_tot,y[trainlen-future:trainlen-future+futureTotal]))
                coor_hourly.append(np.corrcoef(pred_tot,y[trainlen-future:trainlen-future+futureTotal])[0,1])
            if q==2:
                p_two_hourly.append(pred_tot)
                MSE_two_hourly[j]=MSE(pred_tot,y[trainlen-future:trainlen-future+futureTotal])
                r2_two_hourly.append(r2_score(pred_tot,y[trainlen-future:trainlen-future+futureTotal]))
                coor_two_hourly.append(np.corrcoef(pred_tot,y[trainlen-future:trainlen-future+futureTotal])[0,1])
            if q==3:
                p_six_hourly.append(pred_tot)
                MSE_six_hourly[j]=MSE(pred_tot,y[trainlen-future:trainlen-future+futureTotal])
                r2_six_hourly.append(r2_score(pred_tot,y[trainlen-future:trainlen-future+futureTotal]))
                coor_six_hourly.append(np.corrcoef(pred_tot,y[trainlen-future:trainlen-future+futureTotal])[0,1])
            if q==4:
                p_half_day.append(pred_tot)
                MSE_half_day[j]=MSE(pred_tot,y[trainlen-future:trainlen-future+futureTotal])
                r2_half_day.append(r2_score(pred_tot,y[trainlen-future:trainlen-future+futureTotal]))
                coor_half_day.append(np.corrcoef(pred_tot,y[trainlen-future:trainlen-future+futureTotal])[0,1])
            if q==5:
                p_one_day.append(pred_tot)
                MSE_one_day[j]=MSE(pred_tot,y[trainlen-future:trainlen-future+futureTotal])
                r2_one_day.append(r2_score(pred_tot,y[trainlen-future:trainlen-future+futureTotal]))
                coor_one_day.append(np.corrcoef(pred_tot,y[trainlen-future:trainlen-future+futureTotal])[0,1])


# In[27]:


print(r2_hourly)


# In[28]:


def plott(temp,vv,c,l):
    
    range_1=3000-trainlen
    range_2=3144-trainlen
    v=[]
    for i in range(0,range_2-range_1):
        v.append(i/6)

    plt.figure(figsize=(12,5))
    plt.plot(v,temp[trainlen-l+range_1:trainlen-l+range_2],'k',label="Actual raw data", alpha=0.8)
    #plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
    cc=['half hourly','hourly','two hourly','six hourly','half day','1 day']
    xx=['b','g','r','c','m','y']
    for i in range(1):
        
        

        plt.plot(v,vv[range_1:range_2],xx[c],  alpha=0.8, label=cc[c])



    plt.title(r'Prediction by ESN network for 1 day on Anomaly data over all 23 locations', fontsize=25)
    plt.xlabel(r'Time (1 unit = 1 hour)', fontsize=20,labelpad=10)
    plt.ylabel(r'80mWS ', fontsize=20,labelpad=10)
    plt.legend(fontsize='xx-large', loc='best')
    #plt.savefig('one day.png')
    sns.despine()


# In[29]:


print((r2_half_hourly))


# In[30]:


f=p_half_hourly

for i in range(len(f)):
    r=N[i]
    yy=y=np.zeros([len(actual[0])])
    for zz in range(r):
        yy=((yy+actual[distance[zz]]))
    plott(yy,f[i],0,3)


# In[31]:


bb=[]
bb.append(coor_half_hourly)
bb.append(coor_hourly)
bb.append(coor_two_hourly)
bb.append(coor_six_hourly)
bb.append(coor_half_day)
bb.append(coor_one_day)
rr=[]
rr.append(np.square(bb[0]))
rr.append(np.square(bb[1]))
rr.append(np.square(bb[2]))
rr.append(np.square(bb[3]))
rr.append(np.square(bb[4]))
rr.append(np.square(bb[5]))


# In[32]:


plt.figure(figsize=(12,5))
#plt.plot(v,temp[trainlen-l+range_1:trainlen-l+range_2],'k',label="Actual raw data", alpha=0.8)
#plt.plot(range(0,trainlen),pred_training,'.g',  alpha=0.3)
cc=['half hourly','hourly','two hourly','six hourly','half day','1 day']
xx=['b','g','r','c','m','y']
plt.plot(cc,bb,label='correlation coefficient')
plt.plot(cc,rr,label='R squared')
plt.title(r'Correlation coefficient and R squared for Anomaly data', fontsize=25)
plt.xlabel(r'Lead Time (1 unit = 1 hour)', fontsize=20,labelpad=10)
plt.ylabel(r'Goodness of Fit ', fontsize=20,labelpad=10)
plt.legend(fontsize='xx-large', loc='best')
#plt.savefig('goodnes of fit.png')
sns.despine()


# In[ ]:




