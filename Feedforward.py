
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import torch
import torch.utils.data as data_utils
import torch.nn.functional as F
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split


# In[2]:


trainFile = "C:/Users/ajit/Desktop/Problem_05.csv"
pwd = os.getcwd()
os.chdir(os.path.dirname(trainFile))
df=pd.read_csv(os.path.basename(trainFile), parse_dates=['Time'])


# In[3]:


df_1=df[(df['Turbine']==124) & df['NtfOk']==True]


# In[4]:


x=df_1[['RpmB','Energy','WindN']]
y=df_1['WindMfs']


# In[5]:


X_train,X_test,y_train,y_test=train_test_split(np.array(x),np.array(y),random_state=5)


# In[6]:


X_tr,X_te,y_tr,y_te=torch.from_numpy(X_train),torch.from_numpy(X_test),torch.from_numpy(y_train.reshape(-1,1)),torch.from_numpy(y_test.reshape(-1,1))


# In[7]:


trainset=data_utils.TensorDataset(X_tr.float(),y_tr.float())
trainloader=data_utils.DataLoader(trainset,batch_size=512,shuffle=True)
testset=data_utils.TensorDataset(X_te.float(),y_te.float())
testloader=data_utils.DataLoader(testset,batch_size=128)


# In[32]:


class pytorch_model(nn.Module):
    
    def __init__(self,input_size,first_size,second_size,third_size,output_size):
        super().__init__()
        self.fc1=nn.Linear(input_size,first_size)
        self.fc2=nn.Linear(first_size,second_size)
        self.fc3=nn.Linear(second_size,third_size)
        self.fc4=nn.Linear(third_size,output_size)
    def forward(self,x):
        l1=F.relu(self.fc1(x))
        l2=F.relu(self.fc2(l1))
        l3=F.relu(self.fc3(l2))
        out=self.fc4(l3)
        return out


# In[33]:


model=pytorch_model(3,8,5,5,1)


# In[34]:


criteria=nn.MSELoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.001)


# In[35]:


for epoch in range(10):
    running_loss=0.0
    for i,data in enumerate(trainloader):
        inputs,labels=data
        outputs=model(inputs)
        loss=criteria(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss+=loss
        if i %8==7:
            print('[%d, %5d] loss: %.3f' % (epoch+1,i+1,running_loss/8))
            running_loss=0.0
print('Finished training')


# In[37]:


test_output=model(X_te.float())


# In[41]:


(torch.sum((test_output-y_te.float())**2))/len(y_te)

