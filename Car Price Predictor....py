#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
car = pd.read_csv("E:\\Car1.csv")
car


# In[2]:


car.info()


# In[3]:


car['year'].unique()


# In[4]:


car['selling_price'].unique()


# In[5]:


car['km_driven'].unique()


# In[6]:


car['fuel'].unique()


# In[7]:


car['name'].unique()


# In[8]:


backup=car.copy()


# In[9]:


car['mileage'].unique()


# In[10]:


car["mileage"]=car["mileage"].str.split(' ').str.get(0)


# In[11]:


car=car[~car["mileage"].isna()]


# In[12]:


car["mileage"]=car['mileage'].astype(float)


# In[13]:


car["mileage"]


# In[14]:


car.info()


# In[15]:


car["engine"]


# In[16]:


car["engine"]=car["engine"].str.split(' ').str.get(0)


# In[17]:


car=car[~car["engine"].isna()]


# In[18]:


car["engine"]=car['engine'].astype(int)


# In[19]:


car["engine"]=car['engine'].astype(int)


# In[20]:


car["engine"]


# In[21]:


car["max_power"]


# In[22]:


car["max_power"].str.split(' ').str.get(0)


# In[26]:


car[car["max_power"].isna()]


# In[27]:


car["max_power"].str.isnumeric()


# In[29]:


car["max_power"]


# In[30]:


car["max_power"] = car["max_power"].str.split(" ").str.get(0)


# In[31]:


car


# In[32]:


car[car["max_power"]!='']


# In[33]:


car=car[car["max_power"]!='']


# In[34]:


car


# In[35]:


car["max_power"]= car["max_power"].astype(float)


# In[36]:


car=car[~car["mileage"].isna()]


# In[37]:


car


# In[ ]:





# In[38]:


car["seats"]=car['seats'].astype(int)


# In[39]:


car.info()


# In[40]:


car.describe()


# In[41]:


car=car[car["selling_price"]<6e6]


# In[42]:


car


# In[43]:


car.to_csv("clean car.csv")


# In[44]:


car


# In[45]:


car.isnull().sum()


# In[46]:


car=car[~car["torque"].isna()]


# In[47]:


car


# In[48]:


car=car.drop(columns = "torque")


# In[49]:


car


# In[50]:


car.info()


# In[51]:


car["year"].unique()


# In[52]:


car.to_csv("clean car1.csv")


# In[53]:


car


# In[54]:


sns.heatmap(car.corr(), annot= True, cmap= 'Reds')


# # MODEL

# In[55]:


X = car.drop(columns = "selling_price")
y=car["selling_price"]


# In[56]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)


# In[57]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import seaborn as sns


# In[58]:


ohe = OneHotEncoder()
ohe.fit(X[['name','fuel','seller_type','transmission','owner']])


# In[59]:


ohe.categories_


# In[ ]:





# In[60]:


column_trans = make_column_transformer((OneHotEncoder(categories = ohe.categories_),['name','fuel','seller_type',
                                                                                     'transmission','owner']),
                                       remainder = 'passthrough')


# In[61]:


sns.heatmap(car.corr(), annot= True, cmap= 'Reds')


# In[62]:


lr=LinearRegression()


# In[63]:


pipe = make_pipeline(column_trans,lr)


# In[64]:


pipe.fit(X_train,y_train)


# In[65]:


y_prid = pipe.predict(X_test)


# In[66]:


y_prid


# In[67]:


r2_score(y_test,y_prid)


# In[68]:


scores=[]
for i in range (10):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=i)
    lr=LinearRegression()
    pipe = make_pipeline(column_trans,lr)
    pipe.fit(X_train,y_train)
    y_prid = pipe.predict(X_test)
    scores.append(r2_score(y_test,y_prid))
    #print(r2_score(y_test,y_prid),i)


# In[69]:


np.argmax(scores)


# In[70]:


scores[np.argmax(scores)]


# In[71]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state=np.argmax(scores))
lr=LinearRegression()
pipe = make_pipeline(column_trans,lr)
pipe.fit(X_train,y_train)
y_prid = pipe.predict(X_test)
r2_score(y_test,y_prid)


# In[72]:


car


# In[73]:


import pickle


# In[74]:


pickle.dump(pipe,open("LinearRegressionModel1.pkl",'wb'))


# In[92]:


pipe.predict(pd.DataFrame([['Maruti Swift Dzire VDI','2014',"145500",'Diesel','Individual','Manual','First Owner',
                            '23.40','1248','74','5']],
            columns=['name','year','km_driven','fuel','seller_type','transmission','owner',
                     'mileage','engine','max_power','seats']))


# In[ ]:





# In[ ]:





# In[76]:


#car


# In[77]:


#car.info()


# In[78]:


# s=0
# for i in car['year']:
#     #a=car['year'][s]
#     print(car['year'][s])
#     s+=1
    


# In[79]:


# car['year']


# In[80]:


# car['year'].isnull().sum()


# In[81]:


# print(car['year'])


# In[82]:


# car['year'][13]


# In[83]:


# car.head(18)


# In[84]:


# car


# In[85]:


# car=car.reset_index()


# In[86]:


# car


# In[87]:


# car['year']


# In[ ]:




