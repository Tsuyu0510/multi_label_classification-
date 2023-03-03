#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import sklearn
import tensorflow as tf
import keras 
import matplotlib.pyplot as plt 
#import autokeras as ak


# In[3]:


## get data 
## get  data
df_1 = pd.read_csv('route.csv')
df_1


# In[4]:


df_1["表定時間"] = pd.to_datetime(df_1["表定時間"])
#df["進貨指定日"] = pd.to_datetime(df["進貨指定日"])

df_1["表定時間year"] = df_1["表定時間"].dt.year
df_1["表定時間month"] = df_1["表定時間"].dt.month
df_1["表定時間day"] = df_1["表定時間"].dt.day
df_1["表定時間hour"] = df_1["表定時間"].dt.hour
df_1["表定時間minute"] = df_1["表定時間"].dt.minute
df_1["表定時間second"] = df_1["表定時間"].dt.second
df_1_drop = df_1.drop(['表定時間'],axis =1)
## label encoder 
#Label encoding 轉主線編排	
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
#data_le01 = pd.DataFrame(data_le)
#df_drop['車次'] = labelencoder.fit_transform(df_drop['車次'])
#df_drop['店名'] = labelencoder.fit_transform(df_drop['店名'])
df_1_drop['鄰近商圈'] = labelencoder.fit_transform(df_1_drop['鄰近商圈'])

df_1_drop['店名'] = labelencoder.fit_transform(df_1_drop['店名'])


df_1_drop = df_1_drop.reindex(columns=['車次','主線編排','店名','貨量','天氣','檔期','有無節慶','節慶類別',
                                       '鄰近商圈','表定時間year','表定時間month','表定時間day','表定時間hour',
                                       '表定時間minute','表定時間second'])

df_1_drop['concate'] = df_1_drop[['車次','主線編排']].apply(",".join,axis =1) 
df_1_drop_1 = df_1_drop.drop(['車次','主線編排'],axis =1)
#df_drop.drop(['是否為爆量車'],axis =1)
df_1_drop_1


# In[5]:


#df_1_drop['主線編排'].value_counts()


# In[6]:


## dateframe set dummy variables
df_1_drop_1_dummy = pd.get_dummies(df_1_drop_1, columns=['concate'])
df_1_drop_1_dummy


# In[7]:


print(df_1_drop_1_dummy.columns.tolist())


# In[8]:


## get train test dataset 
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
train_data = df_1_drop_1_dummy.iloc[:,0:13]
train_target = df_1_drop_1_dummy.iloc[:,13:]

#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler()
#train_data_sc = scaler.fit_transform(train_data)
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(train_data)
#X_scaled = scaler.transform(train_data)


X_train, X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.2,random_state=44,shuffle=True)
y_train = y_train.astype(np.float64)
y_test = y_test.astype(np.float64)
print(X_train,X_test)
print(y_train,y_test)


# In[9]:


feature_dim = X_train.shape[1]
feature_dim


# In[10]:


label_dim = y_train.shape[1]
label_dim


# In[11]:


## build model 
from keras.models import Sequential
from keras.layers import Dense,Dropout,MaxPooling2D
from keras import optimizers
model = Sequential()
model.add(Dense(1024, activation = 'relu' , input_dim = feature_dim))
model.add(Dense(512, activation = 'relu' ))
#model.add(Dense(128, activation = 'relu' ))




model.add(Dense(label_dim, activation = 'sigmoid' ))
sgd = optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
adamax = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = adam , loss = 'binary_crossentropy', metrics = ['accuracy'])
model.summary()


# In[12]:


epochs = 100
history = model.fit(X_train,y_train,batch_size=4,epochs=epochs,validation_data=(X_test,y_test))
#model.save('multi_route_keras.h5')


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
accuracy = history.history['accuracy']
loss = history.history['loss']
val_accuracy = history.history['val_accuracy']
val_loss = history.history['val_loss']
epochs = range(epochs)
plt.plot(epochs,accuracy,'r-')
plt.plot(epochs,val_accuracy,'b-')
plt.legend()
plt.show()


# In[ ]:


plt.plot(epochs,loss,'r-')
plt.plot(epochs,val_loss,'b-')
plt.legend()
plt.show()


# In[ ]:


model


preds = model.predict(X_test)
df_pred = pd.DateFrame(preds)


# In[ ]:





# In[ ]:




