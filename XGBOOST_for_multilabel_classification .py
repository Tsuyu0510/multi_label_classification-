#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install xgboost
from xgboost import XGBClassifier
import numpy as np 
import pandas as pd 
import sklearn
from sklearn.multiclass import OneVsRestClassifier


# In[2]:


## get previous data
## get  data
#˙pd.set_option('display.width', 1000) # 設定字元顯示寬度
df_1 = pd.read_csv('route.csv')
df_1


# In[3]:


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
                                       '表定時間minute','表定時間second','是否為爆量車'])

#df_drop.drop(['是否為爆量車'],axis =1)
df_1_drop



# In[4]:


df_1_drop['主線編排'].value_counts()


# In[5]:


## dateframe set dummy variables
df_1_drop_dummy = pd.get_dummies(df_1_drop, columns=['車次','主線編排'])
df_1_drop_dummy


# In[6]:


## get train test dataset 
from sklearn.model_selection import train_test_split
train_data = df_1_drop_dummy.iloc[:,0:13]
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(train_data)
#X_scaled = scaler.transform(train_data)
train_target = df_1_drop_dummy.iloc[:,14:54] 

X_train, X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.2, random_state=0)

X_train.shape


# In[7]:


clf_multilabel = OneVsRestClassifier(XGBClassifier())
clf_multilabel.fit(X_train,y_train)
pred = clf_multilabel.predict(X_test)


# In[8]:


pd.set_option('display.max_columns', None) # 設定字元顯示寬度

df_pred = pd.DataFrame(pred,columns=['車次_101', '車次_102', '車次_103', '車次_104', '車次_105', '車次_106', '車次_2M', 
                                                                '車次_2N', '車次_2P', '車次_301', '車次_3L', '車次_8O', '主線編排_2N01', '主線編排_2N02',
                                                                '主線編排_2N03', '主線編排_2N04', '主線編排_2N05', '主線編排_2N06', '主線編排_2N07', 
                                                                '主線編排_2N08', '主線編排_2N09', '主線編排_2N10', '主線編排_2N11', '主線編排_2N12', '主線編排_2N13', '主線編排_2P01', 
                                                                '主線編排_2P02', '主線編排_2P03', '主線編排_2P04', '主線編排_2P05', '主線編排_2P06', '主線編排_2P07', '主線編排_2P08',
                                                                '主線編排_2P09', '主線編排_2P10', '主線編排_2P11', '主線編排_2P12', '主線編排_2P13', '主線編排_2P14'])
df_pred


# In[9]:


pd.set_option('display.max_rows', None) # 設定字元顯示寬度

## append test_x and test_y
X_test.index = range(len(X_test))
X_test

##concat
final_result = pd.concat((X_test,df_pred),axis =1)
## 把label encoding 的店名標籤數值轉回文字
final_result['店名'] = labelencoder.inverse_transform(final_result['店名'])
#final_result

#把透過pandas get dummies 取得one-hot encoding 的車次和主線編排欄位轉回來


def undummify(df, prefix_sep="_"):
    cols2collapse = {
        item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
    }
    series_list = []
    for col, needs_to_collapse in cols2collapse.items():
        if needs_to_collapse:
            undummified = (
                df.filter(like=col)
                .idxmax(axis=1)
                .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                .rename(col)
            )
            series_list.append(undummified)
        else:
            series_list.append(df[col])
    undummified_df = pd.concat(series_list, axis=1)
    return undummified_df
final_result_add = undummify(final_result)
final_result_add.sort_values(by='車次')
#final_result_add_2 = final_result_add.sort_values(by='主線編排')


# In[10]:


from sklearn.metrics import accuracy_score,hamming_loss,recall_score,precision_score,f1_score
print(accuracy_score(y_test,pred))
print(hamming_loss(y_test, pred))
print(recall_score(y_test, pred,average='micro'))
print(precision_score(y_test, pred,average='micro'))
print(f1_score(y_test, pred, average='micro'))


# In[11]:


from sklearn.metrics import multilabel_confusion_matrix
cm = multilabel_confusion_matrix(y_test, pred)
print (cm)
print (cm.shape)


# In[ ]:





# In[ ]:





# In[12]:


df_1


# In[13]:


## add new column for concate col0 and col1
df_1['concate'] = ''


# In[14]:


df_1


# In[15]:


df_1["concate"] = list(df_1["車次"] + "," +df_1["主線編排"])
df_1


# In[16]:


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
#df_1_drop['concate'] = labelencoder.fit_transform(df_1_drop['concate'])


df_1_drop = df_1_drop.reindex(columns=['車次','主線編排','店名','貨量','天氣','檔期','有無節慶','節慶類別',
                                       '鄰近商圈','表定時間year','表定時間month','表定時間day','表定時間hour',
                                       '表定時間minute','表定時間second','是否為爆量車','concate'])

#df_drop.drop(['是否為爆量車'],axis =1)
df_1_drop = df_1_drop.drop(['車次'],axis =1)
df_1_drop = df_1_drop.drop(['主線編排'],axis =1)
df_1_drop


# In[42]:


train_data = df_1_drop.iloc[:,0:13]
train_target = df_1_drop.iloc[:,14:15]
X_train, X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.2, random_state=0)


# In[43]:


from xgboost import XGBClassifier

clf_multilabel = OneVsRestClassifier(XGBClassifier())
clf_multilabel.fit(X_train,y_train)
pred = clf_multilabel.predict(X_test)


# In[44]:


y_test_reindex = y_test.reset_index(drop=True)
y_test_reindex


# In[45]:


df_pred = pd.DataFrame(pred)
df_pred


# In[46]:


df_final = pd.concat([y_test_reindex,df_pred],axis =1)
df_final


# In[47]:


pd.set_option('display.max_rows', None) # 設定字元顯示寬度

## append test_x and test_y
X_test.index = range(len(X_test))
X_test

##concat
final_result = pd.concat((X_test,df_pred),axis =1)
## 把label encoding 的店名標籤數值轉回文字
final_result['店名'] = labelencoder.inverse_transform(final_result['店名'])
#final_result


# In[48]:


final_result


# In[49]:


from sklearn.metrics import accuracy_score,hamming_loss,recall_score,precision_score,f1_score
print(accuracy_score(y_test,pred))
print(hamming_loss(y_test, pred))
print(recall_score(y_test, pred,average='micro'))
print(precision_score(y_test, pred,average='micro'))
print(f1_score(y_test, pred, average='micro'))


# In[50]:


from sklearn.metrics import accuracy_score,hamming_loss,recall_score,precision_score,f1_score
print(accuracy_score(y_test,pred))
print(hamming_loss(y_test, pred))
print(recall_score(y_test, pred,average='macro'))
print(precision_score(y_test, pred,average='macro'))
print(f1_score(y_test, pred, average='macro'))


# In[51]:


from sklearn.metrics import accuracy_score,hamming_loss,recall_score,precision_score,f1_score
print(accuracy_score(y_test,pred))
print(hamming_loss(y_test, pred))
print(recall_score(y_test, pred,average='weighted'))
print(precision_score(y_test, pred,average='weighted'))
print(f1_score(y_test, pred, average='weighted'))


# In[115]:


df_1 = pd.read_csv('route.csv')
df_1

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
df_1_drop['車次'] = labelencoder.fit_transform(df_1_drop['車次'])
df_1_drop['主線編排'] = labelencoder.fit_transform(df_1_drop['主線編排'])

df_1_drop['店名'] = labelencoder.fit_transform(df_1_drop['店名'])
#df_1_drop['concate'] = labelencoder.fit_transform(df_1_drop['concate'])


df_1_drop = df_1_drop.reindex(columns=['車次','主線編排','店名','貨量','天氣','檔期','有無節慶','節慶類別',
                                       '鄰近商圈','表定時間year','表定時間month','表定時間day','表定時間hour',
                                       '表定時間minute','表定時間second','是否為爆量車'])

df_1_drop = df_1_drop.drop(['是否為爆量車'],axis =1)
#df_1_drop = df_1_drop.drop(['車次'],axis =1)
#df_1_drop = df_1_drop.drop(['主線編排'],axis =1)
df_1_drop


# In[116]:


#df_1_drop['concate'] = labelencoder.fit_transform(df_1_drop['concate'])
df_1_drop.columns=['car','route','store','Qty','weather','section','festival','festival_class','Type','Timeyear','Timemonth',
                   'Timeday','Timehour','Timeminute','Timesecond']

df_1_drop


# In[120]:


train_data_num = df_1_drop.iloc[:,2:13]
train_target_num = df_1_drop.iloc[:,0:1]
train_target_num_2 = df_1_drop.iloc[:,1:2]

#X_train_num, X_test_num, y_train_num, y_test_num = train_test_split(train_data_num,train_target_num,test_size=0.2, random_state=44, shuffle=True)
#X_test_num

#print(train_target_num)
#print(train_target_num_2)


# In[121]:


from xgboost import XGBClassifier

clf_multilabel_num = XGBClassifier()
clf_multilabel_num.fit(train_data_num,train_target_num)

clf_multilabel_num_2 = XGBClassifier()
clf_multilabel_num_2.fit(train_data_num,train_target_num_2)


# In[122]:


train_data_num.columns


# In[123]:


import matplotlib.pyplot as plt
plt.barh(train_data_num.columns, clf_multilabel_num.feature_importances_)


# In[124]:


plt.barh(train_data_num.columns, clf_multilabel_num_2.feature_importances_)


# In[ ]:




