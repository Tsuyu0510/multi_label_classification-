#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
from sklearn.ensemble import RandomForestClassifier
from skmultilearn.problem_transform import ClassifierChain
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,multilabel_confusion_matrix
import sklearn.metrics as metrics
import datetime
from sqlalchemy import create_engine
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.multiclass import OneVsRestClassifier


# In[ ]:


#pd.set_option('display.width', 1000) # 設定字元顯示寬度
df_1 = pd.read_csv('route.csv')
df_1


# In[ ]:


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


# In[ ]:


df_1_drop['主線編排'].value_counts()


# In[ ]:


## dateframe set dummy variables
df_1_drop_dummy = pd.get_dummies(df_1_drop, columns=['車次','主線編排'])
df_1_drop_dummy


# In[ ]:


## get train test dataset 
from sklearn.model_selection import train_test_split
train_data = df_1_drop_dummy.iloc[:,0:13]
#from sklearn.preprocessing import StandardScaler
#scaler = StandardScaler().fit(train_data)
#X_scaled = scaler.transform(train_data)
train_target = df_1_drop_dummy.iloc[:,14:54] 

X_train, X_test, y_train, y_test = train_test_split(train_data,train_target,test_size=0.2, random_state=0)

X_train.shape


# In[ ]:


clf_multilabel = OneVsRestClassifier(LGBMClassifier())
clf_multilabel.fit(X_train,y_train)


pred = clf_multilabel.predict(X_test)
df_pred = pd.DataFrame(pred,columns=['車次_101', '車次_102', '車次_103', '車次_104', '車次_105', '車次_106', '車次_2M', 
                                                                '車次_2N', '車次_2P', '車次_301', '車次_3L', '車次_8O', '主線編排_2N01', '主線編排_2N02',
                                                                '主線編排_2N03', '主線編排_2N04', '主線編排_2N05', '主線編排_2N06', '主線編排_2N07', 
                                                                '主線編排_2N08', '主線編排_2N09', '主線編排_2N10', '主線編排_2N11', '主線編排_2N12', '主線編排_2N13', '主線編排_2P01', 
                                                                '主線編排_2P02', '主線編排_2P03', '主線編排_2P04', '主線編排_2P05', '主線編排_2P06', '主線編排_2P07', '主線編排_2P08',
                                                                '主線編排_2P09', '主線編排_2P10', '主線編排_2P11', '主線編排_2P12', '主線編排_2P13', '主線編排_2P14'])


# In[ ]:


df_pred


# In[ ]:


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


# In[ ]:


from sklearn.metrics import accuracy_score,hamming_loss,recall_score,precision_score,f1_score
print(accuracy_score(y_test,pred))
print(hamming_loss(y_test, pred))
print(recall_score(y_test, pred,average='micro'))
print(precision_score(y_test, pred,average='micro'))
print(f1_score(y_test, pred, average='micro'))


# In[ ]:


from sklearn.metrics import multilabel_confusion_matrix
cm = multilabel_confusion_matrix(y_test, pred)
print (cm)
print (cm.shape)


# In[ ]:




