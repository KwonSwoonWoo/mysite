# -*- coding: utf-8 -*- 
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd
import random
import os
import tensorflow as tf
import keras
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import sys
from keras import backend as K
import pickle


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)

# R square 함수
def R_Squared(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#초기 설정

fold = 5
input_data = './DB/DB.csv'
data = pd.read_csv(input_data)
data = data.dropna()
target_data = 'SH'
read = pd.read_csv("./ga/saving/"+target_data+'_Genetic_data.csv') # read.csv를 읽어서

targets_data = 'SH'

targets_data = targets_data.split('/')
data = data.dropna()
t_featurename__ = list(data.columns)
for item in targets_data:
    t_featurename__.remove(item)

y = data.loc[:,[target_data]].values
x = data.loc[:, t_featurename__].values



i = 1

for kf in range(1, fold):
    
    f = open('scaler', 'rb')    #이진 파일 open하여 scailing
    Scaler = pickle.load(f)
    f.close()

    #새 파일이 아니고 기존 파일에 계속 쓸라고 하니까 스케일링 에레 발생!!!!!!!!!!!!!!!!!!----------------------
    model = load_model('./kf_model/'+target_data+'_model_'+str(i)+'.h5', custom_objects={'R_Squared': R_Squared,}) #model load
    S_in = Scaler.transform(read)
    S_out = model.predict(S_in)
    # read[target_data] = S_out

    df_out = pd.DataFrame(S_out)
    df_out.to_csv('./ga/saving/'+target_data+'_data_predict'+str(i)+'.csv',index=False)    
    i +=1


# model을 통해, 2진 파일을 스케일링하여 가중치를 가지고 온 후
# 값을 예측함. 
# createFolder('./saving')
# Target을 포함한 Data값을 씀 

print('Predeiction of Data!')
