# -*- coding: utf-8 -*- 
import warnings
warnings.filterwarnings(action='ignore')
import numpy as np
import pandas as pd
import os
from keras.models import load_model
import sys
import pickle


# 사용자 함수 정의
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
# 안전율 조건
SF1_MIN=0.9
SF1_MAX=1.2

SF2_MIN=0.9
SF2_MAX=1.2

SH1_MIN=0.9
SH1_MAX=1.2

SH2_MIN=0.9
SH2_MAX=1.2
# 안전율 조건

#초기 설정
input_data_1 = 'C:/GEAR_ML1/DB/DB_1.csv'
data = pd.read_csv(input_data_1)
data = data.dropna()
target_data = 'SF1'
targets_data = 'SF1,SF2,SH1,SH2'
read = pd.read_csv('C:/GEAR_ML1/ML_Final_data.csv') 
# Machine learning model load
model_1 = load_model('./models/model_SF1.h5', custom_objects={'R_Squared': R_Squared,}) #model load(SF1)
model_2 = load_model('C:/GEAR_ML1/models/model_SF2.h5', custom_objects={'R_Squared': R_Squared,}) #model load(SF2)
model_3 = load_model('C:/GEAR_ML1/models/model_SH1.h5', custom_objects={'R_Squared': R_Squared,}) #model load(SH1)
model_4 = load_model('C:/GEAR_ML1/models/model_SH2.h5', custom_objects={'R_Squared': R_Squared,}) #model load(SH2)

# TARGET 거르기
targets_data = targets_data.split(',')
data = data.dropna() 
t_featurename__ = list(data.columns)
for item in targets_data:
    t_featurename__.remove(item)


# Scaler들을 오픈
f = open('./Scaler_SF1', 'rb')    
Scaler_SF1 = pickle.load(f)
f.close()
f = open('C:/GEAR_ML1/Scaler/Scaler_SF2', 'rb')    
Scaler_SF2 = pickle.load(f)
f.close()
f = open('C:/GEAR_ML1/Scaler/Scaler_SH1', 'rb')    
Scaler_SH1 = pickle.load(f)
f.close()
f = open('C:/GEAR_ML1/Scaler/Scaler_SH2', 'rb')    
Scaler_SH2 = pickle.load(f)
f.close()

val = 'SH2'
# 10 열 예측
if val == 'SH2': 
    S_in_1 = Scaler_SF1.transform(read) 
    S_out_1 = model_1.predict(S_in_1)

    S_in_2 = Scaler_SF2.transform(read) 
    S_out_2 = model_2.predict(S_in_2)
    S_in_3 = Scaler_SH1.transform(read)  
    S_out_3 = model_3.predict(S_in_3)
    S_in_4 = Scaler_SH2.transform(read)  
    S_out_4 = model_4.predict(S_in_4)

    SF1_DF = pd.DataFrame(S_out_1, columns = ['SF1'])
    SF2_DF = pd.DataFrame(S_out_2, columns = ['SF2'])
    SH1_DF = pd.DataFrame(S_out_3, columns = ['SH1'])    
    SH2_DF = pd.DataFrame(S_out_4, columns = ['SH2'])

    result = pd.concat([read, SF1_DF,SF2_DF,SH1_DF,SH2_DF],axis = 1)
    # 예측값이 MIN(1.0) 보다 작으면 예측 값을 0 처리
    # 예측값이 MAX(1.2) 보다 크면 예측 값을 0처리
    # 0인 데이터를 Drop

    # result['SF1'] = result['SF1'].where(result['SF1'] > SF1_MIN,0) 
    # result['SF1'] = result['SF1'].where(result['SF1'] < SF1_MAX,0)     
    # result['SF2'] = result['SF2'].where(result['SF2'] > SF2_MIN,0)    
    # result['SF2'] = result['SF2'].where(result['SF2'] < SF2_MAX,0) 
    # result['SH1'] = result['SH1'].where(result['SH1'] > SH1_MIN,0)    
    # result['SH1'] = result['SH1'].where(result['SH1'] < SH1_MAX,0) 
    # result['SH2'] = result['SH2'].where(result['SH2'] > SH2_MIN,0)    
    # result['SH2'] = result['SH2'].where(result['SH2'] < SH2_MAX,0) 
    result[result=="0"]=None
    result = result.dropna(axis=0)
    result = result[result.SF1!=0]
    result = result[result.SF2!=0]
    result = result[result.SH1!=0]
    result = result[result.SH2!=0]
    

    createFolder('./prediction')
    result.to_csv('./prediction/SF1_SF2_SH1_SH2_Prediction.csv',index=False)
    # print('./prediction/SF1_SF2_SH1_SH2_Prediction.csv을 확인하세요')

else:
    pass
    exit(0)
    
