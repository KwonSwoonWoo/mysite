# -*- coding: utf-8 -*- 
import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
import pandas as pd
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import plot_model
import keras
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import csv
import seaborn as sns
import os
import subprocess
import sys
from sklearn.preprocessing import MinMaxScaler
import random
import pickle
from keras import regularizers
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
import wandb
from wandb.keras import WandbCallback
from tensorflow.keras.callbacks import TensorBoard
from time import time

#init wandb0
import wandb
from wandb.keras import WandbCallback
# wandb.init(project="kiss")

# -- machine learning 파일에서 wandb 사용법 참조하여 시각화. 


# log_train = './log/train'
# logdir="logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# file_writer = tf.summary.create_file_writer(logdir+"/metrics")
# file_writer.set_as_default()






input_data = './DB/DB.csv'
target_data = 'SH'
x = 4  #변수 개 수. 
X_header = 'mn, pinion, g2, width'
y_header = 'SH'

act_func = 'selu'
learning_rate = 0.005 
bach_size = 32
target = 'SH'
NON = 8
NOL = 24
epochs = 1000 #
fold = 5
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


data = pd.read_csv(input_data)

t_featurename__=list(data.columns)
for item in target_data.split('/'): # TARGET이 여러 개 일 경우 '/' 로 target_data에 명시해야 함
    t_featurename__.remove(item)    # 예 target_data = 'SH/SF'


t_featurename=t_featurename__.copy()
t_featurename.append(target_data)
data = data.dropna()
print(t_featurename__) # without target featurename
print(t_featurename) 

x = data.loc[:, t_featurename__]
y = data.loc[:, [target_data]]

# save x, y
x.to_csv('./csv/x.csv',float_format = '%.2f',sep=',',index=False)
y.to_csv('./csv/y.csv',float_format = '%.2f',sep=',',index=False)

# x,y to numpy
x = x.values
y = y.values

# split data as train and test set
X_train_total, X_test_origin, y_train_total, y_test = train_test_split(x, y, train_size=0.8, shuffle=True, random_state= 0)
X_train_origin, X_val_origin, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state= 0)


#Save the split data
createFolder('./SplitData')
np.savetxt('./SplitData/X_train_total.csv', X_train_total, delimiter=',', fmt = '%.3f', header = X_header, comments='')
np.savetxt('./SplitData/X_test_origin.csv', X_test_origin, delimiter=',', fmt = '%.3f', header = X_header, comments='')
np.savetxt('./SplitData/y_train_total.csv', X_train_total, delimiter=',', fmt = '%.3f', header = y_header, comments='')
np.savetxt('./SplitData/y_test.csv', y_test, delimiter=',', fmt = '%.3f', header = y_header, comments='')
np.savetxt('./SplitData/X_train_origin.csv', X_train_origin, delimiter=',', fmt = '%.3f', header = X_header, comments='')
np.savetxt('./SplitData/X_val_origin.csv', X_val_origin, delimiter=',', fmt = '%.3f', header = X_header, comments='')
np.savetxt('./SplitData/y_train.csv', y_train, delimiter=',', fmt = '%.3f', header = y_header, comments='')
np.savetxt('./SplitData/y_val.csv', y_val, delimiter=',', fmt = '%.3f', header = y_header, comments='')




#Scaling
Scaler = StandardScaler()
Scaler.fit(X_train_origin)
X_train = Scaler.transform(X_train_origin)
X_val = Scaler.transform(X_val_origin)
X_test = Scaler.transform(X_test_origin)


#Scaling saving
np.savetxt('./SplitData/S_X_train.csv', X_train, delimiter=',', fmt = '%.3f', header = X_header, comments='')
np.savetxt('./SplitData/S_X_val.csv', X_val, delimiter=',', fmt = '%.3f', header = X_header, comments='')
np.savetxt('./SplitData/S_X_test.csv', X_test, delimiter=',', fmt = '%.3f', header = X_header, comments='')





f = open('scaler', 'wb') 
pickle.dump(Scaler,f)
f.close()

X_total = np.array(X_train_total)
y_total = np.array(y_train_total) 
X_train = np.array(X_train)
y_train = np.array(y_train)
X_val = np.array(X_val)
y_val = np.array(y_val)

tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='/logs')


def build_model():
    model = Sequential()
    model.add(Dense(NON,activation=act_func))
    #Sequential as relu function
    for i in range(0, NOL):
        model.add(Dense(NON,activation=act_func))
    tf.keras.layers.AlphaDropout(0.2, noise_shape=None, seed=None)
    #output as linear function
    model.add(Dense(1)) 
    model.compile(optimizer='adam', lr=0.0001, loss = 'mean_absolute_error',metrics=['mae',R_Squared])
    return model

def R_Squared(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


loo = LeaveOneOut()
loo.get_n_splits(x)
kf=KFold(n_splits=fold)
kf.get_n_splits(x)

i=1
number_loo = 1
number_kf = 1


from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=100, min_delta = 0.0001, verbose=1)

print("Enter the network model(loo or kf):")
net_model = input()

print("Enter the KFold Number(default:5):")
if net_model == 'kf':
    fold = input()

if net_model == 'loo':
    createFolder('loo_model')
    for train_idx, test_idx in loo.split(x,y):
        print('# of loo:', number_loo)
        X_total = np.array(X_total)
        y_total = np.array(y_train_total) 
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        model = build_model()
        hist = model.fit(X_total, y_total, epochs=epochs, batch_size=32, validation_data=(X_val, y_val))
        
        x = model.evaluate(X_val, y_val)
        print('MAE of test:', x[1])
        model.save('./loo_model/'+target+'_model_'+str(i)+'.h5')
        i +=1
        number_loo +=1
else:
   createFolder('kf_model')
   for train_idx, test_idx in kf.split(x,y): 
        print('# of Fold:', number_kf)
        X_total = np.array(X_total)
        y_total = np.array(y_train_total) 
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_val = np.array(X_val)
        y_val = np.array(y_val)

        model = build_model()
        hist = model.fit(X_total, y_total, epochs=epochs, batch_size=32, validation_data=(X_val, y_val))
        z = model.evaluate(X_val, y_val)
        print('MAE of test:', z[1])
        model.save('./kf_model/'+target+'_model_'+str(i)+'.h5')
        i +=1
        number_kf +=1



# KF model
# from sklearn.model_selection import KFold
# for train_idx, test_idx in kf.split(x,y):
#     X_total = np.array(X_total)
#     y_total = np.array(y_train_total) 
#     X_train = np.array(X_train)
#     y_train = np.array(y_train)
#     X_val = np.array(X_val)
#     y_val = np.array(y_val)

#     model = build_model()
#     hist = model.fit(X_total, y_total, epochs=epochs, batch_size=32, validation_data=(X_val, y_val))
#     model.save('./model_'+str(i)+'.h5')
#     i +=1