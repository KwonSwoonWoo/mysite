#1단 ML


import warnings
warnings.filterwarnings(action='ignore')
import tensorflow as tf
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
# import csv
import seaborn as sns
import os
# from sklearn.preprocessing import MinMaxScaler
import random
# import pickle
from keras import regularizers
# from sklearn.ensemble import RandomForestRegressor
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
import pickle
import sys
import numpy as np
import pandas as pd
from PyQt5.QtCore import QCoreApplication
# import time



act_func ='selu'
learning_rate = 0.0005   
epoch = 700          
bach_size = 32  
unit = 64

input_data = './DB/DB_1.csv'
data = pd.read_csv(input_data)

target_data ='SF1,SF2,SH1,SH2'
t_featurename__=list(data.columns)
for item in target_data.split(','):
    t_featurename__.remove(item)   

# ML - ML_Module
form_class = uic.loadUiType("ML_Module.ui")[0] 
class Module1(QMainWindow, form_class) :
    def __init__(self) :
        super().__init__()
        self.setupUi(self)
        self.lineEdit1.setText(input_data) #DB
        self.pushButton.clicked.connect(self.slot1)
        self.pushButton2.clicked.connect(self.showDialog)
        self.pushButton3.clicked.connect(self.ML)    
        self.lineEdit2.setText(target_data)
        self.pushButton4.clicked.connect(QCoreApplication.instance().quit)
    def showDialog(self):
        target, ok = QInputDialog.getText(self, 'Input Dialog', 'Enter the target:')
        self.lineEdit3.setText(target)
        global t_in
        t_in = target
    
    def slot1(self):
        data = pd.read_csv(self.lineEdit1.text())
    # def slot1(self):        
    #     try:
    #         data = pd.read_csv(self.lineEdit1.text()) 
    #     except FileNotFoundError:
    #         QMessageBox.about(self, "message", "DB명을 잘못 입력 하셨습니다. 종료합니다.")
    #         #sys.exit(app.exec_()) 
    #     t_featurename__ = list(data.columns)
    #     t_featurename = str(t_featurename__) 
    
    def quit(self):
        exit(0)

    def ML(self):
        def createFolder(directory):
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory)
            except OSError:
                print('Error: Creating directory. ' + directory)

        data = pd.read_csv(input_data)
        x = data.loc[:, t_featurename__]
        y = data.loc[:,t_in]
        
        x = x.values
        y = y.values
        
        X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state= 0)
        createFolder('./data/ML_SplitData')
        np.savetxt('./data/ML_SplitData/X_train.csv', X_train, delimiter=',',header=str(t_featurename__), fmt = '%.3f',comments='')
        np.savetxt('./data/ML_SplitData/X_test.csv', X_test, delimiter=',', header=str(t_featurename__),fmt = '%.3f',comments='')
        np.savetxt('./data/ML_SplitData/y_train.csv', y_train, delimiter=',',header=t_in, fmt = '%.3f',comments='') # Target
        np.savetxt('./data/ML_SplitData/y_test.csv', y_test, delimiter=',',header=t_in, fmt = '%.3f',comments='') # Target

        # 6. Scaling 
        if t_in == 'SF1':
            Scaler_SF1 = StandardScaler()
            Scaler_SF1.fit(x)
            X_train = Scaler_SF1.transform(X_train)
            X_test = Scaler_SF1.transform(X_test)
            createFolder('./Scaler')
            f = open('./Scaler/Scaler_SF1', 'wb') 
            pickle.dump(Scaler_SF1,f)
            f.close()

        elif t_in == 'SF2':
            Scaler_SF2 = StandardScaler()
            Scaler_SF2.fit(x)
            X_train = Scaler_SF2.transform(X_train)
            X_test = Scaler_SF2.transform(X_test)
            createFolder('./Scaler')
            f = open('./Scaler/Scaler_SF2', 'wb') 
            pickle.dump(Scaler_SF2,f)
            f.close()

        elif t_in == 'SH1':
            Scaler_SH1 = StandardScaler()
            Scaler_SH1.fit(x)
            X_train = Scaler_SH1.transform(X_train)
            X_test = Scaler_SH1.transform(X_test)
            createFolder('./Scaler')
            f = open('./Scaler/Scaler_SH1', 'wb') 
            pickle.dump(Scaler_SH1,f)
            f.close()
        elif t_in == 'SH2':
            Scaler_SH2 = StandardScaler()
            Scaler_SH2.fit(x)
            X_train = Scaler_SH2.transform(X_train)
            X_test = Scaler_SH2.transform(X_test)
            createFolder('./Scaler')
            f = open('./Scaler/Scaler_SH2', 'wb') 
            pickle.dump(Scaler_SH2,f)
            f.close()
        
        data = data.dropna()
        
        model = Sequential()
        model.add(Dense(unit,activation=act_func)) # ->l1정규화
        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))

        keras.layers.AlphaDropout(0.3, noise_shape=None, seed=None)

        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))

        keras.layers.AlphaDropout(0.3, noise_shape=None, seed=None)

        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))

        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))

        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))
        model.add(Dense(unit,activation=act_func))

        keras.layers.AlphaDropout(0.3, noise_shape=None, seed=None)

        # model.add(Dense(unit,activation=act_func,kernel_regularizer=regularizers.l2(0.001)))
        # model.add(Dense(unit,activation=act_func,kernel_regularizer=regularizers.l2(0.001)))
        # model.add(Dense(unit,activation=act_func,kernel_regularizer=regularizers.l2(0.001)))
        # model.add(Dense(unit,activation=act_func,kernel_regularizer=regularizers.l2(0.001)))

        # model.add(Dense(unit,activation=act_func,kernel_regularizer=regularizers.l2(0.001)))
        # model.add(Dense(unit,activation=act_func,kernel_regularizer=regularizers.l2(0.001)))

        model.add(Dense(1))

        def R_Squared(y_true, y_pred):
            from keras import backend as K
            SS_res =  K.sum(K.square( y_true-y_pred ))
            SS_tot = K.sum(K.square( y_true - K.mean(y_true)))
            return ( 1 - SS_res/(SS_tot + K.epsilon()) )

        X_train = np.array(X_train)
        y_train = np.array(y_train)



        model.compile(loss = 'mean_absolute_error', optimizer=keras.optimizers.Adam(learning_rate), metrics=['mae',R_Squared]) 
        history = model.fit(X_train, y_train, epochs=epoch, batch_size=32, validation_data=(X_test, y_test))
        
        x = model.evaluate(X_train, y_train)
        z = model.evaluate(X_test, y_test)
        
        print("MAE of train: ", x[1])
        print("R2 of train: ", x[2])
        
        print("MAE of test: ", z[1])
        print("R2 of test: ", z[2])



        createFolder('./models')
        model.save('./models/model_'+t_in+'.h5')
        print('Train'+t_in+'complete')
       


if __name__ == "__main__" :
    app = QApplication(sys.argv)
    myWindow = Module1()
    myWindow.show()
    app.exec_()

