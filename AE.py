import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
import seaborn as sns
from pylab import rcParams
from sklearn.model_selection import train_test_split
import os


from keras import regularizers

# Define Create Folder Function
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


df = pd.read_csv("data/creditcard.csv")

# print(df.shape)


# class plot
# label = ['normal','fraud']
# plt.xticks(range(2), label)
# normal = pd.value_counts(df['Class'])
# normal.plot(kind='bar',rot=0)
# plt.xticks(range(2),label)
# plt.show()

#Scaling
df = df.drop(['Time'], axis=1)
from sklearn.preprocessing import StandardScaler
df['Amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1, 1)) # ???

#split data
X_train, X_test = train_test_split(df, test_size=0.2,random_state=42)
featurename = list(df.columns)
#AE는 정상 데이터에 대해서만 학습(X_train의 Class ==0)
X_train = X_train.loc[X_train['Class']==0]
X_train = X_train.drop(['Class'],axis=1)

#X_test는 0 or 1
y_test = X_test['Class']
X_test = X_test.drop(['Class'], axis=1)


X_train = X_train.values
X_test = X_test.values

# print(X_test)
# print(X_train)

# Define model(AE)
epoch = 100
batch_size = 32
# input_dim은 학습의 2번쩨, 즉, 얼마만큼의 feature가 있는 지
input_dim = X_train.shape[1] 
print(input_dim)

encoding_dim = 14 # 왜 14지?


from keras.models import Model, load_model
from keras.layers import Input, Dense
input_layer = Input(shape=(input_dim, ))
#SET ENCODER
encoder = Dense(encoding_dim, activation='tanh')(input_layer)
encoder = Dense(int(encoding_dim/2),activation='relu')(encoder)

#SET DECODER
decoder = Dense(int(encoding_dim/2), activation='tanh')(encoder)
decoder = Dense(input_dim,activation='relu')(decoder) #input_dim으로 회귀. 

#Set AutoEncoder
autoencoder = Model(inputs=input_layer, outputs=decoder)

autoencoder.compile(optimizer='nadam',loss='mean_squared_error',metrics=['accuracy'])



from keras.callbacks import ModelCheckpoint, TensorBoard
checkpointer = ModelCheckpoint(filepath="./model/AEmodel.h5",
                               verbose=0,
                               save_best_only=True)
tensorboard = TensorBoard(log_dir='./logs',
                          histogram_freq=0,
                          write_graph=True,
                          write_images=True)
                               
history = autoencoder.fit(X_train, X_train, epochs=epoch, 
                          batch_size=batch_size,validation_data=(X_test, X_test), 
                          callbacks=[checkpointer])


#평가하기
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right')
plt.show()