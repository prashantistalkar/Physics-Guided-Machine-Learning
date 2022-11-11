# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 01:40:43 2021

@author: prashant
contact : istalkarps@gmail,com
this code predict 10 days of recession flow using PhyLSTM model

"""

# clearing the previous variables 
from IPython import get_ipython
get_ipython().magic('reset -sf')


##########################################################################################################
# to generate same number each time 
import numpy as np
import tensorflow as tf
np.random.seed(1234)
tf.random.set_seed(1234)
###############################################################################################################

import os 
import glob
import pandas as pd
import numpy 
import numpy as np

#########################################################################################################################
# function to find recession event and previous days 
" this code finds the recession events with length of 10 days(excluding peak. The 'Previous-event' seperate out the days before recesseio evert including the peak"
" The last value of previous day is peak value of hydrograph . the input data is 4 column. Last column is Q "

def Recession_function(data,prev_day):
    #prev_day=30 # nedd to delete 
    Q=data[:,3]
    temp_Q = np.zeros((len(Q),2))
    temp_Q[:,0]=Q ;
    Recession_event=np.zeros((10,len(Q)))
    Previous_event=np.zeros((prev_day,len(Q)))
    i=0
    a=0 ;
    while i<len(Q)-1:
        b=1 
        while i<len(Q)-1 and  Q[i,] > Q[i+1,] and  Q[i,]>0 and  Q[i+1,]>0    :
            b+=1 
            i+=1
        if b>10:
               temp_Q[i-b+2:i+1,1]=temp_Q[i-b+2:i+1,0]
               if i-b+2>125:                                             # 125 is to find conistant recession event when prev day is 5,10,30,60 and 120
                  Recession_event[:,a]=temp_Q[i-b+2:i-b+2+10,0] 
                  Previous_event[:,a]=temp_Q[i-b-(prev_day-2):i-b+2,0]
                  a+=1
        i+=1
        
        
    b=  np.where(~Recession_event.any(axis=0))[0]
    b=numpy.transpose(b)
    Recession_event=np.delete(Recession_event,b,1)
    Previous_event=np.delete(Previous_event,b,1)          
    return Recession_event,Previous_event
      
##########################################################################################################################          
## Deviding traning and testing data
" this devides the data into train and test . Percentage is in fraction "
def Train_test(data,percentage):
    total_length=data.shape[1]
    training_length=round(total_length*percentage)       
    Train_data=data[:,0:training_length]    
    Test_data=data[:,training_length:total_length]
    Train_data=np.transpose(Train_data)
    Test_data=np.transpose(Test_data)
    return Train_data, Test_data

    
############################################################################################################################
## varaprasad method
" this uses the varaprasad method to find the mean. Prev 6 days means, prev 5 days and peak and similar for others "
" it compare the mean of 'prev-days' for diffrerent previous days 5, 10,30,60,120.(here peak is not incuded that is why less by one" 
def varaprasad(data,prev_days):
    temp=np.zeros((len(data),2))
    if prev_days==6:
       for  i in range(0,len(data),1):
          if data[i,-2]<numpy.mean(data[i,-6:-1]):
             temp[i,0]=data[i,-2]
             temp[i,1]=data[i,-1]
          else:
              temp[i,0]=numpy.mean(data[i,-6:-1])
              temp[i,1]=data[i,-1]
    elif prev_days==11:
         for  i in range(0,len(data),1):
            if data[i,-2]<numpy.mean(data[i,-6:-1]):
             temp[i,0]=data[i,-2]
             temp[i,1]=data[i,-1]
            elif numpy.mean(data[i,-6:-1]) < numpy.mean(data[i,-11:-1]) :
              temp[i,0]=numpy.mean(data[i,-6:-1])
              temp[i,1]=data[i,-1] 
            else:
              temp[i,0]=numpy.mean(data[i,-11:-1])
              temp[i,1]=data[i,-1] 
    elif prev_days==31:
           for  i in range(0,len(data),1):
            if data[i,-2]<numpy.mean(data[i,-6:-1]):
             temp[i,0]=data[i,-2]
             temp[i,1]=data[i,-1]
            elif numpy.mean(data[i,-6:-1]) < numpy.mean(data[i,-11:-1]) :
              temp[i,0]=numpy.mean(data[i,-6:-1])
              temp[i,1]=data[i,-1] 
            elif numpy.mean(data[i,-11:-1]) < numpy.mean(data[i,-31:-1]) :
              temp[i,0]=numpy.mean(data[i,-11:-1])
              temp[i,1]=data[i,-1] 
            else:
               temp[i,0]=numpy.mean(data[i,-31:-1])
               temp[i,1]=data[i,-1] 
    elif prev_days==61:
        for  i in range(0,len(data),1):
            if data[i,-2]<numpy.mean(data[i,-6:-1]):
             temp[i,0]=data[i,-2]
             temp[i,1]=data[i,-1]
            elif numpy.mean(data[i,-6:-1]) < numpy.mean(data[i,-11:-1]) :
              temp[i,0]=numpy.mean(data[i,-6:-1])
              temp[i,1]=data[i,-1] 
            elif numpy.mean(data[i,-11:-1]) < numpy.mean(data[i,-31:-1]) :
              temp[i,0]=numpy.mean(data[i,-11:-1])
              temp[i,1]=data[i,-1] 
            elif numpy.mean(data[i,-31:-1]) < numpy.mean(data[i,-61:-1]):
               temp[i,0]=numpy.mean(data[i,-31:-1])
               temp[i,1]=data[i,-1]  
            else:
                temp[i,0]=numpy.mean(data[i,-61:-1])
                temp[i,1]=data[i,-1]
    elif prev_days==121:
        for  i in range(0,len(data),1):
            if data[i,-2]<numpy.mean(data[i,-6:-1]):
             temp[i,0]=data[i,-2]
             temp[i,1]=data[i,-1]
            elif numpy.mean(data[i,-6:-1]) < numpy.mean(data[i,-11:-1]) :
              temp[i,0]=numpy.mean(data[i,-6:-1])
              temp[i,1]=data[i,-1] 
            elif numpy.mean(data[i,-11:-1]) < numpy.mean(data[i,-31:-1]) :
              temp[i,0]=numpy.mean(data[i,-11:-1])
              temp[i,1]=data[i,-1] 
            elif numpy.mean(data[i,-31:-1]) < numpy.mean(data[i,-61:-1]):
               temp[i,0]=numpy.mean(data[i,-31:-1])
               temp[i,1]=data[i,-1]  
            elif numpy.mean(data[i,-61:-1]) < numpy.mean(data[i,-121:-1]):
                temp[i,0]=numpy.mean(data[i,-61:-1])
                temp[i,1]=data[i,-1]
            else:
                temp[i,0]=numpy.mean(data[i,-121:-1])
                temp[i,1]=data[i,-1]
    return temp  

def Normalize_data(data):
    mean_data=numpy.mean(data)
    std_data=numpy.std(data)
    data_norm=(data-mean_data)/std_data
    return data_norm,mean_data, std_data

def rescale_nromalisation(data_norm,mean_data,std_data):
    data=data_norm*std_data+mean_data
    return data
                
############################################################################################################################    
## LSTM function 
import os
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import RepeatVector
import pandas
from keras.layers import Dropout
from keras import callbacks 
from tensorflow import keras
import pandas as pd
#  LSTM architecture with encoder-decoder structure
def LSTM_function(XTrain,YTrain,XTest,sample_size_train,sample_size_test,input_seq_len):
    XTrain=XTrain.reshape(sample_size_train,input_seq_len,1)
    YTrain=YTrain.reshape(sample_size_train,10,1)
    XTest=XTest.reshape(sample_size_test,input_seq_len,1)
    model = Sequential()
    model.add(LSTM(64,recurrent_dropout=0.1,input_shape=(input_seq_len, 1)))
    model.add(RepeatVector(10))
    model.add(LSTM(64,recurrent_dropout=0.1, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(1)))
    callback =callbacks.EarlyStopping(monitor='loss', patience=10)
    opt = keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss= 'mse' , optimizer= opt)
    model.fit(XTrain, YTrain, batch_size=8, epochs=1000,callbacks=[callback])
    yhat = model.predict(XTest, verbose=0)
    YTest=yhat.reshape(sample_size_test,10)
    return YTest     
#########################################################################################################################
#NSE
def nse_function(sim,obs):
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator
    return nse_val
def log_nse_function(sim,obs):
    sim[sim<0]=0.01
    sim=np.log(sim)
    obs=np.log(obs)
    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator
    return nse_val 
################################################################################################
# calling functions
import os 
os.chdir(r"E:\Prashant\LSTM run\Revision\Varaprasad") 
path = os.getcwd()
filenames= glob.glob(path + "/*.csv")

Previous_days=121  # including peak 
percentage_fraction=0.6 
input_seq_len=2
nse=np.zeros((len(filenames),1))
nse_log=np.zeros((len(filenames),1))

for i in range(0,len(filenames),1):
    data=numpy.array(pd.read_csv(filenames[i],header=None))
    [Recession,Previous]=Recession_function(data,Previous_days)
    [xtrain,xtest]=Train_test(Previous,percentage_fraction)
    [ytrain,ytest]=Train_test(Recession,percentage_fraction)    

    # mean and std
    [_,xtrain_mean,xtrain_std]=Normalize_data(xtrain)
    [ytrain_norm,ytrain_mean,ytrain_std]=Normalize_data(ytrain)
         
    # varaprad method 
    xtrain=varaprasad(xtrain,Previous_days)
    xtest=varaprasad(xtest,Previous_days)
    xtrain_norm=(xtrain-xtrain_mean)/xtrain_std
    xtest_norm=(xtest-xtrain_mean)/xtrain_std
    
    size=len(xtrain_norm)
    size_test=len(xtest_norm)
    
    y_lstm_norm=LSTM_function(xtrain_norm,ytrain_norm,xtest_norm,size,size_test,input_seq_len)
    y_lstm=(y_lstm_norm*ytrain_std)+ytrain_mean

    pred=y_lstm.reshape(y_lstm.size)
    obs=ytest.reshape(ytest.size)
    nse[i,0]=nse_function(pred, obs)
    nse_log[i,0]=log_nse_function(pred, obs)
    del data,xtrain,xtest,ytrain,ytest,y_lstm,pred,obs
    
    
    
    
