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
 
#Define Global variables 

GMAX = 10          #GMAX : 최대 반복 값
CUG = 1            #현재 세대 값: 현재 반복된 값
gpool = 20         #pooling 횟수 
populating = 10    #인덱싱 횟수
width = 6
module_min =  1
module_max = 20
pinion = 10
safety_min = 0.9 
safety_max = 1.5 

#사용자 정의 함수 선언
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print('Error: Creating directory. ' + directory)


#0. Data load and scailing
input_data = './DB/DB.csv'
data = pd.read_csv(input_data)
target_data = 'SH'

t_featurename__=list(data.columns)
for item in target_data.split('/'):
    t_featurename__.remove(item)
t_featurename=t_featurename__.copy()
t_featurename.append(target_data)
data = data.dropna()

f = open('scaler', 'rb')    #이진 파일 open하여 scailing
Scaler = pickle.load(f)
f.close()

# R square 함수
def R_Squared(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

#1. Initialization
def func(a,b,c):   #min, max, n
    return [np.random.uniform(a,b) for i in range(c)]
def random_pool(n): 
    config_data = pd.read_csv('./CSV/data_minmax.csv') 
    print("input features:") 
    print(t_featurename)
    print("is equal to")
    print(config_data.columns)
    input("ENTER TO START>>")

    isSame=(~config_data.columns.isin(t_featurename__)).sum() # 열의 합계 계산.
    if isSame!=0: # data_minmax에 있는 혜더랑 t_featurename이랑 다르면 에러를 출력.
        print("Error!: DB 특성과 data_minmax.csv의 feature(Target제외)가 같지 않습니다! DB나 data_minmax.csv 값을 수정해 주세요")
        exit(0)  
    temp_init = {}
    for item in t_featurename__:
        temp_init.update({item:func(config_data.loc[0,item],config_data.loc[1,item],n)})  # update:딕셔너리 타입을 한 번에 바꿀 때 씀. 
    return pd.DataFrame(temp_init) 

# model = load_model('model.h5', custom_objects={'R_Squared': R_Squared,}) #model load

#2. selection 함수
def selection(gpool):
    scaled_g = pd.DataFrame(Scaler.transform(gpool)) 
    predicted_target = model.predict(scaled_g)                                                            
    #print ('selection')  
    selection1_pool1=pd.concat([pd.DataFrame(gpool),pd.DataFrame(predicted_target)],axis = 1)
    selection1_pool1.columns=t_featurename # 이름 붙임
    #input(selection1_pool1)
    selection1_pool1 = selection1_pool1.sort_values(by=target_data, ascending=False).iloc[:populating,:][target_data] #Target값 에 따라 Sorting하고, populating만큼 인덱싱 
    #input(selection1_pool1)
    selection1_pool1 = gpool.iloc[selection1_pool1.index,:] # index 함수

    temp_sel = []
    temp_sel.append(np.array(selection1_pool1)[0])                  #인덱싱, 무조건 일단 가장 점수가 높은 것을 내려 보냄. elite selection
    temp_sel.append(np.array(selection1_pool1)[random.randint(1,9)])#그리고 나머지 부모는 점수가 제일 높은 것 나머지에서 보냄

    return temp_sel
    
def crossover(selection1, selection2): 
    rc = random.random()
    
    GA_POOL = len(t_featurename__)    
    Point = random.randint(0,GA_POOL) 
     
    #Arithmetic selection
    for i in range(Point,GA_POOL):        
        selection1[i] = rc*selection1[i]+(1-rc)*selection2[i]
        selection2[i] = rc*selection2[i]+(1-rc)*selection1[i]
        
    return selection1,selection2

def r1():
    return random.random()
def r2():
    return random.random()

def mutations(mutation1,mutation2,cug,gmax):  
    b = 0.005
    CUG = cug
    GMAX = gmax
    fg = (r2()*(1-(CUG/GMAX)))**b
    
    temps = []
    mutation_data = pd.read_csv('./CSV/mutation_minmax.csv')
    for m in (mutation1, mutation2):
        for num ,name in enumerate(mutation_data.columns):
            if name =='SH':    # <----- Target 값 이전까지 끊어줘야 함. 
                break
            if r1() < 0.5:
                m[num] = m[num] + (mutation_data.loc[1,name] - m[num])*fg # 상한 값(a)에 더함. 
                # favg = (mutation_data.loc[1,name]+mutation_data.loc[0,name])/2.0            
            else:
                m[num] = m[num] - (m[num] + mutation_data.loc[0,name])*fg # 하한 값(b)에 더함
                # diff = (mutation_data.loc[1,name]-mutation_data.loc[0,name])/2.0
            
    temps.append(m)    
    return temps

#중요!d
            
gpool_population = random_pool(gpool) #최초 세대의 랜덤 pool 생성
result_data = pd.DataFrame({}, columns=t_featurename__)#
count=0 # 몇 개가 csv파일에 count 되었는 지 세기 위한 변수
facewidth_ratio = int(data.loc[0,'width'] / data.loc[0,'mn'])
print(facewidth_ratio)

while(GMAX > CUG): 
    if count > 1: # 추가
        break     # 추가

    Random_Model = len(t_featurename__)
    Model_number = random.randint(1,Random_Model)

    filename = './'+'kf_model/'+target_data+'_model_'+ str(Model_number) + '.h5'
    model = load_model(filename, custom_objects={'R_Squared': R_Squared})
    S_in = Scaler.transform(gpool_population) # 원본 데이터 스케일링
    S_out = model.predict(S_in)                  # 첫 랜덤 population의 값(TARGET 값) 예측
    gpool_population[target_data]=S_out
    
    ## 조건검사
    ## 파일에저장    
    

    #number_of_feature
    
    condition_SH = (safety_min < gpool_population.loc[:,target_data]).values & (gpool_population.loc[:,target_data] < safety_max ).values 
    gpool_population['pinion'] = gpool_population['pinion'].where(gpool_population['pinion'] > pinion,0) # 피니언 잇수가 'pinion' 보다 큰 값에 대해서는 원래 값들을 넣음. 작은 값들은 0처리 --> coonstraint1
    gpool_population['mn'] = gpool_population['mn'].where(gpool_population['mn'] > module_min,0) # 모듈이 'module_min' - 1  보다 큰 값에 대해서는 원래 값들을 넣음. 작은 값들은 0처리 --> coonstraint2
    gpool_population['mn'] = gpool_population['mn'].where(gpool_population['mn'] < module_max,0) # 모듈이 'module_max' - 20 보다 작은 값들에 대해서는 원래 값들을 넣음. 큰 값들에 대해서는 0처리 --> coonstraint2

    # 0인 값 Drop
    gpool_population[gpool_population=="0"]=None
    gpool_population = gpool_population.dropna(axis=0)
    # gpool_population = gpool_population.drop('0')


    if target_data =="SH": #타겟에 따른 조건 
        condition = condition_SH

    count+=sum(condition) 

    satisfied = gpool_population.loc[condition,t_featurename__]  
    gpool_population=gpool_population.drop(target_data,axis=1)  #  타겟포함인 값은 필요 없음. 
    result_data = result_data.append(satisfied)                 
    
    next_population_list  = [] 
   # print(next_population_list)
    for i in range(int(gpool/2)): # 
    
        #print(gpool_population.shape)    
        parent1, parent2 = selection(gpool_population) #selection 함수를 두 번 돌림. 
        
        mutation_probability = random.random()                 #mutaiton하거나 crossover할 확률

        if mutation_probability < 0.3:
            results = crossover(parent1,parent2)     
            print("Crossover")
            print(results)            
            
        else:
            results = mutations(parent1,parent2,CUG,GMAX) 
            print("Mutation")
            print(results)    

        for result in results:                         #결과 값들을 next_population_list에 append
            next_population_list.append(result)
    gpool_population=pd.DataFrame()
    
    for record in next_population_list:                # 1차원인 next_population_list를 2차원인 dataframe으로 변환하고 이름 붙임
        gpool_population = gpool_population.append(pd.DataFrame(record.reshape(1,-1))) 
    
    gpool_population.reset_index(inplace=True)
    gpool_population = gpool_population.loc[:,gpool_population.columns!='index']
    gpool_population.columns=t_featurename__           #이름 붙임


    print(gpool_population.info()) #C,Cr,V,Ti.....원소 null 값이 있는지 출력

    CUG = CUG +1
    if count > 1 :
        break

result_data = result_data[result_data.mn!=0] # 0 데이터 제거
result_data = result_data.drop_duplicates(keep = "last") # 중복 데이터 제거1

result_data = result_data[result_data.pinion!=0] # 0 데이터 제거
result_data = result_data.drop_duplicates(keep = "last") # 중복 데이터 제거2

result_data = result_data[result_data.g2!=0] # 0 데이터 제거
result_data = result_data.drop_duplicates(keep = "last") # 중복 데이터 제거3

result_data = result_data[result_data.width!=0] # 0 데이터 제거
result_data = result_data.drop_duplicates(keep = "last") # 중복 데이터 제거4

result_data = result_data.replace('0', pd.np.nan).dropna()


module = round(result_data['mn'],2)
GEAR = round(result_data['g2'],2)
pinion = round(result_data['pinion'],2)
WIDTH = result_data['width'].astype('int')


createFolder('./ga/saving')

# #--------------------최종 결과 Save--------------------#
new_result_data = pd.concat([module, pinion,GEAR, WIDTH],axis = 1)
new_result_data.to_csv("./ga/saving/"+target_data+'_Genetic_data.csv',index=False)






#####이하 VBA 아직 시도 중...##
# import xlwings as xw
# def VBAcommand(path, commands):
#     xlApp = xw.App(visible=True) #Macro 파일 보일지 말지
#     wb = xw.books.open(path) # 매크로 파일 경로
#     a = xlApp.api.Application.Run(commands) #명령어



# VBAcommand('./GearExample.xlsm','CalcGeo')
# wb1 = xw.Book('./GearExample.xlsm')

# for i in range(0, count-1):
#      Send_Data = []
#      pd.DataFrame(Send_Data)
#      Send_Data = data.iloc[i,:]
#      sht = wb1.sheets[0]
#      sht.range('M6').value = Send_Data
#      if VBAcommand('./GearExample.xlsm','CalcGeo'):
#          continue




# wb = xw.Book('./GearExample.xlsm')
# sht = wb.sheets[0]
# sht.range('M6').value = Send_Data  # >> 써지긴 하는데 세로로 써짐 -> 가로로 변경/ 인덱스 지워야 함. / 쓴 것 기준으로 아래에 써짐 
