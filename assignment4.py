import numpy as np
import pandas as pd

#Preprocess
def normalization(x,minx,maxx):
    x = (x - minx) / (maxx - minx)
    return x

def initial_weigths(input_node,hidden_node,hidden_layer,output_node):
  node  = input_node

  cp_node = []
  cp_node.append(node)

  [cp_node.append(i) for i in hidden_node]
  
  hidden_node.append(output_node)#append output node to make weigths
  layer_count = []
  velo = []

  for i in range(len(hidden_node)):
    weigths = 2 * np.random.random((cp_node[i], hidden_node[i])) - 1
    velocity = 2 * np.random.random((cp_node[i], hidden_node[i])) - 1
    layer_count.append(weigths)
    velo.append(velocity)
  return layer_count,velo

def splitdata(data,k,Cross):
  length = int(len(data)/k)
  datacopy = data.copy()
  split = []
  g = []
  for i in range(k):
    g.append(datacopy[length*i:(length*(i+1))])
    split.append(g)
    g=[]
  # test = split[Cross]
  return split

def Crossvalidation(xcopy,dcopy,i):
  del xcopy[i]
  del dcopy[i]
  xtrain = sum(xcopy, []) 
  xtrain = np.concatenate(xtrain, axis=0)
  dtrain = sum(dcopy, [])
  dtrain = np.concatenate(dtrain, axis=0)
  return xtrain, dtrain
    

#NN
def NeuralNetwork(data,weights):
  hidden_layer = 2
  # print(weights)
  y = feedfoward(data,weights,hidden_layer)
 
  return y

def feedfoward(data,weights,hidden_layer):   
    feedlayer = []
    dotres = data

    for dotloop in range(hidden_layer+1): # test case มีสองhiddenlayer แสดงว่าต้องdot3ครั้ง จึงเท่ากับhiddenlayer+1
      # print(dotres,'======',weights[dotloop])
      dotres = np.dot(dotres,weights[dotloop]) 
      dotres = sigmoid(dotres)
      # result = np.reshape(dotres,(len(dotres),1))
     
      feedlayer.append(dotres)

    return feedlayer #sigmoid at hidden node and output node

def errorrate(yt,desire):
  et = desire-yt
  return et

def sigmoid(s):
  return 1/(1+np.exp(-s))






#Main
f = pd.read_csv("AirQualityUCI.csv")

x = f[['PT08.S1(CO)','PT08.S2(NMHC)',
            'PT08.S3(NOx)','PT08.S4(NO2)',
            'PT08.S5(O3)','T','RH','AH']]

desire= f[['C6H6(GT)']]

x,desire = np.array(x),np.array(desire)
minx,maxx = np.min(x),np.max(x)
x = normalization(x,minx,maxx)
mind,maxd = np.min(desire),np.max(desire)
desire  = normalization(desire,mind,maxd)

x,desire = pd.DataFrame(x),pd.DataFrame(desire)

d = desire.iloc[120:].reset_index(drop = True)  
x.drop(x.tail(120).index,inplace = True)

x,d = x.to_numpy(),d.to_numpy()


particle = 10
epoch = 10
bird = []
K = 10
Cross = 0

x = splitdata(x,K,Cross)
d = splitdata(d,K,Cross)

hidden_node = [2,3]
input_node = 8
output_node = 1
hidden_layer = 2
bird_weights,bird_velocity = [],[]
 
for i in range(particle):
  weights,velocity = initial_weigths(input_node,hidden_node,hidden_layer,output_node)
  bird_weights.append(weights)
  bird_velocity.append(velocity)

pbest = 100
lbest = 100
position = 100
# print(len(bird_weights))
for i in range(K):
  # print(type(x[i][0]))
 
  xcopy,dcopy = x.copy(),d.copy()
  xtest,dtest = xcopy[i],dcopy[i]
  xtrain,dtrain = Crossvalidation(xcopy,dcopy,i)
  for ep in range(epoch):
    bird = []
    ber = 0
    for b in range(particle):
      for train in range(len(xtrain)):
        yh = NeuralNetwork(xtrain[train],bird_weights[b])
        vh = NeuralNetwork(xtrain[train],bird_velocity[b])
        er = errorrate(yh[-1],dtrain[train])
        ber += abs(er)
      bird.append(ber/len(dtrain))
    # print(len(bird))
    xt = min(bird)
    # print(xt)
    print('|',ep,end=" ")
    if pbest>xt:
      pbest,lbest = xt,xt 
      position = bird_weights.copy()
    
    for k in range(particle):
      r1,r2 = np.random.uniform(0,1),np.random.uniform(0,1)
      c1,c2 = np.random.uniform(0,1),np.random.uniform(0,1)
      p1,p2 = c1*r1,c2*r2
      for m in range(len(bird_velocity[0])):
        bird_velocity[k][m] = (bird_velocity[k][m] 
                                + p1*(position[k][m] - bird_weights[k][m]) 
                                  + p2*(position[k][m] - bird_weights[k][m]))
        bird_weights[k][m] = bird_weights[k][m] + bird_velocity[k][m]

  bestbird = np.argmin(bird)
  bertest = []
  
  for test in range(len(xtest[0])):
    ytest = NeuralNetwork(xtest[0][test],bird_weights[bestbird])
    ert = errorrate(yh[-1],dtest[0][test])
    bertest.append(abs(ert))
  bertest = sum(bertest)/len(xtest[0])  
  print('Cross (',i+1,') Train Error = ',bird[bestbird],' Test Error = ',bertest)