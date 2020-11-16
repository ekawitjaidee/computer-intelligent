import numpy as np
import random
import matplotlib.pyplot as plt

def read_input(fn):
  f = open(fn,'r')
  # print(f.read())
  d=[]
  d.append(f.read().splitlines())
  # print(d)
  p = []
  xinput = []
  desire = []

  for i in range(len(d[0])):
    if i%3==0:
      p.append(d[0][i])
    elif i%3==1:
      xinput.append(d[0][i].split('  '))
    elif i%3==2:
      desire.append(d[0][i].split(' '))
  # print(p)
  # print(xinput)
  # print(desire)
  return xinput,desire

def str_tofloat(data):
  x = data.copy()
  return np.array(list(map(lambda s: list(map(float, s)), x)))

def normalize(d):
    x = d.copy()
    for i in x :
      if i[0] == 0:
        i[0] = 0.1
      else:
        i[0] = 0.9
        
      if i[1] == 0:
        i[1] = 0.1
      else:
        i[1] = 0.9
        
    return x

def initial_weigths(input_node,hidden_node,hidden_layer):
  node  = len(input_node[0])

  cp_node = []
  cp_node.append(node)

  [cp_node.append(i) for i in hidden_node]
  
  hidden_node.append(2)#append output node to make weigths

  layer_count = []

  # np.random.seed(1)
  for i in range(len(hidden_node)):
    weigths = 2 * np.random.random((cp_node[i], hidden_node[i])) - 1
    layer_count.append(weigths)
  
  return layer_count

def crossVar(data,data2,keeprandom):#สุ่ม10เปอเซ็นของข้อมูลเพื่อแบ่งเป็นtest
  d = data.copy()
  d2 = data2.copy()
  r = random.randrange(1,11)
 
  while r in keeprandom:
     r = random.randrange(1,11)
  keeprandom.append(r)
 
  test = []

  dis = round(len(d)/10)
  dist = dis * r   #ระยะของข้อมูลที่จะแบ่ง

  test = d[(dist-dis):dist]
  test2 = d2[(dist-dis):dist]
  print(dist)
  d = np.delete(d,slice((dist-dis),dist),axis=0)
  d2 = np.delete(d2,slice((dist-dis),dist),axis=0)

  return d,test,d2,test2

def initial_bias(hidden_node,hidden_layer):
  resbias = []
  for i in range(len(hidden_node)):
    bias = 2 * np.random.random((1, hidden_node[i])) - 1
    resbias.append(bias)

  return resbias

def randdata(data):
  d  = data.copy()
  resdatarand = []
  rang = int(len(d)/10)
  krang = []
  r2 = random.randrange(0,10)
  for i in range(10):#cross 10 part  
    while r2 in krang:
      r2 = random.randrange(0,10)
    krang.append(r2)
  resdata = []
  for i in krang:
    resdatarand.append(d[i*rang:(i*rang)+rang])
  for i in range(len(resdatarand)):
    for j in resdatarand[i]:
      resdata.append(j)

  return resdata

#NN
class NN():
  def __init__(self,weigths,hidden_node,hidden_layers,learningrate,bias,biasweight,momentumrate,deltaWl,deltabiasWl):
    self.hidden_node = hidden_node 
    self.hidden_layers = hidden_layers 
    self.weigths = weigths
    self.lr = learningrate
    self.bias = bias
    self.biasweight = biasweight
    self.momentumrate = momentumrate  
    self.deltaWl = deltaWl
    self.deltabiasWl = deltabiasWl
    

  def feedfoward(self,data):
   
    feedlayer = []
    dotres = data

    for dotloop in range(self.hidden_layers+1): # test case มีสองhiddenlayer แสดงว่าต้องdot3ครั้ง จึงเท่ากับhiddenlayer+1
      dotres = np.dot(dotres,self.weigths[dotloop]) 
      dotres = self.sigmoid(dotres)
      result = np.reshape(dotres,(len(dotres),1))
     
      feedlayer.append(result)

    return feedlayer #sigmoid at hidden node and output node

  def backward(self,yt,desire,xinput):
  
    gdkeep  =[]    
    deltaW = []
    deltaWbias = []

    tofinddeltaW = []
    [tofinddeltaW.append(i)for i in yt[:-1]]

    #gradient
    #re weigth before output node
    outset = yt[1:]
    outset = np.flip(outset)
    
    gdout = self.diffsigmoid(outset[0]) * self.errorrate(outset[0],desire)
    gdkeep.append(gdout)

    hiddenout = yt[:-1]
    hiddenout = np.flip(hiddenout)
    

    hdweight = np.flip(self.weigths)
  
    for i in range(len(hiddenout)):
      gdkeep.append(np.dot(hdweight[i] ,gdkeep[i]) * self.diffsigmoid(hiddenout[i])) 


    gdinput = gdkeep[-1]
    gdkeep.pop()  
    
    for i in range(len(tofinddeltaW)):  
      deltaW.append((-self.lr)*tofinddeltaW[i]*gdkeep[i])
    
    
    deltaW.append((-self.lr)*np.reshape(xinput,(len(xinput),1))*np.transpose(gdinput))
    deltaW = np.flip(deltaW)
   
    self.weigths = self.weigths 
    self.deltaWl = deltaW
  
    return self.weigths,self.biasweight,self.deltaWl,self.deltabiasWl

  def sigmoid(self,s):
    return 1/(1+np.exp(-s))

  def errorrate(self,yt,desire):
    et = desire-yt
    return et
  
  def diffsigmoid(self, s):
    return (s) * (1 - (s))



def predict(y):
  if y[0] > y[1]:
    predict = [0.9,0.1]
  else:
    predict = [0.1,0.9]
  return predict

def confusionMatrix(d,o):
  truepos,falsepos,falseneg,trueneg = 0,0,0,0
  for i in range(len(d)):
    if d[i][0]>0.5 and d[i][1]<0.5:
      if o[i][0]>0.5 and o[i][1]<0.5:
        truepos = truepos + 1
      else:
        falsepos = falsepos + 1
    else:
      if o[i] == [0.1, 0.9]:
        trueneg = trueneg + 1
      else:
        falseneg = falseneg + 1
  return truepos,falsepos,falseneg,trueneg

#Main
xinput,desire = read_input('cross.txt')

#input
hidden_layers = int(input('layer = '))
hidden_node = []

for i in range(hidden_layers):
  hidden_node.append(int(input('num node :'+str(i+1)+'  = ')))
print(hidden_node)
learningrate = float(input('learningrate = '))
momentumrate = float(input('momentumrate = ')) 
epoch = int(input('Epoch = '))

#variables
epoch = 100
hidden_layers = 2
hidden_node = [2,3]
learningrate = 0.02
momentumrate = 0.01 
oncescross  = epoch/10
epc = 0

errorcatch = []
keeprandom = []
errorcase = 0
rootmeansqure = 0
rms = []
sumsqureerror = []
avgsumsqure = []
avgsumsqureres = []
s =[]
ressumsqure = 0
deltaWl = 0
deltabiasWl = 0
pdValue = []
acc = []


errortest = []
avgerrortest = []

xinput = str_tofloat(xinput)
desire = str_tofloat(desire)
# desire = normalize(desire)

xtrain,xtest,dtrain,dtest = crossVar(xinput,desire,keeprandom)



weigths = initial_weigths(xtrain,hidden_node,hidden_layers) 

bias = initial_bias(hidden_node,hidden_layers)
biasweight = initial_bias(hidden_node, hidden_layers)


NN1 = NN(weigths,hidden_node,hidden_layers,learningrate,bias,biasweight,momentumrate,deltaWl,deltabiasWl)
# deltaWl = 1
# deltabiasWl = 1
# print('1123',dtrain[0])
# out = NN1.feedfoward(xtrain[0])
# outweights,outbiasweights,deltaWl,deltabiasWl = NN1.backward(out,dtrain[0],xtrain[0])
# print(out)


while epc<epoch:
  for j in range(int(oncescross)):
    for i in range(len(xtrain)):#train
      yt =  NN1.feedfoward(xtrain[i])
      outweights,outbiasweights,deltaWl,deltabiasWl = NN1.backward(yt,dtrain[i],xtrain[i])
      pdValue.append(predict(yt[-1]))
      errorcatch.append(NN1.errorrate(yt[-1],dtrain[i]))
      ertr = NN1.errorrate(yt[-1],dtrain[i])
      NN1 = NN(outweights,hidden_node,hidden_layers,learningrate,bias,outbiasweights,momentumrate,deltaWl,deltabiasWl)
    tp,fp,fn,tn = confusionMatrix(dtrain.tolist(),pdValue)
    print("confusionmatrix  0   1")
    print(' 0                 ',tp,'   ',fp)
    print(' 1                 ',fn,'   ',tn)
    acc.append(((tp+tn)/len(dtrain)*100))
    print('Accurancy',((tp+tn)/len(dtrain)*100),'%')
    
    xtrain = randdata(xtrain)
    epc = epc + 1 

    [ s.append((x**2)) for x in errorcatch ]
    sumsqureerror = sum(s)/2
    avgsumsqure.append(sumsqureerror)   
    print('sum squre'+str(sumsqureerror)) 
    
    rootmeansqure = np.sqrt(sum(s)/len(s))
    rms.append(rootmeansqure)     
    
    sumsqureerror = []
    s = []
    errorcatch = []

    print('epc'+str(epc))
    if epc==epoch:
      break
  
  avgsumsqureres.append(sum(avgsumsqure)/len(avgsumsqure))
  print('avgsumsqure',sum(avgsumsqure)/len(avgsumsqure))
  avgsumsqure = []

  for j in range(len(xtest)): #test
    yt = NN1.feedfoward(xtest[j])
    errortest.append(NN1.errorrate(yt[-1],dtest[j]))
    erts = NN1.errorrate(yt[-1],dtrain[i])
  [ s.append((x**2)) for x in errortest ]
  print('sumsqure TEST',(sum(s)/len(s)))
  avgerrortest.append(sum(s)/len(s))
  
  if len(keeprandom)==10: #to Cross validation
    keeprandom = []

  xtrain,xtest,dtrain,dtest = crossVar(xinput,desire,keeprandom) 
  xtrain = randdata(xtrain)
