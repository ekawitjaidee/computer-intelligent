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

def initial_weigths(input_node,hidden_node,hidden_layer):
  node  = len(input_node[0])

  cp_node = []
  cp_node.append(node)

  [cp_node.append(i) for i in hidden_node]
  
  hidden_node.append(2)#append output node to make weigths
  # print(cp_node)
  # print(hidden_node)

  layer_count = []

  # np.random.seed(1)
  for i in range(len(hidden_node)):
    weigths = 2 * np.random.random((cp_node[i], hidden_node[i])) - 1
    # print(weigths)
    layer_count.append(weigths)
  
  return layer_count

def initial_bias(hidden_node,hidden_layer):
  # print(hidden_node)
  resbias = []
  for i in range(len(hidden_node)):
    bias = 2 * np.random.random((1, hidden_node[i])) - 1
    # print(bias.shape)
    resbias.append(bias)
  # print(resbias)
  return resbias


class NN():
  def __init__(self,weigths,hidden_node,hidden_layers,learningrate,bias,biasweight,momentumrate,deltaWl,deltabiasWl):
    self.hidden_node = hidden_node #[2,2]
    self.hidden_layers = hidden_layers #2
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
      dotres = dotres +  self.biasweight[dotloop][0]
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
    tofinddeltaW = np.flip(tofinddeltaW)

    #gradient
    #re weigth before output node
    ytflip = np.flip(yt)
    outset = yt[1:]
    outset = np.flip(outset)
    
    gdout = self.diffsigmoid(outset[0]) * self.errorrate(outset[0],desire)
    gdkeep.append(gdout)

    hiddenout = yt[:-1]
    hiddenout = np.flip(hiddenout)
    
    gdhd = []
    hdweight = np.flip(self.weigths)
  
    for i in range(len(hiddenout)):
      gdkeep.append(np.dot(hdweight[i] ,gdkeep[i]) * self.diffsigmoid(hiddenout[i])) 
 
    gdbias = np.flip(gdkeep)
  
    for i in range(len(gdbias)):
      deltaWbias.append((-self.lr)*gdbias[i][0]*self.biasweight[i])
  

    if self.deltabiasWl == 0:
      self.deltabiasWl = deltaWbias

    resbiasW = []
    for i in range(len(self.biasweight)):
      biasWres = self.biasweight[i] +(self.momentumrate*(deltaWbias[i]-self.deltabiasWl[i]))+deltaWbias[i]
      resbiasW.append(biasWres)

    self.deltabiasWl = deltaWbias
    self.biasweight = resbiasW
    rebiasW = []

    gdinput = gdkeep[-1]
    gdkeep.pop()  
    
    for i in range(len(tofinddeltaW)):  
      deltaW.append((-self.lr)*tofinddeltaW[i]*np.transpose(gdkeep[i]))
    
    deltaW.append((-self.lr)*np.reshape(xinput,(len(xinput),1))*np.transpose(gdinput))
    deltaW = np.flip(deltaW)
      
    # if self.deltaWl == 0:
    #   self.deltaWl = deltaW
 
    self.weigths = self.weigths +(self.momentumrate*(deltaW-self.deltaWl))+deltaW
    self.deltaWl = deltaW
  
    return self.weigths,self.biasweight,self.deltaWl,self.deltabiasWl

  def sigmoid(self,s):
    return 1/(1+np.exp(-s))

  def errorrate(self,yt,desire):
    et = desire-yt
    return et

  def errorcon(self,yt,desire):
    et = desire[0]-yt[0]
    et1 = desire[1]-yt[1]
    return et  ,et1
  
  def diffsigmoid(self, s):
    return (s) * (1 - (s))

def predict(y):
  if y[0] > y[1]:
    o = [0.9,0.1]
  else:
    o = [0.1,0.9]
  return o

def confusion(d,o):
  tp,fp,fn,tn = 0,0,0,0
  for i in range(len(d)):
    if d[i] == [0.9, 0.1]:
      if o[i] == [0.9, 0.1]:
        tp = tp + 1
      else:
        fp = fp + 1
    else:
      if o[i] == [0.1, 0.9]:
        tn = tn + 1
      else:
        fn = fn + 1
  return tp,fp,fn,tn

#Main
xinput,desire = read_input('cross.txt')
#variables
epoch = 100
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

hidden_layers = 2
hidden_node = [2,2]
learningrate = 0.02
momentumrate = 0.01 
errortest = []
avgerrortest = []

#do
xinput = str_tofloat(xinput)
desire = str_tofloat(desire)
desire = normalize(desire)
print('xt',len(xinput))
print('xtest',len(desire))
xtrain,xtest,dtrain,dtest = crossVar(xinput,desire,keeprandom)

print('xt',len(xtrain))
print('xtest',len(xtest))


weigths = initial_weigths(xtrain,hidden_node,hidden_layers) #(data,hiddennode = [2,2],hiddenlayer = 2)
print(type(weigths[0]))
bias = initial_bias(hidden_node,hidden_layers)
biasweight = initial_bias(hidden_node, hidden_layers)


NN1 = NN(weigths,hidden_node,hidden_layers,learningrate,bias,biasweight,momentumrate,deltaWl,deltabiasWl)
deltaWl = 1
deltabiasWl = 1

out = NN1.feedfoward(xtrain[0])
# print(out)
while epc<epoch:
  for j in range(int(oncescross)):
    for i in range(len(xtrain)):#train
      yt =  NN1.feedfoward(xtrain[i])
      outweights,outbiasweights,deltaWl,deltabiasWl = NN1.backward(yt,dtrain[i],xtrain[i])
      print(yt[-1],dtrain[i])
      pdValue = predict(yt[-1])
      errorcatch.append(NN1.errorrate(yt[-1],dtrain[i]))
      ertr = NN1.errorrate(yt[-1],dtrain[i])
      NN1 = NN(outweights,hidden_node,hidden_layers,learningrate,bias,outbiasweights,momentumrate,deltaWl,deltabiasWl)
    xtrain = randdata(xtrain)
    # xtrain,train_y = splitIO(train)
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
    # if abs(errortest[0]) < 0.00001:
    #   print('erroe case')
    #   errorcase = errorcase + 1
    #   break
  [ s.append((x**2)) for x in errortest ]
  print('sumsqure TEST',(sum(s)/len(s)))
  avgerrortest.append(sum(s)/len(s))
  
  if len(keeprandom)==10: #to Cross validation
    keeprandom = []

  xtrain,xtest,dtrain,dtest = crossVar(xinput,desire,keeprandom) 
  xtrain = randdata(xtrain)
  # xtrain,dtrain = splitIO(train)
  # xtest,test_y = splitIO(test)
  
# print('rts 10 Cross',rms)
# print(sum(rms)/len(rms))
print('avgsumsqure',avgsumsqureres)
print('avgtest',avgerrortest)