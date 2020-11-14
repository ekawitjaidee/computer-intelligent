import random
import numpy as np
import matplotlib.pyplot as pl

# random.seed(0)
# np.random.seed(0)

#initial
def read_file(file):
  f = open(str(file),'r')
  t = []

  for x in f:
    t.append(x)
  del t[0] #ลบสองบรรทัดบนสุด
  del t[0]
  return t

def initial_weigths(input_node,hidden_node,hidden_layer):
  node  = len(input_node[0])

  cp_node = []
  cp_node.append(node)

  [cp_node.append(i) for i in hidden_node]
  
  hidden_node.append(1)#append output node to make weigths
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

#tools
def findmax2D(data):
  m = []
  for i in data:
    m.append(max(i))
  return max(m)

def findmin2D(data):
  m = []
  for i in data:
    m.append(min(i))
  return min(m)

#Data Manage
def crossVar(data,keeprandom):#สุ่ม10เปอเซ็นของข้อมูลเพื่อแบ่งเป็นtest
  d = data.copy()
  r = random.randrange(1,11)
 
  while r in keeprandom:
     r = random.randrange(1,11)
  keeprandom.append(r)
 
  test = []

  dis = round(len(d)/10)
  dist = dis * r   #ระยะของข้อมูลที่จะแบ่ง

  test = d[(dist-dis):dist]
  del d[(dist-dis):dist]

  return d,test
  
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

def splitdata(data):
  x = []
  r = []
  for i in data:
    x = i.split()
    r.append(x)
  return r

def splitIO(data):#แยกinputออกจากdesireoutput
  d = []
  desire = []
  for i in data:
    desire.append(i[8])
    d.append(i[:8])
  # print(str(d[-1])+str(desire[-1]))
  return d,desire

def retype(data):#change str to float
  return list(map(lambda sl: list(map(float, sl)), data))  

def Normalization(data,M,m):
  data = np.array(data)
  res = (data -m)/(M-m)
  return res.tolist()

def toarr(data):
  return np.array(data) 



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

    # print(deltaW)  
    if self.deltaWl == 0:
      self.deltaWl = deltaW


    self.weigths = self.weigths +(self.momentumrate*(deltaW-self.deltaWl))+deltaW
    self.deltaWl = deltaW
  
    return self.weigths,self.biasweight,self.deltaWl,self.deltabiasWl

  def sigmoid(self,s):
    return 1/(1+np.exp(-s))

  def errorrate(self,yt,desire):
    et = desire-yt
    return et
  
  def diffsigmoid(self, s):
    return (s) * (1 - (s))

      
      
#main

#input
hidden_layers = int(input('layer = '))
hidden_node = []

for i in range(hidden_layers):
  hidden_node.append(int(input('num node :'+str(i+1)+'  = ')))
print(hidden_node)
learningrate = float(input('learningrate = '))
momentumrate = float(input('momentumrate = ')) 
epoch = int(input('Epoch = '))

# epoch = 100
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


# hidden_layers = 4
# hidden_node = [5,3,4,2]
# learningrate = 0.02
# momentumrate = 0.01 

errortest = []
avgerrortest = []

deltaWl = 0
deltabiasWl = 0

#data
data = splitdata(read_file('Flood_dataset.txt'))
data = retype(data) #str to node

#keep min,max of data
MAX = findmax2D(data)
MIN = findmin2D(data)

#change data formation
data = Normalization(data,MAX,MIN) # normalize data to value in data in -1 to 1 
train,test = crossVar(data,keeprandom) #Crossvalidation split test train  (test 1 in 10) (train 9 in 10)
train = randdata(train)
train_x,train_y = splitIO(train)
train_x = toarr(train_x)
train_y = toarr(train_y)
test_x,test_y = splitIO(test)
test_x = toarr(test_x)
test_y = toarr(test_y)


weigths = initial_weigths(train_x,hidden_node,hidden_layers) #(data,hiddennode = [2,2],hiddenlayer = 2)
print(weigths)
bias = initial_bias(hidden_node,hidden_layers)
biasweight = initial_bias(hidden_node, hidden_layers)

#NN
NN1 = NN(weigths,hidden_node,hidden_layers,learningrate,bias,biasweight,momentumrate,deltaWl,deltabiasWl)
deltaWl = 1
deltabiasWl = 1

while epc<epoch:
  for j in range(int(oncescross)):
    for i in range(len(train_x)):#train
      yt =  NN1.feedfoward(train_x[i])
      outweights,outbiasweights,deltaWl,deltabiasWl = NN1.backward(yt,train_y[i],train_x[i])
      errorcatch.append(NN1.errorrate(yt[-1],train_y[i]))
      ertr = NN1.errorrate(yt[-1],train_y[i])
      NN1 = NN(outweights,hidden_node,hidden_layers,learningrate,bias,outbiasweights,momentumrate,deltaWl,deltabiasWl)
    train = randdata(train)
    train_x,train_y = splitIO(train)
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

  for j in range(len(test_x)): #test
    yt = NN1.feedfoward(test_x[j])
    errortest.append(NN1.errorrate(yt[-1],test_y[j]))
    erts = NN1.errorrate(yt[-1],train_y[i])
    if abs(errortest[0]) < 0.00001:
      print('erroe case')
      errorcase = errorcase + 1
      break
  [ s.append((x**2)) for x in errortest ]
  print('sumsqure TEST',(sum(s)/len(s)))
  avgerrortest.append(sum(s)/len(s))
  
  if len(keeprandom)==10: #to Cross validation
    keeprandom = []

  train,test = crossVar(data,keeprandom) 
  train = randdata(train)
  train_x,train_y = splitIO(train)
  train_x = toarr(train_x)
  train_y = toarr(train_y)
  test_x,test_y = splitIO(test)
  test_x = toarr(test_x)
  test_y = toarr(test_y)
# print('rts 10 Cross',rms)
# print(sum(rms)/len(rms))
print('avgsumsqure',avgsumsqureres)
print('avgtest',avgerrortest)

y = []
[y.append(i[0]) for i in avgsumsqureres]
# print("y",y)
x = [1,2,3,4,5,6,7,8,9,10]
pl.plot(x,y,label='Train',color='deepskyblue',marker='o',markerfacecolor='green',markersize=8)
pl.legend()
pl.grid()
pl.xlabel('Iteration per cross')
pl.ylabel('Error')
pl.show()



