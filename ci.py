import random
import numpy as np



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
  r = random.randrange(0,10)
 
  while r in keeprandom:
     r = random.randrange(0,10)
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
  # MAX = findmax2D(data)
  # return [[sublst/MAX for sublst in lst]for lst in data]
  data = np.array(data)
  res = (data -m)/(M-m)
  return res.tolist()

def toarr(data):
  return np.array(data) 



#NN
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

  def backward(self,yt,desire,xinput,firstround):

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
  

    # print('selfbias'+str(self.biasweight))
    # print('deltabias'+str(deltaWbias))
    if self.deltabiasWl == 0:
      print(1)
      self.deltabiasWl = deltaWbias

    resbiasW = []
    for i in range(len(self.biasweight)):
      # print('selfbiasinside'+str(self.biasweight[i]))
      # print('deltabiasinside'+str(deltaWbias[i]))
      biasWres = self.biasweight[i] +(self.momentumrate*(deltaWbias[i]-self.deltabiasWl[i]))+deltaWbias[i]
      resbiasW.append(biasWres)
    self.deltabiasWl = deltaWbias
    self.biasweight = resbiasW
    # print('selfbias1'+str(self.biasweight))
    rebiasW = []


    gdinput = gdkeep[-1]
    gdkeep.pop()  
    
    # print(self.biasweight)
    for i in range(len(tofinddeltaW)):  
      deltaW.append((-self.lr)*tofinddeltaW[i]*np.transpose(gdkeep[i]))
    
    deltaW.append((-self.lr)*np.reshape(xinput,(len(xinput),1))*np.transpose(gdinput))
    deltaW = np.flip(deltaW)
      
    if self.deltaWl == 0:
      print(2)
      self.deltaWl = deltaW
 
    self.weigths = self.weigths +(self.momentumrate*(deltaW-self.deltaWl))+deltaW
    self.deltaWl = deltaW
  
    return self.weigths,self.biasweight,self.deltaWl,self.deltabiasWl

  def sigmoid(self,s):
    return 1/(1+np.exp(-s))

  def errorrate(self,yt,desire):
    et = desire-yt
    # print(et)
    return et
  
  def diffsigmoid(self, s):
    return (s) * (1 - (s))

      



      
#main

#input
# hidden_layers = int(input('layer = '))
# hidden_node = []
# for i in range(hidden_layers):
#   hidden_node.append(int(input('num node'+str(i+1)+'  = ')))
# print(hidden_node)
epoch = 100
oncescross  = epoch/10
epc = 0
errorcatch = []
reserror = []
keeprandom = []
errorcase = 0

sumsqureerror = []
s =[]
ressumsqure = 0


# hidden_layers = 2
# hidden_node = [2,2]
hidden_layers = 6
hidden_node = [5,3,4,6,4,2]
learningrate = -0.5
momentumrate = 0.4 
errortest = []
avgerrortest = []
firstround = 1

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


# [print(x) for x in train]

weigths = initial_weigths(train_x,hidden_node,hidden_layers) #(data,hiddennode = [2,2],hiddenlayer = 2)
print(type(weigths[0]))
bias = initial_bias(hidden_node,hidden_layers)
biasweight = initial_bias(hidden_node, hidden_layers)
# print(bias)
# print(weigths)


# print('beeeeeeef'+str(biasweight))

#NN
NN1 = NN(weigths,hidden_node,hidden_layers,learningrate,bias,biasweight,momentumrate,deltaWl,deltabiasWl)
deltaWl = 1
deltabiasWl = 1
# for i in range(len(train_x)):

while epc<epoch:
  for j in range(int(oncescross)):
    reserror = []
    errorcatch = []
    s = []
    sumsqureerror = []
    for i in range(len(train_x)):#train
      yt =  NN1.feedfoward(train_x[i])
      outweights,outbiasweights,deltaWl,deltabiasWl = NN1.backward(yt,train_y[i],train_x[i],firstround)
      # print('check'+str(i))
      errorcatch.append(NN1.errorrate(yt[-1],train_y[i]))
      # print(sum(errorcatch)/len(errorcatch))
      ertr = NN1.errorrate(yt[-1],train_y[i])
      NN1 = NN(outweights,hidden_node,hidden_layers,learningrate,bias,outbiasweights,momentumrate,deltaWl,deltabiasWl)
    train = randdata(train)
    train_x,train_y = splitIO(train)
    epc = epc + 1 
    print('epc'+str(epc))
    if epc==epoch:
      break

  # reserror.append(sum(errorcatch)/len(errorcatch))
  [ s.append((x**2)) for x in errorcatch ]
  sumsqureerror.append((sum(s)/2)/len(train_x))
  ressumsqure = sum(sumsqureerror)/len(train_x)  
  # print('Sum error = '+str(reserror))
  print('sum squre'+str(sumsqureerror)) 
  

  for j in range(len(test_x)): #test
    yt = NN1.feedfoward(test_x[j])
    errortest.append(NN1.errorrate(yt[-1],test_y[j]))
    erts = NN1.errorrate(yt[-1],train_y[i])
    # print('error'+str(errortest))
    if abs(errortest[0]) < 0.00001:
      print('erroe case')
      errorcase = errorcase + 1
      break
  avgerrortest.append(sum(errortest)/len(errortest))
  # print('ertr',ertr)
  # print('erts',erts)

  if len(keeprandom)==10:
    #  break
    keeprandom = []


 
   

  train,test = crossVar(data,keeprandom) 
  train = randdata(train)
  train_x,train_y = splitIO(train)
  train_x = toarr(train_x)
  train_y = toarr(train_y)
  test_x,test_y = splitIO(test)
  test_x = toarr(test_x)
  test_y = toarr(test_y)
# print('Sum error = '+str(sum(reserror)/len(reserror)))
# print('sum squre error'+str(ressumsqure))
# print('errorcase'+str(errorcase))
# print('Average errortest'+str(avgerrortest))
# print('Averrage'+str(sum(avgerrortest)/len(avgerrortest)))

# print(pre[0])
