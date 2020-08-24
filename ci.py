import random
import numpy as np



#initial
def initial_node(file):
  s10 = []
  s11 = []
  s12 = []
  s13 = []
  s20 = []
  s21 = []
  s22 = []
  s23 = []
  desire = []

  f = open(str(file),'r')
  s1 = []
  s2 = []

  desire = []
  t = []
  for x in f:
    t.append(x)
  del t[0]
  del t[0]
  for i in t:
    x = i.split()
    s10.append(x[3])
    s11.append(x[2])
    s12.append(x[1])
    s13.append(x[0])
    s20.append(x[7])
    s21.append(x[6])
    s22.append(x[5])
    s23.append(x[4])
    desire.append(x[8])

  s1.append(s10)
  s1.append(s11)
  s1.append(s12)
  s1.append(s13)
  s2.append(s20)
  s2.append(s21)
  s2.append(s22)
  s2.append(s23)
  return s1,s2,desire

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
  r = random.randrange(1,10)
  while r in keeprandom:
     r = random.randrange(1,11)
  keeprandom.append(r)
  print('kepprand'+str(keeprandom))
  # r=1
  # print(r)
  test = []

  dis = round(len(data)/10)
  dist = dis * r   #ระยะของข้อมูลที่จะแบ่ง

  test = data[(dist-dis):dist]
  del data[(dist-dis):dist]

  return data,test
  
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
  def __init__(self,weigths,hidden_node,hidden_layers,learningrate,bias,biasweight,momentumrate):
    self.hidden_node = hidden_node #[2,2]
    self.hidden_layers = hidden_layers #2
    self.weigths = weigths
    self.lr = learningrate
    self.bias = bias
    self.biasweight = biasweight
    self.momentumrate = momentumrate  
    self.deltaWl = 0  
    

  def feedfoward(self,data):
    # print(self.data)
    # print(self.deltaWl)
   
    feedlayer = []
    # print(data)
    dotres = data

    for dotloop in range(self.hidden_layers+1): # test case มีสองhiddenlayer แสดงว่าต้องdot3ครั้ง จึงเท่ากับhiddenlayer+1
      # print(dotloop)
      # print(dotres)
      # print(weigths[dotloop])
      # print(self.bias[dotloop])
      # print(self.weigths)
      # print('before'+str(dotres))
      dotres = np.dot(dotres,self.weigths[dotloop]) 
      # print('dotress'+str(dotres))
      # print('biasss'+str(self.bias[dotloop][0]))
      # print(dotres)
      # print(self.bias[dotloop][0].shape)
      
      # dotres = dotres + (self.bias[dotloop][0] * self.biasweight[dotloop][0])
      # print('aaaa'+str(self.biasweight[dotloop][0]))
      dotres = dotres +  self.biasweight[dotloop][0]
      
      # print('plus bias'+str(dotres))
      dotres = self.sigmoid(dotres)
      # print('bbbb'+str(dotres))
      result = np.reshape(dotres,(len(dotres),1))
      # print('after'+str(result))
      feedlayer.append(result)
      # print(dotres)
    # print('after'+str(feedlayer))
    return feedlayer #sigmoid at hidden node and output node

  def backward(self,yt,desire,xinput,firstround):
    # print('yt' +str(yt))
    gdkeep  =[]    
    deltaW = []
    deltaWbias = []
    tofinddeltaW = []
    # tofinddeltaWinput = []
    # tofinddeltaWinput.append(xinput)
    # tofinddeltaW.append(yt[:-1])
    [tofinddeltaW.append(i)for i in yt[:-1]]
    tofinddeltaW = np.flip(tofinddeltaW)
    # print('tofind'+ str(tofinddeltaW[0]))
    #gradient
    #re weigth before output node
    ytflip = np.flip(yt)
    outset = yt[1:]
    outset = np.flip(outset)
  
    # print('Error = '+str(self.errorrate(outset[0],desire)))
    
    gdout = self.diffsigmoid(outset[0]) * self.errorrate(outset[0],desire)
    gdkeep.append(gdout)

    hiddenout = yt[:-1]
    hiddenout = np.flip(hiddenout)
    

    gdhd = []
    hdweight = np.flip(self.weigths)
    # print('gdout'+str(gdout[0][0]))
    # print('hiddenoutput'+str(self.diffsigmoid(hiddenout[0])))
    # print('hdweight'+str(hdweight))
    # print('gdkeep'+str(gdkeep))
    # print('hiddenout'+str(hiddenout))
    # print(hdweight[0]*hiddenout[0]*gdout[0])
    # print('เวดดดดด'+str(hdweight))
    # print('yhidden'+str(hiddenout))
    for i in range(len(hiddenout)):
    # print(self.diffsigmoid(hiddenout[0][0]))
    # print(gdbefore[0])
      gdkeep.append(np.dot(hdweight[i] ,gdkeep[i]) * self.diffsigmoid(hiddenout[i])) 
    # print('gddddd'+str(gdkeep))
    gdbias = np.flip(gdkeep)
    # print('biasweigth'+str(self.biasweight))
    # print('gdkeppppp'+str(gdbias))
    for i in range(len(gdbias)):
      deltaWbias.append((-self.lr)*gdbias[i][0]*np.transpose(self.biasweight[i]))
    # print('deltabias'+str(deltaWbias[0]))
    # print(len(self.biasweight[0]))
    # print('selfbias'+str(self.biasweight[0].shape[1]))
    # print('selfbias'+str(np.reshape(self.biasweight[0],(self.biasweight[0].shape[1],1))))
  
    # rebiasW = []
    # keepshape = len(self.biasweight)
    # for i in self.biasweight:
    #   rebiasW.append(np.reshape(i,(i.shape[1],1)))
    # self.biasweight = rebiasW

    # deltaWbias = np.reshape(deltaWbias,(1,len(deltaWbias)))
    # self.biasweight = np.reshape(self.biasweight,(len(self.biasweight),1))
    # self.biasweight = toarr(self.biasweight)
    # print(self.biasweight.)

    # print('selfbias'+str(self.biasweight))
    # print('deltabias'+str(deltaWbias))
    resbiasW = []
    for i in range(len(self.biasweight)):
      # print('selfbiasinside'+str(self.biasweight[i]))
      # print('deltabiasinside'+str(deltaWbias[i]))
      biasWres = self.biasweight[i] +(self.momentumrate*(deltaWbias[i]-self.biasweight[i]))+deltaWbias[i]
      resbiasW.append(biasWres)
    self.biasweight = resbiasW
    # print('selfbias1'+str(self.biasweight))

    rebiasW = []
    
    # print('---'+str(self.biasweight))
    
    # rr = np.reshape(self.biasweight,(1,self.biasweight.shape[0]))
    #ต้องปรับ biasweight ให้เป็นรูปแบบเดิมเหมือนกับตอนเข้าตอนแรกขขขขข
      
    # self.biasweight = rr



    gdinput = gdkeep[-1]
    gdkeep.pop()  
    # print(gdinput)
    # print(self.weigths)
    # print(len(gdkeep))
    # # print(len(tofinddeltaW))
    # print(self.biasweight)
    for i in range(len(tofinddeltaW)):
      # print(str(tofinddeltaW[i])+'**********'+str(gdkeep[i]))
      deltaW.append((-self.lr)*tofinddeltaW[i]*np.transpose(gdkeep[i]))
      # resoutnode = self.weigths[-1] + deltaWout + self.momentumrate
    deltaW.append((-self.lr)*np.reshape(xinput,(len(xinput),1))*np.transpose(gdinput))
    # deltaW[-1] = np.reshape(deltaW[-1][0],(8,1))
    # print('deltaW'+str(deltaW[2][0]))
    # print('W'+str(self.weigths[0][0]))
    deltaW = np.flip(deltaW)
      
    if self.deltaWl ==0:
      self.deltaWl = deltaW
    # print('deltaW'+str(deltaW))
    # print('laaassstt'+str(self.deltaWl))
    self.weigths = self.weigths +(self.momentumrate*(deltaW-self.deltaWl))+deltaW
    self.deltaWl = deltaW
    # print('new weight'+str(self.weigths))
    return self.weigths,self.biasweight

  def sigmoid(self,s):
    return 1/(1+np.exp(-s))

  def errorrate(self,yt,desire):
    et = desire-yt
    # print(et)
    return et
  
  def diffsigmoid(self, s):
    return (s) * (1 - (s))

      



      
#main
# s1,s2,desire= initial_node('Flood_dataset.txt')

#input
# hidden_layers = int(input('layer = '))
# hidden_node = []
# for i in range(hidden_layers):
#   hidden_node.append(int(input('num node'+str(i+1)+'  = ')))
# print(hidden_node)
epoch = 10
epc = 0
errorcatch = []
reserror = []
keeprandom = []
sumsqureerror = []
ressumsqure = 0

hidden_layers = 2
hidden_node = [2,2]
learningrate = 0.2
momentumrate = 0.02 
error = 100
firstround = 1


#data
data = splitdata(read_file('Flood_dataset.txt'))
data = retype(data) #str to node

#keep min,max of data
MAX = findmax2D(data)
MIN = findmin2D(data)

#change data formation
data = Normalization(data,MAX,MIN) # normalize data to value in data in -1 to 1 
train,test = crossVar(data,keeprandom) #Crossvalidation split test train  (test 1 in 10) (train 9 in 10)
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
NN1 = NN(weigths,hidden_node,hidden_layers,learningrate,bias,biasweight,momentumrate)
# for i in range(len(train_x)):

while epc<epoch:
  for i in range(len(train_x)):#train
    yt =  NN1.feedfoward(train_x[i])
    outweights,outbiasweights = NN1.backward(yt,train_y[i],train_x[i],firstround)
    # print('obw'+str(outbiasweights))
    errorcatch.append(NN1.errorrate(yt[-1],train_y[i]))
    
   
    # print(sum(errorcatch)/len(errorcatch))
    NN1 = NN(outweights,hidden_node,hidden_layers,learningrate,bias,outbiasweights,momentumrate)
    if epc==epoch:
      break
  reserror.append(sum(errorcatch)/len(errorcatch))
  # sumsqureerror.append(sum(map(abs,errorcatch)/2))
  # ressumsqure = sum(sumsqureerror)/len(train_x)
  for j in range(len(test_x)): #test
    yt = NN1.feedfoward(test_x[j])
    error = NN1.errorrate(yt[-1],test_y[j])
    if abs(error) < 0.00001:
      print('erroe case')
      break
  if len(keeprandom)==10:
    break

  epc = epc + 1 
  print('epc'+str(epc))
  train,test = crossVar(data,keeprandom) 
  train_x,train_y = splitIO(train)
  train_x = toarr(train_x)
  train_y = toarr(train_y)
  test_x,test_y = splitIO(test)
  test_x = toarr(test_x)
  test_y = toarr(test_y)
print('Sum error = '+str(sum(reserror)/len(reserror)))
print('sum squre error'+str(ressumsqure))


# print(pre[0])
