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

  np.random.seed(1)
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
    resbias.append(bias)
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
def crossVar(data):#สุ่ม10เปอเซ็นของข้อมูลเพื่อแบ่งเป็นtest
  r = random.randrange(1,10)
  r=1
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
  def __init__(self,weigths,hidden_node,hidden_layers,learningrate,bias):
    self.hidden_node = hidden_node #[2,2]
    self.hidden_layers = hidden_layers #2
    self.weigths = weigths
    self.lr = learningrate
    self.bias = bias
    

  def feedfoward(self,data):
    # print(self.data)
    # print(weigths)
   
    feedlayer = []
    print(data)
    dotres = data
    for dotloop in range(self.hidden_layers+1): # test case มีสองhiddenlayer แสดงว่าต้องdot3ครั้ง จึงเท่ากับhiddenlayer+1
      # print(dotloop)
      # print(dotres)
      # print(weigths[dotloop])
      # print(self.bias[dotloop][0])
      dotres = np.dot(dotres,self.weigths[dotloop]) 
      dotres = dotres + self.bias[dotloop][0]
      # print('plus bias'+str(dotres))
      dotres = self.sigmoid(dotres)
      feedlayer.append(dotres)
      # print(dotres)
    print(feedlayer)
    return toarr(feedlayer) #sigmoid at hidden node and output node

  def backward(self,yt,desire):
    gd = self.diffsigmoid(yt) * self.errorrate(yt,desire) 
    # print(gd)
    # print(self.weigths)
    # for i in range(len(self.weigths[-1])):
      # print(i) #lr * gd * yt

    return 0

  def sigmoid(self,s):
    return 1/(1+np.exp(-s))

  def errorrate(self,yt,desire):
    et = desire-yt
    # print(et)
    return et
  
  def diffsigmoid(self, s):
    return (s) * (1 - (s))

  def gradient(self):
    return 0
      



      
#main
# s1,s2,desire= initial_node('Flood_dataset.txt')

#input
# hidden_layers = int(input('layer = '))
# hidden_node = []
# for i in range(hidden_layers):
#   hidden_node.append(int(input('num node'+str(i+1)+'  = ')))
# print(hidden_node)
hidden_layers = 2
hidden_node = [2,2]
learningrate = 0.2

#data
data = splitdata(read_file('Flood_dataset.txt'))
data = retype(data)

#keep min,max of data
MAX = findmax2D(data)
MIN = findmin2D(data)

#change data formation
data = Normalization(data,MAX,MIN)
train,test = crossVar(data)
train_x,train_y = splitIO(train)
train_x = toarr(train_x)
train_y = toarr(train_y)

# [print(x) for x in train]

weigths = initial_weigths(train_x,hidden_node,hidden_layers) #(data,hiddennode = [2,2],hiddenlayer = 2)
bias = initial_bias(hidden_node,hidden_layers)
print(bias)
print(weigths)



#NN
NN1 = NN(weigths,hidden_node,hidden_layers,learningrate,bias)
# # for i in train_x:
yt =  NN1.feedfoward(train_x[0])
out = NN1.backward(yt,train_y[0])

# print(pre[0])
