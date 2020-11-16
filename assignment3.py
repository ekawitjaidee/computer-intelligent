import numpy as np



def data():
  f = open('wdbc.data.txt')
  datas = f.readlines()
  ID,M,feature = [],[],[]
  for data in datas:
    data = data.split(',')
    ID.append(data[0])
    M.append(data[1])
    feature.append(data[2:])

  return ID,M,feature

def normalization(x,minx,maxx):
  x = (x - minx) / (maxx - minx)
  return x

def normalizationClass(M):
    M_normalize = []
    for i in M:
        if i == 'M':
            M_normalize.append([0.1])
        else:
            M_normalize.append([0.9])
    return M_normalize

def initial_weigths(input_node,hidden_node,hidden_layer,output_node):
  node  = input_node
  cp_node = []
  cp_node.append(node)
  [cp_node.append(i) for i in hidden_node]
  
  hidden_node.append(output_node)#append output node to make weigths
  layer_count = []

  for i in range(len(hidden_node)):
    weigths = 2 * np.random.random((cp_node[i], hidden_node[i])) - 1
    layer_count.append(weigths)

  return layer_count

def splitdata(data,k):
  length = int(len(data)/k)
  datacopy = data.copy()
  split = []
  for i in range(k):
    g = datacopy[length*i:(length*(i+1))]
    split.append(g)
  return split

def Crossvalidation(xcopy,dcopy,i):
  del xcopy[i]
  del dcopy[i]
  xtrain = np.concatenate(xcopy, axis=0)
  dtrain = np.concatenate(dcopy, axis=0)
  return xtrain, dtrain

def shuffle (x_train,ytrain):
  seed = np.random.randint(1,100000)
  np.random.seed(seed)
  np.random.shuffle(x_train)
  np.random.shuffle(y_train)
  
#NeuralNetwork
#NN
def NeuralNetwork(data,weights,hidden_layer):
   
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

#GA
def sortfitness(poppulation,mae):
  maecop = mae.copy()
  bestfitness = []
  for i in range(len(poppulation)):
    index = np.argmin(maecop)
    bestfitness.append(poppulation[index])
    maecop.pop(index)
  return bestfitness

def crossover(poppulation):
  rand1 = np.random.randint(0,9)
  rand2 = np.random.randint(0,9)
  while rand2 == rand1:
     rand2 = np.random.randint(0,9)
  weight1_2 = poppulation[rand1][int(len(poppulation[rand1])/2):]
  weight1_1 = poppulation[rand1][:int(len(poppulation[rand1])/2)]
  weight2_2 = poppulation[rand1][int(len(poppulation[rand2])/2):]
  weight2_1 = poppulation[rand1][:int(len(poppulation[rand2])/2)]
  poppulation[rand1] = np.array(weight1_1+weight2_2)
  poppulation[rand2] = np.array(weight2_1+weight1_2)

  return poppulation

def mutate(poppulation,hidden_node,hidden_layer,output_node):
  for pop in range(len(poppulation)):
    chance = np.random.randint(0,4)
    if chance ==1 :
      poppulation[pop] = initial_weigths(input_node,hidden_node,hidden_layer,output_node)
  return poppulation

ID,M,feature = data()
feature = np.array(list(map(lambda x: list(map(float, x)), feature)))
feature_normalize = normalization(feature,np.min(feature),np.max(feature))
# print(type(M))
M = normalizationClass(M)

poppulation = []
individual = 10
generation = 15


hidden_layer = 5
hidden_node = [4,6,2,3,4]
input_node = 30
output_node = 1
K = 10 #10Cross

x = splitdata(feature_normalize,K)
y = splitdata(M,K)

for i in range(individual):
  weights = initial_weigths(input_node,hidden_node,hidden_layer,output_node)
  poppulation.append(weights)
  
# print(poppulation)
for k in range (K):
  
  xcopy,ycopy = x.copy(),y.copy()
  x_test,y_test = xcopy[i],ycopy[i]
  x_train,y_train = Crossvalidation(xcopy,ycopy,k)
  # print(x_train.shape)
  for g in range(generation):
    shuffle(x_train,y_train)
    mae =[]
    # print(x_train.shape)
    for p in range (len(poppulation)):
      ber = 0
      for train in range(len(x_train)):
        yh = NeuralNetwork(x_train[train],poppulation[p],hidden_layer)
        er = errorrate(yh[-1],y_train)
        ber += abs(er)
      mae.append(1/(sum(ber)/len(x_train)))
    poppulation = sortfitness(poppulation,mae)
    # poppulation = crossover(poppulation)
    poppulation = mutate(poppulation,hidden_node,hidden_layer,output_node)

  ertests = []
  for test in range(len(x_test)):
    ytest = NeuralNetwork(x_test[test],poppulation[np.argmin(mae)],hidden_layer)
    ert = errorrate(yh[-1],y_test[test])
    ertests.append(abs(ert))
  ertestall = sum(ertests)/len(x_test)
  print('Cross (',k+1,') fitness = ',mae[np.argmin(mae)][-1],' fitness Test = ',1/ertestall)