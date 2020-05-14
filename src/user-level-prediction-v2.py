#!/usr/bin/env python
# coding: utf-8

# In[1]:


# common imports and definitions
import pandas as pd
from random import randrange, shuffle, random
from copy import deepcopy

modelVersion = "V2"
modelStructure = "model/model{0}.json".format(modelVersion)
modelWeights = "model/model{0}.h5".format(modelVersion)

levels = ["A1", "A2", "B1", "B2", "C1", "C2", "TOEFL", "IELTS", "SAT", "undefined"]
cefrLevels = ["A1", "A2", "B1", "B2", "C", "undefined"]
levelsToCefr = {"A1":"A1", "A2":"A2", "B1":"B1", "B2":"B2", "C1":"C", "C2":"C", "TOEFL":"C", "IELTS":"C", "SAT":"C", "undefined":"undefined"}
cefrLevelsCount = {"A1":0, "A2":0, "B1":0, "B2":0, "C":0, "undefined":0}

labels = [0, 1, 2, 3, 4]
levelsToLabels = {"A1":0, "A2":1, "B1":2, "B2":3, "C":4}
labelsToLevels = {0:"A1", 1:"A2", 2:"B1", 3:"B2", 4:"C"}

columnHeaders = ["A1", "A2", "B1", "B2", "C1", "C2", "TOEFL", "IELTS", "SAT", "userLevelByHuman"]
intToColumnHeaders = {0:"A1", 1:"A2", 2:"B1", 3:"B2", 4:"C1", 5:"C2", 6:"TOEFL", 7:"IELTS", 8:"SAT"}


# In[2]:


# loading the parsed data
ue = pd.read_csv("data/_UserEnglishHistory_202005061410.csv").to_dict("record")
print(ue[0])
print(len(ue))


# In[3]:


for u in ue:
  for lvl in levels:
      u[lvl] = 0
  for l in u["userLanguageHistory"].split(";"):
    for lvl in levels:
      if lvl in l:
        u[lvl] = int(l.split(",")[-1])/int(l.split(",")[-2])
  tmp = {k:v for k,v in u.items() if k in levels}
  if tmp[max(tmp, key=tmp.get)] == 0.0:
    u["userLevelByHuman"] = "undefined"

print(ue[0])
print(len(ue))


# In[4]:


pd.DataFrame.from_dict(ue).to_csv("data/UEH.csv", index=False)
# after this step, we are labelling the data manually and load the labelled data at the next step.


# In[5]:


# finalizing the input data after labelled
uel = pd.read_csv("data/UEHLabelled.csv").to_dict("record")
print(uel[0])
print(len(uel))


# In[6]:


uelClean = []

for u in uel:
  tmp = {k:v for k,v in u.items() if k in columnHeaders}
  tmp["userLevelByHuman"] = levelsToCefr[tmp["userLevelByHuman"]]
  cefrLevelsCount[tmp["userLevelByHuman"]] += 1
  if tmp["userLevelByHuman"] != "undefined":
    tmp["userLevelByHuman"] = levelsToLabels[tmp["userLevelByHuman"]]
    uelClean.append(tmp)

print(cefrLevelsCount)
print(uelClean[0])
print(len(uelClean))


# In[7]:


pd.DataFrame.from_dict(uelClean).to_csv("data/UEHClean.csv", index=False)


# In[8]:


# data augmentation
uec = pd.read_csv("data/UEHClean.csv").to_dict("record")
print(uec[0])
print(len(uec))


# In[9]:


uea = {label:[] for label in labels}
print(uea)

for u in uec:
    uea[u["userLevelByHuman"]].append(u)

for label in labels:
    print(label, len(uea[label]))

print(uea[4][0])


# In[10]:


n = 2000

for label in labels:
    for i in range(n-len(uea[label])):
        randomIndex = randrange(0,len(uea[label]))
        tmp = deepcopy(uea[label][randomIndex])
        randomColumn = intToColumnHeaders[randrange(0,9)]
        randomSign = randrange(-1,3,2)
        tmp[randomColumn] = abs(tmp[randomColumn] + randomSign*0.001)
        uea[label].append(tmp)

print(randomIndex, randomColumn, randomSign, tmp)

for label in labels:
    print(label, len(uea[label]))


# In[11]:


ueag = []
for label in labels:
    ueag += uea[label]

print(len(ueag), ueag[-1])


# In[12]:


pd.DataFrame.from_dict(ueag).to_csv("data/UEHAugmented.csv", index=False)


# In[13]:


# training
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
dataset = numpy.loadtxt("data/UEHAugmented.csv", delimiter=",", skiprows=1)
shuffle(dataset, random)

# split into input (X) and output (Y) variables
num_features = 9
num_classes = 5
X = dataset[:,0:num_features]
Y = dataset[:,num_features]

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
# split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.1, random_state=numpy.random.seed(7))

# create model
model = Sequential()
model.add(Dense(12, input_dim=num_features, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X_train, Y_train, epochs=25, batch_size=5, verbose=1)

# evaluate the model
scores = model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[14]:


# serialize model to JSON
model_json = model.to_json()
with open(modelStructure, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelWeights)
print("Saved model to disk")


# In[15]:


# load json and create model
json_file = open(modelStructure, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(modelWeights)
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, Y_test, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[16]:


# make class predictions with the model
predictions = loaded_model.predict_classes(X_test)

# summarize the first n cases
for i in range(50):
    print("predicted {0}, expected {1}, {2}, input {3}".format(predictions[i], Y_test[i], Y_test[i][predictions[i]] == 1.0, X_test[i].tolist()))
