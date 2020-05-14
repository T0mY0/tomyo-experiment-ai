#!/usr/bin/env python
# coding: utf-8

# Data prep

# Getting data ready for labelling

# In[1]:


import pandas as pd

# loading the parsed data
ue = pd.read_csv("data/_UserEnglishHistory_202005061410.csv").to_dict("record")
print(ue[0])
print(len(ue))

levels = ["A1", "A2", "B1", "B2", "C1", "C2", "TOEFL", "IELTS", "SAT"]
modelVersion = "V1"
modelStructure = "model/model{0}.json".format(modelVersion)
modelWeights = "model/model{0}.h5".format(modelVersion)


# In[2]:


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


# In[3]:


pd.DataFrame.from_dict(ue).to_csv("data/UEH.csv", index=False)


# After this step, we are labelling the data manually and load the labelled data at the next step.

# Finalizing the input data after labelled

# In[4]:


uel = pd.read_csv("data/UEHLabelled.csv").to_dict("record")
print(uel[0])
print(len(uel))


# In[5]:


labels = ["A1", "A2", "B1", "B2", "C1", "C2", "TOEFL", "IELTS", "SAT", "userLevelByHuman"]
labelsMap = {"A1":0, "A2":1, "B1":2, "B2":3, "C1":4, "C2":5, "TOEFL":6, "IELTS":7, "SAT":8, "undefined":9}
labelsCount = {"A1":0, "A2":0, "B1":0, "B2":0, "C1":0, "C2":0, "TOEFL":0, "IELTS":0, "SAT":0, "undefined":0}
uelClean = []

for u in uel:
  tmp = {k:v for k,v in u.items() if k in labels}
  labelsCount[tmp["userLevelByHuman"]] += 1
  tmp["userLevelByHuman"] = labelsMap[tmp["userLevelByHuman"]]
  if tmp["userLevelByHuman"] != 9:
    uelClean.append(tmp)

print(labelsCount)
print(uelClean[0])
print(len(uelClean))


# In[6]:


pd.DataFrame.from_dict(uelClean).to_csv("data/UEHClean.csv", index=False)


# Training

# From this onward, it could be on Sagemaker

# In[9]:


from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# load the dataset
# TODO: add the clean data (at https://drive.google.com/open?id=15f8k5VoP0_yPdajnTsQL9GLBGS8MbMDj) to S3 bucket and load it from there
dataset = numpy.loadtxt("data/UEHClean.csv", delimiter=",", skiprows=1)

# split into input (X) and output (Y) variables
num_labels = 9
split_point = 800
X = dataset[:split_point,0:num_labels]
Y = dataset[:split_point,num_labels]
Xt = dataset[split_point:,0:num_labels]
Yt = dataset[split_point:,num_labels]

# create model
model = Sequential()
model.add(Dense(12, input_dim=num_labels, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model
model.fit(X, Y, epochs=25, batch_size=2, verbose=1)

# evaluate the model
scores = model.evaluate(X, Y, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# In[10]:


# serialize model to JSON
model_json = model.to_json()
with open(modelStructure, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelWeights)
print("Saved model to disk")


# In[11]:


# load json and create model
json_file = open(modelStructure, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights(modelWeights)
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, Y, verbose=1)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


# In[12]:


# make class predictions with the model
predictions = loaded_model.predict_classes(Xt)

# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (Xt[i].tolist(), predictions[i], Yt[i]))


# In[ ]:




