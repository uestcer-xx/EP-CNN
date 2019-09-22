import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from gensim.models import Word2Vec

class Sentences():
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            a = [i for i in line.strip()]
            yield a


path = "../train_test_data/word2vec_train"
sentences = Sentences(path)
word_model = Word2Vec(sentences, size=32, window=10, min_count=5)
# output1 = "1214/word_train32.model"
# #output2 = "train.vector"
# word_model.save(output1)



class url_iter():
  def __init__(self, path):
    self.path = path

  def __iter__(self):
    for line in open(self.path):
      y_lable,query=line.strip().split(" ")
      yield y_lable,query





fill_one = [1 for i in range(32)]
fill_zero = [0 for i in range(32)]


def load_TrainData(batch_size):
  while True:
    url_path = "../train_test_data/train_data_2e5"
    #url_iter = url_iter(url_path)
    url_vec = []
    y=[]
    for y_lable,i in url_iter(url_path):
        single_url_vec = []
        for j in i:
            try:
                single_url_vec.append(word_model[j])
            except:
                single_url_vec.append(fill_one)
            if(len(single_url_vec) >= 256):
                break
        while(len(single_url_vec) < 256):
            single_url_vec.append(fill_zero)

        y.append([int(y_lable),1-int(y_lable)])   #normal->[1,0]
        url_vec.append(single_url_vec)
        if len(url_vec)>=batch_size:
          x_train = np.array(url_vec)
          x_train = x_train.reshape(x_train.shape[0], 256, 32, 1)
          y_train=np.array(y)
          yield x_train,y_train
          url_vec=[]
          y=[]

batch_size = 8
num_classes = 2
epochs = 100

img_rows, img_cols =  256,32
input_shape = (img_rows, img_cols, 1)



model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))  
model.add(Conv2D(64, (3, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(128, (3, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(128, (3, 3), activation='relu'))  
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(256, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))  
model.add(Conv2D(256, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(512, (3, 3), activation='relu')) 
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten()) 
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))  


# compile the model 

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# train the model 
model.fit_generator(load_TrainData(batch_size),epochs=epochs,steps_per_epoch=int(200000/batch_size),verbose=1)

model.save('1214/cnn256-32_model_1214.h5')

'''
# test the model 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''

def load_TestData(batch_size):
  while True:
    url_path = "../train_test_data/test_data_2e5"
    url_vec = []
    y=[]
    for y_lable,i in url_iter(url_path):
        single_url_vec = []
        for j in i:
            try:
                single_url_vec.append(word_model[j])
            except:
                single_url_vec.append(fill_one)
            if(len(single_url_vec) >= 256):
                break
        while(len(single_url_vec) < 256):
            single_url_vec.append(fill_zero)
        y.append([int(y_lable),1-int(y_lable)])
        url_vec.append(single_url_vec)
        if len(url_vec)>=batch_size:
          x_test = np.array(url_vec)
          x_test = x_test.reshape(x_test.shape[0], 256,32, 1)
          y_test=np.array(y)
          yield x_test,y_test
          url_vec=[]
          y=[]

score = model.evaluate_generator(load_TestData(batch_size),steps=int(200000/batch_size))
print('test_data_2e5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])

def load_TestData1(batch_size):
  while True:
    url_path = "../train_test_data/test_all_sqli"
    #url_iter = url_iter(url_path)
    url_vec = []
    y=[]
    for y_lable,i in url_iter(url_path):
        single_url_vec = []
        for j in i:
            try:
                single_url_vec.append(word_model[j])
            except:
                single_url_vec.append(fill_one)
            if(len(single_url_vec) >= 256):
                break
        while(len(single_url_vec) < 256):
            single_url_vec.append(fill_zero)

        y.append([int(y_lable),1-int(y_lable)])   #normal->[1,0]
        url_vec.append(single_url_vec)
        if len(url_vec)>=batch_size:
          x_test = np.array(url_vec)
          x_test = x_test.reshape(x_test.shape[0], 256,32, 1)
          y_test=np.array(y)
          yield x_test,y_test
          url_vec=[]
          y=[]

score1 = model.evaluate_generator(load_TestData1(batch_size),steps=int(1201203/batch_size))
print('test_all_sqli')
print('Test loss:', score1[0])
print('Test accuracy:', score1[1])

def load_TestData2(batch_size):
  while True:
    url_path = "../train_test_data/test_all_normal"
    #url_iter = url_iter(url_path)
    url_vec = []
    y=[]
    for y_lable,i in url_iter(url_path):
        single_url_vec = []
        for j in i:
            try:
                single_url_vec.append(word_model[j])
            except:
                single_url_vec.append(fill_one)
            if(len(single_url_vec) >= 256):
                break
        while(len(single_url_vec) < 256):
            single_url_vec.append(fill_zero)

        y.append([int(y_lable),1-int(y_lable)])   #normal->[1,0]
        url_vec.append(single_url_vec)
        if len(url_vec)>=batch_size:
          x_test = np.array(url_vec)
          x_test = x_test.reshape(x_test.shape[0], 256,32, 1)
          y_test=np.array(y)
          yield x_test,y_test
          url_vec=[]
          y=[]

score2 = model.evaluate_generator(load_TestData2(batch_size),steps=int(41307367/batch_size))
print('test_all_normal')
print('Test loss:', score2[0])
print('Test accuracy:', score2[1])

def load_TestData3(batch_size,url_path):
  a=0
  print(a)
  while True:
    url_vec = []
    for y_lable,i in url_iter(url_path):
        single_url_vec = []
        for j in i:
            try:
                single_url_vec.append(word_model[j])
            except:
                single_url_vec.append(fill_one)
            if(len(single_url_vec) >= 256):
                break
        while(len(single_url_vec) < 256):
            single_url_vec.append(fill_zero)

        url_vec.append(single_url_vec)
        if len(url_vec)>=batch_size:
          x_test = np.array(url_vec)
          x_test = x_test.reshape(x_test.shape[0], 256, 32, 1)

          yield x_test
          url_vec=[]


count=41307367
url_path="../train_test_data/test_all_normal"

print(url_path,count)
score = model.predict_generator(load_TestData3(batch_size,url_path),steps=int(count/batch_size))

rate=[score[i][0] for i in range(len(score))]
with open('score','w') as f:
  f.write(",".join(rate))

'''
for i in open('score','r'):
  a=i.strip().split(",")

s=[float(i) for i in a]

n=0
with open('error','w') as f:
  for i in open('x_test.txt','r'):
    if(i.strip().split(" ")[0] =="1"):
      if(s[n]<=0.5):
        f.write(f'{s[n] {i}}')
    else:
      if(s[n]>0.5):
        f.write(f'{s[n] {i}}')
    n+=1
'''