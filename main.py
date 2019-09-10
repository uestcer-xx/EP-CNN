from math import ceil
import numpy as np
import keras
from gensim.models import Word2Vec
from keras.callbacks import EarlyStopping
from keras.layers import (Conv2D, Dense, Dropout, Flatten, Input, MaxPooling2D,
                          Reshape, concatenate)
from keras.models import Model, Sequential, load_model
from keras.utils import plot_model
from ElasticPooling import ElasticPooling

early_stopping=keras.callbacks.EarlyStopping(
    monitor='loss', 
    patience=5, 
    verbose=0, 
    mode='auto'
)


batch_size =1
num_classes = 2
epochs = 30
wordVecNum=256
wordVec=32
dataNum=200000
p1_size=128

img_rows, img_cols =  wordVecNum,wordVec
input_shape = (img_rows, img_cols, 1)

""" 
class Sentences():
    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            a = [i for i in line.strip()]
            yield a


path = "../train_test_data/word2vec_train"
sentences = Sentences(path)
word_model = Word2Vec(sentences, size=wordVec, window=10, min_count=5)
output1 = "word_train64.model"
#output2 = "train.vector"
word_model.save(output1)
"""

word_model= Word2Vec.load("word_train32.model")


class url_iter():
  def __init__(self, path):
    self.path = path
  def __iter__(self):
    for line in open(self.path):
      y_lable,query=line.strip().split(" ")
      yield y_lable,query


fill_one = [1 for i in range(wordVec)]
fill_zero = [0 for i in range(wordVec)]


def load_TrainData(batch_size):
  while True:
    # url_path = "../train_test_data/train_data_2e5"
    url_path ="../train_test_data/b"
    # url_path = "x_train_10000.txt"
    # url_iter = url_iter(url_path)
    url_vec = []
    y=[]
    for y_lable,i in url_iter(url_path):
        single_url_vec = []
        for j in i:
            try:
                single_url_vec.append(word_model[j])
            except:
                single_url_vec.append(fill_one)
            # if(len(single_url_vec) >= wordVecNum):
            #     break
        while(len(single_url_vec) < wordVecNum):
            single_url_vec.append(fill_zero)
        while(len(single_url_vec) < ceil(len(i)/p1_size)*p1_size):
            single_url_vec.append(fill_zero)

        y.append([int(y_lable),1-int(y_lable)])   #normal->[1,0]
        url_vec.append(single_url_vec)
        if len(url_vec)>=batch_size:
            x_train = np.array(url_vec)
            # x_train = x_train.reshape(x_train.shape[0], wordVecNum, wordVec, 1)
            if(len(i)<=wordVecNum):
                x_train = x_train.reshape(x_train.shape[0], wordVecNum, wordVec, 1)
            else:
                x_train = x_train.reshape(x_train.shape[0], ceil(len(i)/p1_size)*p1_size, wordVec, 1)
            y_train=np.array(y)
            yield x_train,y_train
            url_vec=[]
            y=[]


def text_cnn():
    seq = Input(shape=(None,wordVec,1))
    conv_cat=[]
    filter_sizes = [1,3,5]
    for fil in filter_sizes:
        conv1 = Conv2D(filters=32,kernel_size=(fil,fil),padding='same',activation='relu')(seq)
        conv2=Conv2D(64, (fil, fil),padding='same',activation='relu')(conv1)  #
        p1=MaxPooling2D(pool_size=(2,2))(conv2)#行差64,16列

        conv3=Conv2D(128, (3, 3),padding='same', activation='relu')(p1)
        p3=MaxPooling2D(pool_size=(2,2))(conv3)#行差32,8列

        conv4=Conv2D(256, (3, 3),padding='same', activation='relu')(p3)
        p4 = ElasticPooling(32,4)(conv4)

        conv5=Conv2D(512, (3, 3),padding='same', activation='relu')(p4)
        pool=MaxPooling2D(pool_size=(2,2))(conv5)

        drop=Dropout(0.25)(pool)
        fla = Flatten()(drop)
        conv_cat.append(fla)
    merge = concatenate(conv_cat)
    den1 = Dense(128,activation='relu')(merge)
    drop_2 = Dropout(0.5)(den1)
    out = Dense(2,activation='softmax')(drop_2)
    model =Model([seq],out)
    model.compile(loss='categorical_crossentropy',optimizer='Adadelta',metrics=['acc'])
    return model


# model = Sequential()
model=text_cnn()
plot_model(model, to_file='my_model8.png',show_shapes=True)

# train the model 


model.fit_generator(load_TrainData(batch_size),epochs=epochs,steps_per_epoch=int(dataNum/batch_size),verbose=1,callbacks=[early_stopping])

model.save('my8.h5')
model.save_weights('my8_weights.h5')
'''
# test the model 测试模型
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
'''
# model=load_model('my1.h5')

def load_TestData(batch_size):
  while True:
    url_path = "../train_test_data/test_data_2e5"
    # url_path = "x_test_10000.txt"
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
            # if(len(single_url_vec) >= wordVecNum):
            #     break
        while(len(single_url_vec) < wordVecNum):
            single_url_vec.append(fill_zero)
        while(len(single_url_vec) < ceil(len(i)/p1_size)*p1_size):
            single_url_vec.append(fill_zero)

        y.append([int(y_lable),1-int(y_lable)])   #normal->[1,0]
        url_vec.append(single_url_vec)
        if len(url_vec)>=batch_size:
            x_test = np.array(url_vec)
            # x_test = x_test.reshape(1,ceil(len(i)/p1_size)*p1_size,wordVec, 1)
            if(len(i)<=wordVecNum):
                x_test = x_test.reshape(x_test.shape[0], wordVecNum, wordVec, 1)
            else:
                x_test = x_test.reshape(x_test.shape[0], ceil(len(i)/p1_size)*p1_size, wordVec, 1)
            y_test=np.array(y)
            yield x_test,y_test
            url_vec=[]
            y=[]


load_batch_size=20

score = model.evaluate_generator(load_TestData(load_batch_size),steps=int(200000/load_batch_size))

print('Test loss:', score[0])
print('Test accuracy:', score[1])

""" 
# def load_TestData1(batch_size):
#   while True:
#     url_path = "../train_test_data/test_all_sqli"
#     #url_iter = url_iter(url_path)
#     url_vec = []
#     y=[]
#     for y_lable,i in url_iter(url_path):
#         single_url_vec = []
#         for j in i:
#             try:
#                 single_url_vec.append(word_model[j])
#             except:
#                 single_url_vec.append(fill_one)
#             if(len(single_url_vec) >= 256):
#                 break
#         while(len(single_url_vec) < 256):
#             single_url_vec.append(fill_zero)

#         y.append([int(y_lable),1-int(y_lable)])   #normal->[1,0]
#         url_vec.append(single_url_vec)
#         if len(url_vec)>=batch_size:
#           x_test = np.array(url_vec)
#           x_test = x_test.reshape(x_test.shape[0], 256,wordVec, 1)
#           y_test=np.array(y)
#           yield x_test,y_test
#           url_vec=[]
#           y=[]

# score1 = model.evaluate_generator(load_TestData1(batch_size),steps=int(1201203/batch_size))
# print('test_all_sqli')
# print('Test loss:', score1[0])
# print('Test accuracy:', score1[1])

# def load_TestData2(batch_size):
#   while True:
#     url_path = "../train_test_data/test_all_normal"
#     #url_iter = url_iter(url_path)
#     url_vec = []
#     y=[]
#     for y_lable,i in url_iter(url_path):
#         single_url_vec = []
#         for j in i:
#             try:
#                 single_url_vec.append(word_model[j])
#             except:
#                 single_url_vec.append(fill_one)
#             if(len(single_url_vec) >= 256):
#                 break
#         while(len(single_url_vec) < 256):
#             single_url_vec.append(fill_zero)

#         y.append([int(y_lable),1-int(y_lable)])   #normal->[1,0]
#         url_vec.append(single_url_vec)
#         if len(url_vec)>=batch_size:
#           x_test = np.array(url_vec)
#           x_test = x_test.reshape(x_test.shape[0], 256,wordVec, 1)
#           y_test=np.array(y)
#           yield x_test,y_test
#           url_vec=[]
#           y=[]

# score2 = model.evaluate_generator(load_TestData2(batch_size),steps=int(41307367/batch_size))
# print('test_all_normal')
# print('Test loss:', score2[0])
# print('Test accuracy:', score2[1])
"""
