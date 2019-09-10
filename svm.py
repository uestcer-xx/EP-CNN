from sklearn import svm
from gensim.models import Word2Vec
from sklearn.externals import joblib

word_model=Word2Vec.load("train.model")


class url_iter():
  def __init__(self, path):
    self.path = path

  def __iter__(self):
    for line in open(self.path):
      y_lable,query=line.strip("\n").split(" ")
      yield y_lable,query

train_acc=[]
test_acc=[]

url_path="x_train_10000.txt"
y=[]
x=[]
for y_lable,query in url_iter(url_path):
    y.append(int(y_lable))
    single_url_vec=[]
    for i in query:
        try:
            single_url_vec.append(word_model[i])
        except:
            single_url_vec.append(1)
        if(len(single_url_vec)>=256):
            break
    while(len(single_url_vec)<256):
        single_url_vec.append(0)
    x.append(single_url_vec)

url_test_path="test.txt"
y_test=[]
x_test=[]
for y_lable,query in url_iter(url_test_path):
    y_test.append(int(y_lable))
    single_url_vec=[]
    for i in query:
        try:
            single_url_vec.append(word_model[i])
        except:
            single_url_vec.append(1)
        if(len(single_url_vec)>=256):
            break
    while(len(single_url_vec)<256):
        single_url_vec.append(0)
    x_test.append(single_url_vec)


    '''
    #print(c,g)
    clf=joblib.load('clf.model')
    #s_test=clf.score(x_test,y_test)
    #test_acc.append(s_test)
    print(clf.predict(x_test))
    print(y_test)
    '''

def run():
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=0.3, decision_function_shape='ovr')
    clf.fit(x,y,probability=True)
    joblib.dump(clf,'clf.model')
    s_train=clf.score(x,y)
    train_acc.append(s_train)
    print(s_train)
run()
'''
for c in [0.1*i for i in range(1,11)]:
    for g in [0.1*j for j in range(1,11)]:
        run(c,g)
print(train_acc)
print(test_acc)
'''
