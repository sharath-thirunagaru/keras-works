import numpy as np
np.random.seed(1234)
import tensorflow as tf
tf.set_random_seed(1234)
from keras.models import Sequential
from sklearn.datasets import load_iris
from keras.layers import Dense
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class BaseModel():

    def __init__(self,**kwargs):
        self.model=None
        self.class_names=None
        self.target_class_map={}

    def load_data(self,**kwargs):
        pass

    def train(self,**kwargs):
        pass

    def evaluate(self,**kwargs):
        pass


class MlpModel(BaseModel):
    def __init__(self,**kwargs):
        super(MlpModel, self).__init__(**kwargs)

    def load_data(self,**kwargs):
        data = load_iris()
        X,y=data.data,data.target
        self.class_names=data.target_names
        for i,class_name in enumerate(self.class_names):
            self.target_class_map[i]=class_name


        lb = LabelBinarizer()
        labels = lb.fit_transform(y)

        return X,labels
    def train(self,**kwargs):
        x,y = self.load_data()

        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

        model = Sequential()
        model.add(Dense(128,activation='relu',input_shape=(4,)))
        model.add(Dense(128,activation='relu'))
        model.add(Dense(3,activation='softmax'))

        model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

        self.model=model.fit(x_train,y_train,**kwargs)

    def evaluate(self,x_test,y_test,**kwargs):
        y_pred = self.model.predict_classes(x_test)
        y_class_pred = [self.target_class_map[y] for y in y_pred]
        y_class_true = [self.target_class_map[y] for y in y_test]
        print(classification_report(y_class_true,y_class_pred))

