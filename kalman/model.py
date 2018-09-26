import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Conv2D,Flatten
from keras.layers import Dropout
from keras.callbacks import Callback
from ent import get_weights,get_gradients,get_weight_grad
import keras.backend as K


class test_model(object):
	"""docstring for test_model"""
	def __init__(self):
		super(test_model, self).__init__()
	
	def fit(self,X_train,y_train):

		input_shape = X_train[0].shape

		self.model = Sequential()
		self.model.add(Conv2D(32, input_shape=input_shape,kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
		self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
		self.model.add(Flatten())
		self.model.add(Dense(10,activation='softmax'))
		
		opcallback = op_batch_callback(X_train,y_train)
		self.model.compile(optimizer='sgd',loss='mean_squared_error',metrics=['accuracy'])
		self.model.fit(X_train,y_train,epochs=10,batch_size=128,verbose=True,callbacks=[opcallback])


	def evaluate(self,X_test,y_test):
		
		score=self.model.evaluate(X_test,y_test,batch_size=128)
		print("Convolutional neural network test loss:",score[0])
		print('Convolutional neural network test accuracy:',score[1])

		return score[1]



def loss(y_true,y_pred):
	return K.categorical_crossentropy(y_true, y_pred)


class op_batch_callback(Callback):
	"""docstring for op_batch_callback"""
	def __init__(self,X_train,y_train):
		super(op_batch_callback, self).__init__()
		self.pre_grad = []
		self.X_train = X_train
		self.y_train = y_train




	def Kal_gain(self,cur_grad,pre_grad):
		res = []
		for i in range(len(pre_grad)):
			temp = pre_grad[i] / (cur_grad[i] + pre_grad[i])
			temp[np.isnan(temp)] = 0.5
			res.append(temp)
		return res

	def on_epoch_begin(self,epoch,logs={}):
		self.epoch = epoch
		self.pre_g = get_weight_grad(self.model,self.X_train[0*128:(0+1)*128],self.y_train[0*128:(0+1)*128])
		

	def on_batch_begin(self,batch,logs={}):
		self.pre_w = get_weights(self.model)

	def on_batch_end(self,batch,logs={}):
		
		
		self.cur_w = get_weights(self.model)

		self.cur_g = get_weight_grad(self.model,self.X_train[batch*128:(batch+1)*128],self.y_train[batch*128:(batch+1)*128])
		Kalman_gain = self.Kal_gain(self.cur_g,self.pre_g)
		new_w = []

		for P,Z,E in zip(self.pre_w,self.cur_w,Kalman_gain):
			new_w.append(P + (Z-P) * E )

		
		self.model.set_weights(new_w)
		
		new_g = []
		for kal,g in zip(Kalman_gain,self.pre_g):
			new_g.append((1- kal) * g )
		'''
		print('bach -- ', batch)
		print('pre g :', self.pre_g[0][0][0][0][0])
		print('cur g :', self.cur_g[0][0][0][0][0])
		print('pre w :', self.pre_w[0][0][0][0][0:3])
		print('cur w :', self.cur_w[0][0][0][0][0:3])
		print('new w :', new_w[0][0][0][0][0:3])
		print('new g :', new_g[0][0][0][0][0])
		'''
		self.pre_g = new_g
