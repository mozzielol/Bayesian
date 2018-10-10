from keras.models import Sequential
from keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Conv3D,MaxPooling3D,Dropout,core,Reshape,Input
from keras.callbacks import Callback,TensorBoard
from keras import backend as K
from keras.engine import Layer
from utils import *
import numpy as np
import matplotlib.pyplot as plt


def mse(cur_grad,pre_grad):
	res = []
	for i in range(len(pre_grad)):
		res.append(K.sum(K.square(cur_grad[i] - pre_grad[i])))
	return sum(res)/len(res)


def Custom_loss(model,pre_grad,batch,X,y):
	def loss(y_true,y_pred):
		return K.categorical_crossentropy(y_true, y_pred) + mse(get_gradients(model,X[batch*128 : (batch+1)*128],y[batch*128 : (batch+1)*128]),pre_grad)
	return loss


class cnn_model(object):

	def __init__(self):
		self.dim = 64
		self.num_classes = 10
		self.history = None
		self.epoch = 10
		
		

		input_shape = (28, 28, 1)
		noise = K.variable(0)
		
		self.model = Sequential()
		self.model.add(Conv2D(32, input_shape=input_shape,kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
		self.model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal'))
		self.model.add(MaxPooling2D((2, 2)))
		self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
		self.model.add(Conv2D(64, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
		self.model.add(MaxPooling2D(pool_size=(2, 2)))
		self.model.add(Conv2D(self.dim, (3, 3), activation='relu',padding='same',kernel_initializer='he_normal'))
		self.model.add(Flatten())
		self.model.add(Dense(32,activation='relu'))
		self.model.add(Dense(self.num_classes,activation='softmax'))

		self.tbCallBack = TensorBoard(log_dir='./logs/mnist_drift/kal/',  
		histogram_freq=0,  
		write_graph=True,  
		write_grads=True, 
		write_images=True,
		embeddings_freq=0, 
		embeddings_layer_names=None, 
		embeddings_metadata=None)

	def fit(self,X_train,y_train):
		self.pre_x = X_train
		self.pre_y = y_train
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		if self.history is None:
			self.history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,callbacks=[self.tbCallBack],validation_data=(X_train,y_train))
		else:
			history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,callbacks=[self.tbCallBack],validation_data=(X_train,y_train))
			self.history.history['acc'].extend(history.history['acc'])
			self.history.history['val_acc'].extend(history.history['val_acc'])


		global X
		global y
		X = X_train
		y = y_train

	def transfer(self,X_train,y_train):
		opcallback = op_pre_callback(X_train,y_train)
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,callbacks=[opcallback,self.tbCallBack],validation_data=(X,y))
		self.history.history['acc'].extend(history.history['acc'])
		self.history.history['val_acc'].extend(history.history['val_acc'])

	def use_pre(self,X_train,y_train):
		opcallback = op_pre_callback(X,y,True)
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,callbacks=[opcallback,self.tbCallBack],validation_data=(X,y))
		self.history.history['acc'].extend(history.history['acc'])
		self.history.history['val_acc'].extend(history.history['val_acc'])

	def use_cur(self,X_train,y_train):
		opcallback = op_pre_callback(X_train,y_train,True)
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,callbacks=[opcallback,self.tbCallBack],validation_data=(X,y))
		self.history.history['acc'].extend(history.history['acc'])
		self.history.history['val_acc'].extend(history.history['val_acc'])

	def nor_trans(self,X_train,y_train):
		self.model.pop()
		self.model.pop()
		
		for layer in self.model.layers:
			layer.trainable = False
		self.model.add(Dense(32,activation='relu'))
		self.model.add(Dense(self.num_classes,activation='softmax'))
		self.model.compile(optimizer='sgd',loss='categorical_crossentropy',metrics=['accuracy'])
		history = self.model.fit(X_train,y_train,epochs=self.epoch,batch_size=128,verbose=True,callbacks=[self.tbCallBack],validation_data=(X_train,y_train))
		self.history.history['acc'].extend(history.history['acc'])
		self.history.history['val_acc'].extend(history.history['val_acc'])


	def save(self,name):
		self.model.save('./models/{}.h5'.format(name))

	def evaluate(self,X_test,y_test):
		
		score=self.model.evaluate(X_test,y_test,batch_size=128)
		print("Convolutional neural network test loss:",score[0])
		print('Convolutional neural network test accuracy:',score[1])

		return score[1]

	def get_history(self):
		return self.history

	def plot(self,name,model,shift=2):
		plt.subplot(211)
		plt.title('accuracy on current training data')
		for i in range(shift):
			plt.vlines(self.epoch*(i+1),0,1,color='r',linestyles='dashed')
		
		plt.plot(self.history.history['acc'],label='{}'.format(model))
		plt.ylabel('acc')
		plt.xlabel('training time')
		plt.legend(loc='upper right')
		plt.subplot(212)
		plt.title('validation accuracy on original data')
		plt.plot(self.history.history['val_acc'],label='{}'.format(model))
		plt.ylabel('acc')
		plt.xlabel('training time')
		for i in range(shift):
			plt.vlines(self.epoch*(i+1),0,1,color='r',linestyles='dashed')
		plt.legend(loc='upper right')
		plt.subplots_adjust(wspace=1,hspace=1)
		plt.savefig('./images/{}.png'.format(name))
		




class grad_knowledge(Callback):
	"""docstring for grad_knowledge"""
	def __init__(self,pre_x,pre_y,X_train,pre_grad,batch):
		super(grad_knowledge, self).__init__()
		self.pre_grad = pre_grad
		self.pre_x = pre_x
		self.pre_y = pre_y
		self.X_train = X_train
		self.batch = batch
	
	def on_epoch_begin(self,epoch,logs={}):
		self.epoch = epoch

	
	def on_batch_begin(self,batch,logs={}):
		#index = most_similar(self.pre_x,self.X_train[batch*128 : (batch+1)*128])
		grad = get_weight_grad(self.model,self.pre_x[batch*128 : (batch+1)*128],self.pre_y[batch*128 : (batch+1)*128])
		K.set_value(self.pre_grad,np.array(grad))
		K.set_value(self.batch , batch)
		

#

class op_pre_callback(Callback):
	"""docstring for op_batch_callback"""
	def __init__(self,X_train,y_train,use_pre=False):
		super(op_pre_callback, self).__init__()
		self.pre_x = X_train
		self.pre_y = y_train
		self.use_pre = use_pre




	def Kal_gain(self,cur_grad,pre_grad):
		res = []
		for i in range(len(pre_grad)):
			temp = np.absolute(pre_grad[i]) / (np.absolute(cur_grad[i]) + np.absolute(pre_grad[i]))
			temp[np.isnan(temp)] = 0
			res.append(temp)
		return res


	def on_train_begin(self,logs={}):
		self.pre_g = get_weight_grad(self.model,X[0*128:(0+1)*128],y[0*128:(0+1)*128])

	def on_epoch_begin(self,epoch,logs={}):
		self.epoch = epoch
		

	def on_batch_begin(self,batch,logs={}):
		self.pre_w = get_weights(self.model)
		if self.use_pre:
			self.pre_g = get_weight_grad(self.model,self.pre_x[batch*128:(batch+1)*128],self.pre_y[batch*128:(batch+1)*128])
		


	def on_batch_end(self,batch,logs={}):
		
		
		self.cur_w = get_weights(self.model)

		self.cur_g = get_weight_grad(self.model,self.pre_x[batch*128:(batch+1)*128],self.pre_y[batch*128:(batch+1)*128])

		
		Kalman_gain = self.Kal_gain(self.cur_g,self.pre_g)
		new_w = []

		for P,Z,E in zip(self.pre_w,self.cur_w,Kalman_gain):
			new_w.append(P + (Z-P) * E )

		
		self.model.set_weights(new_w)
		
		new_g = []
		for kal,g in zip(Kalman_gain,self.pre_g):
			new_g.append((1- kal) * g )

		self.pre_g = new_g
		
		

	
class grad_check(Callback):
	"""docstring for grad_check"""
	def __init__(self, X_train,y_train):
		super(grad_check, self).__init__()
		self.X_train = X_train
		self.y_train = y_train

	def on_batch_end(self,batch,logs={}):
		cur_g = get_weight_grad(self.model,self.X_train[batch*128:(batch+1)*128],self.y_train[batch*128:(batch+1)*128])
		print('cur g :', self.cur_g[0][0][0][0][0])


class test_callback(Callback):
	"""docstring for test_callback"""
	def __init__(self, X_train,y_train,pre_x, pre_y):
		super(test_callback, self).__init__()
		self.X_train = X_train
		self.y_train = y_train
		self.pre_x = pre_x
		self.pre_y = pre_y

	def on_batch_begin(self,batch,logs={}):
		self.pre_w = get_weights(self.model)
		self.pre_g = get_weight_grad(self.model,self.pre_x[batch*128:(batch+1)*128],self.pre_y[batch*128:(batch+1)*128])
		self.cur_g = get_weight_grad(self.model,self.X_train[batch*128:(batch+1)*128],self.y_train[batch*128:(batch+1)*128])

	def on_batch_end(self,batch,logs={}):
		
		
		self.cur_w = get_weights(self.model)

		self.cur_g = get_weight_grad(self.model,self.pre_x[batch*128:(batch+1)*128],self.pre_y[batch*128:(batch+1)*128])

		
		Kalman_gain = self.Kal_gain(self.cur_g,self.pre_g)
		new_w = []

		for P,Z,E in zip(self.pre_w,self.cur_w,Kalman_gain):
			new_w.append(P + (Z-P) * E )

		
		self.model.set_weights(new_w)
		
		new_g = []
		for kal,g in zip(Kalman_gain,self.pre_g):
			new_g.append((1- kal) * g )

		self.pre_g = new_g

		




