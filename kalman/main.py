from keras.utils import to_categorical
from model import test_model
import keras.backend as K

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def test_optimizer():
	num_classes=10
	from keras.datasets import mnist
	(X_train,y_train),(X_test,y_test) =mnist.load_data()
	img_x, img_y = 28, 28
	input_shape = (img_x,img_y,1)
	X_train = X_train.reshape(X_train.shape[0],img_x,img_y,1)/255
	X_test = X_test.reshape(X_test.shape[0],img_x,img_y,1)/255
	y_train = to_categorical(y_train, num_classes)
	y_test = to_categorical(y_test,num_classes)
	
	model = test_model()
	model.fit(X_train[0:800],y_train[0:800])



if __name__=='__main__':
	test_optimizer()