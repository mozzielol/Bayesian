from utils import load_mnist_drift,TRAIN_NUM,TEST_NUM
from model import cnn_model
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def train(name):
	acc_test = []

	X_train,y_train,X_test,y_test = load_mnist_drift(split=True)
	model = cnn_model()
	model.fit(X_train[:TRAIN_NUM],y_train[:TRAIN_NUM])
	
	
	if name == 'kal':
		model.transfer(X_train[TRAIN_NUM:TRAIN_NUM*2],y_train[TRAIN_NUM:TRAIN_NUM*2])
		model.transfer(X_train[TRAIN_NUM*2:TRAIN_NUM*3],y_train[TRAIN_NUM*2:TRAIN_NUM*3])
	if name == 'kal_pre':
		model.use_pre(X_train[TRAIN_NUM:TRAIN_NUM*2],y_train[TRAIN_NUM:TRAIN_NUM*2])
		model.use_pre(X_train[TRAIN_NUM*2:TRAIN_NUM*3],y_train[TRAIN_NUM*2:TRAIN_NUM*3])
	if name == 'kal_cur':
		model.use_cur(X_train[TRAIN_NUM:TRAIN_NUM*2],y_train[TRAIN_NUM:TRAIN_NUM*2])
		model.use_cur(X_train[TRAIN_NUM*2:TRAIN_NUM*3],y_train[TRAIN_NUM*2:TRAIN_NUM*3])
	if name == 'nor':
		model.fit(X_train[TRAIN_NUM:TRAIN_NUM*2],y_train[TRAIN_NUM:TRAIN_NUM*2])
		model.nor_trans(X_train[TRAIN_NUM*2:TRAIN_NUM*3],y_train[TRAIN_NUM*2:TRAIN_NUM*3])

	acc_test.append(model.evaluate(X_train[:TEST_NUM],y_train[:TEST_NUM]))
	acc_test.append(model.evaluate(X_train[TEST_NUM:TEST_NUM*2],y_train[TEST_NUM:TEST_NUM*2]))
	acc_test.append(model.evaluate(X_train[TRAIN_NUM*2:TRAIN_NUM*3],y_train[TRAIN_NUM*2:TRAIN_NUM*3]))

	model.save(name)
	'''
	plt.subplot(313)
	plt.title('accuracy on test dataset')
	plt.plot(acc_test,label=name)
	plt.legend(loc='upper right')
	'''
	model.plot('res',name)
	
	

if __name__ == '__main__':
	train('kal')
	train('kal_pre')
	train('kal_cur')
	train('nor')
	