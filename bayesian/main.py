from utils import load_mnist_drift
from model import cnn_model

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn



if __name__ == '__main__':
	X_train,y_train,X_test,y_test = load_mnist_drift(split=True)
	model = cnn_model()
	print('---'*20,'first','---'*20)
	model.fit(X_train[:20000],y_train[:20000])
	print('---'*20,'second','---'*20)
	model.transfer(X_train[20000:40000],y_train[20000:40000])
	tran = cnn_model()
	print('---'*20,'first','---'*20)
	tran.fit(X_train[:20000],y_train[:20000])
	print('---'*20,'second','---'*20)
	tran.fit(X_train[20000:40000],y_train[20000:40000])
	#model.fit(X_train[4000:6000],y_train[4000:6000])