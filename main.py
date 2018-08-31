
from ex2_cnn import cnn_model
from utils import load_mnist_drift
from sklearn.externals import joblib

def loss(y_true,y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def test(name):
	from keras.models import load_model
	model = load_model('{}.h5'.format(name),custom_objects={'loss':loss})
	X_train,y_train,X_test,y_test = load_mnist_drift(split=True)
	score = model.evaluate(X_test,y_test,batch_size=128)
	print('Model {}'.format(name))
	print('test loss :', score[0])
	print('test acc :', score[1])


if __name__=='__main__':
	
	X_train,y_train,X_test,y_test = load_mnist_drift(split=True)
	infer = cnn_model('info_gain')
	infer.fit(X_train,y_train)
	infer.evaluate(X_test,y_test)

	para = cnn_model('nor')
	para.fit(X_train,y_train)
	para.evaluate(X_test,y_test)
	
	for name in ['info_gain','nor']:
		test(name)
