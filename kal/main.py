from utils import load_mnist_drift,TRAIN_NUM,TEST_NUM
from model import cnn_model
import matplotlib.pyplot as plt


#igore the warning messages. Cause the kal drawback is slower than normal process, keras 
#will print some warning messages.
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


#Plot the accuracy of test data
#Parameters:
# - name: the name of the model. It will be used in label
# - acc: list of accuracy
# - data_num: which data is plotted(D1,D2 or D3)
def acc_plot(name,acc,data_num):
	plt.figure(1)
	sub = '31'+str(data_num)
	plt.subplot(sub)
	plt.title('test accuracy on {}th dataset'.format(data_num))
	plt.plot(acc,label=name)
	plt.ylabel('acc')
	plt.xlabel('training time')
	for i in range(len(acc)-1):
		plt.vlines((i+1),0,1,color='r',linestyles='dashed')
	plt.legend(loc='upper right')
	plt.subplots_adjust(wspace=1,hspace=1)
	plt.savefig('./images/test.png'.format(name))


#Load the drift data
X_train,y_train,X_test,y_test = load_mnist_drift(split=True)


#record the test accuracy. 
#Parameters:
# - model: the instance of model
# - acc_test_d1: record the accuracy of model on D1
# - acc_test_d2: record the accuracy of model on D2
# - acc_test_d3: record the accuracy of model on D3
def test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3):
	acc_test_d1.append(model.evaluate(X_test[:TEST_NUM],y_test[:TEST_NUM]))
	acc_test_d2.append(model.evaluate(X_test[TEST_NUM:TEST_NUM*2],y_test[TEST_NUM:TEST_NUM*2]))
	acc_test_d3.append(model.evaluate(X_test[TEST_NUM*2:TEST_NUM*3],y_test[TEST_NUM*2:TEST_NUM*3]))


# - Train the model.
# - The validation dataset is alwasy D1_test: model.val_data(X_test[:TEST_NUM],y_test[:TEST_NUM])
# - After the model is trained on each dataset, it will record the accuracy of model on test 
#	data of D1,D2 and D3 by using test_acc()
# - Return: this function will return the accuracy of D1_test,D2_test,D3_test after being trained
#	on each dataset.
def train(name):

	acc_test_d1 = []
	acc_test_d2 = []
	acc_test_d3 = []
	
	model = cnn_model()
	model.val_data(X_test[:TEST_NUM],y_test[:TEST_NUM])
	model.fit(X_train[:TRAIN_NUM],y_train[:TRAIN_NUM])

	test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
	
	if name == 'kal':
		model.transfer(X_train[TRAIN_NUM:TRAIN_NUM*2],y_train[TRAIN_NUM:TRAIN_NUM*2])
		test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
		model.transfer(X_train[TRAIN_NUM*2:TRAIN_NUM*3],y_train[TRAIN_NUM*2:TRAIN_NUM*3])
		test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
	if name == 'kal_pre':
		model.use_pre(X_train[TRAIN_NUM:TRAIN_NUM*2],y_train[TRAIN_NUM:TRAIN_NUM*2])
		test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
		model.use_pre(X_train[TRAIN_NUM*2:TRAIN_NUM*3],y_train[TRAIN_NUM*2:TRAIN_NUM*3])
		test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
	if name == 'kal_cur':
		model.use_cur(X_train[TRAIN_NUM:TRAIN_NUM*2],y_train[TRAIN_NUM:TRAIN_NUM*2])
		test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
		model.use_cur(X_train[TRAIN_NUM*2:TRAIN_NUM*3],y_train[TRAIN_NUM*2:TRAIN_NUM*3])
		test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
	if name == 'nor':
		model.fit(X_train[TRAIN_NUM:TRAIN_NUM*2],y_train[TRAIN_NUM:TRAIN_NUM*2])
		test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)
		model.fit(X_train[TRAIN_NUM*2:TRAIN_NUM*3],y_train[TRAIN_NUM*2:TRAIN_NUM*3])
		test_acc(model,acc_test_d1,acc_test_d2,acc_test_d3)


	model.save(name)
	model.plot('res',name)
	
	return acc_test_d1,acc_test_d2,acc_test_d3
	

if __name__ == '__main__':
	#Train the model. 
	print('--'*10,'kal','--'*10)
	kal_d1,kal_d2,kal_d3 = train('kal')
	print('--'*10,'kal_pre','--'*10)
	kal_pre_d1, kal_pre_d2, kal_pre_d3 = train('kal_pre')
	print('--'*10,'kal_cur','--'*10)
	kal_cur_d1, kal_cur_d2, kal_cur_d3 = train('kal_cur')
	print('--'*10,'nor','--'*10)
	nor_d1,nor_d2,nor_d3 = train('nor')

	#Plot the accuracy on test data.
	acc_plot('nor',nor_d1,1)
	acc_plot('nor',nor_d2,2)
	acc_plot('nor',nor_d3,3)
	acc_plot('kal',kal_d1,1)
	acc_plot('kal',kal_d2,2)
	acc_plot('kal',kal_d3,3)
	acc_plot('kal_pre',kal_pre_d1,1)
	acc_plot('kal_pre',kal_pre_d2,2)
	acc_plot('kal_pre',kal_pre_d3,3)
	acc_plot('kal_cur',kal_cur_d1,1)
	acc_plot('kal_cur',kal_cur_d2,2)
	acc_plot('kal_cur',kal_cur_d3,3)

	
