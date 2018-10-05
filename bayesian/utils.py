import numpy as np
import keras.backend as K




def get_weights(model):
    #weights = [tensor for tensor in model.trainable_weights]
    '''
    weights = []
    for layer in model.layers:
        weights.append(layer.get_weights())
    '''
    return model.get_weights()


def get_gradients(model):
    '''
    Return the gradient of every trainable weight in model
    '''
    weights = [tensor for tensor in model.trainable_weights]
    optimizer = model.optimizer

    return optimizer.get_gradients(model.total_loss, weights)


def get_weight_grad(model, inputs, outputs):
    """ Gets gradient of model for given inputs and outputs for all weights"""
    grads = model.optimizer.get_gradients(model.total_loss, model.trainable_weights)
    symb_inputs = (model._feed_inputs + model._feed_targets + model._feed_sample_weights)
    f = K.function(symb_inputs, grads)
    x, y, sample_weight = model._standardize_user_data(inputs, outputs)
    output_grad = f(x + y + sample_weight)
    return output_grad


from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from skimage.util import random_noise


def load_mnist_drift(train_num=20000,test_num=500,split=False):
    '''
    Return only training data.
    Include:
        - Mnist data
        - Gaussian Mnist data
        - Poisson Mnist data
        - Salt Mnist data

    Para:
        - train_num: number of training samples
        - test_num: number of test samples
        - split: return whole dataset or part of it
    '''

    num_classes = 10    
    (X_train,y_train),(X_test,y_test) =mnist.load_data()

    img_x, img_y = 28, 28
    input_shape = (img_x,img_y,1)
    X_train = X_train.reshape(X_train.shape[0],img_x,img_y,1) /255
    X_test = X_test.reshape(X_test.shape[0],img_x,img_y,1)  /255
    y_train = to_categorical(y_train, num_classes) 
    y_test = to_categorical(y_test,num_classes) 

    if split:
        X_train,y_train = get_new_samples(X_train,y_train,train_num)
        X_test,y_test = get_new_samples(X_test,y_test,test_num)

    X_train_gaussian = random_noise(X_train,mode='gaussian',var=0.2)
    X_train_salt = random_noise(X_train,mode='salt',amount=0.2)
    X_test_gaussian = random_noise(X_test,mode='gaussian',var=0.2)
    X_test_salt = random_noise(X_test,mode='salt',amount=0.2)



    '''
    import matplotlib.pyplot as plt
    fig = plt.figure()
    for i,img in enumerate([X_train,X_train_gaussian,X_train_salt]):
        
        fig.add_subplot(1,3,i+1)
        plt.imshow(img[0].reshape(img_x,img_y))
    plt.savefig('data.png')
    plt.show()
    
    '''
    

    #I did not do preprocess because it will 
    #X_train = preprocess(X_train)
    #X_train_gaussian = preprocess(X_train_gaussian)
    #X_train_poisson = preprocess(X_train_poisson)
    #X_train_salt = preprocess(X_train_salt)

    X_train,y_train = np.concatenate((X_train,X_train_gaussian,X_train_salt)),np.concatenate((y_train,y_train,y_train))
    X_test,y_test = np.concatenate((X_test,X_test_gaussian,X_test_salt)),np.concatenate((y_test,y_test,y_test))

    return X_train,y_train,X_test,y_test

def preprocess(X):
    X = X.astype('float32')
    X /= 255
    return X


def get_new_samples(X,y,num):
    'Function to split the data'
    X_new = X[:num]
    y_new = y[:num]
    return X_new,y_new
    

