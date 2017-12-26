import numpy as np
import scipy.io
import tensorflow as tf

from AE.models.DAE_model_tensorflow import AdditiveDAE


# pre_process data



def loaddata():
    print("Loading data training set")
    matfile = scipy.io.loadmat('../data/preprocessdata/trainset.mat')
    trainset = matfile['trainset']
    load_fn_val = '../data/preprocessdata/valset.mat'
    load_data_val = scipy.io.loadmat(load_fn_val)
    val_set = load_data_val['valset']
    return trainset, val_set

training_epochs = 300
batch_size = 100
display_step = 1
stack_size = 3
hidden_size = [8, 4, 2]
name_scope = ['1','2','3']
input_n_size = []

train_set, val_set = loaddata()


# def get_block_data(data, batch_size):

sdae = []
for i in xrange(stack_size):
    if i == 0:
        ae = AdditiveDAE(n_input= 8, name_scope='name'+name_scope[i],
                         n_hidden=hidden_size[i],
                         transfer_function=tf.nn.softplus,
                         optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                         scale=0.01)
        # ae._initialize_weights()
        sdae.append(ae)

    else:
        ae = AdditiveDAE(n_input=hidden_size[i-1],
                         name_scope='name'+name_scope[i],
                         n_hidden=hidden_size[i],
                         transfer_function=tf.nn.softplus,
                         optimizer=tf.train.AdamOptimizer(learning_rate=0.001),
                         scale=0.01)
        #ae._initialize_weights()
        sdae.append(ae)

w = []
b = []
Hidden_feature = [] # save the feature of every ae
X_train = np.array([0])
for j in xrange(stack_size):
    if j == 0:
        X_train = np.array(train_set)
        X_test = np.array(val_set)
    else:
        X_train_pre = X_train
        X_test_pre = X_test
        X_train = sdae[j-1].transform(X_train_pre)
        X_test = sdae[j-1].transform(X_test_pre)
        print (X_train.shape)
        Hidden_feature.append(X_train)

    for epoch in range(training_epochs):
        avg_cost_train = 0.
        total_bath_train = int(len(X_train) / batch_size)

        for k in range(total_bath_train):
            batch_xs = X_train[k*batch_size:(k+1)*batch_size]
            cost = sdae[j].partial_fit(batch_xs)
            avg_cost_train += cost / len(X_train) * batch_size

        avg_cost_val = 0.
        total_bath_val = int(len(X_test) / batch_size)

        for k in range(total_bath_val):
            batch_xs_val = X_test[k*batch_size:(k+1)*batch_size]
            val_cost = sdae[j].partial_fit_re_cost_only(batch_xs_val)
            avg_cost_val += val_cost / len(X_test) * batch_size

        print "Epoch %02d/%02d average cost: %.6f average val cost: %.6f" % (epoch+1,training_epochs,avg_cost_train,avg_cost_val)

    weight = sdae[j].getWeights()
    w.append(weight)
    print weight
    b.append(sdae[j].getBiases())
    print sdae[j].getBiases()
    print 'end!'