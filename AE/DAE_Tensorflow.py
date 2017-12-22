import numpy as np
import tensorflow as tf
import scipy.io

def loaddata():
    print("Loading data training set")
    matfile = scipy.io.loadmat('../data/preprocessdata/trainset.mat')
    trainset = matfile['trainset']
    load_fn_val = '../data/preprocessdata/valset.mat'
    load_data_val = scipy.io.loadmat(load_fn_val)
    val_set = load_data_val['valset']
    return trainset, val_set


train_set, val_set = loaddata()
print 'Data Ready!'

n_input = 8
n_hidden_1 = 4
n_hidden_2 = 4
n_output = 8

# PLACEHOLDERS
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])
dropout_keep_prob = tf.placeholder("float")

# WEIGHTS
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}


# MODEL
def denoise_auto_encoder(_X, _weights, _biases, _keep_prob):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer_1out = tf.nn.dropout(layer_1, _keep_prob)
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['h2']), _biases['b2']))
    layer_2out = tf.nn.dropout(layer_2, _keep_prob)
    return tf.nn.sigmoid(tf.matmul(layer_2out, _weights['out']) + _biases['out'])


# MODEL AS A FUNCTION
reconstruction = denoise_auto_encoder(x, weights, biases, dropout_keep_prob)
print ("Network Ready!")


# COST
cost = tf.reduce_mean(tf.pow(reconstruction-y, 2)) #???????????
# OPTIMIZER
optm = tf.train.AdamOptimizer(0.01).minimize(cost)
# INITIALIZER
init = tf.initialize_all_variables()
print ("Function Ready!")


# Train
TRAIN_FLAG = 1
epochs = 3000
batch_size = 100
disp_step = 10
noisy_rate = 0.3
dropout_keep_prob_rate = 1.0

savedir = "tmp/"
saver   = tf.train.Saver(max_to_keep=1)
print ("SAVER READY")


sess = tf.Session()
sess.run(init)

print 'Length of training_set:', len(train_set)
print 'Length of val_set:', len(val_set)

if TRAIN_FLAG:
    print ("Training start!")
    for epoch in range(epochs):

        num_batch  = int(len(train_set)/batch_size)
        total_cost = 0.

        for i in range(num_batch):
            batch_xs = train_set[i*batch_size:(i+1)*batch_size]
            batch_xs_noisy = batch_xs + noisy_rate*np.random.randn(batch_size, 8)
            feeds = {x: batch_xs_noisy, y: batch_xs, dropout_keep_prob: dropout_keep_prob_rate}
            sess.run(optm, feed_dict=feeds)
            total_cost += sess.run(cost, feed_dict=feeds)

        num_batch_val = int(len(val_set)/batch_size)
        total_cost_val = 0.

        for i in range(num_batch_val):
            batch_xs_val = val_set[i * batch_size:(i + 1) * batch_size]
            batch_xs_val_noisy = batch_xs_val + noisy_rate * np.random.randn(batch_size, 8)
            feeds_val = {x: batch_xs_val_noisy, y: batch_xs_val, dropout_keep_prob: dropout_keep_prob_rate}
            sess.run(optm, feed_dict=feeds_val)
            total_cost_val += sess.run(cost, feed_dict=feeds_val)

        if epoch % disp_step == 0:
            print ("Epoch %02d/%02d average cost: %.6f average val cost: %.6f"
                   % (epoch, epochs, total_cost/num_batch, total_cost_val/num_batch_val))