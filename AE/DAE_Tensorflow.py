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

params = [
    [8, 4, 4, 1, 1, 8, 0],
    [8, 4, 4, 1, 1, 8, 0.1],
    [8, 4, 4, 1, 1, 8, 0.01],
    [8, 4, 4, 1, 1, 8, 0.001],
    [8, 4, 4, 1, 1, 8, 0.0001],
    [8, 4, 4, 1, 1, 8, 0.00001],
    [8, 4, 4, 2, 2, 8, 0],
    [8, 4, 4, 2, 2, 8, 0.1],
    [8, 4, 4, 2, 2, 8, 0.01],
    [8, 4, 4, 2, 2, 8, 0.001],
    [8, 4, 4, 2, 2, 8, 0.0001],
    [8, 4, 4, 2, 2, 8, 0.00001],
    [8, 6, 6, 4, 4, 8, 0],
    [8, 6, 6, 4, 4, 8, 0.1],
    [8, 6, 6, 4, 4, 8, 0.01],
    [8, 6, 6, 4, 4, 8, 0.001],
    [8, 6, 6, 4, 4, 8, 0.0001],
    [8, 6, 6, 4, 4, 8, 0.00001],
    [8, 7, 7, 6, 6, 8, 0],
    [8, 7, 7, 6, 6, 8, 0.1],
    [8, 7, 7, 6, 6, 8, 0.01],
    [8, 7, 7, 6, 6, 8, 0.001],
    [8, 7, 7, 6, 6, 8, 0.0001],
    [8, 7, 7, 6, 6, 8, 0.00001]
]
results = [None] *24
for pa in range(24):

    n_input = params[pa][0]
    n_hidden_1 = params[pa][1]  #6
    n_hidden_2 = params[pa][2]  #6
    n_hidden_3 = params[pa][3]  #4
    n_hidden_4 = params[pa][4]  #4
    n_output = params[pa][5]
    # PLACEHOLDERS
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_output])
    dropout_keep_prob = tf.placeholder("float")

    # WEIGHTS
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_3])),
        'h3': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
        'h4': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_output]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_3])),
        'b3': tf.Variable(tf.random_normal([n_hidden_4])),
        'b4': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_output]))
    }


    # MODEL
    def denoise_auto_encoder(_X, _weights, _biases, _keep_prob):

        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
        layer_1out = tf.nn.dropout(layer_1, _keep_prob)

        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1out, _weights['h2']), _biases['b2']))
        layer_2out = tf.nn.dropout(layer_2, _keep_prob)

        layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2out, _weights['h3']), _biases['b3']))
        layer_3out = tf.nn.dropout(layer_3, _keep_prob)

        layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3out, _weights['h4']), _biases['b4']))
        layer_4out = tf.nn.dropout(layer_4, _keep_prob)

        return tf.nn.sigmoid(tf.matmul(layer_4out, _weights['out']) + _biases['out'])


    # MODEL AS A FUNCTION
    reconstruction = denoise_auto_encoder(x, weights, biases, dropout_keep_prob)
    print ("Network Ready!")


    # Train
    TRAIN_FLAG = 1
    epochs = 15000
    batch_size = 100
    disp_step = 10
    noisy_rate = params[pa][6]
    dropout_keep_prob_rate = 1
    lr = 0.001

    # COST
    cost = tf.reduce_mean(tf.pow(reconstruction-y, 2)) #???????????
    # OPTIMIZER
    optm = tf.train.AdamOptimizer(learning_rate = lr).minimize(cost)
    # INITIALIZER
    init = tf.initialize_all_variables()
    print ("Function Ready!")




    savedir = "tmp/"
    saver   = tf.train.Saver(max_to_keep=1)
    print ("SAVER READY")


    sess = tf.Session()
    sess.run(init)

    print 'Length of training_set:', len(train_set)
    print 'Length of val_set:', len(val_set)
    best_cost = 1
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

            if (total_cost_val / num_batch_val) < best_cost:
                best_cost = total_cost_val / num_batch_val

            print ("param %02d Epoch %02d/%02d average cost: %.6f average val cost: %.6f"
                 % (pa, epoch, epochs, total_cost/num_batch, total_cost_val/num_batch_val))

            if epoch == epochs - 1:
                print best_cost

    results[pa] = best_cost

print len(results)
for i in range(len(results)):
    print 'params',params[i], results[i]
