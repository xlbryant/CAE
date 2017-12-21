from keras.layers import Input,Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, Activation
from keras.models import Model
from keras import backend as K
import matplotlib.pyplot as plt
import scipy.io
import tensorflow as tf
from keras.callbacks import Callback,warnings
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

conv_filters = 4
kernel_size = 2
Maxpooling_size = 2
input_img = Input(shape=(8, 1))

x = Conv1D(conv_filters, (kernel_size), activation='relu', padding='same')(input_img)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D((Maxpooling_size), padding='same')(x)
x = Conv1D(conv_filters, (kernel_size), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = MaxPooling1D((Maxpooling_size), padding='same')(x)
x = Conv1D(conv_filters, (kernel_size), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
encoded = MaxPooling1D((Maxpooling_size), padding='same')(x)

# at this point the representation is (4, 4, 8) i.e. 128-dimensional

x = Conv1D(conv_filters, (kernel_size), activation='relu', padding='same')(encoded)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling1D((Maxpooling_size))(x)
x = Conv1D(conv_filters, (kernel_size), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling1D((Maxpooling_size))(x)
x = Conv1D(conv_filters, (1), activation='relu')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
x = UpSampling1D((Maxpooling_size))(x)
decoded = Conv1D(1, (kernel_size), activation='relu', padding='same')(x)
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

import numpy as np

# Function to load data
def loaddata():
    print("Loading data training set")
    matfile = scipy.io.loadmat('../data/preprocessdata/trainset.mat')
    trainset = matfile['trainset']
    load_fn_val = '../data/preprocessdata/valset.mat'
    load_data_val = scipy.io.loadmat(load_fn_val)
    val_set = load_data_val['valset']
    return trainset, val_set


# Callback method for reducing learning rate during training
class AdvancedLearnignRateScheduler(Callback):
    def __init__(self, monitor='val_loss', patience=0, verbose=0, mode='auto', decayRatio=0.1):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.wait = 0
        self.decayRatio = decayRatio

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('Mode %s is unknown, '
                          'fallback to auto mode.'
                          % (self.mode), RuntimeWarning)
            mode = 'auto'
        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        current_lr = K.get_value(self.model.optimizer.lr)
        print("\nLearning rate:", current_lr)
        if current is None:
            warnings.warn('AdvancedLearnignRateScheduler requires %s available!' %(self.monitor), RuntimeWarning)
        if self.monitor_op(current, self.best):
            self.best = current
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print('\nEpoch %05d: reducing learning rate' % (epoch))
                    assert hasattr(self.model.optimizer, 'lr'), \
                        'Optimizer must have a "lr" attribute.'
                    current_lr = K.get_value(self.model.optimizer.lr)
                    new_lr = current_lr * self.decayRatio
                    K.set_value(self.model.optimizer.lr, new_lr)
                    self.wait = 0
            self.wait += 1

# Parameters


X_train, X_val = loaddata()
noise_factor = 0.5
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_val_noisy = X_val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_val.shape)

from keras.callbacks import TensorBoard

X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_train_noisy = np.expand_dims(X_train_noisy, axis=2)
X_val_noisy = np.expand_dims(X_val_noisy, axis=2)
autoencoder.fit(X_train_noisy, X_train,
                epochs=100,
                batch_size=8,
                shuffle=True,
                validation_data=(X_val_noisy, X_val),
                callbacks=[
                    # Early stopping definition
                    EarlyStopping(monitor='val_loss', patience=3, verbose=1),
                    # Decrease learning rate by 0.1 factor
                    AdvancedLearnignRateScheduler(monitor='val_loss', patience=1, verbose=1, mode='auto',
                                                  decayRatio=0.1),
                    TensorBoard(log_dir='./tmp/CAE')])
weights_matrix = []
for layer in autoencoder.layers:
    weights = layer.get_weights()
    weights_matrix.append(weights)
scipy.io.savemat('./weights/CAE.mat',mdict={'weights_matrix': weights_matrix})
decoded_imgs = autoencoder.predict(X_val)
# n = 10
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i+1)
#     plt.plot(range(18000), Xval[i].reshape(18000))
#     # plt.imshow(x_test[i].reshape(18000))
#     # plt.gray()
#     # ax.get_xaxis().set_visible(False)
#     # ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i+1 + n)
#     plt.plot(range(18000), decoded_imgs[i].reshape(18000))
#     # plt.imshow(decoded_imgs[i].reshape(18000))
#     # plt.gray()
#     # ax.get_xaxis().set_visible(False)
#     # ax.get_yaxis().set_visible(False)
# plt.savefig('recon_filters256_kernel32.eps', format='eps', dpi=1000)
# # plt.show()