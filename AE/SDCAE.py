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
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

def SDCAE (Hy_W_conv_filters, Hy_W_kernel_size, Hy_W_MaxPooling_size, index):
    input_img = Input(shape=(8, 1))  # adapt this if using `channels_first` image data format
    x = 0
    encoded = x
    for i in range(index):
        if i == 0:
            x = Conv1D(Hy_W_conv_filters[i],(Hy_W_kernel_size[i]), activation='relu', padding='same')(input_img)
        else:
            x = Conv1D(Hy_W_conv_filters[i], (Hy_W_kernel_size[i]), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D((Hy_W_MaxPooling_size[i]),padding='same')(x)
        encoded = x
    decoded = 0
    for i in range(index):
        if i == 0:
            x = Conv1D(Hy_W_conv_filters[index - i - 1], (Hy_W_kernel_size[index - i - 1]), activation='relu', padding='same')(encoded)
        elif index == index:
            x = Conv1D(Hy_W_conv_filters[index - i - 1], (1), activation='relu')(x)
        else:
            x = Conv1D(Hy_W_conv_filters[index - i - 1], (Hy_W_kernel_size[index - i - 1]), activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = UpSampling1D((Hy_W_MaxPooling_size[index - i - 1]))(x)
        decoded = Conv1D(1,(2),activation=None,padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')
    return autoencoder

# Function to load data
def loaddata():
    print("Loading data training set")
    matfile = scipy.io.loadmat('../data/preprocessdata/trainset.mat')
    trainset = matfile['trainset']
    load_fn_val = '../data/preprocessdata/valset.mat'
    load_data_val = scipy.io.loadmat(load_fn_val)
    val_set = load_data_val['valset']
    return trainset,val_set


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
            warnings.warn('AdvancedLearnignRateScheduler'
                          ' requires %s available!' %
                          (self.monitor), RuntimeWarning)
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

import numpy as np
X_train, X_val = loaddata() # Loading data
noise_factor = 0.1
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_val_noisy = X_val + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_val.shape)

from keras.callbacks import TensorBoard

X_train = np.expand_dims(X_train, axis=2)
X_val = np.expand_dims(X_val, axis=2)
X_train_noisy = np.expand_dims(X_train_noisy, axis=2)
X_val_noisy = np.expand_dims(X_val_noisy, axis=2)

wf = [4, 4, 4]
wk = [2, 2, 2]
wp = [2, 2, 2]

weights_matrix = []
for i in range(3):
    autoencoder = SDCAE(wf,wk,wp,i+1)
    for j in range(i):
        layer = autoencoder.layers[j]
        if i != 0:
            weights_matrix = weights_matrix
            if len(weights_matrix[j]) == 1:
                weights = []
                weight1 = weights_matrix[j][0][0]
                weights.append(weight1)
                weight2 = weights_matrix[j][0][1]
                weights.append(np.reshape(weight2, (weight2.shape[1])))
                layer.set_weights(weights)
            else:
                layer.set_weights(weights_matrix[j])
    autoencoder.fit(X_train_noisy, X_train,
                    epochs=1000,
                    batch_size=8,
                    shuffle=True,
                    validation_data=(X_val_noisy, X_val),
                    callbacks=[
                        # Early stopping definition
                        EarlyStopping(monitor='val_loss', patience=30, verbose=1),
                        # Decrease learning rate by 0.1 factor
                        AdvancedLearnignRateScheduler(monitor='val_loss', patience=3, verbose=1, mode='auto',
                                                      decayRatio=0.1),
                        TensorBoard(log_dir='./tmp/DCAE')])
    weights_matrix = []
    print autoencoder.layers
    for layer in autoencoder.layers:
        weights = layer.get_weights()
        print weights
        weights_matrix.append(weights)


scipy.io.savemat('weights/DCAE.mat',mdict={'weights_matrix': weights_matrix})
decoded_imgs = autoencoder.predict(X_val)
