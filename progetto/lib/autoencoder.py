import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

class AutoEncoder:
    def __init__(self,num_features,num_latent_node, activation_fun = 'relu', lamda = 0, min_delta = 0.0004, patience = 5):
        encoder = Dense(units = num_latent_node,
                                   input_shape=(num_features,),
                                   activation = activation_fun,
                        kernel_regularizer = tf.keras.regularizers.l2(lamda))
        
        decoder = Dense(units = num_features)
        self.autoEncoder = Sequential([encoder, decoder])
        self.early = EarlyStopping(monitor='mean_squared_error',min_delta = min_delta, patience = patience)
        
    def model_settings(self, loss_function='mse', opt_methods='adam'):
        self.autoEncoder.compile(opt_methods, loss = loss_function, metrics = ['mse'])
        
    def setting_train_test_DS(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        
    def fit(self, batch_size = 32, epochs = 30, shuffle = True):
        return self.autoEncoder.fit(self.X_train, self.X_train, batch_size=batch_size, epochs=epochs, callbacks = [self.early],
                                    shuffle=shuffle, validation_data=(self.X_test, self.X_test))
        
    def predict(self):
        self.X_test_hat = self.autoEncoder.predict(self.X_test)
                
    def get_error_recustruction_test(self):
        return np.sqrt(np.mean((self.X_test - self.X_test_hat)**2, axis=1))
    