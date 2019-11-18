import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import numpy as np

class AutoEncoder:
    def __init__(self,num_features,num_latent_node, activation_fun = 'relu', lamda = 0):
        self.model = Sequential([Dense(num_latent_node, activation= activation_fun, 
                          input_shape=(num_features,), kernel_regularizer = tf.keras.regularizers.l2(lamda)),
                                 Dense(num_features)])
        
    def model_settings(self, loss_function='mse', opt_methods='sgd'):
        self.model.compile(opt_methods, loss = loss_function)
        
    
    def setting_train_test_DS(self, X_train, X_test):
        self.X_train = X_train
        self.X_test = X_test
        
    def fit(self, batch_size = 32, epochs = 30):
        self.model.fit(self.X_train, self.X_train, batch_size=batch_size, epochs=epochs)
        
    def get_predict_train(self):
        self.recustr_rap_train = self.model.predict(self.X_train)
        return self.recustr_rap_train
    
    def predict_test(self):
        self.recustr_rap_test = self.model.predict(self.X_test)
        return self.recustr_rap_test
        
    def get_error_recustruction_train(self):
        self.error_e_train = np.mean((self.X_train-self.recustr_rap_train)**2, axis=1)
        return self.error_e_test
        
    def get_error_recustruction_test(self):
        self.error_e_test = np.mean((self.X_test-self.recustr_rap_test)**2, axis=1)
        return self.error_e_test
    
