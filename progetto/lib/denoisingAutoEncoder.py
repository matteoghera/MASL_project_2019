from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.ensemble._hist_gradient_boosting.tests import test_loss
import numpy as np


class DenoisingAutoencoder(AutoEncoder):
    def __init__(self, num_features, activation_fun = 'relu', lamda = 0):
        super().__init__(num_features, activation_fun = 'relu', lamda = 0)
        
  
