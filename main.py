'''
CS 559: Deep Learning
Homework: Age Estimation using TensorFlow
Muhammed Cavusoglu (21400653) and Kemal Buyukkaya (21200496)
'''

import os
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model(features, labels, mode):
    # Input Layer
    # Reshape input to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features, [-1, 28, 28, 1])
    
    # Conv1 Layer
    # Compute 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 28, 28, 1]
    # Output Tensor Shape: [batch_size, 28, 28, 32]
    conv1 = tf.contrib.layers.conv2d(
        inputs = input_layer,
        num_outputs = 32,
        kernel_size = [5, 5],
        padding= 'SAME',
        activation_fn = tf.nn.relu
    )
    
    # Pooling1 Layer
    # Max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 28, 28, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 32]
    pool1 = tf.contrib.layers.max_pool2d(inputs = conv1, kernel_size = [2, 2], stride = 2)
    
    # Conv2 Layer
    # Compute 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 14, 14, 32]
    # Output Tensor Shape: [batch_size, 14, 14, 64]
    conv2 = tf.contrib.layers.conv2d(
        inputs = pool1,
        num_outputs = 64,
        kernel_size = [5, 5],
        padding= 'SAME',
        activation_fn = tf.nn.relu
    )
    
    # Pooling 2 Layer
    # Max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 14, 14, 64]
    # Output Tensor Shape: [batch_size, 7, 7, 64]
    pool2 = tf.contrib.layers.max_pool2d(inputs = conv2, kernel_size = [2, 2], stride = 2)
    
    # TODO: Conv3 Layer
    
    # TODO: Pooling 3 Layer
    
    # TODO: Conv4 Layer
    
    # TODO: Pooling 4 Layer
    
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 7, 7, 64]
    # Output Tensor Shape: [batch_size, 7 * 7 * 64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # FC Layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    fc = tf.contrib.layers.fully_connected(inputs = pool2_flat, num_outputs = 1024, activation_fn = tf.nn.relu)
    
    # Dropout (0.6 probability for keeping the element)
    dropout = tf.contrib.layers.dropout(inputs = fc, keep_prob = 0.6, is_training = (mode == tf.estimator.ModeKeys.TRAIN))
    
    # FIXME: Regression Layer
    regression = tf.contrib.layers.fully_connected(inputs = dropout, num_outputs = 1)
    
    # TODO: Predictions and loss
    print regression
    loss = tf.losses.mean_squared_error(labels = labels, predictions = regression)
    
def main():
    training_data, training_labels, validation_data, validation_labels, test_data, test_labels = load_dataset()
    
    # estimator
    cnn_model(training_data, training_labels, None)

def load_dataset():
    # Training set
    training_data = np.array([], dtype="float32")
    training_labels = []
    
    tr_path = 'UTKFace_downsampled/training_set'
    for filename in os.listdir(tr_path):
        training_labels.append(int(filename[:3]))
        img_data = cv2.imread(os.path.join(tr_path, filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
        # RESIZE???
        img_data = cv2.resize(img_data, (28, 28))
        
        np.append(training_data, np.array(img_data))
    
    # FIXME: Validation set
    validation_data = np.array([])
    validation_labels = []
    
    v_path = 'UTKFace_downsampled/validation_set'
    for filename in os.listdir(v_path):
        validation_labels.append(int(filename[:3]))
        img_data = cv2.imread(os.path.join(v_path, filename), cv2.IMREAD_GRAYSCALE)
        
        # RESIZE???
        img_data = cv2.resize(img_data, (28, 28))
        
        np.append(validation_data, np.array(img_data))
    
    # FIXME: Test set
    test_data = np.array([])
    test_labels = []
    
    t_path = 'UTKFace_downsampled/test_set'
    for filename in os.listdir(t_path):
        test_labels.append(int(filename[:3]))
        img_data = cv2.imread(os.path.join(t_path, filename), cv2.IMREAD_GRAYSCALE)
        
        # RESIZE???
        img_data = cv2.resize(img_data, (28, 28))
        
        np.append(test_data, np.array(img_data))
    
    return training_data, training_labels, validation_data, validation_labels, test_data, test_labels
    
if __name__ == "__main__":
  # some run fn. like tf.app.run()
  main()