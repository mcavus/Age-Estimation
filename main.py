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

# For colab
'''
from google.colab import drive
drive.mount('/content/drive')
'''

tf.logging.set_verbosity(tf.logging.INFO)

def cnn_model(features, labels, mode):
    # Input Layer
    # Reshape input to 4-D tensor: [batch_size, width, height, channels]
    input_layer = tf.reshape(features["x"], [-1, 88, 88, 1])
    
    # Conv1 Layer
    # Compute 32 features using a 5x5 filter with ReLU activation.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 88, 88, 1]
    # Output Tensor Shape: [batch_size, 88, 88, 32]
    conv1 = tf.contrib.layers.conv2d(
        inputs = input_layer,
        num_outputs = 32,
        kernel_size = [5, 5],
        padding= 'SAME',
        activation_fn = tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    
    # Pooling1 Layer
    # Max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 88, 88, 32]
    # Output Tensor Shape: [batch_size, 44, 44, 32]
    pool1 = tf.contrib.layers.max_pool2d(inputs = conv1, kernel_size = [2, 2], stride = 2)
    
    bn1 = tf.contrib.layers.batch_norm(inputs = pool1, activation_fn = None)

    # Conv2 Layer
    # Compute 64 features using a 5x5 filter.
    # Padding is added to preserve width and height.
    # Input Tensor Shape: [batch_size, 44, 44, 32]
    # Output Tensor Shape: [batch_size, 44, 44, 64]
    conv2 = tf.contrib.layers.conv2d(
        inputs = bn1,
        num_outputs = 64,
        kernel_size = [5, 5],
        padding= 'SAME',
        activation_fn = tf.nn.relu,
        weights_initializer=tf.contrib.layers.xavier_initializer()
    )
    
    # Pooling 2 Layer
    # Max pooling layer with a 2x2 filter and stride of 2
    # Input Tensor Shape: [batch_size, 44, 44, 64]
    # Output Tensor Shape: [batch_size, 22, 22, 64]
    pool2 = tf.contrib.layers.max_pool2d(inputs = conv2, kernel_size = [2, 2], stride = 2)
    
    bn2 = tf.contrib.layers.batch_norm(inputs = pool2, activation_fn = None)
    
    # Flatten tensor into a batch of vectors
    # Input Tensor Shape: [batch_size, 22, 22, 64]
    # Output Tensor Shape: [batch_size, 22 * 22 * 64]
    pool2_flat = tf.reshape(bn2, [-1, 22 * 22 * 64])

    # FC Layer with 1024 neurons
    # Input Tensor Shape: [batch_size, 7 * 7 * 64]
    # Output Tensor Shape: [batch_size, 1024]
    fc1 = tf.contrib.layers.fully_connected(inputs = pool2_flat, num_outputs = 1024, activation_fn = None, 
                                           weights_initializer=tf.contrib.layers.xavier_initializer())
    
    bn3 = tf.contrib.layers.batch_norm(inputs = fc1, activation_fn = tf.nn.relu)

    fc2 = tf.contrib.layers.fully_connected(inputs = bn3, num_outputs = 1024, activation_fn = None, 
                                           weights_initializer=tf.contrib.layers.xavier_initializer())
    
    bn4 = tf.contrib.layers.batch_norm(inputs = fc2, activation_fn = tf.nn.relu)

    # Dropout (0.6 probability for keeping the element)
    dropout = tf.contrib.layers.dropout(inputs = bn4, keep_prob = 0.6, is_training = (mode == tf.estimator.ModeKeys.TRAIN))
    
    # Regression Layer
    # Input Tensor Shape: [batch_size, 1024]
    # Output Tensor Shape: [batch_size, 1]
    regression = tf.contrib.layers.fully_connected(inputs = dropout, num_outputs = 1, activation_fn = None)
    
    if mode == tf.estimator.ModeKeys.PREDICT:
      return tf.estimator.EstimatorSpec(mode = mode, predictions = regression)
    
    # Loss function
    loss = tf.losses.mean_squared_error(labels = labels, predictions = regression)
    
    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
      optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08, use_locking=False, name='Adam')
      train_op = optimizer.minimize(
          loss = loss,
          global_step = tf.train.get_global_step())
      return tf.estimator.EstimatorSpec(mode = mode, loss = loss, train_op = train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "MAE": tf.metrics.mean_absolute_error(labels = labels, predictions = regression)
    }
    
    return tf.estimator.EstimatorSpec(mode = mode, loss = loss, eval_metric_ops = eval_metric_ops)

def main(argv):
    # Load the data
    # Use below to re-generate npy files
    training_data, training_labels, validation_data, validation_labels, test_data, test_labels = _load_dataset()
    
    # For colab
    '''
    training_data = np.load("/content/drive/My Drive/Colab Notebooks/training_data.npy")
    training_labels = np.load("/content/drive/My Drive/Colab Notebooks/training_labels.npy")
    validation_data = np.load("/content/drive/My Drive/Colab Notebooks/validation_data.npy")
    validation_labels = np.load("/content/drive/My Drive/Colab Notebooks/validation_labels.npy")
    test_data = np.load("/content/drive/My Drive/Colab Notebooks/test_data.npy")
    test_labels = np.load("/content/drive/My Drive/Colab Notebooks/test_labels.npy")
    '''
    
    # For local 
    '''
    training_data = np.load("training_data.npy")
    training_labels = np.load("training_labels.npy")
    validation_data = np.load("validation_data.npy")
    validation_labels = np.load("validation_labels.npy")
    test_data = np.load("test_data.npy")
    test_labels = np.load("test_labels.npy")
    '''
    
    # Estimator
    age_estimator = tf.estimator.Estimator(model_fn = cnn_model)
    
    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x = {"x": training_data},
        y = training_labels,
        batch_size = 256,
        num_epochs = None,
        shuffle=True)
        
    age_estimator.train(
        input_fn = train_input_fn,
        steps=1200)

    # Evaluate the model and print results
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, y = test_labels, num_epochs = 1, shuffle = False)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": validation_data}, y = validation_labels, num_epochs = 1, shuffle = False)
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": training_data}, y = training_labels, num_epochs = 1, shuffle = False)

    eval_results = age_estimator.evaluate(input_fn = eval_input_fn)
    print(eval_results)
    
def _load_dataset():
    # Training set
    training_data = []
    training_labels = []
    
    tr_path = 'UTKFace_downsampled/training_set'
    for filename in os.listdir(tr_path):
        training_labels.append(int(filename[:3]))
        img_data = cv2.imread(os.path.join(tr_path, filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img_data = cv2.resize(img_data, (88, 88)) # Resize
        training_data.append(img_data)
        
    training_data = np.array(training_data, dtype='float32')
    training_labels = np.array(training_labels, dtype='float32')
    training_labels = training_labels.reshape(len(training_labels), 1)
    np.save('training_data', training_data)
    np.save('training_labels', training_labels)
    
    # Validation set
    validation_data = []
    validation_labels = []
    
    v_path = 'UTKFace_downsampled/validation_set'
    for filename in os.listdir(v_path):
        validation_labels.append(int(filename[:3]))
        img_data = cv2.imread(os.path.join(v_path, filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img_data = cv2.resize(img_data, (88, 88)) # Resize
        validation_data.append(img_data)
        
    validation_data = np.array(validation_data, dtype='float32')
    validation_labels = np.array(validation_labels, dtype='float32')
    validation_labels = validation_labels.reshape(len(validation_labels), 1)
    np.save('validation_data', validation_data)
    np.save('validation_labels', validation_labels)
    
    # Test set
    test_data = []
    test_labels = []
    
    t_path = 'UTKFace_downsampled/test_set'
    for filename in os.listdir(t_path):
        test_labels.append(int(filename[:3]))
        img_data = cv2.imread(os.path.join(t_path, filename), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img_data = cv2.resize(img_data, (88, 88)) # Resize
        test_data.append(img_data)
        
    test_data = np.array(test_data, dtype='float32')
    test_labels = np.array(test_labels, dtype='float32')
    test_labels = test_labels.reshape(len(test_labels), 1)
    np.save('test_data', test_data)
    np.save('test_labels', test_labels)
    
    return training_data, training_labels, validation_data, validation_labels, test_data, test_labels
    
if __name__ == "__main__":
  tf.app.run()