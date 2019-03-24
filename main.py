'''
CS 559: Deep Learning
Homework: Age Estimation using TensorFlow
Muhammed Cavusoglu (21400653) and Kemal Buyukkaya (21200496)
'''

import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

def cnn_model():
    print("model")
    
def main():
    training_data, training_labels, validation_data, validation_labels, test_data, test_labels = load_dataset()
    
    print len(training_data), len(training_labels), len(validation_data), len(validation_labels), len(test_data), len(test_labels)

def load_dataset():
    # Training set
    training_data = []
    training_labels = []
    
    tr_path = 'UTKFace_downsampled/training_set'
    for filename in os.listdir(tr_path):
        training_labels.append(int(filename[:3]))
        img_data = cv2.imread(os.path.join(tr_path, filename), cv2.IMREAD_GRAYSCALE)
    
        # RESIZE???
        img_data = cv2.resize(img_data, (28, 28))
        
        training_data.append(img_data)
    
    # Validation set
    validation_data = []
    validation_labels = []
    
    v_path = 'UTKFace_downsampled/validation_set'
    for filename in os.listdir(v_path):
        validation_labels.append(int(filename[:3]))
        img_data = cv2.imread(os.path.join(v_path, filename), cv2.IMREAD_GRAYSCALE)
        
        # RESIZE???
        img_data = cv2.resize(img_data, (28, 28))
        
        validation_data.append(img_data)
    
    # Test set
    test_data = []
    test_labels = []
    
    t_path = 'UTKFace_downsampled/test_set'
    for filename in os.listdir(t_path):
        test_labels.append(int(filename[:3]))
        img_data = cv2.imread(os.path.join(t_path, filename), cv2.IMREAD_GRAYSCALE)
        
        # RESIZE???
        img_data = cv2.resize(img_data, (28, 28))
        
        test_data.append(img_data)
    
    return training_data, training_labels, validation_data, validation_labels, test_data, test_labels
    
if __name__ == "__main__":
  # some run fn. like tf.app.run()
  main()