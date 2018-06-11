"""
This is the primary file for Udacity's Traffic Sign Classifer project
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import tensorflow as tf
import cv2
from sklearn.utils import shuffle

# =================================================
#                   Load Data
# =================================================
data_directory = r"E:\Project\Udacity - Computer Vision\P2 Traffic Sign Classifier\CarND-Traffic-Sign-Classifier-Project\traffic-signs-data"
training_file = data_directory+r"\train.p"
validation_file = data_directory+r"\valid.p"
testing_file = data_directory+r"\test.p"

with open(training_file,mode='rb') as f:
    train = pickle.load(f)
with open(validation_file,mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file,mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'],train['labels']
X_valid, y_valid = valid['features'],valid['labels']
X_test, y_test = test['features'],test['labels']


# =================================================================
#                           A Basic Summary of Data
# =================================================================
n_train = len(y_train)
n_validation = len(y_valid)
n_test = len(y_test)
image_shape = X_train.shape[-3:] # width*length*color channel
n_classes = len(np.unique(y_train))

print("Number of training examples", n_train)
print("Number of testing examples", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
# =================================================================
#                       Exploratory Visualization
# =================================================================


""" frequency plot of traffic sign class """
def class_dist_viz(label_column,data_name):
    """ 
        plot class distribution bar plot in training, validation or testing set
    
    parameters: 
            
    -------------
    
    label_column: a vector of traffic class
            
    data_name: training, validation or testing set. The string will be put as figure
                       title.
    """
    _,ax = plt.subplots()
    ax.set_title(data_name)
    sns.countplot(x=label_column,ax=ax)
    
class_dist_viz(y_train,"training set")
class_dist_viz(y_valid,"validation set")
class_dist_viz(y_test,"testing set")

#
#
#
#

    
""" plot an image for each traffic sign class """

def find_ind(label_column):
    """ return the index of first matched item for each sign class
        
    parameters
    
    ----------
    label_column: an array of traffic class
    
    """
    unique_label = np.unique(label_column)
    label_list = list(label_column)
    return [label_list.index(x) for i,x in enumerate(unique_label)]


def plot_img(img,nrow,ncol):
    """
        plot image in a n(row)*n(column) matrix
    
    parameters:
        
    ----------
    img: a single image (array of pixels)
    
    """
    plt.subplot(nrow,ncol,i+1)
    plt.imshow(img)

class_ind = find_ind(y_train) # index for each class

fig = plt.figure()
num_class = len(class_ind)
for i in range(num_class): # 
    plot_img(X_train[class_ind[i],:,:,:],7,7)
plt.show()

"""
From the plot above, we can see some images are not bright enough. It might potentially cause problem for
the model to recognize them correctly.
"""
# ====================================================================
#                   Data Preprocessing
# ====================================================================

# ======================
# Data Normalization
# ======================
def max_norm(x):
    """
        normalize image data by deviding the maximum value of each pixel
    
    parameters:
        
    ------------
    x: image array of shape: sample_size, length, width, color_channel
    
    """
    x_flat = np.copy(x)
    feature_len = x.shape[1]*x.shape[2]*x.shape[3] # length of flattened image
    x_flat = np.reshape(x_flat,(x.shape[0],feature_len)) # reshape
    x_max = np.amax(x_flat,axis=0) # column max
    x_flat = x_flat/x_max # normalize
    x_norm = np.reshape(x_flat,x.shape)
    return x_norm

X_train = max_norm(X_train)
X_valid = max_norm(X_valid)
X_test = max_norm(X_test)


# ===============================
# RGB to Grayscale Transformation
# ===============================
def grayscale(img):
    """ apply grayscale transform 
	
	parameters:
	-----------------------------
	img: an array of image of shape [batch, height, width, channel]
        
    
	output:
	-----------------------------
	
	return image with only one channel
    """
    img_gray = np.empty((img.shape[0],img.shape[1],img.shape[2],1))
    for ind,each in enumerate(img):
        img_gray[ind] = cv2.cvtColor(each,cv2.COLOR_RGB2GRAY).reshape((img.shape[1],img.shape[2],1))
    
    return img_gray

# X_train = grayscale(X_train) 
# X_valid = grayscale(X_valid)

print(X_train.shape)
# ====================================================================
#                   Define Neural Network Architecture
# ====================================================================
def conv_net(x):
    """
        define a convoluation network for traffic sign classifier
        
    parameters:
    
    -------------
    x: image array 
    
    """
    # x = np.zeros(shape=[100,32,32,3],dtype=np.float32
    # define parameters for weights initialization
    mu = 0
    sigma = 0.1
    
    # =================================================
    # Layer 1:convolution layer + activation + pooling
    # =================================================
    
    # first convoluation layer: input size = 32*32*3, output size = 28*28*6
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5,5,3,6),mean=mu,stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_W,strides=[1,1,1,1],padding="VALID") + conv1_b
    
    # activation
    conv1 = tf.nn.relu(conv1)
    
    # pooling layer input_size = 28*28*6 output_size = 14*14*6
    conv1 = tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    # ==================================================
    # Layer 2:convoluation layer + activation + pooling
    # ==================================================
    
    # second convoluation layer: input_size = 14*14*6, output_size = 10*10*16
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5,5,6,16),mean=mu,stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1,conv2_W,strides=[1,1,1,1],padding="VALID") + conv2_b
    
    # activation 
    conv2 = tf.nn.relu(conv2)
    
    # pooling layer: input_size = 10*10*16 output_size = 5*5*16
    conv2 = tf.nn.max_pool(conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding="VALID")
    
    # ===================================================
    # Flatten Layer. input_size = 5*5*16 output_size = 400
    # ===================================================
    fc0 = tf.contrib.layers.flatten(conv2)
    
    # =====================================================================
    # Layer 3: fully connectted layer: input_size = 400, output_size = 120
    # =====================================================================
    
    # fully connected layer
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400,120),mean=mu,stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0,fc1_W) + fc1_b
    
    # activation
    fc1 = tf.nn.relu(fc1)
    
    # =====================================================================
    # Layer 4: fully connected layer: input_size = 120, output_size = 84
    # =====================================================================
    
    # fully connected layer
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120,84),mean=mu,stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_W) + fc2_b
    
    # activation
    fc2 = tf.nn.relu(fc2)
    
    # ======================================================================
    # Layer 5: fully connected layer: input_size = 84, output_size = 43
    # ======================================================================
    
    # fully connected layer
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84,num_class),mean=mu,stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(num_class))
    logits = tf.matmul(fc2,fc3_W) + fc3_b
    
    # ======================================================================
    # Regularization: add L1/L2 norm
    # ======================================================================
    reg_loss = tf.nn.l2_loss(conv1_W)+tf.nn.l2_loss(conv2_W)+tf.nn.l2_loss(fc1_W)+tf.nn.l2_loss(fc2_W)+tf.nn.l2_loss(fc3_W) 
    
    
    # ======================================================================
    # return logits for each class
    # ======================================================================
    return (logits,reg_loss)
    
# ==========================================================================
#        Building Training Pipeline in TF Computation Graph
# ==========================================================================
    
# ========================================================
#           Define Features and Labels
# ========================================================
x = tf.placeholder(tf.float32,(None,32,32,3)) # features (batch_size*width*height*color_channel)
y = tf.placeholder(tf.int32,(None)) # batch_size
one_hot_y = tf.one_hot(y,num_class) # ont hot encoding
# ========================================================
#           Model training and evaluation
# ========================================================
    
""" Hyperparameters """
rate = 0.05 # learning rate
epoch = 10
batch_size = 128
reg_term = 0.8 # regularization coefficient

""" loss and accuracy part in computational graph """
logits,reg_loss = conv_net(x) # x is the placehoder in the graph defined above
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y,logits=logits) # a vector of cross_entropy loss

# define loss
data_loss = tf.reduce_mean(cross_entropy) # average cross-entropy
loss = data_loss+reg_term*reg_loss

# optimization
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss)
correct_prediction = tf.equal(tf.argmax(one_hot_y,1),tf.argmax(logits,1)) # an array of right or wrong prediction
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) # prediction accuracy
saver = tf.train.Saver() # save trained model

""" helper function to plot accuracy and loss """
def plot_accuracy(train_log,valid_log):
    """
    plot training accuracy and validation accuracy per epoch
    
    parameters:
    -------------
    train_log: an array of training accuracy. This result is obtained from training session
    
    valid_log: an array of validation accuracy. This result is obtained from training session
    """
    
    num_log = len(train_log) # number of element in input array
    
    fig = plt.figure()
    fig.suptitle("Accuracy vs. Epoch")
    ax = fig.add_subplot(111)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.plot(np.arange(0,num_log),train_log,'r-',label='train_accuracy')
    ax.plot(np.arange(0,num_log),valid_log,'b-',label='validation_accuracy')
    ax.legend(loc=4,prop={'size':15})
    
def plot_loss(train_log,valid_log):
    """
    plot training and validation loss per epoch
    
    parameters:
    -------------
    train_log: an array of training loss. This result is obtained from training session
    
    valid_log: an array of validation loss. This result is obtained from training session
    """
    
    num_log = len(train_log) # number of element in input array
    
    fig = plt.figure()
    fig.suptitle("Loss vs. Epoch")
    ax = fig.add_subplot(111)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.plot(np.arange(0,num_log),train_log,'r-',label='train_loss')
    ax.plot(np.arange(0,num_log),valid_log,'b-',label='validation_loss')
    ax.legend(loc=4,prop={'size':15})


# ======================================================================
#                       training the model (training session)
# ======================================================================
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # initialize weights in the network
    num_example = len(X_train) # number of data in the training set
    train_acc,valid_acc,train_loss,valid_loss = [],[],[],[] # evaluation log
    print("Training...")
    for i in range(epoch):
        X_train,y_train = shuffle(X_train,y_train)
        for offset in range(0,num_example,batch_size):
            end = offset + batch_size # end index of each batch
            batch_X, batch_y = X_train[offset:end],y_train[offset:end]
            sess.run(training_operation,feed_dict={x:batch_X,y:batch_y}) # put batch_X in x in the graph. put batch_y in y in the graph
        """ get model performance (accuracy and loss) after every epoch """
        train_acc.append(accuracy.eval(feed_dict={x:X_train,y:y_train}))
        valid_acc.append(accuracy.eval(feed_dict={x:X_valid,y:y_valid}))
        train_loss.append(loss.eval(feed_dict={x:X_train,y:y_train}))
        valid_loss.append(loss.eval(feed_dict={x:X_valid,y:y_valid}))
        print (str(i+1)+"th epoch finished")
    saver.save(sess,"Traffic Sign Classifier")
    
plot_accuracy(train_acc,valid_acc) # plot accuracy
plot_loss(train_loss, valid_loss) # plot loss


        