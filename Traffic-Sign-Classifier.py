# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = '/home/pcvp/code/academic/Data/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/train.p'
validation_file = '/home/pcvp/code/academic/Data/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/valid.p'
testing_file = '/home/pcvp/code/academic/Data/CarND-Traffic-Sign-Classifier-Project/traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

### Replace each question mark with the appropriate value.
### Use python, pandas or numpy methods rather than hard coding the results

# TODO: Number of training examples
n_train = len(X_train)

# TODO: Number of valid examples
n_valid = len(X_valid)

# TODO: Number of testing examples.
n_test = len(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = format(X_train[0].shape)

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(set(y_train))

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import random
import matplotlib.pyplot as plt
import matplotlib
# Visualizations will be shown in the notebook.
#%matplotlib inline

n_sample_images = 10
# Plot random n sample images
#print('%s Sample images', n_sample_images)
for i in range(n_sample_images):
    plt.subplot(2,n_sample_images/2,i+1)
    index = random.randint(0, len(X_train))
    plt.imshow(X_train[index].squeeze())

### Preprocess the data here. Preprocessing steps could include normalization, converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)

import numpy as np
def rgb2gray(imgs):
    # convert to grayscale
    return np.mean(imgs, axis=3, keepdims=True)

def normalize(img, a = -0.5, b = 0.5, Xmin = 0.0, Xmax = 255.0):
    """
    Normalize the image data with Min-Max scaling to a range of [a, b]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    #Xp = np.zeros((img.shape[0], img.shape[1], img.shape[2], 1))

    for c in range(img.shape[2]):
        X = img;
        Xp = a + (X-Xmin)*(b-a)/(Xmax-Xmin)

    return Xp

add_random_brightness = 1
# this function will be used to preprocess all images
def preprocess_images(images, brightness = 0):
    shape = images.shape
    out_img_shape = (shape[1], shape[2], 1)
    batch = np.zeros((shape[0], shape[1], shape[2], shape[3]))

    for i in range(len(images)):
        img = images[i, :]
        img = normalize(img, 0.0, 1.0, 0.0, 255.0)
        if(brightness == 1):
            img_hsv = matplotlib.colors.rgb_to_hsv(img)
            random = 0.5 + np.random.uniform()
            img_hsv[:, :, 2] = img_hsv[:, :, 2] * random
            img_hsv[:, :, 2][img_hsv[:, :, 2] > 1.0] = 1.0
            img= matplotlib.colors.hsv_to_rgb(img_hsv)
        img = normalize(img, -0.5, 0.5, 0.0, 1.0)
        batch[i] = img
    return batch

X_train_p = preprocess_images(X_train[0:n_train], add_random_brightness)
X_valid_p = preprocess_images(X_valid[0:n_valid], add_random_brightness)
X_test_p = preprocess_images(X_test[0:n_test], add_random_brightness)
C = X_train_p[0].shape[2]

# TODO: Number of training examples
n_train_p = len(X_train_p)

# TODO: Number of valid examples
n_valid_p = len(X_valid_p)

# TODO: Number of testing examples.
n_test_p = len(X_test_p)

# TODO: What's the shape of an traffic sign image?
image_shape_p = format(X_train_p[0].shape)

print("Number of training examples =", n_train_p)
print("Number of testing examples =", n_test_p)
print("Image data shape =", image_shape_p)
print("Number of classes =", n_classes)

### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import random
import matplotlib.pyplot as plt
# Visualizations will be shown in the notebook.
#%matplotlib inline

n_sample_images = 10
# Plot random n sample images
#print('%s Sample images', n_sample_images)
for i in range(n_sample_images):
    plt.subplot(2,n_sample_images/2,i+1)
    index = random.randint(0, len(X_train_p))
    img = X_train_p[index]
    img = normalize(img, 0.0, 1.0, -0.5, 0.5)
    plt.imshow(img.squeeze())

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf
from tensorflow.contrib.layers import flatten

def LeNet(x):
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1

    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, C, 6), mean=mu, stddev=sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean=mu, stddev=sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b

    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0 = flatten(conv2)

    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean=mu, stddev=sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc0, fc1_W) + fc1_b

    # SOLUTION: Activation.
    fc1 = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=mu, stddev=sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1, fc2_W) + fc2_b

    # SOLUTION: Activation.
    fc2 = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 10.
    fc3_W = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean=mu, stddev=sigma))
    fc3_b = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b

    return logits

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected,
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
x = tf.placeholder(tf.float32, (None, 32, 32, C))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

EPOCHS = 10
BATCH_SIZE = 128
rate = 0.0009

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=rate)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset + BATCH_SIZE], y_data[offset:offset + BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples


with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    num_examples = len(X_train_p)

    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train_p, y_train = shuffle(X_train_p, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train_p[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})

        validation_accuracy = evaluate(X_valid_p, y_valid)
        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

    saver.save(sess, './lenet')
    print("Model saved")

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test_p, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

### Load the images and plot them here.
### Feel free to use as many code cells as needed.

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.

### Calculate the accuracy for these 5 new images.
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web.
### Feel free to use as many code cells as needed.

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry
