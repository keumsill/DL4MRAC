```python3

# PET-MR
Repo For Code

## Importing Packages
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import scipy
from scipy import misc
from PIL import Image
import skimage
%matplotlib inline



## Encoder

# clear old variables
tf.reset_default_graph()
def autoencoder_mr(input_shape=[None, 360000],
                n_filters=[1, 10, 10, 10],
                filter_sizes=[3, 3, 3, 3],
                corruption=False):
    """Build a deep denoising autoencoder w/ tied weights.
    Parameters
    ----------
    input_shape : list, optional
        Description
    n_filters : list, optional
        Description
    filter_sizes : list, optional
        Description
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    label : Tensor
        Label
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    cost : Tensor
        Overall cost to use for training
    Raises
    ------
    ValueError
        Description
    """
    # %%
    # input to the network
    x = tf.placeholder(
        tf.float32, input_shape, name='x')
    label = tf.placeholder(
        tf.float32, input_shape, name='label')


    # %%
    # ensure 2-d is converted to square tensor.
    if len(x.get_shape()) == 2:
        x_dim = np.sqrt(x.get_shape().as_list()[1])
        if x_dim != int(x_dim):
            raise ValueError('Unsupported input dimensions')
        x_dim = int(x_dim)
        x_tensor = tf.reshape(
            x, [-1, x_dim, x_dim, n_filters[0]])
    elif len(x.get_shape()) == 4:
        x_tensor = x
    else:
        raise ValueError('Unsupported input dimensions')
    current_input = x_tensor
    
    if len(label.get_shape()) == 2:
        label_dim = np.sqrt(label.get_shape().as_list()[1])
        if label_dim != int(label_dim):
            raise ValueError('Unsupported input dimensions')
        label_dim = int(label_dim)
        label_tensor = tf.reshape(
            label, [-1, label_dim, label_dim, n_filters[0]])
    elif len(label.get_shape()) == 4:
        label_tensor = label
    else:
        raise ValueError('Unsupported input dimensions')

    # %%
    # Optionally apply denoising autoencoder
    if corruption:
        current_input = corrupt(current_input)

    # %%
    # Build the encoder
    encoder = []
    shapes = []
    for layer_i, n_output in enumerate(n_filters[1:]):
        n_input = current_input.get_shape().as_list()[3]
        shapes.append(current_input.get_shape().as_list())
        W = tf.Variable(
            tf.random_uniform([
                filter_sizes[layer_i],
                filter_sizes[layer_i],
                n_input, n_output],
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        encoder.append(W)
        output = lrelu(
            tf.add(tf.nn.conv2d(
                current_input, W, strides=[1, 2, 2, 1], padding='SAME'), b))
        current_input = output

    # %%
    # store the latent representation
    z = current_input
    encoder.reverse()
    shapes.reverse()

    # %%
    # Build the decoder using the same weights
    for layer_i, shape in enumerate(shapes):
        W_shape = encoder[layer_i]
        W1 = tf.Variable(
            tf.random_uniform(W_shape.get_shape().as_list(),
                -1.0 / math.sqrt(n_input),
                1.0 / math.sqrt(n_input)))
        b1 = tf.Variable(tf.zeros([W_shape.get_shape().as_list()[2]]))
        output = lrelu(tf.add(
            tf.nn.conv2d_transpose(
                current_input, W1,
                tf.stack([tf.shape(x)[0], shape[1], shape[2], shape[3]]),
                strides=[1, 2, 2, 1], padding='SAME'), b1))
        current_input = output

    # %%
    # now have the reconstruction through the network
    y = current_input
    # cost function measures pixel-wise difference
    cost = tf.reduce_sum(tf.square(y - label_tensor))

    # %%
    return {'x': x, 'z': z, 'label':label, 'y': y, 'cost': cost}
    
    
    
    
    
    ##Test_MR
    def test_mr():
    
    img1 = scipy.misc.imread('mr3_processed.jpg') #SEGMENTED MRI
    #img1 = np.reshape(img1,[1,106400])
    #img1 = img1[35:635,523:1123,0]
    img1 = np.reshape(img1,600*600)
    #print('expand img shape', img1.shape)
    img1 = np.expand_dims(img1, axis=0)
    img = np.repeat(img1, 1000, axis=0)
    
    # label
    img_label = scipy.misc.imread('ac3_processed.jpg') #CT IMAGE (TARGET)
    #img1 = np.reshape(img1,[1,106400])
    #img1 = img1[35:635,523:1123,0]
    img_label = np.reshape(img_label,600*600)
    #print('expand img shape', img1.shape)
    img_label = np.expand_dims(img_label, axis=0)
    label = np.repeat(img_label, 1000, axis=0)
    #print('expand img shape', img.shape)
    ae = autoencoder_mr(input_shape=[None, 600*600])
    # %%
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    batch_size = 100
    n_epochs = 70
    for epoch_i in range(n_epochs):
        #for batch_i in range(mnist.train.num_examples // batch_size):
        for batch_i in range(img.shape[0] // batch_size):
            #batch_xs, _ = mnist.train.next_batch(batch_size)
            train = img[batch_i*batch_size:(batch_i + 1)*batch_size,:]
            train_label = label[batch_i*batch_size:(batch_i + 1)*batch_size,:]
            sess.run(optimizer, feed_dict={ae['x']: train, ae['label']: train_label})
        print(epoch_i, sess.run(ae['cost'], feed_dict={ae['x']: train, ae['label']: train_label}))

    # Plot example reconstructions
    n_examples = 2
    img_test1 = scipy.misc.imread('ac3_processed.jpg')
    img_test2 = scipy.misc.imread('mr3_processed.jpg')
    #img1 = np.reshape(img1,[1,106400])
    #img_test1 = img_test1[35:635,523:1123,0]
    #print('expand img shape', img_test1.shape)
    img_test1 = np.reshape(img_test1,[1,600*600])
    #img_test2 = img_test2[0:600,185:785,0]
    #print('expand img shape', img_test2.shape)
    img_test2 = np.reshape(img_test2,[1,600*600])
    #img_test1 = np.expand_dims(img_test1, axis=0)
    #img_test = np.repeat(img_test1, n_examples, axis=0)
    img_test = np.concatenate((img_test1,img_test2),axis=0)
    img_label = np.concatenate((img_test1,img_test1),axis=0)
    #print('expand img shape', img_test.shape)
    #img = np.repeat(img, 5, axis=0)
    #test_xs, _ = mnist.test.next_batch(n_examples)
    #test_xs_norm = np.array([img - mean_img for img in test_xs])
    recon = sess.run(ae['y'], feed_dict={ae['x']: img_test, ae['label']: img_label})
    fig, axs = plt.subplots(2, n_examples, figsize=(100, 20))
    for example_i in range(n_examples):
        axs[0][example_i].imshow(
            np.reshape(img_test[example_i, :], (600, 600)))
        axs[1][example_i].imshow(
            np.reshape(recon[example_i, :], (600, 600)))
        result = Image.fromarray(np.reshape(recon[example_i, :], (600, 600)).astype(np.uint8))
        name = 'test7e' + str(example_i) +'.jpg'
        result.save(name)
    #fig.show()
    #print(img)
    #print(recon)
    plt.draw()
    #plt.waitforbuttonpress()
    
```
