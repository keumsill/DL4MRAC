## Importing Packages
import tensorflow as tf
import numpy as np
import math
import timeit
import matplotlib.pyplot as plt
import scipy
import scipy.io
from scipy import misc
from PIL import Image
import skimage
import os
import os.path
import dicom
from dicom.dataset import Dataset, FileDataset
import datetime, time
%matplotlib inline

In [22]:
"""Activations for TensorFlow.
Parag K. Mital, Jan 2016."""

def lrelu(x, leak=0.2, name="lrelu"):
    """Leaky rectifier.

    Parameters
    ----------
    x : Tensor
        The tensor to apply the nonlinearity to.
    leak : float, optional
        Leakage parameter.
    name : str, optional
        Variable scope to use.

    Returns
    -------
    x : Tensor
        Output of the nonlinearity.
    """
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

In [27]:
## Encoder

# clear old variables
tf.reset_default_graph()
def autoencoder_mr(input_shape=[None, 110*110],
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
    cost =(tf.reduce_sum(tf.square(y - label_tensor)))
    # %%input
    return {'x': x, 'z': z, 'label':label, 'y': y, 'cost': cost}

In [28]:
def write_dicom(pixel_array,filename):
    """
    INPUTS:
    pixel_array: 2D numpy ndarray.  If pixel_array is larger than 2D, errors.
    filename: string name for the output file.
    """

    ## Metadata is captured from Osirix
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    file_meta.MediaStorageSOPInstanceUID = '1.2.276.0.7230010.3.1.4.680961.1356.1477424584.260802'
    file_meta.ImplementationClassUID = '1.2.276.0.7230010.3.0.3.6.0'
    ds = FileDataset(filename, {},file_meta = file_meta,preamble="\0"*128)
    ds.Modality = 'MR'
    ds.ContentDate = str(datetime.date.today()).replace('-','')
    ds.ContentTime = str(time.time()) #milliseconds since the epoch
    ds.StudyInstanceUID =  '1.2.840.113619.2.40.20410.4.1101731106.1478904046.976759'
    ds.SeriesInstanceUID = '1.2.840.113619.2.40.20410.4.1101731106.1478904045.846068'
    ds.SOPInstanceUID =    '1.2.840.113619.2.40.20410.4.1101731106.1478904123.170427'
    ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.4'
    ds.SecondaryCaptureDeviceManufctur = 'Python 2.7.3'
    ds.SliceThickness = 2.4
    ds.SpacingBetweenSlices = 2.4

    ## These are the necessary imaging components of the FileDataset object.
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 1
    ds.HighBit = 15
    ds.BitsStored = 16
    ds.BitsAllocated = 16
    ds.SmallestImagePixelValue = '\\x00\\x00'
    ds.LargestImagePixelValue = '\\xff\\xff'
    ds.Columns = pixel_array.shape[0]
    ds.Rows = pixel_array.shape[1]
    if pixel_array.dtype != np.uint16:
        pixel_array = pixel_array.astype(np.uint16)
    ds.PixelData = pixel_array.tostring()

    ds.save_as(filename)
    return

In [31]:
##Test_MR
def test_mr():
    
    #train
    train_names = []
    #Bringing in images from the Blank_ZTE folder
    for jpg in os.listdir('./Blank_ZTE'):
        train_names.append(jpg)
        
    #Sorting the order of the images (can try randomizing order in the future)
    train_names_sorted = sorted(train_names)

    train = []
    #Pulling in array values from Images
    for name in train_names_sorted:
        img_cur = scipy.misc.imread('./Blank_ZTE/' + name)
        img_cur = np.reshape(img_cur,110*110)
        img_cur = np.expand_dims(img_cur, axis=0)
        train.append(img_cur)
    
    train = np.concatenate(train)
    #Repeating current values to create larger dataset
    train = np.repeat(train, 10, axis=0)

        
    # label
    target_names = []
    #Bringing in images from Segemented_ZTE folder
    for jpg in os.listdir('./Segmented_ZTE'):
        target_names.append(jpg)
    
    #Ordering images
    target_names = sorted(target_names)

    target = []
    #Gettting arrays from images
    for name in target_names:
        img_cur = scipy.misc.imread('./Segmented_ZTE/' + name)
        img_cur = np.reshape(img_cur,110*110)
        img_cur = np.expand_dims(img_cur, axis=0)
        target.append(img_cur)
    
    target = np.concatenate(target)
    target = np.repeat(target, 10, axis=0)
    
    #initializing encoder
    ae = autoencoder_mr(input_shape=[None, 110*110])
    # %%
    learning_rate = 0.01
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(ae['cost'])

    # %%
    # We create a session to use the graph
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # %%
    # Fit all training data
    n_epochs = 100
    batch_size = 40
    save_path = '/home/jimbest-devereux/petmr/PhotoDump'
    save_path2 = '/home/jimbest-devereux/petmr/MatDump'
    save_path3 = '/home/jimbest-devereux/petmr/DcmDump'
    for epoch_i in range(n_epochs):
        for batch_i in range(train.shape[0] // batch_size):
            train_set = train[batch_i*batch_size:(batch_i + 1)*batch_size,:]
            train_label = target[batch_i*batch_size:(batch_i + 1)*batch_size,:]
            sess.run(optimizer, feed_dict={ae['x']: train_set, ae['label']: train_label})
            
        #creating different file outputs for the entire image set
        for img_name in train_names_sorted:
            if epoch_i % 10 == 0: #can change % value to get more or less outputs
                img_testing = scipy.misc.imread('./Blank_ZTE/'+ img_name)
                img_testing = np.reshape(img_testing, [1,110*110])
                recon_testing = sess.run(ae['y'], feed_dict={ae['x']: img_testing})
                recon_testing.tofile(save_path2 + '/' + str(epoch_i) + '_' + img_name.partition('.')[0])
                result_testing = Image.fromarray(np.reshape(recon_testing, (110,110)).astype(np.uint8))
                result_testing.save(save_path + '/' + str(epoch_i) + '_' + img_name, 'JPEG')
                write_dicom(np.reshape(recon_testing,(110,110)), save_path3 + '/' + str(epoch_i) + '_' + img_name.partition('.')[0]+ '.dcm')
        
#         if epoch_i != 0:
#             if prev_cost > sess.run(ae['cost'], feed_dict={ae['x']: train_set, ae['label']: train_label}):
#                 learning_rate = learning_rate * 1.05
#             else:
#                 learning_rate = learning_rate * 0.9
        
        prev_cost = sess.run(ae['cost'], feed_dict={ae['x']: train_set, ae['label']: train_label})                    
        print(epoch_i, np.sqrt(sess.run(ae['cost'], feed_dict={ae['x']: train_set, ae['label']: train_label})/(batch_size*img_testing.shape[1])), learning_rate)
                #imgplot = plt.imshow(result_testing)
                #plt.show()
                #result_testing.save('ImgTest' + str(epoch_i) + '.jpg')
           
In [32]:
##Running the network
test_mr()