from __future__ import print_function
import tensorflow as tf
import numpy as np

import TensorflowUtils as utils
import read_MITSceneParsingData as scene_parsing
import datetime
import BatchDatsetReader as dataset
from six.moves import xrange

"""
Tuan import for crf
"""
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax


from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt

import cv2



FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("batch_size", "2", "batch size for training")
tf.flags.DEFINE_string("logs_dir", "logs/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
#tf.flags.DEFINE_string("data_dir", "Data_zoo\\MIT_SceneParsing\\", "path to dataset")
#tf.flags.DEFINE_string("data_dir", "Data_zoo\\ClothParsing\\", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1000)
NUM_OF_CLASSESS = 18 # human parsing  59 #cloth   151  # MIT Scene
IMAGE_SIZE = 224

"""
Tuan - Function which returns the labelled image after applying CRF

"""


#
#  building cross matrix
#
def _calcCrossMat(gtimage, predimage, n):
    crossMat = []
    for i in range(n):
        crossMat.append([0] * n)
    # print(crossMat)
    height, width = gtimage.shape
    gtlabel = prelabel = 0

    for y in range(height):
        # print(crossMat)
        for x in range(width):
            gtlabel = gtimage[y, x]
            predlabel = predimage[y, x]
            if predlabel >= n or gtlabel >= n:
                print('gt:%d, pr:%d' % (gtlabel, predlabel))
            else:
                crossMat[gtlabel][predlabel] = crossMat[gtlabel][predlabel] + 1;

    return crossMat

# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D fucntions will be applied
# else Generic functions will be applied

def crf(original_image, annotated_image, use_2d=True):
    # Converting annotated image to RGB if it is Gray scale
    print("crf function")

    #if (len(annotated_image.shape) < 3):
    #    annotated_image = gray2rgb(annotated_image)

    #imsave("testing2.png", annotated_image)

    # Converting the annotations RGB color to single 32 bit integer
    annotated_label = annotated_image[:, :, 0] + (annotated_image[:, :, 1] << 8) + (annotated_image[:, :, 2] << 16)

    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    #print("colors: ",colors )
    #print("labels: ",labels )
    # Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:, 0] = (colors & 0x0000FF)
    colorize[:, 1] = (colors & 0x00FF00) >> 8
    colorize[:, 2] = (colors & 0xFF0000) >> 16

    # Gives no of class labels in the annotated image
    #n_labels = len(set(labels.flat))
    n_labels = NUM_OF_CLASSESS
    print("No of labels in the Image are ")
    #print(n_labels)
    #print("annotated shape:", annotated_image.shape)


    # Setting up the CRF model
    if use_2d:
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        processed_probabilities = annotated_image
        softmax = processed_probabilities.transpose((2, 0, 1))

        U = unary_from_softmax(softmax)
        #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(10, 10), srgb=(13, 13, 13), rgbim=original_image,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(20)
    #print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)
    #print(">>>>>>>>MAP shape: ", MAP.shape)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP, :]

    #print(">>>>>>>>>COlorized map shape: ", MAP.shape)
    #print(">>>>>>>>>map ount non zero0: ", np.count_nonzero(MAP,axis = 0))
    #print(">>>>>>>>>map ount non zero1: ", np.count_nonzero(MAP,axis = 1))


    output = MAP.reshape(original_image.shape)
    output = rgb2gray(output)
    #plt.imshow(output, cmap = "nipy_spectral")
    #print(">>>>>>>>>COlorized map shape: ", output.shape)
    #print(">>>>>>>>>map ount non zero0: ", np.count_nonzero(output, axis=0))
    #print(">>>>>>>>>map ount non zero1: ", np.count_nonzero(output, axis=1))
    #import cv2;
    #hist_crf = cv2.calcHist([output.astype(np.uint8)], [0], None, [256], [0, 256])
    #print("hist)crf:", hist_crf)
    #plt.show()
    #Get output


    """
    annotated_image  = np.reshape(annotated_image, [224*224,-1] )

    print(">>>>>>>>>>>>>>>>>>>shape annotated image0: ", annotated_image)
    MAP_bf = np.argmax(annotated_image, axis=0)
    print(">>>>>>>>>>>>>>>>>>>shape annotated image1: ", MAP_bf)
    MAP_bf = colorize[MAP_bf, :]
    print(">>>>>>>>>>>>>>>>>>>shape annotated image2: ", MAP_bf.shape)
    output_bf = MAP_bf.reshape(original_image.shape)
    output_bf = rgb2gray(output_bf)

    fig = plt.figure(figsize=(1,3))
    #error = output - annotated_image
    fig.add_subplot(1,1,original_image)
    fig.add_subplot(1,2, output_bf)
    fig.add_subplot(1,3, output)

    #fig.add_subplot(1,4,rgb2gray(error.reshape(original_image.shape)))
    plt.show()
    """
    #imsave(output_image, MAP.reshape(original_image.shape))
    #imsave(output_image + ".png", output)
    return MAP.reshape(original_image.shape), output

def vgg_net(weights, image):
    layers = (
        'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

        'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

        'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
        'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

        'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
        'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

        'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
        'relu5_3', 'conv5_4', 'relu5_4'
    )

    net = {}
    current = image
    for i, name in enumerate(layers):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = utils.get_variable(np.transpose(kernels, (1, 0, 2, 3)), name=name + "_w")
            bias = utils.get_variable(bias.reshape(-1), name=name + "_b")
            current = utils.conv2d_basic(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current, name=name)
            if FLAGS.debug:
                utils.add_activation_summary(current)
        elif kind == 'pool':
            current = utils.avg_pool_2x2(current)
        net[name] = current

    # added for resume better
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter
    return net

def Unetinference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    # 1. donwload VGG pretrained model from network if not did before
    #    model_data is dictionary for variables from matlab mat file
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    # 2. construct model graph
    with tf.variable_scope("inference"):
        # 2.1 VGG
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        #
        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)


        # prob = tf.nn.softmax(conv_t3, axis =3)
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3, image_net

def inference(image, keep_prob):
    """
    Semantic segmentation network definition
    :param image: input image. Should have values in range 0-255
    :param keep_prob:
    :return:
    """
    # 1. donwload VGG pretrained model from network if not did before
    #    model_data is dictionary for variables from matlab mat file
    print("setting up vgg initialized conv layers ...")
    model_data = utils.get_model_data(FLAGS.model_dir, MODEL_URL)

    mean = model_data['normalization'][0][0][0]
    mean_pixel = np.mean(mean, axis=(0, 1))

    weights = np.squeeze(model_data['layers'])

    processed_image = utils.process_image(image, mean_pixel)

    # 2. construct model graph
    with tf.variable_scope("inference"):
        # 2.1 VGG
        image_net = vgg_net(weights, processed_image)
        conv_final_layer = image_net["conv5_3"]
        #
        pool5 = utils.max_pool_2x2(conv_final_layer)

        W6 = utils.weight_variable([7, 7, 512, 4096], name="W6")
        b6 = utils.bias_variable([4096], name="b6")
        conv6 = utils.conv2d_basic(pool5, W6, b6)
        relu6 = tf.nn.relu(conv6, name="relu6")
        if FLAGS.debug:
            utils.add_activation_summary(relu6)
        relu_dropout6 = tf.nn.dropout(relu6, keep_prob=keep_prob)

        W7 = utils.weight_variable([1, 1, 4096, 4096], name="W7")
        b7 = utils.bias_variable([4096], name="b7")
        conv7 = utils.conv2d_basic(relu_dropout6, W7, b7)
        relu7 = tf.nn.relu(conv7, name="relu7")
        if FLAGS.debug:
            utils.add_activation_summary(relu7)
        relu_dropout7 = tf.nn.dropout(relu7, keep_prob=keep_prob)

        W8 = utils.weight_variable([1, 1, 4096, NUM_OF_CLASSESS], name="W8")
        b8 = utils.bias_variable([NUM_OF_CLASSESS], name="b8")
        conv8 = utils.conv2d_basic(relu_dropout7, W8, b8)
        # annotation_pred1 = tf.argmax(conv8, dimension=3, name="prediction1")

        # now to upscale to actual image size
        deconv_shape1 = image_net["pool4"].get_shape()
        W_t1 = utils.weight_variable([4, 4, deconv_shape1[3].value, NUM_OF_CLASSESS], name="W_t1")
        b_t1 = utils.bias_variable([deconv_shape1[3].value], name="b_t1")
        conv_t1 = utils.conv2d_transpose_strided(conv8, W_t1, b_t1, output_shape=tf.shape(image_net["pool4"]))
        fuse_1 = tf.add(conv_t1, image_net["pool4"], name="fuse_1")

        deconv_shape2 = image_net["pool3"].get_shape()
        W_t2 = utils.weight_variable([4, 4, deconv_shape2[3].value, deconv_shape1[3].value], name="W_t2")
        b_t2 = utils.bias_variable([deconv_shape2[3].value], name="b_t2")
        conv_t2 = utils.conv2d_transpose_strided(fuse_1, W_t2, b_t2, output_shape=tf.shape(image_net["pool3"]))
        fuse_2 = tf.add(conv_t2, image_net["pool3"], name="fuse_2")

        shape = tf.shape(image)
        deconv_shape3 = tf.stack([shape[0], shape[1], shape[2], NUM_OF_CLASSESS])
        W_t3 = utils.weight_variable([16, 16, NUM_OF_CLASSESS, deconv_shape2[3].value], name="W_t3")
        b_t3 = utils.bias_variable([NUM_OF_CLASSESS], name="b_t3")
        conv_t3 = utils.conv2d_transpose_strided(fuse_2, W_t3, b_t3, output_shape=deconv_shape3, stride=8)


        # prob = tf.nn.softmax(conv_t3, axis =3)
        annotation_pred = tf.argmax(conv_t3, dimension=3, name="prediction")

    return tf.expand_dims(annotation_pred, dim=3), conv_t3, image_net

"""inference
  optimize with trainable paramters (Check which ones)
  loss_val : loss operator (mean(
  
"""
def train(loss_val, var_list, global_step):
    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
    grads = optimizer.compute_gradients(loss_val, var_list=var_list)
    if FLAGS.debug:
        # print(len(var_list))
        for grad, var in grads:
            utils.add_gradient_summary(grad, var)
    return optimizer.apply_gradients(grads, global_step= global_step)


def main(argv=None):

    # 1. input placeholders
    keep_probability = tf.placeholder(tf.float32, name="keep_probabilty")
    image = tf.placeholder(tf.float32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 3), name="input_image")
    annotation = tf.placeholder(tf.int32, shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1), name="annotation")

    # 2. construct inference network
    pred_annotation, logits, net = inference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)
    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    # 3. loss measure
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation, squeeze_dims=[3]),
                                                                          name="entropy")))
    tf.summary.scalar("entropy", loss)

    # 4. optimizing
    trainable_var = tf.trainable_variables()
    if FLAGS.debug:
        for var in trainable_var:
            utils.add_to_regularization_and_summary(var)
    train_op = train(loss, trainable_var, net['global_step'])

    print("Setting up summary op...")
    summary_op = tf.summary.merge_all()

    print("Setting up image reader from ", FLAGS.data_dir, "...")
    train_records, valid_records = scene_parsing.read_dataset(FLAGS.data_dir)
    print("data dir:", FLAGS.data_dir)
    print("train_records length :", len(train_records))
    print("valid_records length :", len(valid_records))

    print("Setting up dataset reader")
    image_options = {'resize': True, 'resize_size': IMAGE_SIZE}
    if FLAGS.mode == 'train':
        train_dataset_reader = dataset.BatchDatset(train_records, image_options)
    validation_dataset_reader = dataset.BatchDatset(valid_records, image_options)


    sess = tf.Session()

    print("Setting up Saver...")
    saver = tf.train.Saver()
    summary_writer = tf.summary.FileWriter(FLAGS.logs_dir, sess.graph)

    # 5. paramter setup
    # 5.1 init params
    sess.run(tf.global_variables_initializer())
    # 5.2 restore params if possible
    ckpt = tf.train.get_checkpoint_state(FLAGS.logs_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Model restored...")

    # 6. train-mode
    if FLAGS.mode == "train":

        global_step = sess.run(net['global_step'])

        for itr in xrange(global_step, MAX_ITERATION):
            # 6.1 load train and GT images
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            #print("train_image:", train_images.shape)
            #print("annotation :", train_annotations.shape)

            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
            # 6.2 traininging

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)

            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
                global_step = sess.run(net['global_step'])
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=global_step)

    # test-mode
    elif FLAGS.mode == "visualize":

        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(5+itr))
            utils.save_image(valid_annotations[itr].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(5+itr))
            utils.save_image(pred[itr].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(5+itr))
            print("Saved image: %d" % itr)

    elif FLAGS.mode == "test":  # heejune added
        print(">>>>>>>>>>>>>>>>Test mode")
        validation_dataset_reader.reset_batch_offset(0)
        #print(">>>>>>>>>>>>>>>>Test mode")
        for itr1 in range(validation_dataset_reader.get_num_of_records()//2):
            #print(">>>>>>>>>>>>>>>>Test mode")
            valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
            pred, logits = sess.run([pred_annotation, logits], feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            logits = np.squeeze(logits)
            pred = np.squeeze(pred)
            print("logits shape:", logits.shape)
            np.set_printoptions(threshold=np.inf)
            print("logits:", logits)
            for itr2 in range(FLAGS.batch_size):
                """
                Tuan add CRF post processing
                
                import sys
                path = "D:/Python/Anaconda/envs/tf\ws/FCN-Tuan/pydensecrf/dense_crf_python/"
                sys.path.append(path)
                import pydensecrf.densecrf as dcrf
                from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, softmax_to_unary
                import skimage.io as io
                image = valid_images[itr2].astype(np.uint8)
                processed_probabilities = pred
                softmax = processed_probabilities.transpose((2, 0, 1))
                
                unary = softmax_to_unary(softmax)
                # The inputs should be C-continious -- we are using Cython wrapper
                unary = np.ascontiguousarray(unary)

                d = dcrf.DenseCRF(image.shape[0] * image.shape[1], 2)
                print(">>>>>>>>>>>", unary.shape)
                d.setUnaryEnergy(unary)


                # This potential penalizes small pieces of segmentation that are
                # spatially isolated -- enforces more spatially consistent segmentations
                feats = create_pairwise_gaussian(sdims=(10, 10), shape=image.shape[:2])

                d.addPairwiseEnergy(feats, compat=3,
                                    kernel=dcrf.DIAG_KERNEL,
                                    normalization=dcrf.NORMALIZE_SYMMETRIC)

                # This creates the color-dependent features --
                # because the segmentation that we get from CNN are too coarse
                # and we can use local color features to refine them
                feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20, 20, 20),
                                                   img=image, chdim=2)

                d.addPairwiseEnergy(feats, compat=10,
                                     kernel=dcrf.DIAG_KERNEL,
                                     normalization=dcrf.NORMALIZE_SYMMETRIC)
                Q = d.inference(5)
                res = np.argmax(Q, axis=0).reshape((image.shape[0], image.shape[1]))

                cmap = plt.get_cmap('bwr')

                f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
                ax1.imshow(res, vmax=1.5, vmin=-0.4, cmap=cmap)
                ax1.set_title('Segmentation with CRF post-processing')
                #probability_graph = ax2.imshow(np.dstack((train_annotation,)*3)*100)
                #ax2.set_title('Ground-Truth Annotation')
                plt.show()

                
                end Tuan
                """
                print("Output file: ", FLAGS.logs_dir + "crf_" + str(itr1*2+itr2))
                crfimage, crfoutput = crf(valid_images[itr2].astype(np.uint8),logits[itr2])

                fig = plt.figure()
                pos = 240 + 1
                plt.subplot(pos)
                plt.imshow(valid_images[itr2].astype(np.uint8))
                plt.axis('off')
                plt.title('Original')

                pos = 240 + 2
                plt.subplot(pos)
                plt.imshow(valid_annotations[itr2].astype(np.uint8),cmap=plt.get_cmap('nipy_spectral'))
                plt.axis('off')
                plt.title('GT')

                pos = 240 + 3
                plt.subplot(pos)
                plt.imshow(pred[itr2].astype(np.uint8),cmap=plt.get_cmap('nipy_spectral'))
                plt.axis('off')
                plt.title('Prediction')

                pos = 240 + 4
                plt.subplot(pos)
                plt.imshow(crfoutput, cmap=plt.get_cmap('nipy_spectral'))
                plt.axis('off')
                plt.title('CRFPostProcessing')

                pos = 240 + 6
                plt.subplot(pos)
                ret, errorImage = cv2.threshold(cv2.absdiff(pred[itr2].astype(np.uint8), valid_annotations[itr2].astype(np.uint8)), 0.5,
                                                255, cv2.THRESH_BINARY)
                plt.imshow(errorImage, cmap=plt.get_cmap('gray'))
                plt.axis('off')
                plt.title('Pred Error:' + str(np.count_nonzero(errorImage)))

                # Calculate cross matrix between gtimage and prediction image and store to file
                #pred_cross_matrix = _calcCrossMat(valid_annotations[itr2].astype(np.uint8), pred[itr2].astype(np.uint8),
                #                                  NUM_OF_CLASSESS)
                #np.save(FLAGS.logs_dir + "pred_cross_matrix_" + str(itr1*2 + itr2), pred_cross_matrix)

                # ---End calculate cross matrix


                #Convert image from float to 0->255 uint8
                #nfo = np.info(crfoutput.dtype)
                #crfoutput = crfoutput.astype(np.float64)/ info.max()
                #crfoutput = 255*crfoutput
                #crfoutput = crfoutput.astype(np.uint8)

                crfoutput = cv2.normalize(crfoutput,None, 0, 255, cv2.NORM_MINMAX)
                valid_annotations[itr2] = cv2.normalize(valid_annotations[itr2],None, 0, 255, cv2.NORM_MINMAX)

                pos = 240 + 8
                plt.subplot(pos)
                ret, errorImage = cv2.threshold(cv2.absdiff(crfoutput.astype(np.uint8), valid_annotations[itr2].astype(np.uint8)),
                                                10, 255, cv2.THRESH_BINARY)
                plt.imshow(errorImage, cmap=plt.get_cmap('gray'))
                plt.axis('off')
                plt.title('CRF Error:'+ str(np.count_nonzero(errorImage)))

                # Calculate cross matrix between gtimage and CRF postprocessing of prediction image and store to file
                #crfpred_cross_matrix = _calcCrossMat(valid_annotations[itr2].astype(np.uint8), crfoutput.astype(np.uint8),
                #                                     NUM_OF_CLASSESS)
                #np.save(FLAGS.logs_dir + "crfpred_cross_matrix_" + str(itr1 * 2 + itr2), pred_cross_matrix)

                # ---End calculate cross matrix

                np.set_printoptions(threshold=np.inf)
                #print(cv2.absdiff(crfoutput.astype(np.uint8), valid_annotations[itr2].astype(np.uint8)))

                #fig.show()
                plt.show()
                #print(FLAGS.logs_dir + "resultSum_" + str(itr1 * 2 + itr2))
                plt.savefig(FLAGS.logs_dir + "resultSum_" + str(itr1 * 2 + itr2))
                #---------------------------------------------
                utils.save_image(valid_images[itr2].astype(np.uint8), FLAGS.logs_dir, name="inp_" + str(itr1*2+itr2))
                utils.save_image(valid_annotations[itr2].astype(np.uint8), FLAGS.logs_dir, name="gt_" + str(itr1*2+itr2))
                utils.save_image(pred[itr2].astype(np.uint8), FLAGS.logs_dir, name="pred_" + str(itr1*2 + itr2))
                utils.save_image(crfoutput, FLAGS.logs_dir, name="crf_" + str(itr1 * 2 + itr2))

                #---End calculate cross matrix
                print("Saved image: %d" % (itr1*2 + itr2))

if __name__ == "__main__":
    tf.app.run()
