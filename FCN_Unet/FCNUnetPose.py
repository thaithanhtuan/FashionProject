from __future__ import print_function
import tensorflow as tf
import numpy as np
from PIL import Image
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
tf.flags.DEFINE_string("logs_dir", "logs/UN/", "path to logs directory")
tf.flags.DEFINE_string("data_dir", "Data_zoo/MIT_SceneParsing/", "path to dataset")
#tf.flags.DEFINE_string("data_dir", "Data_zoo\\MIT_SceneParsing\\", "path to dataset")
#tf.flags.DEFINE_string("data_dir", "Data_zoo\\ClothParsing\\", "path to dataset")
tf.flags.DEFINE_float("learning_rate", "1e-4", "Learning rate for Adam Optimizer")
tf.flags.DEFINE_string("model_dir", "Model_zoo/", "Path to vgg model mat")
tf.flags.DEFINE_bool('debug', "False", "Debug mode: True/ False")
tf.flags.DEFINE_string('mode', "train", "Mode train/ test/ visualize")

MODEL_URL = 'http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat'

MAX_ITERATION = int(1e5 + 1001)
NUM_OF_CLASSESS = 18 # human parsing  59 #cloth   151  # MIT Scene
IMAGE_SIZE = 224


"""
    Tuan - calculate IoU
"""

def _calcIOU(gtimage, predimage, n):
    IoUs = []
    for i in range(n):
        IoUs.append([0] * n)

    height, width = gtimage.shape
    gtlabel = prelabel = 0
    for label in range(n): #0--> 17
        intersection = 0
        union = 0
        for y in range(height):#0->223
            # print(crossMat)
            for x in range(width):#=-->223
                gtlabel = gtimage[y, x]
                predlabel = predimage[y, x]

                if predlabel >= n or gtlabel >= n:
                    print('gt:%d, pr:%d' % (gtlabel, predlabel))
                else:
                    if(gtlabel==label and predlabel == label):
                        intersection = intersection + 1
                    if(gtlabel==label or predlabel == label):
                        union = union + 1
        if(union == 0):
            IoUs[label] = 0.0
        else:
            #print("label:", label , "intersection:", intersection, " - union:", union)
            IoUs[label] = (float)(intersection) / union
    return IoUs



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
                crossMat[gtlabel][predlabel] = (int)(crossMat[gtlabel][predlabel]) + 1;

    return crossMat


"""
Tuan - Function which returns the labelled image after applying CRF

"""
# Original_image = Image which has to labelled
# Annotated image = Which has been labelled by some technique( FCN in this case)
# Output_image = The final output image after applying CRF
# Use_2d = boolean variable
# if use_2d = True specialised 2D fucntions will be applied
# else Generic functions will be applied

def crf(original_image, annotated_image, use_2d=True):
    # Converting annotated image to RGB if it is Gray scale
    print("crf function")


    # Gives no of class labels in the annotated image
    #n_labels = len(set(labels.flat))
    n_labels = NUM_OF_CLASSESS

    # Setting up the CRF model

    d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

    # get unary potentials (neg log probability)
    processed_probabilities = annotated_image
    softmax = processed_probabilities.transpose((2, 0, 1))
    #print(softmax.shape)
    U = unary_from_softmax(softmax, scale=None, clip=1e-5)

    U = np.ascontiguousarray(U)
    #U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
    d.setUnaryEnergy(U)

    # This potential penalizes small pieces of segmentation that are
    # spatially isolated -- enforces more spatially consistent segmentations
    feats = create_pairwise_gaussian(sdims=(3, 3), shape=original_image.shape[:2])

    d.addPairwiseEnergy(feats, compat=3,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This creates the color-dependent features --
    # because the segmentation that we get from CNN are too coarse
    # and we can use local color features to refine them
    feats = create_pairwise_bilateral(sdims=(3, 3), schan=(13, 13, 13),
                                      img=original_image, chdim=2)

    d.addPairwiseEnergy(feats, compat=10,
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)

    # Run Inference for 5 steps
    Q = d.inference(5)
    #print(">>>>>>>>Qshape: ", Q.shape)
    # Find out the most probable class for each pixel.
    output = np.argmax(Q, axis=0).reshape((original_image.shape[0], original_image.shape[1]))

    return output





"""

  UNET  

"""
def conv(inputs, filters, kernel_size=[3, 3], activation=tf.nn.relu, l2_reg_scale=None, batchnorm_istraining=None):
    if l2_reg_scale is None:
        regularizer = None
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
    conved = tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=activation,
        kernel_regularizer=regularizer
    )
    if batchnorm_istraining is not None:
        conved = bn(conved, batchnorm_istraining)

    return conved


def bn(inputs, is_training):
    normalized = tf.layers.batch_normalization(
        inputs=inputs,
        axis=-1,
        momentum=0.9,
        epsilon=0.001,
        center=True,
        scale=True,
        training=is_training,
    )
    return normalized


def pool(inputs):
    pooled = tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)
    return pooled


def conv_transpose(inputs, filters, l2_reg_scale=None):
    if l2_reg_scale is None:
        regularizer = None
    else:
        regularizer = tf.contrib.layers.l2_regularizer(scale=l2_reg_scale)
    conved = tf.layers.conv2d_transpose(
        inputs=inputs,
        filters=filters,
        strides=[2, 2],
        kernel_size=[2, 2],
        padding='same',
        activation=tf.nn.relu,
        kernel_regularizer=regularizer
    )
    return conved

def unetinference(image, keep_prob):
    net = {}
    l2_reg = FLAGS.learning_rate
    # added for resume better
    global_iter_counter = tf.Variable(0, name='global_step', trainable=False)
    net['global_step'] = global_iter_counter
    with tf.variable_scope("inference"):
        inputs = image
        teacher = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, 18])
        is_training = True

        # 1, 1, 3
        conv1_1 = conv(inputs, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv1_2 = conv(conv1_1, filters=64, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool1 = pool(conv1_2)

        # 1/2, 1/2, 64
        conv2_1 = conv(pool1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv2_2 = conv(conv2_1, filters=128, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool2 = pool(conv2_2)

        # 1/4, 1/4, 128
        conv3_1 = conv(pool2, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv3_2 = conv(conv3_1, filters=256, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool3 = pool(conv3_2)

        # 1/8, 1/8, 256
        conv4_1 = conv(pool3, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        conv4_2 = conv(conv4_1, filters=512, l2_reg_scale=l2_reg, batchnorm_istraining=is_training)
        pool4 = pool(conv4_2)

        # 1/16, 1/16, 512
        conv5_1 = conv(pool4, filters=1024, l2_reg_scale=l2_reg)
        conv5_2 = conv(conv5_1, filters=1024, l2_reg_scale=l2_reg)
        concated1 = tf.concat([conv_transpose(conv5_2, filters=512, l2_reg_scale=l2_reg), conv4_2], axis=3)

        conv_up1_1 = conv(concated1, filters=512, l2_reg_scale=l2_reg)
        conv_up1_2 = conv(conv_up1_1, filters=512, l2_reg_scale=l2_reg)
        concated2 = tf.concat([conv_transpose(conv_up1_2, filters=256, l2_reg_scale=l2_reg), conv3_2], axis=3)

        conv_up2_1 = conv(concated2, filters=256, l2_reg_scale=l2_reg)
        conv_up2_2 = conv(conv_up2_1, filters=256, l2_reg_scale=l2_reg)
        concated3 = tf.concat([conv_transpose(conv_up2_2, filters=128, l2_reg_scale=l2_reg), conv2_2], axis=3)

        conv_up3_1 = conv(concated3, filters=128, l2_reg_scale=l2_reg)
        conv_up3_2 = conv(conv_up3_1, filters=128, l2_reg_scale=l2_reg)
        concated4 = tf.concat([conv_transpose(conv_up3_2, filters=64, l2_reg_scale=l2_reg), conv1_2], axis=3)

        conv_up4_1 = conv(concated4, filters=64, l2_reg_scale=l2_reg)
        conv_up4_2 = conv(conv_up4_1, filters=64, l2_reg_scale=l2_reg)
        outputs = conv(conv_up4_2, filters=18, kernel_size=[1, 1], activation=None)
        annotation_pred = tf.argmax(outputs, dimension=3, name="prediction")

        return tf.expand_dims(annotation_pred, dim=3), outputs, net
        #return Model(inputs, outputs, teacher, is_training)


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
    # global_step = tf.Variable(0, trainable=False, name='global_step')

    # 2. construct inference network
    pred_annotation, logits, net = unetinference(image, keep_probability)
    tf.summary.image("input_image", image, max_outputs=2)
    tf.summary.image("ground_truth", tf.cast(annotation, tf.uint8), max_outputs=2)

    tf.summary.image("pred_annotation", tf.cast(pred_annotation, tf.uint8), max_outputs=2)

    # 3. loss measure
    loss = tf.reduce_mean((tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                          labels=tf.squeeze(annotation,
                                                                                            squeeze_dims=[3]),
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
        valid = list()
        step  = list()
        lo = list()

        global_step = sess.run(net['global_step'])
        global_step = 0
        for itr in xrange(global_step, MAX_ITERATION):
            # 6.1 load train and GT images
            train_images, train_annotations = train_dataset_reader.next_batch(FLAGS.batch_size)
            # print("train_image:", train_images.shape)
            # print("annotation :", train_annotations.shape)

            feed_dict = {image: train_images, annotation: train_annotations, keep_probability: 0.85}
            # 6.2 traininging

            sess.run(train_op, feed_dict=feed_dict)

            if itr % 10 == 0:
                train_loss, summary_str = sess.run([loss, summary_op], feed_dict=feed_dict)
                print("Step: %d, Train_loss:%g" % (itr, train_loss))
                summary_writer.add_summary(summary_str, itr)
                if itr % 500 ==0:
                    lo.append(train_loss)
            if itr % 500 == 0:
                valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
                valid_loss = sess.run(loss, feed_dict={image: valid_images, annotation: valid_annotations,
                                                       keep_probability: 1.0})
                print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))

                global_step = sess.run(net['global_step'])
                saver.save(sess, FLAGS.logs_dir + "model.ckpt", global_step=global_step)

                #Save the accuracy to the figure
                #print("valid", valid, "step", step)

                valid.append(valid_loss)
                step.append(itr)
                #print("valid", valid, "step", step)

                plt.plot(step, valid)
                plt.ylabel("Accuracy")
                plt.xlabel("Step")
                plt.title('Training Loss')
                plt.savefig(FLAGS.logs_dir + "Faccuracy.jpg")

                plt.clf();
                plt.plot(step, lo)
                plt.title('Loss')
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.savefig(FLAGS.logs_dir + "Floss.jpg")

                plt.clf();

                plt.plot(step, lo)
                plt.plot(step, valid)
                plt.ylabel("Loss")
                plt.xlabel("Step")
                plt.title('Result')
                plt.legend(['Training Loss','Accuracy'], loc='upper right')
                plt.savefig(FLAGS.logs_dir + "Fmerge.jpg")

    # test-mode
    elif FLAGS.mode == "visualize":

        valid_images, valid_annotations = validation_dataset_reader.get_random_batch(FLAGS.batch_size)
        pred = sess.run(pred_annotation, feed_dict={image: valid_images, annotation: valid_annotations,
                                                    keep_probability: 1.0})
        valid_annotations = np.squeeze(valid_annotations, axis=3)
        pred = np.squeeze(pred, axis=3)

        for itr in range(FLAGS.batch_size):
            utils.save_image(valid_images[itr].astype(np.uint8), FLAGS.logs_dir + "Image/", name="inp_" + str(5 + itr))
            utils.save_image((valid_annotations[itr].astype(np.uint8))*255/18, FLAGS.logs_dir + "Image/", name="gt_" + str(5 + itr))
            utils.save_image((pred[itr].astype(np.uint8))*255/18, FLAGS.logs_dir + "Image/", name="pred_" + str(5 + itr))
            print("Saved image: %d" % itr)

    elif FLAGS.mode == "test":  # heejune added
        print(">>>>>>>>>>>>>>>>Test mode")
        crossMats = list()
        mIOU_all = list()
        validation_dataset_reader.reset_batch_offset(0)
        pred_e = list()
        # print(">>>>>>>>>>>>>>>>Test mode")
        for itr1 in range(validation_dataset_reader.get_num_of_records() // 2):
            # print(">>>>>>>>>>>>>>>>Test mode")
            valid_images, valid_annotations = validation_dataset_reader.next_batch(FLAGS.batch_size)
            pred, logits1 = sess.run([pred_annotation, logits],
                                    feed_dict={image: valid_images, annotation: valid_annotations,
                                               keep_probability: 1.0})
            valid_annotations = np.squeeze(valid_annotations, axis=3)
            #logits1 = np.squeeze(logits1)
            pred = np.squeeze(pred)
            print("logits shape:", logits1.shape)
            np.set_printoptions(threshold=np.inf)
            # print("logits:", logits)
            for itr2 in range(FLAGS.batch_size):

                print("Output file: ", FLAGS.logs_dir + "crf_" + str(itr1 * 2 + itr2))
                crfoutput = crf(valid_images[itr2].astype(np.uint8), logits1[itr2])

                fig = plt.figure()
                pos = 240 + 1
                plt.subplot(pos)
                plt.imshow(valid_images[itr2].astype(np.uint8))
                plt.axis('off')
                plt.title('Original')

                pos = 240 + 2
                plt.subplot(pos)
                plt.imshow(valid_annotations[itr2].astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
                plt.axis('off')
                plt.title('GT')

                pos = 240 + 3
                plt.subplot(pos)
                plt.imshow(pred[itr2].astype(np.uint8), cmap=plt.get_cmap('nipy_spectral'))
                plt.axis('off')
                plt.title('Prediction')

                pos = 240 + 4
                plt.subplot(pos)
                plt.imshow(crfoutput, cmap=plt.get_cmap('nipy_spectral'))
                plt.axis('off')
                plt.title('CRFPostProcessing')

                pos = 240 + 6
                plt.subplot(pos)
                ret, errorImage = cv2.threshold(
                    cv2.absdiff(pred[itr2].astype(np.uint8), valid_annotations[itr2].astype(np.uint8)), 0.5,
                    255, cv2.THRESH_BINARY)
                plt.imshow(errorImage, cmap=plt.get_cmap('gray'))
                plt.axis('off')
                plt.title('Pred Error:' + str(np.count_nonzero(errorImage)))
                pred_e.append(np.count_nonzero(errorImage))

                crossMat = _calcCrossMat(valid_annotations[itr2].astype(np.uint8), pred[itr2].astype(np.uint8), NUM_OF_CLASSESS)
                crossMats.append(crossMat)
                #print(crossMat)
                IoUs = _calcIOU(valid_annotations[itr2].astype(np.uint8), pred[itr2].astype(np.uint8),
                                NUM_OF_CLASSESS)
                mIOU_all.append(IoUs)


                crfoutput = cv2.normalize(crfoutput, None, 0, 255, cv2.NORM_MINMAX)
                valid_annotations[itr2] = cv2.normalize(valid_annotations[itr2], None, 0, 255, cv2.NORM_MINMAX)

                pos = 240 + 8
                plt.subplot(pos)
                ret, errorImage = cv2.threshold(cv2.absdiff(crfoutput.astype(np.uint8), valid_annotations[itr2].astype(np.uint8)),
                   10, 255, cv2.THRESH_BINARY)
                plt.imshow(errorImage, cmap=plt.get_cmap('gray'))
                plt.axis('off')
                plt.title('CRF Error:' + str(np.count_nonzero(errorImage)))





                #np.set_printoptions(threshold=np.inf)

                #plt.show()

                np.savetxt(FLAGS.logs_dir + "Image/Crossmatrix" + str(itr1 * 2 + itr2) + ".csv", crossMat, fmt='%4i', delimiter=',')
                np.savetxt(FLAGS.logs_dir + "Image/IoUs" + str(itr1 * 2 + itr2) + ".csv", IoUs, fmt='%4f',
                           delimiter=',')
                plt.savefig(FLAGS.logs_dir + "Image/resultSum_" + str(itr1 * 2 + itr2))
                # ---------------------------------------------
                utils.save_image(valid_images[itr2].astype(np.uint8), FLAGS.logs_dir + "Image/",
                                 name="inp_" + str(itr1 * 2 + itr2))
                utils.save_image(valid_annotations[itr2].astype(np.uint8), FLAGS.logs_dir + "Image/",
                                 name="gt_" + str(itr1 * 2 + itr2))
                utils.save_image(pred[itr2].astype(np.uint8), FLAGS.logs_dir + "Image/", name="pred_" + str(itr1 * 2 + itr2))
                utils.save_image(crfoutput, FLAGS.logs_dir + "Image/", name="crf_" + str(itr1 * 2 + itr2))

                plt.close('all')
                print("Saved image: %d" % (itr1 * 2 + itr2))
                # save list of error to file
        with open(FLAGS.logs_dir + 'pred_e.txt', 'w') as file:
            for error in pred_e:
                file.write("%i\n" % error)
        np.savetxt(FLAGS.logs_dir + "Crossmatrix.csv", np.sum(crossMats, axis=0), fmt='%4i',
                   delimiter=',')
        np.savetxt(FLAGS.logs_dir + "mIoUs" + ".csv", np.mean(mIOU_all, axis=0), fmt='%4f',
                   delimiter=',')

if __name__ == "__main__":
    """valid = []
    step = []
    valid.append(0.5)
    valid.append(1.5)
    step.append(0)
    step.append(500)
    plt.plot(step, valid)
    plt.savefig(FLAGS.logs_dir + "accuracy.jpg")"""
    tf.app.run()
