# -*- coding: utf-8 -*-
"""
Created on Fri Aug 17 12:06:14 2018

@author: aduser
Slim unet evaluation using .dat files
"""

from __future__ import print_function

import tensorflow as tf
from PIL import Image

import os
import glob
import argparse

import data.dataset_loader as dataset_loader
from modelBase import unet2d
import numpy as np
from scipy.misc import imsave
import cv2

# Basic model parameters as external flags.
FLAGS = None

#other constants
datapath = 'G:\\Arindam\\Projects\\ImageAveragingUsingSLIM\\Train\\patches128\\'
width  = 128
height = 128
depth  = 1
NumData = 981


def maybe_save_images(images, filenames,sess):
    """
    Save images to disk
    -------------
    Args:
        images: numpy array     [batch_size, image_size, image_size]
        filenames: numpy string array, filenames corresponding to the images   [batch_size]
    """

    if FLAGS.output_dir is not None:
        batch_size = images.shape[0]
        print ( 'bs ', batch_size, 'images shape ', images.shape)
        for i in range(batch_size):
            image_array = images[i, :, :,:]
            print ('Eval FileName ', filenames[i])
            file_path = os.path.join(FLAGS.output_dir, filenames[i].decode("utf-8"))
            print ( image_array.shape)
            image_array = tf.squeeze ( image_array )
            print ( image_array.shape)
            #image = Image.fromarray(np.uint8(image_array))
            #image.save(file_path)
            Image.fromarray(np.asarray(image_array.eval(sess))).save(file_path)

def getdata(datapath):
    x = np.zeros((FLAGS.batch_size,width,height,depth))
    y = np.zeros((FLAGS.batch_size,width,height,depth))  
    order = np.random.permutation(NumData)+1    
    for it in range(0,FLAGS.batch_size):
        number = order[it]
        X = np.array(np.fromfile(datapath + 'image' +  str(int(number)) + '.dat',dtype=np.float32)).reshape(width,height,depth)
        Y = np.array(np.fromfile(datapath + 'label' +  str(int(number)) + '.dat',dtype=np.float32)).reshape(width,height,depth)
        # shuffle
        #shuf = numpy.random.permutation(21)
        #X = X[:,:,shuf]
        #Y = Y[shuf]
        x[it,:,:,:] = X
        y[it,:,:,:] = Y
    return x,y

    
def evaluate():
    """
    Eval unet using specified args:
    """

    inputs, targets = getdata(datapath)
    
    #    logits = unet.build(images, FLAGS.num_classes, False)
    predictions, variables_to_restore,conv2_upsampledTwice,conv7_upsampledTwice=unet2d.unet2d_AB( inputs, 3, True, False, 5)
    predicted_images = predictions # this is because I have removed the last softmax layer
    
    #predicted_images = unet.predict(logits, FLAGS.batch_size, FLAGS.image_size)
    #AB_DEBUG NOT MEASURING ACCURACY
    #accuracy = unet.accuracy(logits, labels)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    sess = tf.Session()

    sess.run(init_op)

    saver = tf.train.Saver()

    if not tf.gfile.Exists(FLAGS.checkpoint_path + '.meta'):
        raise ValueError("Can't find checkpoint file")
    else:
        print('[INFO    ]\tFound checkpoint file, restoring model.', FLAGS.checkpoint_path)
        saver.restore(sess, FLAGS.checkpoint_path)
    
    coord = tf.train.Coordinator()

    threads = tf.train.start_queue_runners(sess = sess, coord = coord)

    global_accuracy = 0.0

    step = 1
    num = 0
    
    try:
    
        while not coord.should_stop():
            #acc_seg_value, predicted_images_value, filenames_value = sess.run([accuracy, predicted_images, filenames])
            #global_accuracy += acc_seg_value
            #global_accuracy += acc_seg_value
            #print('[PROGRESS]\tAccuracy for current batch: %.5f' % (acc_seg_value))
            predicted_images_value = sess.run(predictions)
            
            #maybe_save_images(predicted_images_value, filenames_value, sess)
            if FLAGS.output_dir is not None:
                batch_size = predicted_images_value.shape[0]
                print ( 'batch size ', batch_size, 'images shape ', predicted_images_value.shape)
                for i in range(batch_size):
                    num=num+1;
                    image_array = predicted_images_value[i, :, :,:]
                    
                    file_path = os.path.join(FLAGS.output_dir, str (num)+'.dat')
                    
                    #timg = tf.image.encode_png(image_array)
                    #dataimg = sess.run (timg)
                    np.save(file_path, image_array)
                    

            
            step += 1

    except tf.errors.OutOfRangeError:
        print('[INFO    ]\tDone evaluating in %d steps.' % step)

    finally:
        # When done, ask the threads to stop.
        coord.request_stop()

    #global_accuracy = global_accuracy / step

    #print('[RESULT  ]\tGlobal accuracy = %.5f' % (global_accuracy))

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()


def main(_):
    """
    Run unet prediction on input tfrecords
    """

    if FLAGS.output_dir is not None:
        if not tf.gfile.Exists(FLAGS.output_dir):
            print('[INFO    ]\tOutput directory does not exist, creating directory: ' + os.path.abspath(FLAGS.output_dir))
            tf.gfile.MakeDirs(FLAGS.output_dir)
        
    evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Eval Unet on given tfrecords.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory')
    parser.add_argument('--tfrecords_prefix', help = 'Tfrecords prefix', default = 'training')
    parser.add_argument('--checkpoint_path', help = 'Path of checkpoint to restore. (Ex: ../Datasets/checkpoints/unet.ckpt-80000)')
    parser.add_argument('--num_classes', help = 'Number of segmentation labels', type = int, default = 2)
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 128)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 1)
    parser.add_argument('--output_dir', help = 'Output directory for the prediction files. If this is not set then predictions will not be saved')
    
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
