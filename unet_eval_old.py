# ============================================================== #
#                         Fusnet eval                            #
#                                                                #
#                                                                #
# Eval fusnet with processed dataset in tfrecords format         #
#                                                                #
# Author: Karim Tarek                                            #
# ============================================================== #

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


def load_datafiles(type):
    """
    Get all tfrecords from tfrecords dir:
    """

    tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '%s-*' % type)
    data_files = tf.gfile.Glob(tf_record_pattern)

    data_size = 0
    for fn in data_files:
        for record in tf.python_io.tf_record_iterator(fn):
            data_size += 1

    return data_files, data_size


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
            print ( image_array.dtype )
            image_array_uint = tf.image.convert_image_dtype (image_array,  dtype=tf.uint8)
            
            print ( image_array_uint.shape)
            print ( image_array_uint.dtype )
            
            #image = Image.fromarray(np.asarray(image_array, ))
            #image.save(file_path)
            Image.fromarray(np.asarray(image_array_uint)).save(file_path)

    
    
def evaluate():
    """
    Eval unet using specified args:
    """
    """
    data_files, data_size = load_datafiles(FLAGS.tfrecords_prefix)
    images, labels, filenames = dataset_loader.inputs(
                                    data_files = data_files,
                                    image_size = FLAGS.image_size,
                                    batch_size = FLAGS.batch_size,
                                    num_epochs = 1,
                                    train = False)

    """
    ##
    ## read from file 
    ##
    
    X_data = []
    
    files = glob.glob ("G:\\Arindam\\Projects\\ImageAveragingUsingSLIM\\Tests\\dim128SetA\\*.png")
    for myFile in files:
        print(myFile)
        image = cv2.imread (myFile)
        X_data.append (image)
    
    print('X_data shape:', np.array(X_data).shape)
    images=np.array(X_data)
    file_list=["1.png","2.png","3.png","4.png","5.png", "6.png", "7.png", "8.png", "9.png", "10.png"]
    
    #filenames= tf.convert_to_tensor(file_list)
    #logits, variables_to_restore = unet2d.unet2d_AB(images, 3, False, False, 5)
    logits, variables_to_restore = unet2d.unet2d_AB(np.array(X_data, dtype=float), 3, False, False, 5)

    predicted_images = logits # this is because I have removed the last softmax layer
    
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

    step = 0
    
    
    try:
    
        while not coord.should_stop():
            #acc_seg_value, predicted_images_value, filenames_value = sess.run([accuracy, predicted_images, filenames])
            #global_accuracy += acc_seg_value
            #global_accuracy += acc_seg_value
            #print('[PROGRESS]\tAccuracy for current batch: %.5f' % (acc_seg_value))
            #predicted_images_value, filenames_value = sess.run([predicted_images, filenames])
            predicted_images_value = sess.run([predicted_images])
            
            
            # save the original images to see if they are different from just feed dict
    
            #maybe_save_images ( images, filenames_value, sess)
            #images_value = sess.run ( images )
            #print (images_value.shape)
            #print (images_value.dtype)
            
            print ('images shape ', images.shape)
            print ('images type  ', images.dtype)
            
            
            ##print (predicted_images_value.shape)
            #print (predicted_images_value.dtype)
            
            #np.savetxt('data.csv',images_value[0,:,:,0],delimiter=',',fmt='%f')  
            in_image = tf.image.convert_image_dtype (images,  dtype=tf.uint16)
            #in_image = images
            print ('in image type ', in_image.dtype)
            print ('in image type ', in_image.shape)
            in_image = tf.squeeze (in_image)
            print ('in image type  after squeeze ', in_image.dtype)
            print ('in image type after squeeze ', in_image.shape)
            enc = tf.image.encode_png((in_image[1,:,:,:]))
            fname = tf.constant('2.png')
            fwrite = tf.write_file(fname, enc)
            result = sess.run(fwrite)
            
            #maybe_save_images(predicted_images_value, filenames_value, sess)
            if FLAGS.output_dir is not None:
                print ( len( predicted_images_value) ) 
                L = len ( predicted_images_value)
                batch_size = L[0]#predicted_images_value.shape[0]
                print ( 'batch size ', batch_size, 'images shape ', predicted_images_value.shape)
                for i in range(batch_size):
                    image_array = predicted_images_value[i, :, :,:]
                    print ( 'output image type ', image_array.dtype)
                    #print ( 'output image shape ', image_array.size)
                    print ( 'Eval FileName ', filenames_value[i])
                    
                    
                    print ( 'output image type ', image_array.dtype)
                    #print ( 'output image shape ', image_array.shape)
                    
                    timg = tf.image.encode_png(image_array)
                    #file_path = os.path.join(FLAGS.output_dir, filenames_value[i].decode("utf-8"))
                    file_path = os.path.join(FLAGS.output_dir, file_list[i])
                    dataimg = sess.run (timg)
                    f = open(file_path, "wb+")
                    f.write(dataimg)
                    f.close()
                    

            
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
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 224)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 1)
    parser.add_argument('--output_dir', help = 'Output directory for the prediction files. If this is not set then predictions will not be saved')
    
    FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
