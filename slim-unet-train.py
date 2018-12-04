# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 09:38:29 2018

@author: aduser

Unet 2 D using SLIM AB_DEBUG Custom Unet using SLIM
"""

# The following snippet trains the U-net model using a mean_squared_error loss.
import tensorflow as tf
import sys
import os
import glob
import data.dataset_loader as dataset_loader

from absl import flags
# This is needed since the notebook is stored in the `tensorflow/models/gan` folder.
sys.path.append('D:\\ArindamData\\models-master-dirty\\models-master\\research\\')
sys.path.append(os.path.join('D:\\ArindamData\\models-master-dirty\\models-master\\research\\', 'slim'))

from tensorflow.contrib import slim
#import from the current folder
from modelBase import slim_unet2D_deepSupervision
import numpy as np

#ckpt_dir = '/tmp/regression_model/'

#input data location - pairs of images and labels
flags.DEFINE_string('tfrecords_dir', 'D:\\ArindamData\\Code\\ImageAveragingSlimExp\\Datasets\\', 'DIR.')
flags.DEFINE_string('tfrecords_prefix', 'avg128setA', 'prefix')
flags.DEFINE_string('train_log_dir', 'D:\ArindamData\Code\ImageAveragingSlimExp\Checkpoint\\Averaging_DSN_bilinear\\',
                    'Directory where to write event logs.')
flags.DEFINE_integer('batch_size', 8, 'The number of images in each batch.')
flags.DEFINE_integer('patch_size', 128, 'The size of the patches to train on.')
# default 
#flags.DEFINE_string('train_log_dir', 'G:\\Arindam\Projects\\ImageAveragingUsingSLIM\\Checkpoint\\gteavsrdsgtr\\',
#                    'Directory where to write event logs.')
FLAGS = flags.FLAGS

#other constants
#datapath = 'G:\\Arindam\\Projects\\ImageAveragingUsingSLIM\\Train\\patches128\\'
#width  = 128
#height = 128
#depth  = 1
#NumData = 981
#Helper functions.
def load_datafiles(type):
    """
    Get all tfrecords from tfrecords dir:
    """

    tf_record_pattern = os.path.join(FLAGS.tfrecords_dir, '%s-*' % type)
    print ( tf_record_pattern)
    data_files = tf.gfile.Glob(tf_record_pattern)

    data_size = 0
    for fn in data_files:
        print ( fn )
        for record in tf.python_io.tf_record_iterator(fn):
            data_size += 1

    return data_files, data_size


def getdata(datapath):
    x = np.zeros((FLAGS.batch_size,width,height,depth))
    y = np.zeros((FLAGS.batch_size,width,height,depth))  
    
    order = np.random.permutation(NumData)+1    
    print ( 'order ', str ( int ( order[0])))
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

def main(_):
    with tf.Graph().as_default():
        if not tf.gfile.Exists(FLAGS.train_log_dir):
            tf.gfile.MakeDirs(FLAGS.train_log_dir)
        tf.logging.set_verbosity(tf.logging.INFO)
        
        #Get the data. 
        #inputs, targets = convert_data_to_tensors(x_train, y_train)
        
        
        data_files, data_size = load_datafiles(FLAGS.tfrecords_prefix)
        inputs, targets, filenames = dataset_loader.inputs(
                                            data_files = data_files,
                                            image_size = FLAGS.patch_size,
                                            batch_size = FLAGS.batch_size,
                                            num_epochs = 5000000,
                                            train = True)
        """
        inputs, targets = getdata(datapath)
        """
        #change the input to read data file (from Lars ) 
        # Make the model.
        #predictions, nodes = regression_model(inputs, is_training=True)
        #predictions=slimunet.build(inputs, 3, True)
        predictions, variables_to_restore, conv2_upsampledTwice, conv7_upsampledTwice = slim_unet2D_deepSupervision.unet2D_A( inputs, 3, True, False, 5)
        print ( 'prediction ', predictions.shape)
        print ( 'label shape ', targets.shape)
        #print(''tf.shape ( predictions))
    
        # Add the loss function to the graph.
        loss = tf.losses.mean_squared_error(labels=targets, predictions=predictions)
        deep_loss = tf.losses.mean_squared_error(labels=targets, predictions=conv7_upsampledTwice) +  0.2*tf.losses.mean_squared_error(labels=targets, predictions=conv2_upsampledTwice)
        
        total_loss = loss + 0.25 * deep_loss
        
        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.scalar('losses/loss', loss)
        tf.summary.scalar('losses/deep_loss', deep_loss)
        
        #tf.summary.image('predictions', predictions)
        tf.summary.image("input_image",inputs,  max_outputs=8)
        tf.summary.image("ground_truth",targets,  max_outputs=8)
        tf.summary.image("pred_annotation", predictions, max_outputs=8)
        tf.summary.image("pred_deep_7", conv7_upsampledTwice, max_outputs=8)
        tf.summary.image("pred_deep_2", conv2_upsampledTwice, max_outputs=8)
    
         
        #tf.contrib.layers.summarize_tensors()
        tf.contrib.layers.summarize_activations()
        
        # The total loss is the user's loss plus any regularization losses.
        #total_loss = slim.losses.get_total_loss()
        global_step = slim.get_or_create_global_step()
        gen_lr = tf.train.exponential_decay(learning_rate=0.00001,
                global_step=tf.train.get_or_create_global_step(),
                decay_steps=10000,
                decay_rate=0.9, staircase=True,)
        # Specify the optimizer and create the train op:
        optimizer = tf.train.AdamOptimizer(learning_rate=gen_lr)
        train_op  = slim.learning.create_train_op(total_loss, optimizer,summarize_gradients=True) 
        """
        Here is another version of it 
        slim.learning.train(train_tensor, 
                      logdir=train_log_dir,
                      local_init_op=tf.initialize_local_variables(),
                      save_summaries_secs=FLAGS.save_summaries_secs,
                      save_interval_secs=FLAGS.save_interval_secs)
        """
        # Run the training inside a session.
        final_loss = slim.learning.train(
            train_op,
            logdir=FLAGS.train_log_dir,
            local_init_op=tf.initialize_local_variables(),
            number_of_steps=50000,
            save_summaries_secs=15,
            log_every_n_steps=500)
      
    print("Finished training. Last batch loss:", final_loss)
    print("Checkpoint saved in %s" % FLAGS.train_log_dir)


if __name__ == '__main__':
  tf.app.run()