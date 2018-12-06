# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 09:47:14 2018
noise2noise_slim_unet2D_deep_supervision.py

@author: aduser
Unet with deep supervision using unet
"""
import tensorflow as tf
import functools

slim = tf.contrib.layers



def deconv_upsample(inputs, factor, name, padding = 'SAME', activation_fn = None):
    """
    Convolution Transpose upsampling layer with bilinear interpolation weights:
    ISSUE: problems with odd scaling factors
    ----------
    Args:
        inputs: Tensor, [batch_size, height, width, channels]
        factor: Integer, upsampling factor
        name: String, scope name
        padding: String, input padding
        activation_fn: Tensor fn, activation function on output (can be None)
    Returns:
        outputs: Tensor, [batch_size, height * factor, width * factor, num_filters_in]
    """

    with tf.variable_scope(name):
        stride_shape   = [1, factor, factor, 1]
        input_shape    = tf.shape(inputs)
        num_filters_in = inputs.get_shape()[-1].value
        output_shape   = tf.stack([input_shape[0], input_shape[1] * factor, input_shape[2] * factor, num_filters_in])

        curr_shape = inputs.shape.as_list()
        #print ( 'curr shape ', curr_shape.shape)
        outputs  = tf.image.resize_images(inputs, size=[factor*curr_shape[1], factor*curr_shape[2]], method=tf.image.ResizeMethod.BILINEAR, align_corners=False)
        #print ( 'outputs shape ', outputs.shape)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs
    
def train(loss, learning_rate, global_step):
    """
    Train opetation:
    ----------
    Args:
        loss: loss to use for training
        learning_rate: Float, learning rate
    Returns:
        train_op: Training operation
    """
    
    #decayed_learning_rate = tf.train.exponential_decay(learning_rate, global_step, 
    #                        learning_rate_decay_steps, learning_rate_decay_rate, staircase = True)

    # execute update_ops to update batch_norm weights
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        optimizer   = tf.train.AdamOptimizer(learning_rate)
        train_op    = optimizer.minimize(loss, global_step = global_step)

    tf.summary.scalar("learning_rate", learning_rate)

    return train_op    
    
    
def noise2noise_unet2D_1(image_batch_tensor,
           number_of_classes,
           is_training=False,
           reuse=False,
           weight_decay=0.2
           ):

    unet_batch_norm = functools.partial(
        slim.batch_norm, is_training=is_training)
    
    with tf.variable_scope('unet', reuse=reuse) as unet_scope:
        net = tf.to_float(image_batch_tensor)
        with tf.contrib.framework.arg_scope([slim.conv2d,  slim.conv2d_transpose],
                                stride=1, padding='SAME',
                                normalizer_fn=unet_batch_norm,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_initializer=tf.zeros_initializer()):
            conv0 = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], activation_fn=tf.nn.relu, scope='conv0')
            pool0 = slim.max_pool2d(conv0, [2, 2], padding='SAME', scope='pool0')  # 1/2

            conv1 = slim.repeat(pool0, 2, slim.conv2d, 64, [3, 3], activation_fn=tf.nn.relu, scope='conv1')
            pool1 = slim.max_pool2d(conv1, [2, 2],padding='SAME', scope='pool1')  # 1/4

            conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], activation_fn=tf.nn.relu, scope='conv2')
            pool2 = slim.max_pool2d(conv2, [2, 2],padding='SAME', scope='pool2')  # 1/8

            conv3 = slim.repeat(pool2, 2, slim.conv2d, 256, [3, 3],activation_fn=tf.nn.relu, scope='conv3')
            pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME', scope='pool3')  # 1/16

            conv4 = slim.repeat(pool3, 2, slim.conv2d, 512, [3, 3], activation_fn=tf.nn.relu, scope='conv4')
            pool4 = slim.max_pool2d(conv4, [2, 2],padding='SAME', scope='pool4')  # 1/32

            conv_t1 = deconv_upsample(pool4, 2,  'upsample1') 
            merge1 = tf.concat([conv_t1, conv4], 3, name='merge1_5')
            conv5 = slim.repeat(merge1, 2, slim.conv2d, 512, [3, 3], activation_fn=tf.nn.relu, scope='conv5')

            conv_t2 = deconv_upsample(conv5, 2,  'upsample2')
            merge2 = tf.concat([conv_t2, conv3], 3,name='merge2_6')
            conv6 = slim.repeat(merge2, 2, slim.conv2d, 256, [3, 3], activation_fn=tf.nn.relu, scope='conv6')

            conv_t3 = deconv_upsample(conv6, 2,  'upsample3')
            merge3 = tf.concat([conv_t3, conv2], 3, name='merge3_7')
            conv7 = slim.repeat(merge3, 2, slim.conv2d, 128, [3, 3], activation_fn=tf.nn.relu, scope='conv7')

            conv_t4 = deconv_upsample(conv7, 2,  'upsample4')
            merge4 = tf.concat([conv_t4, conv1], 3,name='merge4_8')
            conv8 = slim.repeat(merge4, 2, slim.conv2d, 64, [3, 3], activation_fn=tf.nn.relu, scope='conv8')


            conv_t5  =  deconv_upsample(conv8, 2,  'upsample5') 
            merge5 = tf.concat([conv_t5, conv0], 3, name='merge5_9')
            conv9 = slim.repeat(merge5, 2, slim.conv2d, 32, [3, 3], activation_fn=tf.nn.relu, scope='conv9')


            conv10 = slim.conv2d(conv9, number_of_classes, [1,1], normalizer_fn=None, activation_fn=None,scope='output') # 2 CLASSES_NUM
            conv11 = conv10 + image_batch_tensor

            conv7_out = slim.conv2d(conv7, number_of_classes, [1,1], normalizer_fn=None, activation_fn=None,scope='hidden7')
            conv7_upsampledTwice = deconv_upsample(conv7_out, 4,  'hidden7_upsample4')
            conv2_out = slim.conv2d(conv2, number_of_classes, [1,1], normalizer_fn=None, activation_fn=None,scope='hidden2')
            conv2_upsampledTwice = deconv_upsample(conv2_out, 4,  'hidden2_upsample4')
                   
    return conv11, tf.contrib.framework.get_variables(unet_scope), conv2_upsampledTwice, conv7_upsampledTwice

