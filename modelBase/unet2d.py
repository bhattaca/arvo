import tensorflow as tf
import functools

slim = tf.contrib.slim


def unet2d_conv(net, layer_idx, filter_count, filter_size, atrous_filter_size, atrous_filter_rate):
    layer_idx += 1

    # print("conv_layer {}: filter_count={}, filter_size={}, atrous_filter_size={}, atrous_filter_rate={}".
    #      format(layer_idx, filter_count, filter_size, atrous_filter_size, atrous_filter_rate))

    net = slim.conv2d(net, filter_count, filter_size,
                      scope='conv{}'.format(layer_idx))

    layer_idx += 1

    net_out_stdconv = slim.conv2d(net, filter_count, filter_size,
                                  scope='conv{}'.format(layer_idx))

    net_out_atrous = slim.conv2d(net, filter_count, atrous_filter_size,
                                 rate=atrous_filter_rate,
                                 scope='aconv{}'.format(layer_idx))

    filter_count *= 2

    net = tf.concat([net_out_stdconv, net_out_atrous], axis=3)

    return net, layer_idx, filter_count


def unet2d_down(net, layer_idx, filter_count, filter_size, atrous_filter_size, atrous_filter_rate, stride):
    net, layer_idx, filter_count = unet2d_conv(
        net, layer_idx, filter_count, filter_size, atrous_filter_size, atrous_filter_rate)

    direct = net

    net = slim.max_pool2d(net, stride, scope='pool{}'.format(layer_idx))

    return net, direct, layer_idx, filter_count


def unet2d_up(net, direct, layer_idx, filter_count, filter_size,
              stride):

    # Use nn upsampling and conv2d instead of conv2d_transpose to avoid checkerboard artifacts
    # https://distill.pub/2016/deconv-checkerboard/
    if stride > 1:
        target_size = tf.shape(net)[1:3] * stride
        net = tf.image.resize_nearest_neighbor(net, target_size)

    net = slim.conv2d(net, filter_count,
                      kernel_size=stride, stride=1,
                      scope='deconv{}'.format(layer_idx))

    filter_count //= 2

    net = tf.concat([net, direct], axis=3)
    layer_idx += 1

    net = slim.conv2d(net, filter_count, filter_size,
                      scope='conv{}'.format(layer_idx))

    layer_idx += 1
    net = slim.conv2d(net, filter_count, filter_size,
                      scope='conv{}'.format(layer_idx))

    return net, layer_idx, filter_count


def unet2d(image_batch_tensor,
           number_of_classes,
           is_training=False,
           reuse=False,
           depth=3,
           filter_size=3,
           atrous_filter_size=3,
           atrous_filter_rate=2,
           stride=2,
           dropout_keep_probability=None,  # dummy
           reuseVariables=None,  # dummy
           batch_norm_scale=True,
           weight_decay=0.02,
           initial_filter_count=16,
           use_batch_norm=True  # if applying batch normalization at all
           ):

    """
    2d-UNET inspired by the 3D version as described in Cicek et al, "2d U-NET: Learning Dense
    Volumentric Segmentation from Sparse Annotation", MICCAI 2016". In
    particular this is a TensorFlow Slim port of Caffe prototxt
    https://lmb.informatik.uni-freiburg.de/resources/opensource/2dUnet_miccai2016_with_BN.prototxt

    Args:
        image_batch_tensor  (tensor): Batch 2d volume input
        number_of_classes      (int): Number of classes, e.g. 2 for Foreground/Background types of problems
        is_training           (bool): Training or test version of the network (e.g. with dropout layers)
        reuse                 (bool): Reuse TensorFlow variables
        depth                  (int): Depth of the network (default 3)
        filter_size            (int): Filter size of the conv layers (default 3)
        atrous_filter_size     (int): Filter size of the atrous convolutions of the conv layers (default 3)
        atrous_filter_rate     (int): Input stride of the atrous convolutions of the conv layers (default 2)
        stride                 (int): Stride of the pooling layers (default 2)
        dropout_keep_probability (float or none): Dropout For bottleneck layer (default: None)
        batch_norm_scale      (bool): Include scaling operation in the batch norm (default True)
        weight_decay         (float): Strength of L2 weight regularizer
        use_batch_norm        (bool): if applying batch normalization at all (default: True)
    """

    if use_batch_norm:
        unet_batch_norm = functools.partial(
            slim.batch_norm, scale=batch_norm_scale, is_training=is_training)
    else:
        unet_batch_norm = None

    with tf.variable_scope('unet', reuse=reuse) as unet_scope:

        # Convert image to float32 before doing anything else
        # This is necessary, since conv2d does not accept int inputs:
        #   input: A Tensor. Must be one of the following types: half, bfloat16, float32.
        #   A 4-D tensor. The dimension order is interpreted according to the value of data_format, see below for details.
        #   (From https://www.tensorflow.org/api_docs/python/tf/nn/conv2d)
        net = tf.to_float(image_batch_tensor)
        """
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.crelu,
                        normalizer_fn=slim.batch_norm,
                        normalizer_params={'is_training': is_training, 'decay': 0.95}):
        """
        with slim.arg_scope([slim.conv2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=unet_batch_norm,
                            weights_regularizer=slim.l2_regularizer(weight_decay),
                            biases_initializer=tf.zeros_initializer()):

            direct_connections = []
            layer_idx = 0
            filter_count = initial_filter_count  # this is the initial filter count only

            # Down
            for d in range(depth):
                # print("down {} / {}".format(d, depth))
                net, direct, layer_idx, filter_count = unet2d_down(
                    net, layer_idx, filter_count, filter_size, atrous_filter_size, atrous_filter_rate, stride)

                direct_connections.append(direct)

            # Drop out
            if dropout_keep_probability:
                net = slim.dropout(net, dropout_keep_probability, is_training=is_training, scope='dropoutAtBottleneck')

            # Bottom
            net, layer_idx, filter_count = unet2d_conv(
                net, layer_idx, filter_count, filter_size, atrous_filter_size, atrous_filter_rate)

            # Up
            for d in reversed(range(depth)):
                # print("up {} / {}".format(d, depth))
                net, layer_idx, filter_count = unet2d_up(
                    net, direct_connections[d], layer_idx, filter_count,
                    filter_size, stride)
        print ( 'net ', net.shape)        
        # Output
        net = slim.conv2d(net, number_of_classes, [1, 1],
                          normalizer_fn=None,
                          activation_fn=None,
                          scope='output')
        print ( 'net2 ', net.shape)
        #print ( 'net 3', lambda x: x, slim.get_variables(unet_scope))

    #return net, list((lambda x: x, slim.get_variables(unet_scope)))
    return net, slim.get_variables(unet_scope)
"""
def unet2d_AB(image_batch_tensor,
           number_of_classes,
           is_training=False,
           reuse=False,
           depth=3,
           filter_size=3,
           atrous_filter_size=3,
           atrous_filter_rate=2,
           stride=1,
           dropout_keep_probability=None,  # dummy
           reuseVariables=None,  # dummy
           batch_norm_scale=True,
           weight_decay=0.02,
           initial_filter_count=64,
           use_batch_norm=True  # if applying batch normalization at all
           ):
    
    unet_batch_norm = functools.partial(
            slim.batch_norm, is_training=is_training, center=True, scale=True, epsilon=1e-5, decay=0.9)
    
    with tf.variable_scope('unet', reuse=reuse) as unet_scope:

        net = tf.to_float(image_batch_tensor)
        with slim.arg_scope([slim.conv2d,  slim.conv2d_transpose],
                                stride=1, padding='SAME',
                                
                                normalizer_fn=unet_batch_norm,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_initializer=tf.zeros_initializer()):

            filter_count = initial_filter_count  # this is the initial filter count only
            
            conv0 = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], activation_fn=tf.nn.relu, scope='conv0')
            pool0 = slim.max_pool2d(conv0, [2, 2], scope='pool0')  # 1/2
            print ( 'conv0:', conv0.shape)
            print ( 'pool0:', pool0.shape)
            
         
            conv1 = slim.repeat(pool0, 2, slim.conv2d, 64, [3, 3], activation_fn=tf.nn.relu, scope='conv1')
            pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')  # 1/4
            print ( 'conv1:', conv1.shape)
            print ( 'pool1:', pool1.shape)
            
            
            conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], activation_fn=tf.nn.relu, scope='conv2')
            pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')  # 1/8
            print ( 'conv2:', conv2.shape)
            print ( 'pool2:', pool2.shape)
            
            conv3 = slim.repeat(pool2, 2, slim.conv2d, 256, [3, 3],activation_fn=tf.nn.relu, scope='conv3')
            pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')  # 1/16
            print ( 'conv3:', conv3.shape)
            print ( 'pool3:', pool3.shape)  
            
            
            conv4 = slim.repeat(pool3, 2, slim.conv2d, 512, [3, 3], activation_fn=tf.nn.relu, scope='conv4')
            pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')  # 1/32
            print ( 'conv4:', conv4.shape)
            print ( 'pool4:', pool4.shape)     # 10 x 2 x 2 x 512 
            
            # upsampling
            conv_t1 = slim.conv2d_transpose(pool4, num_outputs=512, kernel_size=[3,3], stride=2, activation_fn=tf.nn.relu, scope='conv_t1') # up to 1/16 + conv3
            print ( 'conv_t1:', conv_t1.shape)
            merge1 = tf.concat([conv_t1, conv4], 3)
            conv5 = slim.stack(merge1, slim.conv2d, [(1024, [3, 3]),(512, [3,3])], scope='conv5')
            print ( 'conv5:', conv5.shape)
            
            filter_count = filter_count/2;
            conv_t2 = slim.conv2d_transpose(conv5,  num_outputs=256, kernel_size=[3,3], stride=2, activation_fn=tf.nn.relu, scope='conv_t2') # up to 1/8 + conv2
            merge2 = tf.concat([conv_t2, conv3], 3)
            conv6 = slim.stack(merge2, slim.conv2d, [(512, [3,3]), (256, [3,3])], scope='conv6')
            print ( 'conv6:', conv6.shape)
            
            filter_count = filter_count/2;
            conv_t3 = slim.conv2d_transpose(conv6,  num_outputs=128, kernel_size=[3,3], stride=2, activation_fn=tf.nn.relu, scope='conv_t3') # up to 1/4 + conv1
            merge3 = tf.concat([conv_t3, conv2], 3)
            conv7 = slim.stack(merge3, slim.conv2d, [(256, [3,3]), (128, [3,3])], scope='conv7')
            print ( 'conv7:', conv7.shape)
            
            filter_count = filter_count/2;
            conv_t4 = slim.conv2d_transpose(conv7,  num_outputs=64, kernel_size=[3,3], stride=2,activation_fn=tf.nn.relu, scope='conv_t4')  # up to 1/2 + conv0
            merge4 = tf.concat([conv_t4, conv1], 3)
            conv8 = slim.stack(merge4, slim.conv2d, [(128, [3,3]), (64, [3,3])], scope='conv8')
            print ( 'conv8:', conv8.shape)
            
            conv_t5 = slim.conv2d_transpose(conv8,  num_outputs=32, kernel_size=[3,3], stride=2, activation_fn=tf.nn.relu, scope='conv_t5')  # up to 1/2 + conv0
            merge5 = tf.concat([conv_t5, conv0], 3)
            conv9 = slim.stack(merge5, slim.conv2d, [(64, [3,3]), (32, [3,3])], scope='conv9')
            print ( 'conv9:', conv9.shape)
            
            # output layer scoreMap
            #conv10 = slim.conv2d(conv9, 1, [1,1], normalizer_fn=None, activation_fn=None,scope='output') # 2 CLASSES_NUM
            conv10 = slim.conv2d(conv9, 1, [1,1], normalizer_fn=None, activation_fn=None,scope='output') # 2 CLASSES_NUM
            
            
            
            #conv8_out = slim.conv2d(conv8, 1, [1,1], normalizer_fn=None, activation_fn=None,scope='garb8') # 2 CLASSES_NUM
            #conv7_out = slim.conv2d(conv7, 1, [1,1], normalizer_fn=None, activation_fn=None,scope='garb7') # 2 CLASSES_NUM
            
            

        
    return conv10, slim.get_variables(unet_scope)
"""

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
    
    
    
def unet2d_AB(image_batch_tensor,
           number_of_classes,
           is_training=False,
           reuse=False,
           depth=3,
           filter_size=3,
           atrous_filter_size=3,
           atrous_filter_rate=2,
           stride=1,
           dropout_keep_probability=None,  # dummy
           reuseVariables=None,  # dummy
           batch_norm_scale=True,
           weight_decay=0.02,
           initial_filter_count=64,
           use_batch_norm=True  # if applying batch normalization at all
           ):
    #version 2 of the image translation 
#    unet_batch_norm = functools.partial(
#            slim.batch_norm, is_training=is_training, center=True, scale=True, epsilon=1e-5, decay=0.9)
    unet_batch_norm = functools.partial(
        slim.batch_norm, is_training=is_training)
    
    with tf.variable_scope('unet', reuse=reuse) as unet_scope:

        net = tf.to_float(image_batch_tensor)
        with slim.arg_scope([slim.conv2d,  slim.conv2d_transpose],
                                stride=1, padding='SAME',
                                
                                normalizer_fn=unet_batch_norm,
                                weights_regularizer=slim.l2_regularizer(weight_decay),
                                biases_initializer=tf.zeros_initializer()):

            filter_count = initial_filter_count  # this is the initial filter count only
            
            conv0 = slim.repeat(net, 2, slim.conv2d, 32, [3, 3], activation_fn=tf.nn.relu, scope='conv0')
            pool0 = slim.max_pool2d(conv0, [2, 2], scope='pool0')  # 1/2
            print ( 'conv0:', conv0.shape)
            print ( 'pool0:', pool0.shape)
            
         
            conv1 = slim.repeat(pool0, 2, slim.conv2d, 64, [3, 3], activation_fn=tf.nn.relu, scope='conv1')
            pool1 = slim.max_pool2d(conv1, [2, 2], scope='pool1')  # 1/4
            print ( 'conv1:', conv1.shape)
            print ( 'pool1:', pool1.shape)
            
            
            conv2 = slim.repeat(pool1, 2, slim.conv2d, 128, [3, 3], activation_fn=tf.nn.relu, scope='conv2')
            pool2 = slim.max_pool2d(conv2, [2, 2], scope='pool2')  # 1/8
            print ( 'conv2:', conv2.shape)
            print ( 'pool2:', pool2.shape)
            
            conv3 = slim.repeat(pool2, 2, slim.conv2d, 256, [3, 3],activation_fn=tf.nn.relu, scope='conv3')
            pool3 = slim.max_pool2d(conv3, [2, 2], scope='pool3')  # 1/16
            print ( 'conv3:', conv3.shape)
            print ( 'pool3:', pool3.shape)  
            
            
            conv4 = slim.repeat(pool3, 2, slim.conv2d, 512, [3, 3], activation_fn=tf.nn.relu, scope='conv4')
            pool4 = slim.max_pool2d(conv4, [2, 2], scope='pool4')  # 1/32
            print ( 'conv4:', conv4.shape)
            print ( 'pool4:', pool4.shape)     # 10 x 2 x 2 x 512 
            
            # upsampling
            # conv_t1 = slim.conv2d_transpose(pool4, num_outputs=512, kernel_size=[3,3], stride=2, activation_fn=tf.nn.relu, scope='conv_t1') # up to 1/16 + conv3
            
            conv_t1 = deconv_upsample(pool4, 2,  'upsample1')
            print ( 'conv_t1:', conv_t1.shape)
            merge1 = tf.concat([conv_t1, conv4], 3, name='merge1_5')
            conv5 = slim.repeat(merge1, 2, slim.conv2d, 512, [3, 3], activation_fn=tf.nn.relu, scope='conv5')
            print ( 'conv5:', conv5.shape)
            

            #conv_t2 = slim.conv2d_transpose(conv5,  num_outputs=256, kernel_size=[3,3], stride=2, activation_fn=tf.nn.relu, scope='conv_t2') # up to 1/8 + conv2
            conv_t2 = deconv_upsample(conv5, 2,  'upsample2')
            print ( 'conv_t2:', conv_t2.shape)
            merge2 = tf.concat([conv_t2, conv3], 3,name='merge2_6')
            conv6 = slim.repeat(merge2, 2, slim.conv2d, 256, [3, 3], activation_fn=tf.nn.relu, scope='conv6')
            print ( 'conv6:', conv6.shape)
            
            #conv_t3 = slim.conv2d_transpose(conv6,  num_outputs=128, kernel_size=[3,3], stride=2, activation_fn=tf.nn.relu, scope='conv_t3') # up to 1/4 + conv1
            conv_t3 = deconv_upsample(conv6, 2,  'upsample3')
            merge3 = tf.concat([conv_t3, conv2], 3, name='merge3_7')
            conv7 = slim.repeat(merge3, 2, slim.conv2d, 128, [3, 3], activation_fn=tf.nn.relu, scope='conv7')
            print ( 'conv7:', conv7.shape)
            

            #conv_t4 = slim.conv2d_transpose(conv7,  num_outputs=64, kernel_size=[3,3], stride=2,activation_fn=tf.nn.relu, scope='conv_t4')  # up to 1/2 + conv0
            conv_t4 = deconv_upsample(conv7, 2,  'upsample4')
            merge4 = tf.concat([conv_t4, conv1], 3,name='merge4_8')
            conv8 = slim.repeat(merge4, 2, slim.conv2d, 64, [3, 3], activation_fn=tf.nn.relu, scope='conv8')
            print ( 'conv8:', conv8.shape)
            
            #conv_t5 = slim.conv2d_transpose(conv8,  num_outputs=32, kernel_size=[3,3], stride=2, activation_fn=tf.nn.relu, scope='conv_t5')  # up to 1/2 + conv0
            conv_t5  =  deconv_upsample(conv8, 2,  'upsample5') 
            merge5 = tf.concat([conv_t5, conv0], 3, name='merge5_9')
            conv9 = slim.repeat(merge5, 2, slim.conv2d, 32, [3, 3], activation_fn=tf.nn.relu, scope='conv9')
            print ( 'conv9:', conv9.shape)
            
            # output layer scoreMap
            #conv10 = slim.conv2d(conv9, 1, [1,1], normalizer_fn=None, activation_fn=None,scope='output') # 2 CLASSES_NUM
            conv10 = slim.conv2d(conv9, 1, [1,1], normalizer_fn=None, activation_fn=None,scope='output') # 2 CLASSES_NUM
            
            
            
            
            conv7_out = slim.conv2d(conv7, 1, [1,1], normalizer_fn=None, activation_fn=None,scope='hidden7')
            conv7_upsampledTwice = deconv_upsample(conv7_out, 4,  'hidden7_upsample4')
            print ( 'conv7_out: ', conv7_out.shape)
            print ( 'conv7_upsampledTwice: ', conv7_upsampledTwice.shape)
            #w = tf.Variable([128, 128])
            #conv7_upsampledTwice = tf.image.resize_images( conv7_out, size=w)
            #conv7_upsampledOnce  = slim.conv2d_transpose(conv7_out,  num_outputs=4, kernel_size=[3,3], stride=2, activation_fn=None,scope='hidden7_UP1')
            #conv7_upsampledTwice = slim.conv2d_transpose(conv7_upsampledOnce,  num_outputs=1, kernel_size=[3,3], stride=2, activation_fn=None,scope='hidden7_UP2')
            
            
            conv2_out = slim.conv2d(conv2, 1, [1,1], normalizer_fn=None, activation_fn=None,scope='hidden2')
            print ( 'conv2_out: ', conv2_out.shape)
            conv2_upsampledTwice = deconv_upsample(conv2_out, 4,  'hidden2_upsample4')
            print ( 'conv2_upsampledTwice: ', conv2_upsampledTwice.shape)
            #conv2_upsampledTwice = tf.image.resize_images( conv2_out, size=w)
            #conv2_upsampledOnce= slim.conv2d_transpose(conv2_out,  num_outputs=4, kernel_size=[3,3], stride=2, activation_fn=None,scope='hidden2_UP1')
            #conv2_upsampledTwice= slim.conv2d_transpose(conv2_upsampledOnce,  num_outputs=1, kernel_size=[3,3], stride=2, activation_fn=None,scope='hidden2_UP2')
            

        
    return conv10, slim.get_variables(unet_scope), conv2_upsampledTwice, conv7_upsampledTwice
