
from __future__ import print_function

import configparser
import numpy as np
import tensorflow as tf

import argparse
import os
import time
import glob
import sys

import data.dataset_loader as dataset_loader
from modelBase import arvo_slim_unet2D_deepSupervision



# Basic model parameters as external flags.
from absl import flags
# This is needed since the notebook is stored in the `tensorflow/models/gan` folder.
sys.path.append('D:\\ArindamData\\models-master-dirty\\models-master\\research\\')
sys.path.append(os.path.join('D:\\ArindamData\\models-master-dirty\\models-master\\research\\', 'slim'))
#flags.DEFINE_string('train_log_dir', 'D:\ArindamData\Code\ImageAveragingSlimExp\Checkpoint\\ck_6mm_1\\',
#                    'Directory where to write event logs.') 

FLAGS = flags.FLAGS

def getdata(datapath):
    x = np.zeros((FLAGS.batch_size,FLAGS.width,FLAGS.height,FLAGS.depth))
    y = np.zeros((FLAGS.batch_size,FLAGS.width,FLAGS.height,FLAGS.depth))  
    
    order = np.random.permutation(FLAGS.NumData)+1    
   
    for it in range(0,FLAGS.batch_size):
        number = order[it]
        X = np.array(np.fromfile(datapath + 'image' +  str(int(number)) + '.dat',dtype=np.float32)).reshape(FLAGS.width,FLAGS.height,FLAGS.depth)
        Y = np.array(np.fromfile(datapath + 'label' +  str(int(number)) + '.dat',dtype=np.float32)).reshape(FLAGS.width,FLAGS.height,FLAGS.depth)
        # shuffle
        #shuf = numpy.random.permutation(21)
        #X = X[:,:,shuf]
        #Y = Y[shuf]
        x[it,:,:,:] = X
        y[it,:,:,:] = Y
    return x,y



def evaluate():
    inputs = tf.placeholder(tf.float32, shape=[None, FLAGS.width,FLAGS.height,FLAGS.depth])
    predictions, variables_to_restore, conv2_upsampledTwice, conv7_upsampledTwice = arvo_slim_unet2D_deepSupervision.unet2D_arvo_6mm_test5( inputs, FLAGS.num_class, False, False)     
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    sess = tf.Session()
    sess.run(init_op)
    saver = tf.train.Saver()

      
    
    if not tf.gfile.Exists(FLAGS.checkpointpath + '.meta'):
        raise ValueError("Can't find checkpoint file")
    else:
        print('[INFO    ]\tFound checkpoint file, restoring model.')
        saver.restore(sess, FLAGS.checkpointpath)
    

    # Read images 
    testPath=FLAGS.evaldatapath
    testOutPath = os.path.join(testPath, 'out')
    testDeep2Path = os.path.join(testPath, 'out','deep2')
    full_path = os.path.join(testPath, '*.dat')
    
    if not tf.gfile.Exists(testOutPath):
        print('[INFO    ]\tOut directory does not exist, creating directory: ' + os.path.abspath(testOutPath))
        tf.gfile.MakeDirs(testOutPath)
    if not tf.gfile.Exists(testDeep2Path):
        print('[INFO    ]\tOut directory does not exist, creating directory: ' + os.path.abspath(testDeep2Path))
        tf.gfile.MakeDirs(testDeep2Path)
    # Use glob to read in all the dat images
    


    for file in glob.glob(full_path):
        print(file)
        x=np.zeros((1,FLAGS.width,FLAGS.height,FLAGS.depth));
        
        X = np.array(np.fromfile(file,dtype=np.float32)).reshape(FLAGS.width,FLAGS.height,FLAGS.depth)        
        x[0,:,:,:]=X;
        predicted_values, conv2_upsampledTwice_values=sess.run([predictions,conv2_upsampledTwice], feed_dict={inputs:x.astype(FLAGS.dtyp)})
        
        segmented_image = (predicted_values/tf.sqrt(1+tf.pow(predicted_values,2))+1)/2
        segmented_image_value = segmented_image.eval(session=sess)
        
        outname=os.path.basename(file)
        outpath=os.path.join(testOutPath, outname)
        print  ( outpath )
        fid = open(outpath,"w")
        predicted_values.tofile(fid)
        #file.write(predicted_values)
        fid.close()
        print ( outpath)
        #### out the deep layer
        outname=os.path.basename(file)
        outpath=os.path.join(testDeep2Path,outname)
        print  ( outpath )
        fid = open(outpath,"w")
        conv2_upsampledTwice_values.tofile(fid)
        #file.write(predicted_values)
        fid.close()
        print ( outpath)
    sess.close()
            
            
def gradientdistance(v1,v2):
    a,b = tf.image.image_gradients(v1)
    t1 = tf.concat([a,b],axis = 3)
    c,d = tf.image.image_gradients(v2)
    t2 = tf.concat([c,d],axis = 3)
    c = tf.losses.mean_squared_error(t1,t2)
    return c        

def train ():
    Learningrate = 1e-4
    
    inputs = tf.placeholder(tf.float32, shape=[None, FLAGS.width,FLAGS.height,FLAGS.depth]) 
    targets = tf.placeholder(tf.float32, shape=[None, FLAGS.width,FLAGS.height,FLAGS.depth])
    learning_rate = tf.placeholder(tf.float32, shape=[])
    
    #logits,score = unet.build(inputs,1,True)    
    #loss = tf.losses.mean_squared_error(i1,score)
    predictions, variables_to_restore, conv2_upsampledTwice, conv7_upsampledTwice = arvo_slim_unet2D_deepSupervision.unet2D_arvo_6mm_test5( inputs, FLAGS.num_class, True, False)
    
    
    loss = tf.losses.mean_squared_error(labels=targets, predictions=predictions)
    deep_loss = tf.losses.mean_squared_error(labels=targets, predictions=conv7_upsampledTwice) +  0.25*tf.losses.mean_squared_error(labels=targets, predictions=conv2_upsampledTwice)
    
    gradient_loss = gradientdistance(targets, predictions)
    #6mm_test_5
    #total_loss = 0.9*loss + 0.1*deep_loss
    #6mm_test_6
    #calls unet2D_arvo_6mm_test5
    # test 8 
	#total_loss = 0.9*loss + 0.1*deep_loss + 1000.0*gradient_loss
	# test 1 for noise to noise
    #total_loss = 0.9*loss + 0.1*deep_loss 
    total_loss = 0.4*tf.reduce_mean(tf.abs(targets - predictions))+ 0.1*tf.reduce_mean(tf.abs(targets - conv7_upsampledTwice))+0.5*tf.reduce_mean(tf.abs(targets - conv2_upsampledTwice))
	
	
    #Averaging
    #total_loss = 0.75*loss + 0.25 * deep_loss
	#total_loss = 0.8*loss + 0.1 * deep_loss + 0.1*tf.reduce_mean(tf.abs(targets - predictions))
    #total_loss = loss + 0.25*deep_loss + 0.25*tf.reduce_mean(tf.abs(targets - predictions))
    #noisetonoise
    #6mm 128x128
    #noise to noise
    #first test the choroid was not perfect upping the  conv2 
    #total_loss = 0.6*tf.reduce_mean(tf.abs(targets - predictions))+ 0.2*tf.reduce_mean(tf.abs(targets - conv7_upsampledTwice))+0.2*tf.reduce_mean(tf.abs(targets - conv2_upsampledTwice))
    #print ( 'target: ', targets.shape)
    #total_loss = 0.4*tf.reduce_mean(tf.abs(targets - predictions))+ 0.1*tf.reduce_mean(tf.abs(targets - conv7_upsampledTwice))+0.5*tf.reduce_mean(tf.abs(targets - conv2_upsampledTwice))
    
    ### SEGMENTATION LOSS ####
    #predictions = (predictions/tf.sqrt(1+tf.pow(predictions,2))+1)/2
    #deep_prediction_2 = (conv7_upsampledTwice/tf.sqrt(1+tf.pow(conv7_upsampledTwice,2))+1)/2
    #deep_prediction_7 = (conv7_upsampledTwice/tf.sqrt(1+tf.pow(conv7_upsampledTwice,2))+1)/2
    #loss = tf.losses.hinge_loss(targets, predictions) 
    #deep_loss = 0.75*tf.losses.hinge_loss(targets, deep_prediction_7) + 0.25*tf.losses.hinge_loss(targets, deep_prediction_2)
    #total_loss = 0.75*loss + 0.25*deep_loss
    
    
    tf.summary.scalar('losses/total_loss', total_loss)
    tf.summary.scalar('losses/loss', loss)
    tf.summary.scalar('losses/deep_loss', deep_loss)
    tf.summary.scalar('losses/gradient_loss', gradient_loss)
    
    tf.summary.image('predictions', predictions)
    tf.summary.image("input_image",inputs,  max_outputs=8)
    tf.summary.image("ground_truth",targets,  max_outputs=8)
    tf.summary.image("pred_annotation", predictions, max_outputs=8)
    tf.summary.image("pred_deep_7", conv7_upsampledTwice, max_outputs=8)
    tf.summary.image("pred_deep_2", conv2_upsampledTwice, max_outputs=8)

    
    
    global_step = tf.Variable(0, name = 'global_step', trainable = False)
    train_op = arvo_slim_unet2D_deepSupervision.train(total_loss, learning_rate, global_step)

    
    init_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

    saver = tf.train.Saver(max_to_keep=3,keep_checkpoint_every_n_hours=1)
    session_manager = tf.train.SessionManager(local_init_op = tf.local_variables_initializer())
    sess = session_manager.prepare_session("", init_op = init_op, saver = saver, checkpoint_dir = FLAGS.checkpointdir)
    

    writer = tf.summary.FileWriter(FLAGS.checkpointdir + "/train_logs", sess.graph)
    merged = tf.summary.merge_all()
    totalstart = time.clock();
    for it in range(0,30000,1):
        start = time.clock()
        step = tf.train.global_step(sess, global_step)
    
        print ( 'step/batch : ', str ( int  ( it)))
        print ( 'step/batch : ', step)
        x,y = getdata(FLAGS.traindatapath)
        _,loss_value, summary= sess.run([train_op,total_loss,merged],feed_dict={inputs:x.astype(FLAGS.dtyp),targets:y.astype(FLAGS.dtyp),learning_rate: Learningrate})
        print ('per batch ', time.clock() - start)
        print ('Total  ', time.clock() - totalstart)
        writer.add_summary(summary, it)
        if it%1000==0:
            Learningrate = Learningrate*0.99
        if it%500==0:        
            checkpoint_path = os.path.join(FLAGS.checkpointdir, 'unet.ckpt')
            saver.save(sess,  checkpoint_path, global_step=step)

    writer.close()
    sess.close()
                

# Helper function to read the configs
def ConfigSectionMap(Config, section):
    dict1 = {}
    options = Config.options(section)
    for option in options:
        try:
            dict1[option] = Config.get(section, option)
            if dict1[option] == -1:
                print("skip: %s" % option)
        except:
            print("exception on %s!" % option)
            dict1[option] = None
    return dict1
     
def main(_):
    """
    Download processed dataset if missing & train
    """
    if FLAGS.Mode == "TRAIN":
        if not tf.gfile.Exists(FLAGS.checkpointdir):
            print('[INFO    ]\tCheckpoint directory does not exist, creating directory: ' + os.path.abspath(FLAGS.checkpointdir))
            tf.gfile.MakeDirs(FLAGS.checkpointdir)
        print ('train')
        train()
    else:
        print  ('evaluate')
        evaluate()
        
        

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Config file')
    parser.add_argument('--configPath', help = 'path to a ini file with params.')
    args = parser.parse_args()
    print ( 'path: ', args.configPath )
    if not tf.gfile.Exists(args.configPath):
        print ( 'path to ini is wrong')
    else:
        Config = configparser.ConfigParser()
        Config.read(args.configPath)
        flags.DEFINE_string('Mode',   Config['DEFAULT']['Mode'],'Mode')
        flags.DEFINE_integer('width', Config['DEFAULT']['Width'],'Width')
        flags.DEFINE_integer('height', Config['DEFAULT']['Height'],'Height')
        flags.DEFINE_integer('depth', Config['DEFAULT']['Depth'],'Depth')
        flags.DEFINE_string('dtyp', Config['DEFAULT']['Dtype'],'Dtype')
        flags.DEFINE_integer('num_class', 1,'Num classes')
        if Config['DEFAULT']['Mode'] == 'TRAIN':
            print ( 'trainModel')
            flags.DEFINE_integer('NumData', Config['TRAIN_PARAMS']['NumData'],'NumData')
            flags.DEFINE_integer('batch_size', Config['TRAIN_PARAMS']['BatchSize'],'BatchSize')
            flags.DEFINE_string('traindatapath', Config['TRAIN_PARAMS']['DataPath'],'Datapath')
            flags.DEFINE_string('checkpointdir', Config['TRAIN_PARAMS']['CkptDir'],'ckeckpointDir')
        else:
            print ('evaluateMode')
            flags.DEFINE_integer('NumData', Config['EVAL_PARAMS']['NumData'],'NumData')           
            flags.DEFINE_integer('batch_size', Config['EVAL_PARAMS']['BatchSize'],'BatchSize')
            flags.DEFINE_string( 'evaldatapath', Config['EVAL_PARAMS']['DataPath'],'DataPath')
            flags.DEFINE_string( 'checkpointpath',Config['EVAL_PARAMS']['CkptPath'],'ckeckpointDir')
        tf.app.run()
        
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Train Unet on given tfrecords directory.')
    parser.add_argument('--tfrecords_dir', help = 'Tfrecords directory')
    parser.add_argument('--tfrecords_prefix', help = 'Tfrecords prefix', default = 'training')
    parser.add_argument('--checkpoint_dir', help = 'Checkpoints directory')
    parser.add_argument('--num_classes', help = 'Number of segmentation labels', type = int, default = 2)
    parser.add_argument('--class_weights', help = 'Weight per class for weighted loss. .npy file that contains single array [num_classes]')
    parser.add_argument('--image_size', help = 'Target image size (resize)', type = int, default = 224)
    parser.add_argument('--learning_rate', help = 'Learning rate', type = float, default = 1e-4)
    parser.add_argument('--learning_rate_decay_steps', help = 'Learning rate decay steps', type = int, default = 10000)
    parser.add_argument('--learning_rate_decay_rate', help = 'Learning rate decay rate', type = float, default = 0.9)
    parser.add_argument('--weight_decay_rate', help = 'Weight decay rate', type = float, default = 0.0005)
    parser.add_argument('--batch_size', help = 'Batch size', type = int, default = 8)
    parser.add_argument('--num_epochs', help = 'Number of epochs', type = int, default = 1000)

    #FLAGS, unparsed = parser.parse_known_args()

    tf.app.run()
"""
