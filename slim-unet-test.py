# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 01:06:40 2018

@author: aduser
"""



"""
  # Create model and obtain the predictions:
images, labels = LoadData(...)
predictions = MyModel(images)
# Choose the metrics to compute:
names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({"accuracy": slim.metrics.accuracy(predictions, labels),
  "mse": slim.metrics.mean_squared_error(predictions, labels),})


initial_op = tf.group(
  tf.global_variables_initializer(),
  tf.local_variables_initializer())
with tf.Session() as sess:
    metric_values = slim.evaluation(
            sess,
            num_evals=1,
            initial_op=initial_op,
            eval_op=names_to_updates.values(),
            final_op=name_to_values.values())
"""


import tensorflow as tf
import cv2
from modelBase import unet2d
import numpy as np
import glob
from PIL import Image

#sess=tf.Session()    

with tf.Graph().as_default() as graph: # Set default graph as graph
    with tf.Session() as sess:
        #First let's load meta graph and restore weights
        saver = tf.train.import_meta_graph('G:\\Arindam\\Projects\\ImageAveragingUsingSLIM\\Checkpoint\\gteavsrdsgtr\\model.ckpt-50000.meta')
        saver.restore(sess,tf.train.latest_checkpoint('G:\\Arindam\\Projects\\ImageAveragingUsingSLIM\\Checkpoint\\gteavsrdsgtr\\'))
       
        #saver = tf.train.import_meta_graph('D:\\ArindamData\\Code\\U-Net\\Unet\\Datasets\\checkpoints\\sCXz9mG46KExHKTG\\unet.ckpt-138800.meta')
        #saver.restore(sess,tf.train.latest_checkpoint('D:\\ArindamData\\Code\\U-Net\\Unet\\Datasets\\checkpoints\\sCXz9mG46KExHKTG\\'))
        
        
        
        graph = tf.get_default_graph (); 
        
        test = tf.get_default_graph().get_tensor_by_name("batch_processing/shuffle_batch/random_shuffle_queue:0")
        print ( test )
        
        print("Load Image...")
        
        X_data = []
        
        files = glob.glob ("G:\\Arindam\\Projects\\ImageAveragingUsingSLIM\\Tests\\dim128SetA\\*.png")
        for myFile in files:
            print(myFile)
            image = cv2.imread (myFile)
            X_data.append (image)
        
        print('X_data shape:', np.array(X_data).shape)
        
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        
        #saver = tf.train.Saver()
        
        #predictions, variables_to_restore=unet2d.unet2d_AB( np.array(X_data), 3, False, False, 5)
        #Session_out = sess.run( predictions )#X_data})
        f = open("demofile.txt", "w")
        
        for op in graph.get_operations():
            #print ("Operation Name :",op.name)         # Operation name
            #print ("Tensor Stats :",str(op.values()))     # Tensor name
            f.write(str(op.name))
            f.write("\n")
            for l in op.values():
               f.write ( str(l))
               f.write("\n")
        f.close () 
        
        l_input = graph.get_tensor_by_name('batch_processing/shuffle_batch:0') # Input Tensor
        l_output = graph.get_tensor_by_name('unet/output/BiasAdd:0') # Output Tensor
        
        Session_out = sess.run(l_output,  feed_dict={l_input : (np.array(X_data,dtype=np.float))})#X_data})
        print ( Session_out.shape)
        np.savetxt('dataout.csv',Session_out[0,:,:,0],delimiter=',',fmt='%f')  
        img = Session_out[1,:,:,:]
        timg = tf.image.encode_png(img)
        dataimg = sess.run (timg)
        f = open("G:\\Arindam\\Projects\\ImageAveragingUsingSLIM\\Tests\\out.png", "wb+")
        f.write(dataimg)
        f.close()

        
        
