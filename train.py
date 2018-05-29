import tensorflow as tf
import numpy as np
from six.moves import cPickle as pickle
import skimage.io
from misc import *
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from model import *
from global_vars import *

def train(train_dataset,train_labels,valid_dataset,valid_labels,input_pl, output_pl, train_step,loss,preds, num_epochs=10,print_every=100):
    TRAIN_SIZE = train_labels.shape[0]
    VALID_SIZE = valid_labels.shape[0]
    iter_per_epoch = TRAIN_SIZE//BATCH_SIZE
    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.latest_checkpoint("checkpoints/")
        if ckpt:
            print("Loading model checkpoint...")
            saver.restore(sess,ckpt)
        else:
            print("No checkpoint found")
            
        for i in range(num_epochs):
            batch = 0
            for j in range(iter_per_epoch):
                x_batch = train_dataset[batch:(batch+BATCH_SIZE)%TRAIN_SIZE]
                y_batch = train_labels[batch:(batch+BATCH_SIZE)%TRAIN_SIZE]
                feed_dict = {input_pl:x_batch,output_pl:y_batch}
                train_loss,_ = sess.run([loss,train_step],feed_dict)
                if (j+1)%print_every==0:
                    print("Epoch ",i," Iteration ",j,"Loss: ",train_loss)
                    
                batch += BATCH_SIZE
            saver.save(sess,'checkpoints/logo')
            #Testing on validation set
            iters = VALID_SIZE//BATCH_SIZE
            batch = 0
            avg_accuracy = 0
            for j in range(iters):
                val_x_batch = valid_dataset[batch:(batch+BATCH_SIZE)%VALID_SIZE]
                val_y_batch = valid_labels[batch:(batch+BATCH_SIZE)%VALID_SIZE]
                feed_dict = {input_pl: val_x_batch, output_pl: val_y_batch}
                val_preds = sess.run(preds,feed_dict)
                val_preds = np.argmax(val_preds,axis = 1)
                accuracy = np.mean(val_preds==val_y_batch)
                avg_accuracy += accuracy
            avg_accuracy = avg_accuracy/iters;
            print(avg_accuracy)


def main():
	#Reading dataset into memory
	with open('logo.pickle','rb') as pk:
	    pick = pickle.load(pk)
	    train_dataset = pick['train_dataset']
	    train_labels = pick['train_labels']
	    valid_dataset = pick['valid_dataset']
	    valid_labels = pick['valid_labels']
	    test_dataset = pick['test_dataset']
	    test_labels = pick['test_labels']
	    del pick
	    print('Training set', train_dataset.shape, train_labels.shape)
	    print('Valid set', valid_dataset.shape, valid_labels.shape)
	    print('Test set', test_dataset.shape, test_labels.shape)



	input_pl = tf.placeholder(tf.float32, shape = [None,32,64,3])
	output_pl = tf.placeholder(tf.int32,shape = [None])

	preds,loss = forward_pass(input_pl,output_pl)
	solver = get_solver(0.0001)
	train_step =solver.minimize(loss)

	train(train_dataset,train_labels,valid_dataset,valid_labels,input_pl,output_pl,train_step,loss,preds)

if __name__ == '__main__':
    main()