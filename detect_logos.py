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
import os


def logo_recognition(sess, img, obj_proposal,input_pl,preds):
	recog_results = {}
	recog_results['region'] = obj_proposal
	img = scale(img)

	pred = sess.run(
		[preds], feed_dict={input_pl: img})
	recog_results['pred_class'] = CLASS_NAME[np.argmax(pred)]
	recog_results['prob'] = np.max(pred)
	return recog_results

def detect_logos(images,input_pl,preds,sess):
	for image in images:
		img = skimage.io.imread(image)
		img = img[:,:,0:3]
		print(img.shape)
		proposals = list(get_region_proposals(img))
		preds_prob = []
		preds_label = []
		results =[]
		for region in proposals:
			x,y,w,h = region
			crop_img = img[y:y+h,x:x+w]
			crop_img = imresize(crop_img,(32,64,3),interp='bicubic')
			crop_img = crop_img[None,:]
			results.append(
				logo_recognition(sess, crop_img, region,input_pl,preds))

		del_idx = []
		for i, result in enumerate(results):
			if result['pred_class'] == CLASS_NAME[-1]:
				del_idx.append(i)
		results = np.delete(results, del_idx)
		nms_results = nms(list(results), 0.8, 0.1)
		fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
		ax.imshow(img)

		for result in nms_results:
			(x, y, w, h) = result['region']
			ax.text(
				x,
				y,
				result['pred_class'],
				fontsize=13,
				bbox=dict(facecolor='blue', alpha=0.7))
			rect = mpatches.Rectangle(
				(x, y), w, h, fill=False, edgecolor='blue', linewidth=5)
			ax.add_patch(rect)
		plt.savefig('results/' + image[len('test_images/'):], bbox_inches='tight')
		plt.close(fig)



def main():
	images = os.listdir("test_images")
	images = ['test_images/'+image for image in images]
	input_pl = tf.placeholder(tf.float32, shape = [None,32,64,3])
	output_pl = tf.placeholder(tf.int32,shape = [None])

	preds,loss = forward_pass(input_pl,output_pl)

	with tf.Session() as sess:
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		ckpt = tf.train.latest_checkpoint("checkpoints/")
		if ckpt:
			print("Loading model checkpoint...")
			saver.restore(sess,ckpt)
		else:
			print("No checkpoint found")
		detect_logos(images,input_pl,preds,sess)

if __name__ == '__main__':
	main()