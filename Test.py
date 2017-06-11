from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
import math
import os
import argparse
import sys
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

FLAGS = None
val = [11,30,40,3,1800,3,130,140,0]
def quit():
	c=2
	key = cv2.waitKey(1)
	if key == ord('q'):
		return True
	chars = ['a','s','d','f','g','h','j','k','l']
	for i in range(0,9):
		if(key == ord(chars[i])):
			val[i] += c
			print(chars[i],': ',val[i])
		elif(key == ord('1')+i):
			val[i] -= c
			print(chars[i],': ',val[i])
	return False
	
x = 0
y = 0
idx = 0
path = 'S:\\Box\\Documents\\code\\Python\\'
def show(title,img,size=10,unique=True, arrange=False):
	shape = [x//size for x in img.shape[1::-1]]
	if unique:
		title=str(idx)+" "+title
	cv2.imshow(title, cv2.resize(img, tuple(shape), interpolation = cv2.INTER_AREA))
	if arrange:
		global wx,wy
		cv2.moveWindow(title,wx,wy)
		wx+=shape[0]
		if wx>1400:
			wx=0
			wy+=shape[1]+30

def largest_contour(img,all=False):
	_, contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
	best = 0
	best_idx = 0
	for idx in range(len(contours)):
		a = cv2.contourArea(contours[idx])
		if a > best:
			best = a
			best_idx = idx
	if all:
		return best_idx,best,contours,hierarchy
	else:
		return contours[best_idx]

def extract_boundaries(blob):	
	axis = [[1,-1],[1,1]]
	points = []
	for i in range(4):
		best = (i%2)*999999999
		best_p = [0,0]
		for p in blob:
			p=p[0]
			cur = np.dot(axis[i//2],p)/math.sqrt(2)
			if i%2==0 and cur > best:
				best = cur
				best_p = p
			if i%2==1 and cur < best:
				best = cur
				best_p = p
		points.append(best_p)
	return np.float32([points[3], points[0], points[1], points[2]])

def compute_edges(img):
	gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray,(val[0],val[0]),0)
	#show("blurred",blurred,5)
	edges = cv2.Canny(blurred,val[1],val[2])
	kernel = np.ones((val[3],val[3]),np.uint8)
	return cv2.dilate(edges,kernel,iterations = 1)

def find_center(edges,i,j):
	x = offset+j*cell_size+cell_size//2
	y = offset+i*cell_size+cell_size//2
	
	frame = edges[y-cell_size:y+cell_size,x-cell_size//2:x+cell_size//2]
	res = cv2.matchTemplate(frame,template,cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	y += max_loc[1]-cell_size//2
	
	frame = edges[y-cell_size//2:y+cell_size//2,x-cell_size:x+cell_size]
	res = cv2.matchTemplate(frame,template.transpose(),cv2.TM_CCOEFF_NORMED)
	min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
	x += max_loc[0]-cell_size//2
	return (x,y)
	
def train():
	global knn
	knn = cv2.ml.KNearest_create()
	
	images = []
	labels = []
	for digit in range(1,10):
		folder=path+'output\\'+str(digit)
		for filename in os.listdir(folder):
			img = cv2.imread(os.path.join(folder,filename),0)
			if img is not None:
				images.append(img.ravel())
				labels.append(digit)
	train = np.array(images).astype(np.float32)
	labels = np.array(labels)
	knn.train(train, cv2.ml.ROW_SAMPLE, labels)
	
	global my,sess,mx
	mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
	mx = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))
	my = tf.matmul(mx, W) + b
	y_ = tf.placeholder(tf.float32, [None, 10])
	cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=my))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	for _ in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={mx: batch_xs, y_: batch_ys})

def main(_):
	global offset,cell_size,template,mx,my
	train()
	while True:
		idx = 12
		wx = 0
		wy = 0
		k = [0]*9
		
		while idx<13:
			img = cv2.imread(path+'dataset\\'+str(idx)+'.jpg')
			#lines = open(path+'dataset\\'+str(idx)+'.txt').readlines()
			edges = compute_edges(img)
			#show("edges",edges,7)
			blob = largest_contour(edges)
			offset = 90#divisible by 9
			points = extract_boundaries(blob)		
			points2 = np.float32([[offset,offset],[val[4]-offset,offset],[offset,val[4]-offset],[val[4]-offset,val[4]-offset]])
			m = cv2.getPerspectiveTransform(points,points2)
			perspective = cv2.warpPerspective(img,m,(val[4],val[4]))
			edges = compute_edges(perspective)
			#show("edges",edges,5)
			cell_size = (val[4]-2*offset)//9
			template = cv2.imread(path+'template.png',0)
			w, h = template.shape[1::-1]		
			for i in range(9):
				for j in range(9):
					x,y = find_center(edges,i,j)
					#cv2.circle(perspective,(x,y),10,(255,0,0),10)
					x -= cell_size//2
					y -= cell_size//2
					digit = np.zeros((cell_size,cell_size))
					best_cnt,size,cntrs,h = largest_contour(edges[y+10:y+cell_size-9,x+10:x-9+cell_size].copy(),all=True)
					
					color_digit = perspective[y:y+cell_size,x:x+cell_size]
					hsv = cv2.cvtColor(color_digit, cv2.COLOR_BGR2HSV)
					
					lower_red = np.array([160,60,50])
					upper_red = np.array([180,255,255])
					mask = cv2.inRange(hsv,lower_red,upper_red)
					res = cv2.bitwise_and(color_digit,color_digit,mask=mask)
					res = cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
					result = 0
					if cv2.countNonZero(res) > 140:
						#cv2.circle(perspective,(x+cell_size//2,y+cell_size//2),10,(0,0,255),10)
						M = cv2.moments(cntrs[best_cnt])
						cx = int(M['m10']/M['m00'])
						cy = int(M['m01']/M['m00'])
						#cv2.circle(perspective,(x+10+cx,y+10+cy),10,(255,0,0),10)
						digit = cv2.cvtColor(perspective[y+30+cy-cell_size//2:y-9+cy+cell_size//2,x+30+cx-cell_size//2:x-9+cx+cell_size//2], cv2.COLOR_BGR2GRAY)
						digit = cv2.resize(digit, (28,28))
						_, digit = cv2.threshold(digit,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
						#digit = cv2.normalize(digit.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)
						#show("digits"+str(i)+','+str(j),digit,1,arrange=True)
						'''
						lbl = int(lines[i][j])-1
						k[lbl] += 1
						cv2.imwrite(path + "output\\hand\\"+str(lbl+1)+"i"+str(k[lbl])+".jpg",digit)
						'''
						tensor = []
						tensor.append(digit.ravel())
						tensor = np.array(tensor)
						ans = int(sess.run(tf.argmax(my,1), feed_dict={mx: tensor}))
					elif size>2000:
						#cv2.circle(perspective,(x+cell_size//2,y+cell_size//2),70,(0,255,0),8)
						cv2.drawContours(digit,cntrs,best_cnt,255,3,8,h,3)
						x,y,w,h = cv2.boundingRect(cntrs[best_cnt])
						digit = cv2.resize(digit[y:y+h,x:x+w],(20,20))
						_,res,_,_ = knn.findNearest(np.array([digit.ravel().astype(np.float32)]),5)
						ans = int(res[0])
						#show("digits"+str(i)+','+str(j),digit,1,arrange=True)
						#cv2.imwrite("S:\\Box\\Documents\\code\\Python\\output\\"+lines[i][j]+"\\"+str(idx)+"i_"+str(i)+"_"+str(j)+".jpg",digit)
					else:
						ans='_'
					print(ans,end='')
				print()
			#show("perspective",perspective,5)
			idx+=1
		#if(quit()):
		break

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='/tmp/tensorflow/mnist/input_data', help='Directory for storing input data')
	FLAGS, _ = parser.parse_known_args()
	tf.app.run(main=main, argv=[sys.argv[0]])