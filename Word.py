import numpy as np
from PIL import Image
import PIL.ImageOps    
import matplotlib.pyplot as plt
import cv2
import pickle
import scipy.misc
import getCharacter

def addPadding(ch):
	y, x = ch.shape



	
	
	if( y == x):
		return ch
	flag = 1
	if( y > x):
		diff = y - x
		flag = diff & 1 #odd
		diff = diff/2
		if(diff > 0):			
			ch = np.hstack((np.zeros((y,diff)), ch))
			ch = np.hstack((ch, np.zeros((y, diff))))

		if(flag == 1):
			ch = np.hstack((np.zeros((y, 1)), ch))
	else:
		diff = x - y
		flag = diff & 1 #odd
		diff = diff/2
		if(diff > 0):
			ch = np.vstack((np.zeros((diff, x)), ch))
			ch = np.vstack((ch, np.zeros((diff, x))))

		if(flag == 1):
			ch = np.vstack((np.zeros((1, x)), ch))

	return ch

def size(ch):
	i_width = 24
	i_height = 24
	padding = 4
	ch = scipy.misc.imresize(ch, (i_height, i_width))
	ch = np.vstack((np.zeros((padding, i_width)), ch))
	ch = np.vstack((ch, np.zeros((padding, i_width))))

	ch = np.hstack((np.zeros((i_height + 2*padding , padding)), ch))
	ch = np.hstack((ch, (np.zeros((i_height + 2*padding , padding)))))

	return ch

#with open("test.txt", "rb") as fp:
#	chars = pickle.load(fp)



'''
for ch in chars:
	plt.imshow(ch, cmap='gray')
	plt.show()
'''

def getWord(chars):
	word = []
	for ch in chars:
		ch = ~ch #inverting image
		
		ch = addPadding(ch)
		fit = size(ch)
		
		word.append(getCharacter.getChar(fit))
	print  "".join(word),






