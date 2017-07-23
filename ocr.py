import numpy as np
from PIL import Image
import PIL.ImageOps    
import matplotlib.pyplot as plt
import cv2
import pickle
import Word

def next(arr, ind):
	y, x = arr.shape
	while(ind < x):
		if(arr[0][ind] == 1):
			end = ind + 1
			while((end < x) and (arr[0][end] == 1)):
				end += 1
			return ind, end
		ind += 1
	return -1, -1

def greater(a, b):
    momA = cv2.moments(a)        
    (xa,ya) = int(momA['m10']/momA['m00']), int(momA['m01']/momA['m00'])

    momB = cv2.moments(b)        
    (xb,yb) = int(momB['m10']/momB['m00']), int(momB['m01']/momB['m00'])
    if xa>xb:
        return 1

    if xa == xb:
        return 0
    else:
        return -1


def cut(img):
	y, x  = img.shape
	i=0
	j = y

	su = img.sum(axis = 1)

	while(su[i] == 0):
		i += 1
	i -= 1
	if i < 0:
		i = 0
	j -= 1
	while(su[j] == 0):
		j -= 1
	j += 2

	return i, j


def chkprev(prev_x, prev_y, cur_x, cur_y):
	if( ( cur_x - prev_x ) < 10 and (cur_y - prev_y) < 10):
		return True
	return False

img = Image.open("img6.png")
img = img.convert("L")	#convert the image to greyscale

img = PIL.ImageOps.invert(img) #invert the image

#plt.imshow(img, cmap='gray')
#plt.show()

img = np.asarray(img)
#plt.imshow(img, cmap='gray')
#plt.show()

#img[0][0] = 1
img = img.copy()
y, x =  img.shape

img[:,x-1] = 0 #for a single image
original = img.copy()


#print img
#plt.imshow(img, cmap='gray')
#plt.show()


#creating horizontal lines where character is present

for i in range(y):
	for j in range(x):
		if(img[i][j] != 0):
			img[i][:] = 1
			continue


				
#plt.imshow(img, cmap='gray')
#plt.show()

lines=[]
i = 0
while(i < y):
	if(img[i][0] == 1):
		j = i + 1
		while((j < y ) and (img[j][0] == 1)):
			j +=1
		arr = original[i:j]
		lines.append(arr)
		i = j - 1
	i += 1

'''
#printing lines one by one
for i in range(len(lines)):
	tmp = lines[i]
	plt.imshow(tmp, cmap='gray')
	plt.show()
'''

#now individaual words by verticle line


words = []

for i in range(len(lines)):
	cpy = lines[i].copy()
	tmp = lines[i]
	y, x = cpy.shape
	for c in range(x):
		for r in range(y):
			if(cpy[r,c] != 0):
				cpy[:,c] = 1
				continue
	st, en = next(cpy, 0)

	while(True):
		nst, nen = next(cpy, en + 1)
		if(nst == -1):
			words.append(tmp[:,st:en])
			break
		if(nst - en < 8):
			en = nen
		else:
			words.append(tmp[:,st:en])
			st = nst
			en = nen

	

	'''
	tmp = lines[i]
	tmp = tmp[:,st:en]
	plt.imshow(tmp, cmap='gray')
	plt.show()
	'''

chars = []
currentChars = []
for i in range(len(words)):
	tmp = words[i]
	tmp = cv2.bitwise_not(tmp)
	#plt.imshow(tmp, cmap='gray')
	#plt.show()
	img = tmp
	orig = img.copy()
	currentChars = []
	

	# smooth the image to avoid noises
	ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	thresh_color = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)

	#plt.imshow(thresh, cmap='gray')
	#plt.show()

	im_floodfill = thresh.copy()
 
	# Mask used to flood filling.
	# Notice the size needs to be 2 pixels than the image.
	h, w = thresh.shape[:2]
	mask = np.zeros((h+2, w+2), np.uint8)
	 
	# Floodfill from point (0, 0)
	cv2.floodFill(im_floodfill, mask, (0,0), 255);
	 
	# Invert floodfilled image
	im_floodfill_inv = cv2.bitwise_not(im_floodfill)
	 
	# Combine the two images to get the foreground.
	im_out = thresh | im_floodfill_inv
	'''
	# Display images.
	cv2.imshow("Thresholded Image", thresh)
	cv2.imshow("Floodfilled Image", im_floodfill)
	cv2.imshow("Inverted Floodfilled Image", im_floodfill_inv)
	cv2.imshow("Foreground", im_out)
	cv2.waitKey(0)
	'''
	orig_processsed = thresh
	thresh = im_out


	#plt.imshow(thresh, cmap='gray')
	#plt.show()
	# Find the contours


	contours = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)


	#previous location of the contor to avoid duplicates
	prev_x = 0 
	prev_y = 0

	contours = contours[1]
	# For each contour, find the bounding rectangle and draw it
	contours.sort(greater)
	for cnt in contours:
		x,y,w,h = cv2.boundingRect(cnt)



		if( chkprev(prev_x, prev_y, x + w/2.0, y + h/2.0) ):
			continue

		if( w < 10 and h < 10):
			continue

		prev_x = x + w/2.0
		prev_y = y + h/2.0

		tmp = orig[y:y+h, x:x+w]
		whole = orig_processsed[:, x:x+w]
		test = whole[:y -1].sum()
		
		i, j = cut(whole)
		#print i, j
		'''
		if(i > 2):
			i -=2
		j += 2
'''
		tmp = orig[i:j, x:x + w]
		
		'''
		plt.imshow(tmp, cmap='gray')
		plt.show()

		
		plt.imshow(whole, cmap='gray')
		plt.show()
		
		
		
		
		plt.imshow(process, cmap='gray')
		plt.show()
		'''

		#check
		
		chars.append(tmp)
		currentChars.append(tmp)
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
		cv2.rectangle(thresh_color,(x,y),(x+w,y+h),(0,255,0),1)

	Word.getWord(currentChars)	
	print " "
	plt.imshow(img, cmap='gray')
	plt.show()



'''
for i in range(len(words)):
	chars = []

	wrd = words[i]

	y, x = wrd.shape
	plt.imshow(wrd, cmap='gray')
	plt.show()

	st = 0
	en = 1
	sum = 0.0

	flag = False;

	for i in range(x):
		for j in range(y):
			sum += wrd[j][i]/255.0
		en = i
		if((sum < 1.5) & flag):
			
			chars.append(wrd[:,st:en])
			st = en + 1
			flag = False
		elif(sum > 2.5):
			flag = True
		sum = 0
	if(st + 3 < en):
		chars.append(wrd[:,st:en])
	print len(chars)


	for i in range(len(chars)):
		tmp = chars[i]
		plt.imshow(tmp, cmap='gray')
		plt.show()
'''


