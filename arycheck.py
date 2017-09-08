# -*- coding:utf-8 -*-
import numpy as np
import cv2
size = 256
#b = np.random.randint(0, 255, (size*size))
b = np.arange(65536)
g = np.random.randint(0, 255, (size*size))
r = np.random.randint(0, 255, (size*size))

print b
print g
print r

#b = np.reshape(b, (size * size,1))
#g = np.reshape(g, (size * size,1))
#r = np.reshape(r, (size * size,1))

#b = np.reshape(b, (size , size))
#g = np.reshape(g, (size , size))
#r = np.reshape(r, (size , size))

#b = np.array(b, dtype='float32')
#print b
#
#list = np.array([], dtype='uint32')
#list = np.append(list, b, axis=0)
#list = np.append(list, g, axis=0)
#list = np.append(list, r, axis=0)
#list = np.reshape(list, (256, 256, 3))
#list = np.array(list, dtype='uint32')
#list = np.clip(list, 0, 255)
#print list

a = np.c_[b,g,r]


out = [a[x:x + size] for x in xrange(0, len(a), size)]
out = np.array(out)

print out

#out = out.astype(np.uint8)
#out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
#out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
#cv2.imwrite('ary.png', out)
list1=[[1,2], [-4,4], [5,6], [7,8]]
list=[1, 2, 3, 4]
#print np.clip(np.array(list1), 0, 5)
list2=[[2,2], [3,4], [5,6], [7,8]]
list3=[[3,2], [3,4], [5,6], [7,8]]
#for i, j, k in zip(list1, list2, list3):
#    print i, j, k


src = cv2.imread('ary.png', 1)
BGR = cv2.split(src)
BGR = np.array(BGR)
#print BGR[0][0]
#print BGR[1][0]
#print BGR[2][0]
#print BGR[2][0].shape
#print np.ravel(BGR[2][::][::]).shape
