import matplotlib.pyplot as plt
import numpy as np
import copy

def redimensionare_imagine(im_p, s):

    row, col, dim = im_p.shape

    row_new_r = row % s
    col_new_r = col % s

    return im[row_new_r: , col_new_r:]

def averaging(im_p, s):

    row, col, dim = im_p.shape

    boxes_h = row // s
    boxes_w = col // s

    im_filtrata = np.zeros((boxes_h, boxes_w, 3))

    for i in range(0, boxes_h):
        for j in range(0, boxes_w):

            im_window = im_p[i*s : (i+1)*s, j*s: (j+1)*s] # sau im_window[i*s : (i+1)*s][j*s: (j+1)*s][] ??

            im_filtrata[i][j][0] = np.average(im_window[ : , : , 0]) # sau im_window[][][0] ??
            im_filtrata[i][j][1] = np.average(im_window[ : , : , 1]) # sau im_window[][][1] ??
            im_filtrata[i][j][2] = np.average(im_window[ : , : , 2]) # sau im_window[][][2] ??

    return im_filtrata


img_path = "C:/Users/Vlad/Desktop/lena.png"

im = plt.imread(img_path)

print('Dimensiune imagine originala:')
print(im.shape)

im_copy = copy.copy(im)

s = 6

im_copy = redimensionare_imagine(im_copy, s)

print('Dimensiune imagine redimensionata:')
print(im_copy.shape)

im_copy = averaging(im_copy, s)

plt.figure()
plt.subplot(1,2,1)
plt.imshow(im)

plt.subplot(1,2,2)
plt.imshow(im_copy)

plt.figure()
plt.imshow(im[:,:,2])

plt.show()
