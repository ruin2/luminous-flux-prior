import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm



def calculate_EF(img,rad=2):
    s = rad**2
    kernel = np.ones((rad,rad), dtype=np.uint8)
    _min = cv2.erode(img, kernel, iterations=1)
    _max = cv2.dilate(img, kernel, iterations=1)
    sigma = cv2.filter2D(img,-1,kernel/(rad**2))
    theta_F = _min/_max
    A =  np.max(sigma,axis=-1)
    return sigma/s , theta_F , A


def calculate_ET(im, sz ,k=1):
    #k是一个参数，主要由F_0 * e^{-x}构成
    sigma,dF,A = calculate_EF(im,sz)
    t = np.sqrt(2*sigma**(k*dF))
    return t,A,sigma,dF


def Recover(im, t,A,EF,dF):
    res = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        res[:, :, ind] = 1*(im[:, :, ind] + A*(t[:, :, ind]-1)) / t[:, :, ind] 
        # res[:, :, ind] += 0.5*(im[:, :, ind] + dF[:, :, ind]*(t[:, :, ind]-1)) / t[:, :, ind]
    return res


if __name__ == '__main__':
    window = 7
    # num = 10# 图片编号
    k = 0.5
    
    import os
    import time
    start = time.time()
    root_dir = 'pot'
    save_dir = 'pot'
    for img in tqdm(os.listdir(root_dir)):
        path = '{}/{}'.format(root_dir,img)
        src = cv2.imread(path)
        src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        I = src.copy()/255
        t,A,EF,dF = calculate_ET(I,window,k)
        J = Recover(I,t,A,EF,dF)
        J[J<0] = 0
        J[J>1] = 1
        # plt.axis('off')
        plt.imshow(J)
        # plt.imsave('{}/dehaze{}'.format(save_dir,img),J)#*np.reshape(A,A.shape+(1,))