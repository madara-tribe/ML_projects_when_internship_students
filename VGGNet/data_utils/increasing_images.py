from PIL import Image
import os
import _pickle as cPickle
import sys
import pickle
import numpy as np
import cv2
import scipy.misc
import matplotlib.pyplot as plt

# In[77]: # contrast

def change_contrast(img_batch):
    min_table = 10
    max_table = 205
    diff_table = max_table - min_table

    LUT_HC = np.arange(256, dtype = 'uint8' )
    LUT_LC = np.arange(256, dtype = 'uint8' )

    # ハイコントラストLUT作成
    for i in range(0, min_table):
        LUT_HC[i] = 0
    for i in range(min_table, max_table):
        LUT_HC[i] = 255 * (i - min_table) / diff_table
    for i in range(max_table, 255):
        LUT_HC[i] = 255

    # ローコントラストLUT作成
    for i in range(256):
        LUT_LC[i] = min_table + i * (diff_table) / 255

    high_cont_img = []
    for i in img_batch:
        high_cont_img.append(cv2.LUT(i, LUT_HC))
    low_cont_img = []
    for i in img_batch:
        low_cont_img.append(cv2.LUT(i, LUT_LC))

    return high_cont_img, low_cont_img


def chage_gamma_contrast(img_batch, gamma = 1.5):
    look_up_table = np.zeros((256, 1), dtype = 'uint8')

    for i in range(256):
        look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
    ganma_img = []
    for i in img_batch:
        ganma_img.append(cv2.LUT(i, look_up_table))
    return ganma_img


# In[79]:

def average_contrast(img_batch):
    average_square = (2,2)
    averaged_img=[]
    for i in img_batch:
        averaged_img.append(cv2.blur(i, average_square))
    return averaged_img


# In[80]: sharp

def sharp(img_batch):
    # シャープの度合い
    k = 0.3
    # 粉雪（シャープ化）
    shape_operator = np.array([[0, -k, 0],[-k, 1 + 4 * k, -k],[0,-k, 0]])

    img_sharp=[]
        # 作成したオペレータを基にシャープ化
    for i in img_batch:
        img_sharp.append(cv2.convertScaleAbs(cv2.filter2D(i, -1, shape_operator)))
    return img_sharp


def sift_angle(img_batch):
    SHAPE=img_batch[0].shape[1]
    # 画像サイズの取得(横, 縦)
    size = tuple(np.array([SHAPE, SHAPE]))
    print(size)
    # 回転させたい角度
    rad=np.pi/20
    # x軸方向に平行移動させたい距離
    move_x = 20
    # y軸方向に平行移動させたい距離
    move_y = SHAPE* -0.1

    matrix = [[np.cos(rad), -1 * np.sin(rad), move_x],
                   [np.sin(rad), np.cos(rad), move_y]]

    affine_matrix = np.float32(matrix)


    chage_angle = [cv2.warpAffine(i, affine_matrix, size, flags=cv2.INTER_LINEAR) for i in img_batch]
    return chage_angle


def flip(img_batch):
    flip_img=[cv2.flip(i, 1) for i in img_batch]
    return flip_img

    seen10000 = np.r_[seen5000, flip_img]

def increase_image(img_batch):
    low_cont_img, high_cont_img = change_contrast(img_batch)
    gamma_img = chage_gamma_contrast(img_batch)
    average_img = average_contrast(img_batch)
    sharp_img = sharp(img_batch)
    sift_angle_img = sift_angle(img_batch)
    conbined_img = np.concatenate((img_batch, low_cont_img, high_cont_img, gamma_img,
                           average_img, sharp_img, sift_angle_img))
    flip_combined = flip(conbined_img)
    increased_images = np.r_[conbined_img, flip_combined]
    return increased_images


# save to pickle file
def increase():
    path1="/Users/Downloads/DCGAN-tensorflow/data/celebA"
    images = os.listdir(path1) # Set directory path here
    namedic = {int(name.split(".")[0]):name for name in images}
    lobby_name_order=[]
    for lst in sorted(namedic.items()):
        lobby_name_order.append(lst[1])  # Set images in numerical order
    # In[76]: crop images for ideal size
    input_height = 108  # height and wide size to zoom images
    input_wide = 108
    img_batch = [get_image(path1+'/'+batch, input_height, input_wide, resize_height=64, resize_width=64,
                  crop=True,grayscale=False) for batch in lobby_name_order]
    with open('ロビーtrain128.pickle', mode='wb') as f:
        pickle.dump(seen10000, f)

        
if __name__ == '__main__':
    increase()
