from PIL import Image
import os
import _pickle as cPickle
import sys
import pickle
import numpy as np
import cv2
import scipy.misc
import matplotlib.pyplot as plt



def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              crop=True, grayscale=False):
  image = imread(image_path, grayscale)
  return transform(image, input_height, input_width, resize_height, resize_width, crop)



def imread(path, grayscale = False):
    if (grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

    
    
def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, crop=True):
  if crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)




def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])




def unpickle(file):
    fp = open(file, 'rb')
    if sys.version_info.major == 2:
        data = pickle.load(fp)
    elif sys.version_info.major == 3:
        data = pickle.load(fp, encoding='latin-1')
    fp.close()

    return data


path1="/Users/Downloads/DCGAN-tensorflow/data/celebA"
images = os.listdir(path1) # Set directory path here
namedic = {int(name.split(".")[0]):name for name in images}
lobby_name_order=[]
for lst in sorted(namedic.items()):
    lobby_name_order.append(lst[1])  # Set images in numerical order



# In[76]: crop images for ideal size

input_height=108  # height and wide size to zoom images
input_wide=108

img_batch = [get_image(path1+'/'+batch, input_height, input_wide, resize_height=64, resize_width=64,
              crop=True,grayscale=False) for batch in lobby_name_order]


# In[77]: # contrast


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

# 変換

high_cont_img = []
for i in img_batch:
    high_cont_img.append(cv2.LUT(i, LUT_HC))
low_cont_img = []
for i in img_batch:
    low_cont_img.append(cv2.LUT(i, LUT_LC))


# In[78]: ganmma, ganmma2


gamma = 1.5
gamma1 = 0.75
look_up_table = np.zeros((256, 1), dtype = 'uint8')

for i in range(256):
    look_up_table[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma)
ganma=[]
for i in img_batch:
    ganma.append(cv2.LUT(i, look_up_table))


look_up_table2 = np.zeros((256, 1), dtype = 'uint8')

for i in range(256):
    look_up_table2[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma1)
ganma1=[]
for i in img_batch:
    ganma1.append(cv2.LUT(i, look_up_table2))


# In[79]:


average_square = (2,2)
blur_img=[]
for i in img_batch:
    blur_img.append(cv2.blur(i, average_square))


# In[80]: sharp


# シャープの度合い
k = 0.3
# 粉雪（シャープ化）
shape_operator = np.array([[0, -k, 0],[-k, 1 + 4 * k, -k],[0,-k, 0]])
 
img_tmp=[]
    # 作成したオペレータを基にシャープ化
for i in img_batch:
    img_tmp.append(cv2.convertScaleAbs(cv2.filter2D(i, -1, shape_operator))) 



# In[81]: sharp2


# シャープの度合い
k = 0.1
# 粉雪（シャープ化）
shape_operator = np.array([[0, -k, 0],[-k, 1 + 4 * k, -k],[0,-k, 0]])
 
img_tmp2=[]
    # 作成したオペレータを基にシャープ化
for i in img_batch:
    img_tmp2.append(cv2.convertScaleAbs(cv2.filter2D(i, -1, shape_operator))) 



# In[82]: ganmma2, ganmma3


gamma2 = 1.4
gamma3 = 0.8
look_up_table3 = np.zeros((256, 1), dtype = 'uint8')

for i in range(256):
    look_up_table3[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma2)
ganma2=[]
for i in img_batch:
    ganma2.append(cv2.LUT(i, look_up_table3))


look_up_table4 = np.zeros((256, 1), dtype = 'uint8')

for i in range(256):
    look_up_table4[i][0] = 255 * pow(float(i) / 255, 1.0 / gamma3)
ganma3=[]
for i in img_batch:
    ganma3.append(cv2.LUT(i, look_up_table4))

    
# In[82]: # shifting and rotation (change value to use many times)


SHAPE=imgs[0].shape[1]
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


afn = [cv2.warpAffine(i, affine_matrix, size, flags=cv2.INTER_LINEAR) for i in imgs]

# In[83]: connection of all parts


seen5000 = np.concatenate((img_batch, low_cont_img, high_cont_img,ganma, ganma1,
                           blur_img, img_tmp, img_tmp2,ganma2, ganma3, afn))
seen5000.shape


# In[84]: flip right and left which its images become twice


flip_img=[]
for i in seen5000:
    flip_img.append(cv2.flip(i, 1))
seen10000 = np.r_[seen5000, flip_img]
plt.imshow(seen10000[1611], 'gray')#32×32
plt.show()
seen10000.shape


# In[86]: save to pickle file


with open('ロビーtrain128.pickle', mode='wb') as f:
    pickle.dump(seen10000, f)
