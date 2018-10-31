import cv2
import numpy as np
from matplotlib import pyplot as plt
import image_analysis
import os

from_dir_path = r"D:\SublimeWorkSpace\spine\side_img\SBJ_0019\01383139_9_All2_16_01.png"
img =cv2.imread(from_dir_path, 0)

# normalizedImg = img.copy()
# normalizedImg = cv2.normalize(img,normalizedImg, 0, 255, cv2.NORM_MINMAX)

# 計算直方圖每個 bin 的數值
hist = cv2.calcHist([img], [0], None, [256], [0, 256])

# 畫出直方圖
plt.plot(hist,color = 'r')
plt.xlim([0, 256])
plt.ylim([0, 200000])
plt.show()



# def find_img_files(directory):
#     return (f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png'))

# def find_line_left_right_x(cols_pixels,line_x):
#     left_x=None
#     if np.any(np.where(cols_pixels[line_x-10 :line_x] <60)):
#         left_x = line_x-10 + np.max(np.where(cols_pixels[line_x-10 :line_x] <60))
#     right_x=None 
#     if np.any(np.where(cols_pixels[line_x :line_x+10] <60)):        
#         right_x =line_x +  np.min(np.where(cols_pixels[line_x :line_x+10] <60))

#     if left_x and right_x:
#         return left_x,right_x
#     else:
#         return None,None
# def remove_lines(or_img_array,img_top,img_bottom):
#     #去除右邊的不明直線=  =
#     cols_pixels = []
#     for i in range(int(or_img_array.shape[1] - or_img_array.shape[1]/3),or_img_array.shape[1]):
#         temp_count = cv2.countNonZero(or_img_array[img_top:img_bottom, i:i + 1])        
#         cols_pixels.append(temp_count)

#     cols_pixels = np.array(cols_pixels)
#     lines_x = None
#     if np.any(np.where(cols_pixels > img_bottom-img_top-20)):
#         lines_x = np.where(cols_pixels > img_bottom-img_top-20)

#     if lines_x:
#         for idx in range(len(lines_x[0])):
#             left_x ,right_x = find_line_left_right_x(cols_pixels,lines_x[0][idx])
            
#             if left_x and right_x :
#                 or_img_array[img_top:img_bottom, int(or_img_array.shape[1] - or_img_array.shape[1]/3) + left_x:int(or_img_array.shape[1] - or_img_array.shape[1]/3)+ right_x] =0
#     return or_img_array

# from_dir_path ='D:/SublimeWorkSpace/spine/side_img/SBJ_0015'
# to_dir_path ='./test'


# total_file = find_img_files(from_dir_path)
# first_img_tag = True
# side_set =None

# img_top =0
# img_bottom=None
# img_left=0
# img_right=None

# for file in total_file:
#     img =cv2.imread('{}/{}'.format(from_dir_path, file), 0)
#     or_img_array = np.array(img)
#     #blur = cv2.blur(img,(1,3))
#     ret,thresh1=cv2.threshold(img,0,255,cv2.THRESH_BINARY)
    
#     img_array = np.array(thresh1)

#     if first_img_tag:
#         #去除影像最上及最下黑邊
#         #上邊
#         rows_pixels = []
#         for i in range(int(or_img_array.shape[0]*0.3)):
#             temp_count = cv2.countNonZero(or_img_array[i:i+1, :])
#             rows_pixels.append(temp_count)

#         rows_pixels = np.array(rows_pixels)
        
#         if np.any(np.where(rows_pixels==0)):  

#             img_top = np.max( np.where(rows_pixels==0) )
            
#         #下邊
#         rows_pixels = []
#         for i in range(or_img_array.shape[0] -int(or_img_array.shape[0]*0.3),or_img_array.shape[0]):
#             temp_count = cv2.countNonZero(or_img_array[i:i+1, :])
#             rows_pixels.append(temp_count)

#         rows_pixels = np.array(rows_pixels)

#         img_bottom =or_img_array.shape[0]
#         if np.any(np.where(rows_pixels==0)):
#             img_bottom = or_img_array.shape[0] -int(or_img_array.shape[0]*0.3) + np.min(np.where(rows_pixels==0))
             

#         #左右邊
#         img_left=0
#         img_right=img_array.shape[1]        
        
#         #去除不明直線
#         or_img_array = remove_lines(or_img_array,img_top,img_bottom)

#         side_set = np.array([or_img_array[img_top:img_bottom,img_left:img_right]])
#         first_img_tag=False
#     else:
#         #去除右邊的不明直線=  =
#         or_img_array = remove_lines(or_img_array,img_top,img_bottom)
#         #print(side_set.shape,or_img_array[img_left:img_right,img_top:img_bottom].shape)
#         side_set =np.concatenate((side_set, [or_img_array[img_top:img_bottom,img_left:img_right]]))
#     cv2.imwrite('./test/img/new_{}'.format(file), or_img_array)
# print('側面照維數：',side_set.shape)

# front_set = side_set.swapaxes(0,2)
# print('正面照維數：',front_set.shape)




# first_img_tag = True
# resize_front_set =None
# for idx in range(0,front_set.shape[0]):
#     resize_img = cv2.resize(front_set[idx], (front_set.shape[2]*9,front_set.shape[1]),interpolation=cv2.INTER_LINEAR)            
#     #cv2.imwrite('{}/side_img_{}.png'.format(to_dir_path, idx), resize_img)
#     img_array = np.array(resize_img)
#     if first_img_tag:
#         resize_front_set = np.array([img_array])
#         first_img_tag=False
#     else:
#         resize_front_set =np.concatenate((resize_front_set, [img_array]))


# resize_front_set = resize_front_set.swapaxes(0,2)
# list_side = resize_front_set.tolist()

# one_img =[]
# for i in range(0,len(list_side)):
#     one_img.append([])
#     for j in range(0, len(list_side[0])):
#         one_img[i].append(max(list_side[i][j]))        

# one_img= np.asarray(one_img)
# one_img = one_img.swapaxes(0,1)
# print('維數：',one_img.shape)
# cv2.imwrite('{}/coronal.png'.format(to_dir_path), one_img)