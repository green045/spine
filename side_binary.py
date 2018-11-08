import cv2
import numpy as np
from matplotlib import pyplot as plt
import image_analysis
import os
#import thinning


def high_pass(img):#, filestr):
    high_pass_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, high_pass_filter)
    # cv2.imwrite(os.path.join(output_path, filestr+'_highpass.png'), img)
    return img

def find_binary_value(origin_img):  


    # 測試先做高通濾波後再二值化
    origin_img = cv2.add(origin_img, high_pass(origin_img)) #high_pass(origin_img) #cv2.add(origin_img, high_pass(origin_img))


    y, x = np.histogram(origin_img, bins=np.arange(256))

    for index in range(len(y))[::-1]:
        for index2 in range(index + 1):
            y[index] += y[index2]
    # print(filestr, np.median(origin_img), origin_img.mean())
    binary_value = y[-1] // 10 * 9
    for index in range(len(y)):
        if y[-index-1] < binary_value <= y[-index]:
            binary_value = 255-index
            break

    ret, origin_img = cv2.threshold(origin_img, binary_value, 255, cv2.THRESH_BINARY)

    # 刪除小的獨立點(小於某個範圍的contours就變黑)
    image, contours, hierarchy = cv2.findContours(origin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    zero_img = np.zeros_like(origin_img)  # Create mask where white is what we want, black otherwise
    for contour in contours:
        contour_rows = contour[:, 0, 0]
        contour_cols = contour[:, 0, 1]
        if max(contour_rows) - min(contour_rows) <= 20 and max(contour_cols) - min(contour_cols) <= 20:
            cv2.drawContours(zero_img, [contour], -1, 255, -1)
    zero_img = cv2.bitwise_not(zero_img)
    origin_img = cv2.bitwise_and(origin_img, zero_img)

    color_img = cv2.cvtColor(origin_img, cv2.COLOR_GRAY2BGR)   
  
    #thinning
    #thinning_img = thinning_spine(origin_img)
    #cv2.imwrite(os.path.join(output_path, filestr + '_thinning.png'), thinning_img)
    return color_img
    


data_path = './side_img'
output_path = './bin_img'

total_dir = os.listdir(data_path)

for each_dir in total_dir:
    print(each_dir)
    each_dir_path = data_path+'/'+ each_dir
    save_dir_path =output_path+'/'+ each_dir
    if not os.path.isdir(save_dir_path):
        os.mkdir(save_dir_path)


    total_file = os.listdir(each_dir_path)
    total_file = filter(lambda x: x.endswith('png'), total_file) #只抓png檔

    for file in total_file: #跑出二值化的值
        
        #filestr = file.split('.')[0]   #file[:-4]

        origin_img = cv2.imread(os.path.join(each_dir_path, file), 0)
        # Test binary value
        color_img = find_binary_value(origin_img)


        cv2.imwrite(os.path.join(save_dir_path, 'bin_'+file), color_img)