#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import sys
import random
import math

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import skimage.io
from skimage import img_as_float
import train_spine
import utils
import model as modellib
import visualize
import cv2
import image_analysis
import numpy as np

def find_rids_y_val(rect_img,opposite_img):
    #分為左邊肋骨及右邊肋骨，計算不同    

    #從所有輪廓鍾取出單純肋骨的輪廓
    rid_contour=[]
    _, contours, hierarchy = cv2.findContours(rect_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not (contours is None or len(contours) == 0):
        #print(len(contours))

        area_bound = rect_img.shape[0]*rect_img.shape[1]*0.005
        for each_contour in contours:
            area = cv2.contourArea(each_contour)
            if area<area_bound:
                continue

            rect = cv2.minAreaRect(each_contour)   
            box = cv2.boxPoints(rect)
            box = np.int0(box)         
            W = rect[1][0]
            H = rect[1][1]
            angle = rect[2]
            
            #框格mask
            rect_mask = np.zeros(rect_img.shape, np.uint8)
            cv2.drawContours(rect_mask, [box], 0, 255, -1)
            #rect_mask = cv2.cvtColor(rect_mask, cv2.COLOR_BGR2GRAY)
            _, rect_mask = cv2.threshold(rect_mask, 127, 1, cv2.THRESH_BINARY)

            min_w_size = rect_img.shape[1]
            if opposite_img.shape[1] <rect_img.shape[1]:
                min_w_size = opposite_img.shape[1]
           
            temp_oppo = opposite_img[:,:min_w_size] * rect_mask[:, :min_w_size]
            opp_count = cv2.countNonZero(temp_oppo)

            opp_count = opp_count/(W*H)
            ori_count = area / (W*H)
            if (opp_count < 0.1):
                continue 
            
            if H/W >2 and -70<angle and angle<-20:                  
                rid_contour.append(each_contour)
            
            
        #rect_img = cv2.cvtColor(rect_img,cv2.COLOR_GRAY2RGB)
            
        #計算每根肋骨的頂點，並找出所有肋骨中最低的頂點y值
        res_y = 0
        res_idx =0
        for idx,contour in enumerate(rid_contour):
            extTop = tuple(contour[contour[:, :, 1].argmin()][0])           
            if extTop[1] > res_y:
                res_y = extTop[1]
                res_idx = idx

        if(rid_contour ==[]):
            return None,res_y,res_idx
        return rid_contour,res_y,res_idx

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou


def borns_modify(file_name,mid_x,csv_path =""):
    #csv_writer = open(csv_path, 'a')
    #csv_writer.write("X" + "," + "X2-X1" + "," + "mean" + "," + "std"+ '\n')
    csv_writer.write(file_name+ '\n')
    subX =[]
    csv_writer.write("X: "+ ",")
    for idx,x in enumerate(mid_x):
        csv_writer.write(str(x)+ ",")
        if idx>0:
            subX.append(mid_x[idx] - mid_x[idx-1])

    csv_writer.write('\n'+"subX: "+ ",")
    for idx,x in enumerate(subX):
        csv_writer.write(str(x)+ ",")

    subX = np.array(subX)
    abs_subX = np.abs(subX)
    std = np.std(abs_subX)
    mean = np.mean(abs_subX)
    csv_writer.write('\n'+"mean: "+","+ str(mean)+","+"std: "+","+ str(std)+'\n')
    #csv_writer.close()

if __name__ == '__main__':
       
    
    front_dir_path ='./front_img_half'
    # data_path = args.dataset
    data_path = './bin_img'
    # img_save_dir = args.save
    output_path = './result_spine'
    # Root directory of the project

    csv_path ="./borns_result.csv"
    csv_writer = open(csv_path, 'w')
    

    ROOT_DIR = os.getcwd()
    # Directory to save logs and trained model
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    
    
    
    # Configurations
    
    class InferenceConfig(train_spine.EngineConfig):
        # Set batch size to 1 since we'll be running inference on
        # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    config = InferenceConfig()
    config.display()
    # Create model object in inference mode.
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)
    
    
    model_path = './model/mask_rcnn_spine.h5'
    print(model_path)
    # Load weights trained on MS-COCO
    model.load_weights(model_path, by_name=True)
    
    # COCO Class names
    # Index of the class in the list is its ID. For example, to get ID of
    # the teddy bear class, use: class_names.index('teddy bear')
    class_names = ['background', 'spine']
    
    


    
    total_dir = os.listdir(front_dir_path)
    total_dir = filter(lambda x: x.endswith('png'), total_dir)


    for each_front_img in sorted(total_dir):
        gray_image=cv2.imread('{}/{}'.format(front_dir_path, each_front_img), 0)

        ret,thresh1=cv2.threshold(gray_image,254,255,cv2.THRESH_BINARY)


        #去雜訊
        kernel = np.ones((3,3),np.uint8)
        opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

        _, contours, hierarchy  = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = image_analysis.max_contour(contours)
        spine_contour = np.zeros(gray_image.shape, np.uint8)
        cv2.drawContours(spine_contour, [max_contour], 0, (255, 255, 255), -1)
        

        #找影像最上黑邊
        #上邊
        rows_pixels = []
        for i in range(int(gray_image.shape[0]*0.3)):
            temp_count = cv2.countNonZero(gray_image[i:i+1, :])
            rows_pixels.append(temp_count)

        rows_pixels = np.array(rows_pixels)
        img_top = 0
        if np.any(np.where(rows_pixels==0)):  

            img_top = np.max( np.where(rows_pixels==0) ) 

        #img_btn = spine_contour.shape[0]
        #下邊
        rows_pixels = []
        for i in range(gray_image.shape[0] -int(gray_image.shape[0]*0.3),gray_image.shape[0]):
            temp_count = cv2.countNonZero(gray_image[i:i+1, :])
            rows_pixels.append(temp_count)

        rows_pixels = np.array(rows_pixels)

        img_btn =gray_image.shape[0]
        if np.any(np.where(rows_pixels==0)):
            img_btn = gray_image.shape[0] -int(gray_image.shape[0]*0.3) + np.min(np.where(rows_pixels==0))


        #統計每行pixel數，找出主骨幹左右邊界，為了將影像分為左右邊肋骨
        cols_pixels = []
        for i in range(spine_contour.shape[1]):
            temp_count = cv2.countNonZero(spine_contour[img_top:img_btn, i:i + 1])
            cols_pixels.append(temp_count)

        cols_pixels = np.array(cols_pixels)

        cols_pixels[np.where(cols_pixels[:]<spine_contour.shape[0]*2/3)] = 0

        #左邊肋骨的邊界與右邊肋骨的邊界   
    
        if ~np.any(np.where(cols_pixels>0)):
            continue
        left_edge = np.min(np.where(cols_pixels>0))
        right_edge = np.max(np.where(cols_pixels>0))


        #統計每列pixel數，為了找出肋骨的下邊界
        rows_pixels = []
        for i in range(spine_contour.shape[0]):
            temp_count = cv2.countNonZero(spine_contour[i:i + 1, :])
            rows_pixels.append(temp_count)    

        rows_pixels = np.array(rows_pixels)

        #肋骨下邊界
        bottom_edge = spine_contour.shape[0]
        if np.any(np.where(rows_pixels[int(spine_contour.shape[0]/2):]>spine_contour.shape[1]*0.9)):
            bottom_edge = int(spine_contour.shape[0]/2) + np.min( np.where(rows_pixels[int(spine_contour.shape[0]/2):]>spine_contour.shape[1]*0.9))
        top_edge = img_top#int(bottom_edge - spine_contour.shape[0]/2)


        #肋骨們
        # left_edge -= 5
        # right_edge += 5
        ribs_img = opening.copy() #- main_spine_contour 
        ribs_img[top_edge  :bottom_edge ,left_edge: right_edge] =0
        kernel = np.ones((1,3),np.uint8)
        #ribs_img = cv2.morphologyEx(ribs_img, cv2.MORPH_OPEN, kernel)
        main_spine = cv2.erode(opening,kernel,iterations = 1)
        main_spine = cv2.dilate(main_spine,kernel,iterations = 1)
        if cv2.countNonZero(ribs_img) <50:
            continue

        #取出左右肋骨
        left_img =ribs_img[ top_edge  :bottom_edge ,0 : left_edge].copy()
        right_img =ribs_img[ top_edge :bottom_edge ,right_edge : ribs_img.shape[1]].copy()


        #水平翻轉,rect_img肋骨靠左對齊,如右肋骨圖
        left_img = cv2.flip(left_img,1)


        #左邊肋骨
        if cv2.countNonZero(left_img) <50:
            continue
        left_rid_contour,left_rids_y,left_contour_idx =find_rids_y_val(left_img,right_img)      


        #右邊肋骨
        
        if cv2.countNonZero(right_img) <50:
            continue
        right_rid_contour,right_rids_y,right_contour_idx =find_rids_y_val(right_img,left_img)
      



        #畫出肋骨外接矩形
        '''
        #圈出肋骨外接矩形
        draw_img = cv2.cvtColor(ribs_img, cv2.COLOR_GRAY2BGR)#gray_image * ribs_img[:, :] 
        #draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2RGB)
        draw_left_img =draw_img[ top_edge  :bottom_edge ,0 : left_edge]
        draw_left_img = cv2.flip(draw_left_img,1)
        draw_right_img =draw_img[ top_edge  :bottom_edge ,right_edge : ribs_img.shape[1]]


        #左邊肋骨
        if left_rid_contour!=None:
            for idx , contour in enumerate(left_rid_contour):
                #圈出輪廓的外接矩形  
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)  
                #cv2.drawContours(draw_right_img, [box], 0, (255, 0 , 255), 2)
                if(left_contour_idx == idx) :
                    cv2.drawContours(draw_left_img, [box], 0, (255, 0 , 0), 2)
                else:
                    cv2.drawContours(draw_left_img, [box], 0, (0, 255, 0), 2)

        #右邊肋骨
        if right_rid_contour!=None:
            for idx , contour in enumerate(right_rid_contour):
                #圈出輪廓的外接矩形  
                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)  
                #cv2.drawContours(draw_left_img, [box], 0, (255,0 , 255), 2)
                if(right_contour_idx == idx):
                    cv2.drawContours(draw_right_img, [box], 0, (255, 0 , 0), 2)
                else:
                    cv2.drawContours(draw_right_img, [box], 0, (0, 255, 0), 2)
        #把左側肋骨轉回來
        draw_left_img = cv2.flip(draw_left_img,1)
        '''

                    
        real_y=None
        if left_rids_y ==0 and right_rids_y ==0:
            real_y=None
        elif left_rids_y ==0:
            real_y = top_edge + (int)(right_rids_y)
        elif right_rids_y ==0:
            real_y = top_edge + (int)(left_rids_y)
        else:
            real_y = (top_edge + left_rids_y) if left_rids_y>right_rids_y else top_edge +right_rids_y
            
        
        if real_y==None:            
            continue





        #讀側面影像檔並寫檔
        each_dir = each_front_img[:8]
        print(each_dir)
        each_dir_path = data_path+'/'+ each_dir
        save_dir_path =output_path+'/'+ each_dir
        if not os.path.isdir(save_dir_path):
            os.mkdir(save_dir_path)


        total_file = os.listdir(each_dir_path)
        total_file = filter(lambda x: x.endswith('png'), total_file) #只抓png檔


        spine_box_list = []
        first_img = True
        for file_name in sorted(total_file):
            image = skimage.io.imread(os.path.join(each_dir_path, file_name))

            '''
            cv2_img = cv2.imread(os.path.join(each_dir_path, file_name))
            cv2_rect =cv2_img.copy()
            ski_rect_img = cv2_rect[real_y:, :, ::-1]
            '''
            ski_rect_img = image[real_y:, :, :]


            # Run detection
            results = model.detect([ski_rect_img], verbose=0)

            # Visualize results
            r = results[0]

            boxes = r['rois']
            N = boxes.shape[0]
            if not N:
                continue
            if first_img :
                for each_box in boxes:
                    spine_box_list.append(each_box)                    
                first_img = False
            else:
                for each_box in boxes:
                    list_num = len(spine_box_list)
                    
                    add_box_flag = True
                    for compare_box in spine_box_list:
                        IOU = bb_intersection_over_union(each_box, compare_box)
                        if IOU > 1/4:
                            add_box_flag = False
                            break
                    if add_box_flag:
                        spine_box_list.append(each_box)
            '''
            img_save_path = os.path.join(save_dir_path, 'res_'+file_name[4:])
            visualize.save_instances(img_save_path, ski_rect_img, r['rois'], r['masks'], r['class_ids'], 
                                        class_names)
            '''
        spine_box_list = sorted(spine_box_list, key = lambda x : x[0])   # sort by y1       
        print("spine Num : " +str(len(spine_box_list)))        
        borns_mid_x = np.array([],dtype = np.int)
        borns_mid_y = np.array([],dtype = np.int)
        for spine_box in spine_box_list:
            y1, x1, y2, x2 = spine_box
            temp_mid_x = int((x1+x2)/2)
            temp_mid_y = int( (y1+y2)/2)
            borns_mid_x = np.append(borns_mid_x, temp_mid_x)    
            borns_mid_y = np.append(borns_mid_y,temp_mid_y)


        
        borns_modify(each_dir,borns_mid_x,csv_path)


        poly = np.poly1d(np.polyfit(borns_mid_x, borns_mid_y, 1)) #三次多項式
        print(poly)
        csv_writer.write("poly: "+","+ str(poly)+'\n'+'\n'+'\n')
        total_file = os.listdir(each_dir_path)
        total_file = filter(lambda x: x.endswith('png'), total_file) #只抓png檔
        for file_name in sorted(total_file):
            
            cv2_img = cv2.imread(os.path.join(each_dir_path, file_name))
            #print(file_name)
            cv2_rect =cv2_img.copy()
            draw_image = cv2_rect[real_y:, :, :]
            label ="L"
            i = 1            
            for idx,spine_box in enumerate(spine_box_list):
                y1, x1, y2, x2 = spine_box
                cv2.circle(draw_image, (borns_mid_x[idx], borns_mid_y[idx]), 3, (255, 0, 0), -1, 8, 0)
                cv2.rectangle(draw_image,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(draw_image,label+str(i),(x2,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,255),2,cv2.LINE_AA)
                cv2.putText
                cv2.putText
                i +=1
            for t in range(0, draw_image.shape[1], 1):
                y_ = np.int(poly(t))
                cv2.circle(draw_image, (t, y_), 1, (0, 0, 255), 1, 8, 0)
            img_save_path = os.path.join(save_dir_path, 'res_'+file_name[4:])
            cv2.imwrite(img_save_path, draw_image)
    csv_writer.close()