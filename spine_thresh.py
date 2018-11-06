import cv2
import numpy as np
from matplotlib import pyplot as plt
import image_analysis
import os

def find_rids_y_val(rect_img,opposite_img):
    #分為左邊肋骨及右邊肋骨，計算不同

    '''
    #水平翻轉,rect_img肋骨靠左對齊,如右肋骨圖
    if is_left:
        rect_img = cv2.flip(rect_img,1)
    else:
        opposite_img = cv2.flip(opposite_img,1)
    '''

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

            '''
            if is_left:
                if W/H >2 and -70<angle and angle<-20:
                    rid_contour.append(each_contour)

            else:
            '''
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

def find_img_files(directory):
        return (f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.png'))


from_dir_path ='./front_img_half'
to_dir_path ='./line_front_img'
total_file = find_img_files(from_dir_path)


for file in total_file:

    gray_image=cv2.imread('{}/{}'.format(from_dir_path, file), 0)


    # 計算直方圖每個 bin 的數值
    hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

    # 畫出直方圖
    plt.plot(hist,color = 'r')
    plt.xlim([0, 256])
    plt.ylim([0, 200000])
    save_file_name ='{}/cor_hist_{}.png'.format(to_dir_path,file)
    plt.savefig(save_file_name )
    plt.close()
            

    ret,thresh1=cv2.threshold(gray_image,254,255,cv2.THRESH_BINARY)
    find_line_frame = thresh1.copy()
    find_line_frame = cv2.cvtColor(find_line_frame,cv2.COLOR_GRAY2RGB)

    #去雜訊
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh1, cv2.MORPH_OPEN, kernel)

    _, contours, hierarchy  = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = image_analysis.max_contour(contours)
    spine_contour = np.zeros(gray_image.shape, np.uint8)
    cv2.drawContours(spine_contour, [max_contour], 0, (255, 255, 255), -1)

    #找主骨幹
    kernel = np.ones((11,3),np.uint8)   
    main_spine = cv2.erode(opening,kernel,iterations = 3)
    main_spine = cv2.dilate(main_spine,kernel,iterations = 3)

    _, contours, hierarchy  = cv2.findContours(main_spine, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    max_contour = image_analysis.max_contour(contours)
    main_spine_contour = np.zeros(gray_image.shape, np.uint8)
    cv2.drawContours(main_spine_contour, [max_contour], 0, (255, 255, 255), -1)

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
    

    for idx in range(len(cols_pixels)):
        plt.plot(idx, cols_pixels[idx], '-bo')
    filename = '{}/col_pixels_{}.png'.format(to_dir_path,file)
    plt.savefig(filename)
    plt.close()
    

    cols_pixels[np.where(cols_pixels[:]<spine_contour.shape[0]*2/3)] = 0

    '''
    for idx in range(len(cols_pixels)):
        plt.plot(idx, cols_pixels[idx], '-bo')
    filename = "./binary/spine_middle.png"
    plt.savefig(filename)
    plt.close()
    '''


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

    '''
    for idx in range(len(rows_pixels)):
        plt.plot(idx, rows_pixels[idx], '-ro')
    filename = "./binary/row_pixels.png"
    plt.savefig(filename )
    plt.close()
    '''
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
    #print(left_rids_y)



    

    #圈出肋骨外接矩形
    draw_img = cv2.cvtColor(ribs_img, cv2.COLOR_GRAY2BGR)#gray_image * ribs_img[:, :] 
    #draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2RGB)
    draw_left_img =draw_img[ top_edge  :bottom_edge ,0 : left_edge]
    draw_left_img = cv2.flip(draw_left_img,1)
    draw_right_img =draw_img[ top_edge  :bottom_edge ,right_edge : ribs_img.shape[1]]



    if left_rid_contour!=None:
        for idx , contour in enumerate(left_rid_contour):
            #圈出輪廓的外接矩形  
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)  
            cv2.drawContours(draw_right_img, [box], 0, (255, 0 , 255), 2)
            if(left_contour_idx == idx) :
                cv2.drawContours(draw_left_img, [box], 0, (255, 0 , 0), 2)
            else:
                cv2.drawContours(draw_left_img, [box], 0, (0, 255, 0), 2)



    #右邊肋骨
    
    if cv2.countNonZero(right_img) <50:
        continue
    right_rid_contour,right_rids_y,right_contour_idx =find_rids_y_val(right_img,left_img)
    #print(right_rids_y)


    
    if right_rid_contour!=None:
        for idx , contour in enumerate(right_rid_contour):
            #圈出輪廓的外接矩形  
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int0(box)  
            cv2.drawContours(draw_left_img, [box], 0, (255,0 , 255), 2)
            if(right_contour_idx == idx):
                cv2.drawContours(draw_right_img, [box], 0, (255, 0 , 0), 2)
            else:
                cv2.drawContours(draw_right_img, [box], 0, (0, 255, 0), 2)


    real_y=None
    if left_rids_y ==0 and right_rids_y ==0:
        real_y=None
    elif left_rids_y ==0:
        real_y = top_edge + (int)(right_rids_y)
    elif right_rids_y ==0:
        real_y = top_edge + (int)(left_rids_y)
    else:
        real_y = top_edge + left_rids_y if left_rids_y>right_rids_y else right_rids_y

    
    if real_y:
        
        cv2.line(find_line_frame,(0,real_y),(spine_contour.shape[1],real_y),(255,0,0),5)




    cv2.imwrite('{}/line_{}'.format(to_dir_path,file), find_line_frame)


    #將線畫在側面圖上
    



    side_set_path ='./side_img/'+ file[:8]
    save_path ='./resImg/'+ file[:8]
    if not os.path.isdir(save_path):
        os.mkdir(save_path)
    total_file = os.listdir(side_set_path)
    print(total_file)

    first_tag = True
    for img_file in total_file:
        img =cv2.imread('{}/{}'.format(side_set_path, img_file), 0)            

        tmp_line_frame = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        if real_y and first_tag:
            first_tag = False
            or_img_array = np.array(img)

            '''
            #找影像最上黑邊
            #上邊
            rows_pixels = []
            for i in range(int(or_img_array.shape[0]*0.3)):
                temp_count = cv2.countNonZero(or_img_array[i:i+1, :])
                rows_pixels.append(temp_count)

            rows_pixels = np.array(rows_pixels)
            img_top = 0
            if np.any(np.where(rows_pixels==0)):  

                img_top = np.max( np.where(rows_pixels==0) ) 
            
            real_y += img_top 
            '''  
        if real_y:         
            cv2.line(tmp_line_frame,(0,real_y),(tmp_line_frame.shape[1],real_y),(255,0,0),5)
        cv2.imwrite('{}/res_{}'.format(save_path,img_file), tmp_line_frame)
    


    #影像處理的步驟
    
    titles = ['Image','BINARY','opening_contour','main_spine','ribs_img','left_ribs','right_ribs','final']
    images = [gray_image, thresh1,opening,main_spine_contour,ribs_img,draw_left_img,draw_right_img,find_line_frame]
    for i in range(len(titles)):
       plt.subplot(2,4,i+1),plt.imshow(images[i],'gray')
       plt.title(titles[i])
       plt.xticks([]),plt.yticks([])
    plt.savefig('{}/detail_{}'.format(to_dir_path,file) )
    plt.close()
    