#dicom轉出影像檔

import numpy as np
import tkinter as tk
from tkinter import filedialog as fd
import pydicom
import cv2
import os
import json

class DicomToPng:
    def __init__(self):
        self.root = tk.Tk()
        self.init_window()
        self.data_output_path = ''
        self.dicom_file_path = ''

    
    def choose_input_dir(self):
        self.dicom_file_path = fd.askdirectory(title="選擇dicom路徑",initialdir = "./Reformated_Data_89")
        input_dir = tk.Label(self.root, text=self.dicom_file_path).grid(row=0, column=1,padx=5,pady=5)
    
    def choose_output_dir(self):
        self.data_output_path = fd.askdirectory(title="選擇png存放路徑",initialdir = "./side_data")
        output_dir = tk.Label(self.root, text=self.data_output_path).grid(row=1, column=1,padx=5,pady=5)

    def find_dicom_files(self,directory):
        return (f for f in os.listdir(directory) if f.endswith('.dcm') or f.endswith('.IMA'))
        
    def read_dicom_file(self):

        total_dir = os.listdir(self.dicom_file_path)
        print(total_dir)

        for each_dir in total_dir:
            from_dir_path = self.dicom_file_path+'/'+each_dir
            total_file = self.find_dicom_files(from_dir_path)

            to_dir_path = self.data_output_path+'/'+each_dir
            if not os.path.isdir(to_dir_path):
                os.mkdir(to_dir_path)

            i=1
            for file in total_file:

                ds = pydicom.dcmread('{}/{}'.format(from_dir_path, file))
                if not file.endswith("dcm"):
                    os.rename('{}/{}'.format(from_dir_path, file),'{}/{}.dcm'.format(from_dir_path, file))

                    ds = pydicom.dcmread('{}/{}.dcm'.format(from_dir_path, file))

                pixel_ds =  np.frombuffer(ds.PixelData, 'uint{}'.format(ds.BitsAllocated))[:ds.Rows*ds.Columns*ds.SamplesPerPixel].reshape(ds.Rows, ds.Columns, ds.SamplesPerPixel)

                try:
                    pixel_ds = pixel_ds*ds.RescaleSlope+ds.RescaleIntercept
                except Exception as e:
                    pass
                try:
                    pixel_ds = np.where(pixel_ds <= ds.WindowCenter-0.5-(ds.WindowWidth-1)/2, 0, np.where(pixel_ds > ds.WindowCenter-0.5+(ds.WindowWidth-1)/2, 255, ((pixel_ds-(ds.WindowCenter-0.5))/(ds.WindowWidth-1)+0.5)*255))
                except Exception as e:
                    pass
                if ds.SeriesDescription.lower().find('snapshot') != -1:
                    ds.SeriesDescription = ''
                
                print('{}/{}_{}_{}_{}_{:0>2}.png'.format(to_dir_path, ds.PatientID, ds.SeriesNumber, ds.SeriesDescription, ds.BitsStored,i))
                
                cv2.imwrite('{}/{}_{}_{}_{}_{:0>2}.png'.format(to_dir_path, ds.PatientID, ds.SeriesNumber, ds.SeriesDescription, ds.BitsStored,i), pixel_ds)
                i+=1

        print(os.listdir(self.data_output_path))


    def init_window(self):
        self.root.title('dicom轉png')
        
        btn1 = tk.Button(self.root, text='選擇讀取資料夾',width=16,height=2,command=self.choose_input_dir).grid(row=0, column=0,padx=5,pady=5)
        input_dir = tk.Label(self.root, text='  '*50).grid(row=0, column=1,padx=5,pady=5)
        btn2 = tk.Button(self.root, text='選擇存放資料夾',width=16,height=2,command=self.choose_output_dir).grid(row=1, column=0,padx=5,pady=5)
        output_dir = tk.Label(self.root, text='  '*50).grid(row=1, column=1,padx=5,pady=5)
        btn3 = tk.Button(self.root, text='dicom 轉 png',width=16,height=2,command=self.read_dicom_file).grid(row=2, column=0,columnspan=2,padx=5,pady=5)
        
        
        self.root.mainloop()

if __name__ == "__main__":
    dicom_png = DicomToPng()

        