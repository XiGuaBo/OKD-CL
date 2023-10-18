import os,sys,pathlib
import numpy as np
import cv2 as cv
from utils.para import *


class DataLoader:
    def __init__(self,data_root,lab_type=lab_type):
        data_path = pathlib.Path(data_root)
        class_path = [str(dir_name) for dir_name in data_path.glob('*')]
        classlb = [lab.split('/')[-1] for lab in class_path]
        classlb.sort()
        lab_dict = dict([ (labname,idx) for (idx,labname) in enumerate(classlb) ])
        self.lab_dict = dict([ (idx,labname) for (idx,labname) in enumerate(classlb) ])
        print ('class len : {} class list : {}'.format(len(classlb),lab_dict))
        self.img_files = []
        for pth in class_path:
            pth_root = pathlib.Path(pth)
            for files in pth_root.glob('*'):
                self.img_files.append(str(files))
        
        self.img_files.sort()
        # ../train/n02492660/n02492660_5.jpeg
        # ../lab_type/n02492660_n02492660_5.jpeg
        # ../test/n02492660/ILSVRC2012_val_00003118.jpeg
        # ../lab_type/n02492660_ILSVRC2012_val_00003118.jpeg
        self.lab_files = [ lab.replace(data_root.split('/')[-1]+'/'+lab.split('/')[-2],lab_type) for lab in self.img_files ]
        for idx in range(len(self.img_files)):
            self.lab_files[idx] = self.lab_files[idx].replace(self.lab_files[idx].split('/')[-1],self.img_files[idx].split('/')[-2]+ '_' +self.lab_files[idx].split('/')[-1])
        # self.lab_files.sort()
        self.lab_type = lab_type
        
        if (debug):
            print ('img list : {} ...'.format(self.img_files[check_idx_range[0]:check_idx_range[1]]))
            print ('lab list : {} ...'.format(self.lab_files[check_idx_range[0]:check_idx_range[1]]))
        
        self.lab_vector = []
        for img in self.img_files:
            label = np.zeros((1,classNums))
            label[0][lab_dict[img.split('/')[-2]]] = 1
            self.lab_vector.append(label)
        self.lab_vector = np.array(self.lab_vector,dtype=np.float32)
        if (debug):
            # print ('lab_vec list : {} ...'.format(self.lab_vector[check_idx_range[0]:check_idx_range[1]]))
            class_lab_list = []
            for idx in range(check_idx_range[0],check_idx_range[1]):
                class_lab_list.append(self.lab_vector[idx][0].argmax())
            print (class_lab_list)
            del class_lab_list
            
        
    def getDateSets(self,resize=size_holder):
        if (resize!=None):
            imgs = np.array([cv.resize(cv.imdecode(np.fromfile(file,dtype=np.uint8),cv.IMREAD_COLOR),resize) for file in self.img_files])
            labs = np.array([cv.resize(cv.imdecode(np.fromfile(file,dtype=np.uint8),cv.IMREAD_COLOR),resize) for file in self.lab_files])
        else:
            imgs = np.array([cv.imdecode(np.fromfile(file,dtype=np.uint8),cv.IMREAD_COLOR) for file in self.img_files])
            labs = np.array([cv.imdecode(np.fromfile(file,dtype=np.uint8),cv.IMREAD_COLOR) for file in self.lab_files])
        return imgs,labs,self.lab_vector
    
    def getDataSize(self):
        return len(self.img_files)
