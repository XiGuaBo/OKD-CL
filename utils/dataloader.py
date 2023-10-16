import os,sys,pathlib
import numpy as np
import cv2 as cv
from utils.para import *

sub_set = ["n02481823","n02085782","n02422106","n01484850","n01749939","n02089867",
           "n01728920","n01740131","n02483708","n02690373","n02422699","n03937543",
           "n02408429","n02101006","n02101006","n03594945","n02493509","n01843065",
           "n01698640","n01644373","n01494475","n02480855","n02009229","n03983396",
           "n02484975","n04065272","n02025239","n01748264","n02092339","n02607072",
           "n04612504","n02134418","n01632458","n02125311","n04037443","n02412080",
           "n01664065","n02098105","n02132136","n02437312","n04552348","n01688243",
           "n01630670","n01669191","n01641577","n02129604","n03977966","n01828970",
           "n02091831","n01644900","n01491361","n02114367","n04465501","n01756291",
           "n02835271","n01855672","n02823428","n01614925","n04252225","n02120079",
           "n04146614","n01443537","n04147183","n03445924","n02130308","n02814533",
           "n02033041","n03947888","n02356798","n04482393","n03785016","n02510455",
           "n03444034","n03792782","n02397096","n02701002","n04509417","n02090379",
           "n01689811","n02483362","n02514041","n02487347","n02071294","n02442845",
           "n02058221","n03770679","n04483307","n02002724","n01667114","n03791053",
           "n02101388","n02930766","n02109525","n02109961","n02441942","n02100583",
           "n02102040","n02444819","n02097474","n02490219"] # 100 kinds

class DataLoader:
# @param lab_type -> ['labelmask','instancemask','componentmask']
    def __init__(self,data_root,lab_type=lab_type):
        data_path = pathlib.Path(data_root)
        # class_path = [str(dir_name) for dir_name in data_path.glob('*')]
        # class_path = [str(dir_name) for dir_name in data_path.glob('*')][-128:]
        class_path = [os.path.join(data_root,subc) for subc in sub_set]
        classlb = [lab.split('/')[-1] for lab in class_path]
        classlb.sort()
        lab_dict = dict([ (labname,idx) for (idx,labname) in enumerate(classlb) ])
        self.lab_dict = dict([ (idx,labname) for (idx,labname) in enumerate(classlb) ])
        
        ###############################
        self.lab_dict[38] = 'reserve'
        ###############################
        
        print ('class len : {} class list : {}'.format(len(classlb),lab_dict))
        print ('class len : {} lab dict : {}'.format(len(classlb),self.lab_dict))
        self.img_files = []
        for pth in class_path:
            pth_root = pathlib.Path(pth)
            for files in pth_root.glob('*'):
                self.img_files.append(str(files))
        
        self.img_files.sort()
        # ../train/001/1-0.png
        # ../lab_type/1-0.png
        self.lab_files = [ lab.replace(data_root.split('/')[-1]+'/'+lab.split('/')[-2],lab_type).replace('JPEG','png') for lab in self.img_files ]
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
