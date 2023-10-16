import os,sys,pathlib
import numpy as np
import cv2 as cv
import shutil

data_root = "PartImageNet"
data_type = ['train','val','test']
data_ctx = ['images']

# for type in data_type:
#     if not (os.path.exists(type)):
#         os.mkdir(type)
    
# for ctx in data_ctx:
#     data_path = os.path.join(data_root,ctx)
#     for type in data_type:
#         catsId = []
#         file_path = os.path.join(data_path,type)
#         file_path_root = pathlib.Path(file_path)
#         for file in file_path_root.glob('*'):
#             file_str = str(file)
#             file_cat = file_str.split('/')[-1].split('_')[0]
#             if (file_cat in catsId):
#                 shutil.copyfile(file_str,os.path.join(type,file_cat,file_str.split('/')[-1]))
#             else:
#                 catsId.append(file_cat)
#                 os.makedirs(os.path.join(type,file_cat),exist_ok=True)
#                 shutil.copyfile(file_str,os.path.join(type,file_cat,file_str.split('/')[-1]))
#         print (type+" has categories : {}".format(len(catsId)))
            
if not (os.path.exists('labelmask')):
    os.mkdir('labelmask')
    
label_ctx = ['annotations']

for ctx in label_ctx:
    data_path = os.path.join(data_root,ctx)
    for type in data_type:
        file_path = os.path.join(data_path,type)
        file_path_root = pathlib.Path(file_path)
        cnt = 0
        for file in file_path_root.glob('*.png'):
            file_str = str(file)
            # shutil.copyfile(file_str,os.path.join("labelmask",file_str.split('/')[-1]))
            out_path = os.path.join("labelmask",file_str.split('/')[-1])
            lab = cv.imdecode(np.fromfile(file_str,dtype=np.uint8),cv.IMREAD_COLOR)
            lab[np.where(lab < 40)] = 255
            lab[np.where(lab == 40)] = 0
            cv.imwrite(out_path,lab)
            cnt+=1
        print (type+" has samples : {}".format(cnt))