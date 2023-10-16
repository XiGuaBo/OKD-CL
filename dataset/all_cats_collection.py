import os,sys,pathlib
import numpy as np
import cv2 as cv
import shutil
import json

data_root = "PartImageNet"

f = open(os.path.join(data_root,"labels_dict.txt"),'r')
lab_dict = {}
for line in f.readlines():
    catId = line.split(' ')[1]
    catNm = line.split(catId)[-1][1:-1]
    lab_dict[catId] = catNm
    
# with open("idx2name.json",'w') as f:
#     f.write(json.dumps(lab_dict,ensure_ascii=False))
# f.close()
    
data_path = os.path.join("train")
cats = []
for ctx in pathlib.Path(data_path).glob('*'):
    str_ctx = str(ctx).split('/')[-1]
    cats.append(str_ctx + ":" + lab_dict[str_ctx])

print ("all catagoties : {}".format(len(cats)))
f = open('supcat_and_fine_cat.txt','w')
for cat in cats:
    print (cat)
    f.write(cat+'\n')
f.close()