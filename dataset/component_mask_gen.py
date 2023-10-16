import numpy as np
import cv2 as cv
import pathlib
import os
import matplotlib.pyplot as plt
import json

def dataloader(root,type=['train','val','test'],resize=None):
    root_dir = pathlib.Path(root)
    tar_dir = []
    for file in root_dir.glob('*'):
        tar_dir.append(str(file))
    tar_dir.sort()
    print (tar_dir)
    lab = []
    img = []
    lab_dir = []
    for idx,img_dir in enumerate(tar_dir):
        lab_dir.append(img_dir)
        if (resize==None):
            lab.append(cv.imread(img_dir,cv.IMREAD_COLOR))
        else:
            lab.append(cv.resize(cv.imread(img_dir,cv.IMREAD_COLOR),resize))
        fg = 0
        for t in type:
            img_data_path = img_dir.replace('labelmask',t+'/'+img_dir.split('/')[-1].split('_')[0]).replace('png','JPEG')
            if (os.path.exists(img_data_path)):
                fg = 1
                if (resize==None):
                    img.append(cv.imread(img_data_path,cv.IMREAD_COLOR))
                else:
                    img.append(cv.resize(cv.imread(img_data_path,cv.IMREAD_COLOR),resize))
                break
        if not fg:
            print ("{} corresponding image missed!".format(img_dir))
            exit(-1)
    return img,lab,lab_dir

def Gaoth(dx,dy,cx,cy,p,m00):
    return np.exp(-1*(np.power(dx-cx,2)+np.power(dy-cy,2))/(p*m00))

def main(heatmap_mode=False):
    img,lab,lab_dir = dataloader("labelmask",resize=None)
    ds = []
    for idx in range(len(lab)):
        ds.append((img[idx],lab[idx],lab_dir[idx]))
    del img,lab,lab_dir
    
    if (heatmap_mode):
        newDir = "componentlabel_visable"
    else:
        newDir = "componentlabel"
    os.makedirs(newDir,exist_ok=True)
    # f = open("dataset/PartImageNet/seg_labels_dict.json")
    # seg_labels_dict = json.loads(f.read())
    # f.close()

    
    for img,label,dir in ds:
        export_dir = os.path.join(newDir,dir.split('/')[-1])
        print (export_dir)
        if (os.path.exists(export_dir)):
            continue
        mask = np.sum(label,axis=2).squeeze()
        mask = mask / 3
        export_lab = np.zeros(mask.shape,dtype=np.float32)
        for i in range(0,40):
            comp = np.zeros(mask.shape)
            comp[mask==i] = 1
            # comp[comp>=1] = 1
            if (np.sum(comp.flatten()) > 50):
                # plt.subplot(1,3,1)
                # plt.imshow(mask)
                # plt.subplot(1,3,2)
                # plt.imshow(comp)
                # 求取组件几何中心
                c,_ = cv.findContours(np.array(comp,dtype=np.uint8),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
                for c_ in c:
                    M = cv.moments(c_)
                    if (M['m00'] <= 1):
                        continue
                    cx,cy = int(M['m10'] / M['m00']),int(M['m01'] / M['m00'])
                    rad = int(20*M['m00'])
                    # print (mask.shape,export_lab.shape)
                    for dy in range(max(cy-rad,0),min(cy+rad,mask.shape[0]-1)):
                        for dx in range(max(cx-rad,0),min(cx+rad,mask.shape[1]-1)):
                            # print (dy,dx)
                            export_lab[dy,dx] = max(export_lab[dy,dx],Gaoth(dx,dy,cx,cy,0.6,M["m00"]))
                    # print (cx,cy)
                    # comp = cv.circle(np.array(comp,dtype=np.uint8),(cx,cy),5,0.5,thickness=cv.FILLED)
                    # export_lab[comp==0.5] = 0.5
                    # export_lab[comp==1] = 1
                    # plt.subplot(1,3,3)
                    # plt.imshow(cv.circle(np.array(comp,dtype=np.uint8),(cx,cy),5,0.5,thickness=cv.FILLED))
                    # plt.show()
        # plt.subplot(1,2,1)
        # plt.imshow(mask)
        # plt.subplot(1,2,2)
        # plt.imshow(export_lab)
        # plt.show()
        if (heatmap_mode):
            export_lab = np.expand_dims(export_lab, axis=2)
            export_lab = np.tile(export_lab, [1, 1, 3])
            export_lab = np.uint8(255 * export_lab)
            export_lab = cv.applyColorMap(export_lab, cv.COLORMAP_JET)
            if not (export_lab.shape == img.shape):
                tmp = np.zeros(export_lab.shape)
                for y in range(tmp.shape[0]):
                    for x in range(tmp.shape[1]):
                        tmp[y,x] = img[x,y]
                for y in range(tmp.shape[0]):
                    for x in range(int(tmp.shape[1]/2)):
                        temp_pix = [tmp[y,x][0],tmp[y,x][1],tmp[y,x][2]]
                        tmp[y,x] = tmp[y,tmp.shape[1]-1-x]
                        tmp[y,tmp.shape[1]-1-x] = temp_pix
                img = tmp
            try:
                concat_export = export_lab * 0.3 + img * 0.5
                cv.imwrite(export_dir,concat_export)
            except:
                print ("{} is failed to get the component soft mask!".format(dir))
        else:
            export_lab = np.expand_dims(export_lab, axis=2)
            export_lab = np.uint8(255 * export_lab)
            cv.imwrite(export_dir,export_lab)
    return

if __name__ == "__main__":
    main(heatmap_mode=True)