import numpy as np
import cv2 as cv
import pathlib
import os
import matplotlib.pyplot as plt

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
        # print (img_data_path)
        # if (os.path.exists(img_data_path)):
        #     if (resize==None):
        #         img.append(cv.imread(img_data_path,cv.IMREAD_COLOR))
        #     else:
        #         img.append(cv.resize(cv.imread(img_data_path,cv.IMREAD_COLOR),resize))
        # else:
        #     lab.pop()
        #     lab_dir.pop()
            # tar_dir.pop(idx)
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
        newDir = "instancelabel_visable"
    else:
        newDir = "instancelabel"
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
        comp = np.zeros(mask.shape)
        for i in range(0,40):
            comp[mask==i] = 1
        del mask
        mask = comp
        # cv.imwrite("test.png",mask*255)
        # import pdb
        # pdb.set_trace()
        c,_ = cv.findContours(np.array(mask,dtype=np.uint8),cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        mcx=0
        mcy=0
        mm00 = 0
        for c_ in c:
            M = cv.moments(c_)
            if (M['m00'] <= 1):
                continue
            cx,cy = int(M['m10'] / M['m00']),int(M['m01'] / M['m00'])
            mcx += cx/len(c)
            mcy += cy/len(c)
            mm00 += M['m10']/len(c)
        rad = int(100*mm00)
        for dy in range(max(mcy-rad,0),min(mcy+rad,mask.shape[0]-1)):
            for dx in range(max(mcx-rad,0),min(mcx+rad,mask.shape[1]-1)):
                export_lab[dy,dx] = max(export_lab[dy,dx],Gaoth(dx,dy,mcx,mcy,0.03,mm00))
                
        if (heatmap_mode):
            export_lab = np.expand_dims(export_lab, axis=2)
            export_lab = np.tile(export_lab, [1, 1, 3])
            export_lab = np.uint8(255 * export_lab)
            export_lab = cv.applyColorMap(export_lab, cv.COLORMAP_JET)
            # print (export_lab.shape,img.shape)
            # if ("5154" in export_dir):
            #     cv.imwrite("test_lab.png",export_lab)
            #     cv.imwrite("test_img.png",img)
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