from model.classfier import *
from model.gradCam import *
import utils.para as para
from utils.dataloader import *

import tqdm
import matplotlib.pyplot as plt
import random

UNLIMIT = 9999999999

def disturbance():
    dl = DataLoader(para.test_data)
    imgs,labmasks,labs = dl.getDateSets()
    ds_len_ = dl.getDataSize()
    img_files = dl.img_files
    lab_files = dl.lab_files
    ds = []
    for idx in range(ds_len_):
        ds.append(np.array([imgs[idx],labmasks[idx],labs[idx]]))
    ds = np.array(ds)
    print (ds.shape,ds[0].shape,ds[0][0].shape,ds[0][1].shape,ds[0][2].shape)
    
    # vgg = VGG_16(feature_out=True,dim_out=para.classNums)
    # weights_save_dir = "./ckpt/classifier-vgg-100c-40-retrain/classifier-vgg-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-vgg-100c-constrainted-by-labelmask-union/classifier-vgg-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-vgg-100c-constrainted-by-instancemask-union/classifier-vgg-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-vgg-100c-constrainted-by-componentmask-union/classifier-vgg-100c-constrainted-by-componentmask-union"
    # vgg.load_weights(weights_save_dir)

    # inceptionv3 = Inception_V3(feature_out=True,dim_out=para.classNums)
    # weights_save_dir = "./ckpt/classifier-inception_v3-100c-40-retrain/classifier-inception_v3-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-inception_v3-100c-constrainted-by-labelmask-union/classifier-inception_v3-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-inception_v3-100c-constrainted-by-instancemask-union/classifier-inception_v3-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-inception_v3-100c-constrainted-by-componentmask-union/classifier-inception_v3-100c-constrainted-by-componentmask-union"
    # inceptionv3.load_weights(weights_save_dir)
    
    # resnet50 = ResNet_50(feature_out=True,dim_out=para.classNums)
    # weights_save_dir = "./ckpt/classifier-resnet50-100c-40-retrain/classifier-resnet50-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-resnet50-100c-constrainted-by-labelmask-union/classifier-resnet50-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-resnet50-100c-constrainted-by-instancemask-union/classifier-resnet50-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-resnet50-100c-constrainted-by-componentmask-union/classifier-resnet50-100c-constrainted-by-componentmask-union"
    # resnet50.load_weights(weights_save_dir)
    
    # resnet101 = ResNet_101(feature_out=True,dim_out=para.classNums)
    # weights_save_dir = "./ckpt/classifier-resnet101-100c-40-retrain/classifier-resnet101-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-resnet101-100c-constrainted-by-labelmask-union/classifier-resnet101-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-resnet101-100c-constrainted-by-instancemask-union/classifier-resnet101-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-resnet101-100c-constrainted-by-componentmask-union/classifier-resnet101-100c-constrainted-by-componentmask-union"
    # resnet101.load_weights(weights_save_dir)

    # resnet152 = ResNet_152(feature_out=True,dim_out=para.classNums)
    # weights_save_dir = "./ckpt/classifier-resnet152-100c-40-retrain/classifier-resnet152-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-resnet152-100c-constrainted-by-labmask-union/classifier-resnet152-100c-constrainted-by-labmask-union"
    # weights_save_dir = "./ckpt/classifier-resnet152-100c-constrainted-by-instancemask-union/classifier-resnet152-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-resnet152-100c-constrainted-by-componentmask-union/classifier-resnet152-100c-constrainted-by-componentmask-union"
    # resnet152.load_weights(weights_save_dir)

    # v1 = MobileNet_v1(feature_out=True,dim_out= para.classNums)
    # weights_save_dir = "./ckpt/classifier-mobilenet_v1-100c-40-retrain/classifier-mobilenet_v1-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v1-100c-constrainted-by-labelmask-union/classifier-mobilenet_v1-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v1-100c-constrainted-by-instancemask-union/classifier-mobilenet_v1-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v1-100c-constrainted-by-componentmask-union/classifier-mobilenet_v1-100c-constrainted-by-componentmask-union"
    # v1.load_weights(weights_save_dir)

    v2 = MobileNet_v2(feature_out=True,dim_out= para.classNums)
    weights_save_dir = "./ckpt/classifier-mobilenet_v2-100c-40-retrain/classifier-mobilenet_v2-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v2-100c-constrainted-by-labelmask-union/classifier-mobilenet_v2-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v2-100c-constrainted-by-instancemask-union/classifier-mobilenet_v2-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v2-100c-constrainted-by-componentmask-union/classifier-mobilenet_v2-100c-constrainted-by-componentmask-union"
    v2.load_weights(weights_save_dir)
    
    
    P = 0
    N = 0
    
    foreground = 0
    background = 0

    process = tqdm.tqdm(enumerate(ds),total=ds_len_)

    for idx,(img,lab_mask,lab) in process:

        lab_mask = lab_mask/255.0
        m = (np.sum(lab_mask,axis=2)/3)
        m = np.expand_dims(m,axis=2)
        m[m>0] = 1.0
        

        ### (1) Cover foreground and contours
        # index = np.where(m>0)
        # y = index[0]
        # x = index[1]
        # try:
        #     upper_left_x = np.min(x)
        #     upper_left_y = np.min(y)
        #     lower_right_x = np.max(x)
        #     lower_right_y = np.max(y)
        # except:
        #     print ("Warn: Sample Exception!")
        #     continue
        # dx = int((lower_right_x - upper_left_x)*mask_ratio/2)
        # dy = int((lower_right_y - upper_left_y)*mask_ratio/2)
        # img = cv.rectangle(img,(upper_left_x+dx,upper_left_y+dy),(lower_right_x-dx,lower_right_y-dy),mask_pixels,thickness=cv.FILLED)
        
        ### (2) Cover the whole foreground
        # img = img * (1.0-m)

        ### (3) Cover the whole background
        # img = img * m 

        ### (4) Cover 50% background randomly
        # sum_fore = np.sum(m)
        # sum_50 = (img.shape[0] * img.shape[1] - sum_fore) * 0.5
        # sum = 0
        # i = random.randint(0,img.shape[0]-1)
        # while(sum < sum_50):
        #     for j in range(img.shape[1]):
        #         if m[i][j] == 0:
        #             img[i][j] = 0
        #             sum += 1
        #     if i < img.shape[0]-1:
        #        i = i + 1
        #     else:
        #        i = 0
        
        ### (5) Cover 20% background randomly
        # sum_fore = np.sum(m)
        # print(sum_fore)
        # sum_20 = (img.shape[0] * img.shape[1] - sum_fore) * 0.2
        # sum = 0
        # i = random.randint(0,img.shape[0]-1)

        # while(sum < sum_20):
        #     for j in range(img.shape[1]):
        #         if m[i][j] == 0:
        #             img[i][j] = 0
        #             sum += 1
        #     if i < img.shape[0]-1:
        #        i = i + 1
        #     else:
        #        i = 0

        ### (6) gasuss noise on background
        # img = gasuss_noise(img,maskimg=m,flag=0)
        
        # ### (7) gasuss noise on foreground
        # img = gasuss_noise(img,maskimg=m,flag=1)

        # ### (8) spiced_salt noise on background
        # img = spiced_salt_noise(img,maskimg=m,flag=0)

        # ### (9) spiced_salt noise on foreground
        # img = spiced_salt_noise(img,maskimg=m,flag=1)

        img = img/255.0
        # img = preprocess_input(img)
        img = np.expand_dims(img,axis=0)
        lab_mask = np.expand_dims(lab_mask,axis=0)

        with tf.GradientTape() as t:
            cam3,cam_mask,preds = compute_heatmap(v2,img,np.argmax(lab.squeeze(axis=0)),size_holder,tape=t,training=False)
            cam = np.where(cam3 > np.percentile(cam3, 100 - keep_percent), 1, 0)
            cam_mask = cam_mask.numpy().squeeze()
            cam_mask = cv.resize(cam_mask,size_holder)

        if (np.argmax(lab.squeeze(axis=0)) == np.argmax(preds.numpy().squeeze(axis=0))):
            P+=1
        else:
            N+=1
        
        mask = (np.sum(lab_mask,axis=3)/3)
        mask = mask.squeeze()
        if (dl.lab_type == 'Hard_masks'):
            mask[mask > 0] = 1.0
        
        foreground += np.sum(cam_mask * mask)/ds_len_
        background += (np.sum(cam_mask))/ds_len_

    process.close()
    print ('Test Acc : {:.2f}%'.format(P/ds_len_*100))
    print ('Test Foreground Reasoning Rate : {:.2f}%'.format(foreground/(background)*100))
        

def main():
    disturbance()
    return 
    
if __name__ == "__main__":
    main()
