from model.classfier import *
from model.gradCam import *
import utils.para as para
from utils.dataloader import *

import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import preprocess_input
import tensorflow.keras as keras

UNLIMIT = 9999999999

def metrics_test():
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
    
    #vgg = VGG_16(feature_out=True,dim_out=para.classNums)
    #weights_save_dir = "./ckpt/classifier-vgg-100c-40-retrain/classifier-vgg-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-vgg-100c-constrainted-by-labelmask-union/classifier-vgg-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-vgg-100c-constrainted-by-instancemask-union/classifier-vgg-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-vgg-100c-constrainted-by-componentmask-union/classifier-vgg-100c-constrainted-by-componentmask-union"
    #vgg.load_weights(weights_save_dir)

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

    resnet152 = ResNet_152(feature_out=True,dim_out=para.classNums)
    # weights_save_dir = "./ckpt/classifier-resnet152-100c-40-retrain/classifier-resnet152-100c-40-retrain"
    weights_save_dir = "./ckpt/classifier-resnet152-100c-constrainted-by-labelmask-union/classifier-resnet152-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-resnet152-100c-constrainted-by-instancemask-union/classifier-resnet152-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-resnet152-100c-constrainted-by-componentmask-union/classifier-resnet152-100c-constrainted-by-componentmask-union"
    resnet152.load_weights(weights_save_dir)

    # v1 = MobileNet_v1(feature_out=True,dim_out= para.classNums)
    # weights_save_dir = "./ckpt/classifier-mobilenet_v1-100c-40-retrain/classifier-mobilenet_v1-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v1-100c-constrainted-by-labelmask-union/classifier-mobilenet_v1-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v1-100c-constrainted-by-instancemask-union/classifier-mobilenet_v1-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v1-100c-constrainted-by-componentmask-union/classifier-mobilenet_v1-100c-constrainted-by-componentmask-union"
    # v1.load_weights(weights_save_dir)

    # v2 = MobileNet_v2(feature_out=True,dim_out= para.classNums)
    # weights_save_dir = "./ckpt/classifier-mobilenet_v2-100c-40-retrain/classifier-mobilenet_v2-100c-40-retrain"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v2-100c-constrainted-by-labelmask-union/classifier-mobilenet_v2-100c-constrainted-by-labelmask-union"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v2-100c-constrainted-by-instancemask-union/classifier-mobilenet_v2-100c-constrainted-by-instancemask-union"
    # weights_save_dir = "./ckpt/classifier-mobilenet_v2-100c-constrainted-by-componentmask-union/classifier-mobilenet_v2-100c-constrainted-by-componentmask-union"
    # v2.load_weights(weights_save_dir)
    

    os.makedirs("./data_collection",exist_ok=True)
    sample_prob_file = open("./data_collection/sample_prob_union.txt",'w+')
    sample_frr_file = open("./data_collection/sample_frr_union.txt",'w+')
    # sample_prob_file = open("./data_collection/sample_prob_base.txt",'w+')
    # sample_frr_file = open("./data_collection/sample_frr_base.txt",'w+')
    
    P = 0
    N = 0
    
    foreground = 0
    background = 0
    
    process = tqdm.tqdm(enumerate(ds),total=ds_len_)
    for idx,(img,lab_mask,lab) in process:
        # img = img/255.0
        img_ = img / 255.0
        img = preprocess_input(img)
        img = np.expand_dims(img,axis=0)
        lab_mask = lab_mask/255.0
        lab_mask = np.expand_dims(lab_mask,axis=0)
        with tf.GradientTape() as t:
            # cam3,cam_mask,preds = compute_heatmap(vgg,img,np.argmax(lab.squeeze(axis=0)),size_holder,tape=t,training=False)
            # cam3,cam_mask,preds = compute_heatmap(inceptionv3,img,np.argmax(lab.squeeze(axis=0)),size_holder,tape=t,training=False)
            # cam3,cam_mask,preds = compute_heatmap(resnet50,img,np.argmax(lab.squeeze(axis=0)),size_holder,tape=t,training=False)
            # cam3,cam_mask,preds = compute_heatmap(resnet101,img,np.argmax(lab.squeeze(axis=0)),size_holder,tape=t,training=False)
            cam3,cam_mask,preds = compute_heatmap(resnet152,img,np.argmax(lab.squeeze(axis=0)),size_holder,tape=t,training=False)
            # cam3,cam_mask,preds = compute_heatmap(v1,img,np.argmax(lab.squeeze(axis=0)),size_holder,tape=t,training=False)
            # cam3,cam_mask,preds = compute_heatmap(v2,img,np.argmax(lab.squeeze(axis=0)),size_holder,tape=t,training=False)
            
            cam = np.where(cam3 > np.percentile(cam3, 100 - keep_percent), 1, 0)
            cam_mask = cam_mask.numpy().squeeze()
            cam_mask = cv.resize(cam_mask,size_holder)
            # seg,heatmap = overlay_gradCAM(img_,cam3=cam3,cam=cam)

        if (np.argmax(lab.squeeze(axis=0)) == np.argmax(preds.numpy().squeeze(axis=0))):
            P+=1
            img_name = dl.img_files[idx].split('/')[-1].split('.')[0]
            prob_str = img_name + " prob : {:.2f}".format(preds.numpy().squeeze(axis=0)[np.argmax(lab.squeeze(axis=0))]*100) + '\r\n'
        else:
            img_name = dl.img_files[idx].split('/')[-1].split('.')[0]
            prob_str = img_name + " prob : {:.2f}".format(preds.numpy().squeeze(axis=0)[np.argmax(lab.squeeze(axis=0))]*100)
            prob_str += " fake prob : {:.2f} ({})".format(preds.numpy().squeeze(axis=0)[np.argmax(preds.numpy().squeeze(axis=0))]*100,dl.lab_dict[np.argmax(preds.numpy().squeeze(axis=0))])  +'\r\n'
            N+=1
        
        mask = (np.sum(lab_mask,axis=3)/3)
        mask = mask.squeeze()
        if (dl.lab_type == 'Hard_masks'):
            mask[mask > 0] = 1.0

        img_name = dl.img_files[idx].split('/')[-1].split('.')[0]
        frr_str = img_name + " frr : {:.2f}".format((np.sum(cam_mask * mask)/np.sum(cam_mask))*100) + '\r\n'
        
        sample_prob_file.write(prob_str)
        
        sample_frr_file.write(frr_str)
        
        foreground += np.sum(cam_mask * mask)/ds_len_
        background += (np.sum(cam_mask))/ds_len_
        
        # cv.imwrite(res_dir+'/'+lab_files[idx].split('/')[-1],heatmap)

    process.close()
    print ('Test Acc : {:.2f}%'.format(P/ds_len_*100))
    print ('Test Foreground Reasoning Rate : {:.2f}%'.format(foreground/(background)*100))
    sample_frr_file.close()
    sample_prob_file.close()
    
def main():
    metrics_test()
    return 
    
if __name__ == "__main__":
    main()
