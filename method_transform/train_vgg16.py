from model.classfier import *
from model.gradCam import *
import utils.para as para
from utils.dataloader import *

import tqdm
import matplotlib.pyplot as plt


UNLIMIT = 9999999999

def classifier_train_with_constraint(mode = None):

    # train data
    dl = DataLoader(para.train_data,lab_type=lab_type)
    imgs,labmasks,labs = dl.getDateSets()
    ds_len_ = dl.getDataSize()
    ds = []
    for idx in range(ds_len_):
        ds.append(np.array([imgs[idx],labmasks[idx],labs[idx]]))
    ds = np.array(ds)
    print (ds.shape,ds[0].shape,ds[0][0].shape,ds[0][1].shape,ds[0][2].shape)
    np.random.shuffle(ds)
    
    # val data
    ts_dl = DataLoader(para.test_data)
    ts_imgs,ts_labmasks,ts_labs = ts_dl.getDateSets()
    ts_ds_len_ = ts_dl.getDataSize()
    ts_ds = []
    for idx in range(ts_ds_len_):
        ts_ds.append(np.array([ts_imgs[idx],ts_labmasks[idx],ts_labs[idx]]))
    ts_ds = np.array(ts_ds)
    print (ts_ds.shape,ts_ds[0].shape,ts_ds[0][0].shape,ts_ds[0][1].shape,ts_ds[0][2].shape)
    np.random.shuffle(ts_ds)
    
    vgg = VGG_16(feature_out=True,vector_out = True,dim_out=para.classNums)
    
    for idx,layer in enumerate(vgg.layers):
        if ('vgg' in layer.name):
            for layer_ in layer.layers:
                layer_.trainable = True
        else:       
            layer.trainable = True

    weights_dir_dict = { "HC":"labelmask","KI":"knowledge-inject","KI_HC":"labelmask-union"}

    weights_save_dir = "./ckpt/classifier-vgg-15-constrainted-by-" + weights_dir_dict[mode] + "/" + "classifier-vgg-15-constrainted-by-" + weights_dir_dict[mode]
    
    vgg.load_weights("./ckpt/classifier-vgg-15-20-retrain/classifier-vgg-15-20-retrain")

    global_loss = UNLIMIT
    
    for epoch in range(para.constraint_epochs):
        mean_loss = tf.constant(0,dtype=tf.float32)
        batch_loss = tf.constant(0,dtype=tf.float32)
        
        opt = keras.optimizers.Adam(learning_rate=lr_decay_specific(para.fine_tune_lr,epoch,0.90))
        
        process = tqdm.tqdm(enumerate(ds),total=ds_len_)
        for idx,(img,lab_mask,lab) in process:
            img = img/255.0
            img = np.expand_dims(img,axis=0)

            lab_mask = lab_mask/255.0
            lab_mask = cv.resize(lab_mask,(7,7))
            gray_mask = np.sum(lab_mask,axis=2)/3
            if (dl.lab_type == 'Hard_masks'):
                gray_mask[gray_mask>0] = 1.0
            gray_mask = np.expand_dims(gray_mask,axis=0)
            gray_mask = np.expand_dims(gray_mask,axis=3)
            del lab_mask

            with tf.GradientTape(persistent=True) as tape:
                vgg.feature_out = True
                vgg.vector_out = True
                (preds,convOuts,hiddenout) = vgg(img)  # preds after softmax
                
                if "KI" in mode:
                    # Align Loss
                    classidx = np.argmax(lab[0])
                    knowledge = np.loadtxt('./txt-15/{}.txt'.format(classidx))
                    knowledge = tf.convert_to_tensor([knowledge],dtype=tf.float32)
                    loss_kll = para.mseLoss(knowledge,hiddenout)
                
                if mode != "KI":
                    # constrained loss
                    loss = preds[:, np.argmax(lab.squeeze(axis=0))]
                    if (tape==None):
                        print ("Please Transmit Tape First!")
                        exit(-1)
                    grads = tape.gradient(loss, convOuts)
                    convOuts = convOuts[0]
                    grads = grads[0]
                    norm_grads = tf.divide(grads, tf.reduce_mean(tf.square(grads)) + tf.constant(1e-7))
                    weights = tf.reduce_mean(norm_grads, axis=(0, 1))
                    cam = tf.reduce_sum(tf.multiply(weights, convOuts), axis=-1, keepdims=True)
                    cam = tf.math.maximum(cam,0)
                    if (tf.reduce_max(cam)!=0):
                        cam = cam / (tf.reduce_max(cam))
                    loss_am = para.mseLoss(gray_mask,cam)

                # Accuracy loss
                loss_acc = para.categoricalCrossentropyLoss(lab,preds)

                # total loss
                if mode == "KI":
                    loss = loss_acc +  0.2 * loss_kll
                elif "KI" in mode:
                    loss = loss_acc + 0.2 * loss_am +  0.2 * loss_kll
                else:
                    loss = loss_acc + 0.2 * loss_am

                mean_loss += loss/ds_len_
                batch_loss += loss

            if ((idx+1) % para.fine_tune_batch_size == 0):
                gd = tape.gradient(batch_loss,vgg.trainable_variables)
                opt.apply_gradients(zip(gd,vgg.trainable_variables))
                batch_loss = tf.constant(0,dtype=tf.float32)
            
            process.set_description_str('epochs {} - step {}'.format(str(epoch+1),str(idx+1)))
            if mode == "KI":
                process.set_postfix_str('prob {:.3f} kll {:.2f}% mean loss {:.3f}'.format(1-loss_acc.numpy(),(1-loss_kll.numpy())*100,mean_loss.numpy()))
            elif "KI" in mode:
                process.set_postfix_str('prob {:.3f} rr {:.2f}% kll {:.2f}% mean loss {:.3f}'.format(1-loss_acc.numpy(),(1-loss_am.numpy())*100,(1-loss_kll.numpy())*100,mean_loss.numpy()))
            else:
                process.set_postfix_str('prob {:.3f} rr {:.2f}% mean loss {:.3f}'.format(1-loss_acc.numpy(),(1-loss_am.numpy())*100,mean_loss.numpy()))
        
        process.close()
        
        if (global_loss > mean_loss.numpy()):
            print ('save weights - {:.3f}'.format(mean_loss.numpy()))
            vgg.save_weights(weights_save_dir)
        global_loss = min(global_loss,mean_loss)
        
        # val metrics
        P = 0
        foreground = 0
        background = 0
        ts_process = tqdm.tqdm(enumerate(ts_ds),total=ts_ds_len_)
        for ts_idx,(ts_img,ts_labmask,ts_lab) in ts_process:
            ts_img = ts_img/255.0
            ts_labmask = ts_labmask/255.0
            ts_img = np.expand_dims(ts_img,axis=0)
            ts_labmask = np.expand_dims(ts_labmask,axis=0)
            
            with tf.GradientTape() as tape:
                vgg.feature_out = True
                vgg.vector_out = False
                _,cam_mask,ts_pred = compute_heatmap(vgg,ts_img,np.argmax(ts_lab.squeeze(axis=0)),size_holder,tape=tape,training=False)
            cam_mask = cam_mask.numpy().squeeze()
            cam_mask = cv.resize(cam_mask,size_holder)
            # cam_mask = np.where(cam_mask > np.percentile(cam_mask, 100 - keep_percent), 1, 0)
            mask = (np.sum(ts_labmask,axis=3)/3)
            mask = mask.squeeze()
            if (ts_dl.lab_type == 'Hard_masks'):
                mask[mask > 0] = 1
            # print (cam_mask.shape,mask.shape)
            foreground += np.sum(cam_mask * mask)/ts_ds_len_
            background += np.sum(cam_mask)/ts_ds_len_
            
            if (ts_pred[0].numpy().argmax() == ts_lab[0].argmax()):
                P+=1
        ts_process.close()
        print ('test acc - {:.2f}%'.format(P/ts_ds_len_*100))
        print ('Test Foreground Reasoning Rate : {:.2f}%'.format(foreground/(background)*100))


def classifier_train_without_constraint():
    # train data
    dl = DataLoader(para.train_data)
    imgs,_,labs = dl.getDateSets()
    ds_len_ = dl.getDataSize()
    ds = []
    for idx in range(ds_len_):
        ds.append(np.array([imgs[idx],labs[idx]]))
    ds = np.array(ds)
    print (ds.shape,ds[0].shape,ds[0][0].shape,ds[0][1].shape)
    np.random.shuffle(ds)
    
    # val data
    ts_dl = DataLoader(para.test_data)
    ts_imgs,_,ts_labs = ts_dl.getDateSets()
    ts_ds_len_ = ts_dl.getDataSize()
    ts_ds = []
    for idx in range(ts_ds_len_):
        ts_ds.append(np.array([ts_imgs[idx],ts_labs[idx]]))
    ts_ds = np.array(ts_ds)
    print (ts_ds.shape,ts_ds[0].shape,ts_ds[0][0].shape,ts_ds[0][1].shape)
    np.random.shuffle(ts_ds)
    

    vgg = VGG_16(feature_out=False,dim_out=classNums)
    
    # param block
    # 20 eps block all vgg's layers(only train classfier head)

    for idx,layer in enumerate(vgg.layers):
        if ('vgg16' in layer.name):
            for layer_ in layer.layers:
                layer_.trainable = False
        else:       
            layer.trainable = True
    
    weights_save_dir = "./ckpt/classifier-vgg-15-20-retrain/classifier-vgg-15-20-retrain"

    # weights_save_dir = "./ckpt/classifier-vgg-15-40-retrain/classifier-vgg-15-40-retrain"
    # vgg.load_weights("./ckpt/classifier-vgg-15-20-retrain/classifier-vgg-15-20-retrain")

    global_loss = UNLIMIT


    for epoch in range(20):
        mean_loss = tf.constant(0.,dtype=tf.float32)
        batch_loss = tf.constant(0.,dtype=tf.float32)
        # 20 eps
        opt = keras.optimizers.Adam(learning_rate=lr_decay(base_linear_lr,epoch))
        
        # 20-40 epochs
        # opt = keras.optimizers.Adam(learning_rate=lr_decay_specific(para.fine_tune_lr,epoch,0.90))
        process = tqdm.tqdm(enumerate(ds),total=ds_len_)
        for idx,(img,lab) in process:
            img = img/255.0
            img = np.expand_dims(img,axis=0)
            
            with tf.GradientTape() as tape:
                pred = vgg(img)
                loss = categoricalCrossentropyLoss(lab,pred)
                mean_loss += loss/ds_len_
                batch_loss += loss
            if ((idx+1) % batch_size == 0):
                gd = tape.gradient(batch_loss,vgg.trainable_variables)
                opt.apply_gradients(zip(gd,vgg.trainable_variables))
                batch_loss = tf.constant(0.,dtype=tf.float32)
            process.set_description_str('epochs {} - step {}'.format(str(epoch+1),str(idx+1)))
            process.set_postfix_str('cur loss {:.3f} mean loss {:.3f}'.format(loss.numpy(),mean_loss.numpy()))
        process.close()
        if (global_loss > mean_loss.numpy()):
            print ('save weights - {:.3f}'.format(mean_loss.numpy()))
            vgg.save_weights(weights_save_dir)
        global_loss = min(global_loss,mean_loss)

        # val metrics
        P = 0
        ts_process = tqdm.tqdm(enumerate(ts_ds),total=ts_ds_len_)
        for ts_idx,(ts_img,ts_lab) in ts_process:
            ts_img = ts_img/255.0
            ts_img = np.expand_dims(ts_img,axis=0)
            ts_pred = vgg(ts_img)
            if (ts_pred[0].numpy().argmax() == ts_lab[0].argmax()):
                P+=1
        ts_process.close()
        print ('test acc - {:.2f}%'.format(P/ts_ds_len_*100))


def metrics_test():
    dl = DataLoader(para.test_data)
    imgs,labmasks,labs = dl.getDateSets()
    ds_len_ = dl.getDataSize()
    img_files = dl.img_files
    lab_files = dl.lab_files
    ds = []
    for idx in range(ds_len_):
        ds.append(np.array([imgs[idx],labmasks[idx],labs[idx]]))
    # print (imgs.shape,classLab.shape)
    ds = np.array(ds)
    print (ds.shape,ds[0].shape,ds[0][0].shape,ds[0][1].shape,ds[0][2].shape)

    vgg = VGG_16(feature_out=True,dim_out=para.classNums,pretrain_load=None)

    weights_save_dir = "./ckpt/classifier-vgg-15-40-retrain/classifier-vgg-15-40-retrain"
    # weights_save_dir = "./ckpt/classifier-vgg-15-constrainted-by-knowledge-inject/classifier-vgg-15-constrainted-by-knowledge-inject"
    # weights_save_dir = "./ckpt/classifier-vgg-15-constrainted-by-labelmask-union/classifier-vgg-15-constrainted-by-labelmask-union"
   
    vgg.load_weights(weights_save_dir)

    res_dir = './result/vgg-baseline-40'
    # res_dir = './result/vgg-labelmask'
    # res_dir = './result/vgg-knowledge-inject'
    # res_dir = './result/vgg-labelmask-union'
    os.makedirs(res_dir,exist_ok=True)
    
    P = 0
    N = 0
    
    foreground = 0
    background = 0
    
    process = tqdm.tqdm(enumerate(ds),total=ds_len_)
    for idx,(img,lab_mask,lab) in process:
        img = img/255.0
        lab_mask = lab_mask/255.0
        img = np.expand_dims(img,axis=0)
        lab_mask = np.expand_dims(lab_mask,axis=0)
        with tf.GradientTape() as t:
            cam3,cam_mask,preds = compute_heatmap(vgg,img,np.argmax(lab.squeeze(axis=0)),size_holder,tape=t,training=False)
            cam = np.where(cam3 > np.percentile(cam3, 100 - keep_percent), 1, 0)
            cam_mask = cam_mask.numpy().squeeze()
            cam_mask = cv.resize(cam_mask,size_holder)
            # cam_mask = np.where(cam_mask > np.percentile(cam_mask, 100 - keep_percent), 1, 0)
            seg,heatmap = overlay_gradCAM(img.squeeze(axis=0),cam3=cam3,cam=cam)
        # GradResShow(img,heatmap,seg)
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
        
        cv.imwrite(res_dir+'/'+lab_files[idx].split('/')[-1],heatmap)

    process.close()
    print ('Test Acc : {:.2f}%'.format(P/ds_len_*100))
    print ('Test Foreground Reasoning Rate : {:.2f}%'.format(foreground/(background)*100))

def main():
    # step1:  baseline train:
    # first 20 epochs: first 20 eps block all vgg's layers(only train classfier head)
    # 20--40 epochs: After training the model for 20 epochs, further train it for 20 epochs using only accuracy loss, to compare with the guided trained model
    #classifier_train_without_constraint()

    #step2(20 epochs): Guided training is carried out on the basis of the model trained for 20 epochs
    #mode : HC / KI / KI_HC 
    classifier_train_with_constraint(mode = "HC")
    
    #step3: metric test
    metrics_test()
     
    return 
    
if __name__ == "__main__":
    main()
