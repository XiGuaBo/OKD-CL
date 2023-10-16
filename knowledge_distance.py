import torch
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from model.classfier import *
from model.gradCam import *
import utils.para as para
from utils.dataloader import *
import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.applications.resnet import preprocess_input
import tensorflow.keras as keras
import json
import torch

color_codes = [
    '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF', '#800000', '#008000', '#000080',
    '#FFA500', '#FFC0CB', '#800080', '#FFD700', '#008080', '#FF1493', '#FF4500', '#DC143C', '#FF69B4',
    '#6A5ACD', '#ADFF2F', '#00FF7F', '#CD5C5C', '#F08080', '#FA8072', '#E9967A', '#FFA07A', '#8B0000',
    '#FF8C00', '#FFA500', '#FFD700', '#FFFF00', '#9ACD32', '#32CD32', '#00FF00', '#7FFF00', '#00FF7F',
    '#7CFC00', '#ADFF2F', '#556B2F', '#8FBC8F', '#006400', '#228B22', '#00FF00', '#008000', '#006400',
    '#008080', '#008B8B', '#00FFFF', '#0000FF', '#000080', '#00008B', '#191970', '#4682B4', '#1E90FF',
    '#ADD8E6', '#87CEEB', '#6495ED', '#0000CD', '#00008B', '#4169E1', '#8A2BE2', '#9932CC', '#9400D3',
    '#800080', '#BA55D3', '#800080', '#9370DB', '#4B0082', '#8B008B', '#6B8E23', '#808000', '#556B2F',
    '#6B8E23', '#9ACD32', '#808000', '#BDB76B', '#DAA520', '#FFD700', '#F0E68C', '#FFFFE0', '#FFFACD',
    '#FAFAD2', '#FFEFD5', '#FFE4B5', '#FFE4C4', '#FFDEAD', '#F5DEB3', '#DEB887', '#D2B48C', '#BC8F8F',
    '#F4A460', '#D2691E', '#CD853F', '#B8860B', '#DAA520', '#FF8C00', '#FF7F50', '#FF6347', '#FF4500',
    '#FFA07A', '#FF69B4', '#FF1493', '#C71585', '#DB7093', '#FFC0CB', '#FFB6C1', '#FF69B4', '#FF00FF',
    '#DA70D6', '#BA55D3', '#9370DB', '#8A2BE2'
]

idx_to_name = json.load(open("./idx2name.json",'r'))
knowledge_path = "./knowledge/human_knowledge/"

vgg = VGG_16(feature_out=True,vector_out=True,dim_out=para.classNums,pretrain_load=None)
vgg.load_weights("./ckpt/classifier-vgg-100c-40-retrain/classifier-vgg-100c-40-retrain")

# inception_v3 = Inception_V3(feature_out=True,vector_out=True,dim_out=para.classNums)
# inception_v3.load_weights("./ckpt/classifier-inception_v3-100c-40-retrain/classifier-inception_v3-100c-40-retrain")

# res50 = ResNet_50(feature_out=True,vector_out=True,dim_out=classNums)
# res50.load_weights("./ckpt/classifier-resnet50-100c-40-retrain/classifier-resnet50-100c-40-retrain")

# res101 = ResNet_101(feature_out=True,vector_out=True,dim_out=classNums)
# res101.load_weights("./ckpt/classifier-resnet101-100c-40-retrain/classifier-resnet101-100c-40-retrain")

# res152 = ResNet_152(feature_out=True,vector_out=True,dim_out=classNums)
# res152.load_weights("./ckpt/classifier-resnet152-100c-40-retrain/classifier-resnet152-100c-40-retrain")

# mobilenet_v1 = MobileNet_v1(feature_out=True,vector_out=True,dim_out=para.classNums)
# mobilenet_v1.load_weights("./ckpt/classifier-mobilenet_v1-100c-40-retrain/classifier-mobilenet_v1-100c-40-retrain")

# mobilenet_v2 = MobileNet_v2(feature_out=True,vector_out=True,dim_out=para.classNums)
# mobilenet_v2.load_weights("./ckpt/classifier-mobilenet_v2-100c-40-retrain/classifier-mobilenet_v2-100c-40-retrain")



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

# term_to_name = ["001","002","de Gaulle","Kiev","Garibaldi"]
# terms_chosen = [0,1,2,3,4]

######################### BaseLine Process ##################################
process = tqdm.tqdm(enumerate(ds),total=ds_len_)
for idx,(img,lab_mask,lab) in process:
    
#     # For VGG16 InceptionV3 MobileNetV1-2
    img = img / 255.0 
#     # For ResNet50,101,152
#     # img = keras.applications.resnet.preprocess_input(img)
    
    img = np.expand_dims(img,axis=0)
    tar_category = np.argmax(lab.squeeze(axis=0))
    vector_file = knowledge_path + idx_to_name[str(dl.lab_dict[tar_category])].split(',')[-1].lower().replace(' ','_') + '.pt'
    kv = tf.reduce_mean(torch.load(vector_file),axis=0)

    (pred,convouts,hiddenout) = vgg(img)
    # (pred,convouts,hiddenout) = inception_v3(img)
    # (pred,convouts,hiddenout) = res50(img)
    # (pred,convouts,hiddenout) = res101(img)
    # (pred,convouts,hiddenout) = res152(img)
    # (pred,convouts,hiddenout) = mobilenet_v1(img)
    # (pred,convouts,hiddenout) = mobilenet_v2(img)
    # print(pred.shape)
    distance = para.mseLoss(kv,hiddenout)
    plt.scatter(tar_category,distance,c='#4169E1',s=10 ,alpha=0.7)
    # print(X.shape)
process.close()


######################### KI Model Process ##################################
vgg.load_weights("./ckpt/classifier-vgg-100c-constrainted-by-knowledge-inject/classifier-vgg-100c-constrainted-by-knowledge-inject")
# inception_v3.load_weights("./ckpt/classifier-inception_v3-100c-constrainted-by-knowledge-inject/classifier-inception_v3-100c-constrainted-by-knowledge-inject")
# res50.load_weights("./ckpt/classifier-resnet50-100c-constrainted-by-knowledge-inject/classifier-resnet50-100c-constrainted-by-knowledge-inject")
# res101.load_weights("./ckpt/classifier-resnet101-100c-constrainted-by-knowledge-inject/classifier-resnet101-100c-constrainted-by-knowledge-inject")
# res152.load_weights("./ckpt/classifier-resnet152-100c-constrainted-by-knowledge-inject/classifier-resnet152-100c-constrainted-by-knowledge-inject")
# mobilenet_v1.load_weights("./ckpt/classifier-mobilenet_v1-100c-constrainted-by-knowledge-inject/classifier-mobilenet_v1-100c-constrainted-by-knowledge-inject")
# mobilenet_v2.load_weights("./ckpt/classifier-mobilenet_v2-100c-constrainted-by-knowledge-inject/classifier-mobilenet_v2-100c-constrainted-by-knowledge-inject")
process = tqdm.tqdm(enumerate(ds),total=ds_len_)
for idx,(img,lab_mask,lab) in process:

    # For VGG16 InceptionV3 MobileNetV1-2
    img = img / 255.0 
    # For ResNet50,101,152
    # img = keras.applications.resnet.preprocess_input(img)

    img = np.expand_dims(img,axis=0)
    tar_category = np.argmax(lab.squeeze(axis=0))
    vector_file = knowledge_path + idx_to_name[str(dl.lab_dict[tar_category])].split(',')[-1].lower().replace(' ','_') + '.pt'
    kv = tf.reduce_mean(torch.load(vector_file),axis=0)

    (pred,convouts,hiddenout) = vgg(img)
    # (pred,convouts,hiddenout) = inception_v3(img)
    # (pred,convouts,hiddenout) = res50(img)
    # (pred,convouts,hiddenout) = res101(img)
    # (pred,convouts,hiddenout) = res152(img)
    # (pred,convouts,hiddenout) = mobilenet_v1(img)
    # (pred,convouts,hiddenout) = mobilenet_v2(img)
    # print(pred.shape)
    distance = para.mseLoss(kv,hiddenout)
    plt.scatter(tar_category,distance,c='#CD0000',s=10 ,alpha=0.7)
    # print(X.shape)
process.close()

######################### KI & HC Model Process ##################################
# vgg.load_weights("./ckpt/classifier-vgg-100c-constrainted-by-labelmask-union/classifier-vgg-100c-constrainted-by-labelmask-union")
# # inception_v3.load_weights("./ckpt/classifier-inception_v3-100c-constrainted-by-labelmask-union/classifier-inception_v3-100c-constrainted-by-labelmask-union")
# # res50.load_weights("./ckpt/classifier-resnet50-100c-constrainted-by-labelmask-union/classifier-resnet50-100c-constrainted-by-labelmask-union")
# # res101.load_weights("./ckpt/classifier-resnet101-100c-constrainted-by-labelmask-union/classifier-resnet101-100c-constrainted-by-labelmask-union")
# # res152.load_weights("./ckpt/classifier-resnet152-100c-constrainted-by-labelmask-union/classifier-resnet152-100c-constrainted-by-labelmask-union")
# # mobilenet_v1.load_weights("./ckpt/classifier-mobilenet_v1-100c-constrainted-by-labelmask-union/classifier-mobilenet_v1-100c-constrainted-by-labelmask-union")
# # mobilenet_v2.load_weights("./ckpt/classifier-mobilenet_v2-100c-constrainted-by-labelmask-union/classifier-mobilenet_v2-100c-constrainted-by-labelmask-union")
# process = tqdm.tqdm(enumerate(ds),total=ds_len_)
# for idx,(img,lab_mask,lab) in process:

# #     # For VGG16 InceptionV3 MobileNetV1-2
#     img = img / 255.0 
# #     # For ResNet50,101,152
# #     # img = keras.applications.resnet.preprocess_input(img)
    
#     img = np.expand_dims(img,axis=0)
#     tar_category = np.argmax(lab.squeeze(axis=0))
#     knowledge = np.loadtxt('./txt-99/{}.txt'.format(tar_category))
#     kv = tf.convert_to_tensor([knowledge],dtype=tf.float32)

#     (pred,convouts,hiddenout) = vgg(img)
#     # (pred,convouts,hiddenout) = inception_v3(img)
#     # (pred,convouts,hiddenout) = res50(img)
#     # (pred,convouts,hiddenout) = res101(img)
#     # (pred,convouts,hiddenout) = res152(img)
#     # (pred,convouts,hiddenout) = mobilenet_v1(img)
#     # (pred,convouts,hiddenout) = mobilenet_v2(img)
#     # print(pred.shape)
#     distance = para.mseLoss(kv,hiddenout)
#     plt.scatter(tar_category,distance,c='#A020F0',s=10 ,alpha=0.7)
#     # print(X.shape)
# process.close()


# plt.xticks([])
# plt.yticks([])

# plt.savefig("./knowledge_distance_vgg16", dpi=600)
# plt.savefig("./knowledge_distance_inceptionv3", dpi=600)
# plt.savefig("./knowledge_distance_resnet50", dpi=600)
# plt.savefig("./knowledge_distance_resnet101", dpi=600)
# plt.savefig("./knowledge_distance_resnet152", dpi=600)
# plt.savefig("./knowledge_distance_mobilenetv1", dpi=600)
# plt.savefig("./knowledge_distance_mobilenetv2", dpi=600)

plt.savefig("./union_knowledge_distance_vgg16", dpi=600)
# plt.savefig("./union_knowledge_distance_inceptionv3", dpi=600)
# plt.savefig("./union_knowledge_distance_resnet50", dpi=600)
# plt.savefig("./union_knowledge_distance_resnet101", dpi=600)
# plt.savefig("./union_knowledge_distance_resnet152", dpi=600)
# plt.savefig("./union_knowledge_distance_mobilenetv1", dpi=600)
# plt.savefig("./union_knowledge_distance_mobilenetv2", dpi=600)

#plt.savefig("./guided_knowledge_distance_vgg16", dpi=600)
# plt.savefig("./guided_knowledge_distance_inceptionv3", dpi=600)
# plt.savefig("./guided_knowledge_distance_resnet50", dpi=600)
# plt.savefig("./guided_knowledge_distance_resnet101", dpi=600)
# plt.savefig("./guided_knowledge_distance_resnet152", dpi=600)
# plt.savefig("./guided_knowledge_distance_mobilenetv1", dpi=600)
# plt.savefig("./guided_knowledge_distance_mobilenetv2", dpi=600)









