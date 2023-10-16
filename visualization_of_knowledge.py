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

# vgg = VGG_16(feature_out=True,vector_out=True,dim_out=para.classNums,pretrain_load=None)
# vgg.load_weights("./ckpt/classifier-vgg-100c-40-retrain/classifier-vgg-100c-40-retrain")
# vgg.load_weights("./ckpt/classifier-vgg-100c-constrainted-by-knowledge-inject/classifier-vgg-100c-constrainted-by-knowledge-inject")
# vgg.load_weights("./ckpt/classifier-vgg-100c-constrainted-by-labelmask-union/classifier-vgg-100c-constrainted-by-labelmask-union")

# inception_v3 = Inception_V3(feature_out=True,vector_out=True,dim_out=para.classNums)
# inception_v3.load_weights("./ckpt/classifier-inception_v3-100c-40-retrain/classifier-inception_v3-100c-40-retrain")
# inception_v3.load_weights("./ckpt/classifier-inception_v3-100c-constrainted-by-knowledge-inject/classifier-inception_v3-100c-constrainted-by-knowledge-inject")
# inception_v3.load_weights("./ckpt/classifier-inception_v3-100c-constrainted-by-labelmask-union/classifier-inception_v3-100c-constrainted-by-labelmask-union")

# res50 = ResNet_50(feature_out=True,vector_out=True,dim_out=classNums)
# res50.load_weights("./ckpt/classifier-resnet50-100c-40-retrain/classifier-resnet50-100c-40-retrain")
# res50.load_weights("./ckpt/classifier-resnet50-100c-constrainted-by-knowledge-inject/classifier-resnet50-100c-constrainted-by-knowledge-inject")
# res50.load_weights("./ckpt/classifier-resnet50-100c-constrainted-by-labelmask-union/classifier-resnet50-100c-constrainted-by-labelmask-union")

# res101 = ResNet_101(feature_out=True,vector_out=True,dim_out=classNums)
# res101.load_weights("./ckpt/classifier-resnet101-100c-40-retrain/classifier-resnet101-100c-40-retrain")
# res101.load_weights("./ckpt/classifier-resnet101-100c-constrainted-by-knowledge-inject/classifier-resnet101-100c-constrainted-by-knowledge-inject")
# res101.load_weights("./ckpt/classifier-resnet101-100c-constrainted-by-labelmask-union/classifier-resnet101-100c-constrainted-by-labelmask-union")

# res152 = ResNet_152(feature_out=True,vector_out=True,dim_out=classNums)
# res152.load_weights("./ckpt/classifier-resnet152-100c-40-retrain/classifier-resnet152-100c-40-retrain")
# res152.load_weights("./ckpt/classifier-resnet152-100c-constrainted-by-knowledge-inject/classifier-resnet152-100c-constrainted-by-knowledge-inject")
# res152.load_weights("./ckpt/classifier-resnet152-100c-constrainted-by-labelmask-union/classifier-resnet152-100c-constrainted-by-labelmask-union")

# mobilenet_v1 = MobileNet_v1(feature_out=True,vector_out=True,dim_out=para.classNums)
# mobilenet_v1.load_weights("./ckpt/classifier-mobilenet_v1-100c-40-retrain/classifier-mobilenet_v1-100c-40-retrain")
# mobilenet_v1.load_weights("./ckpt/classifier-mobilenet_v1-100c-constrainted-by-knowledge-inject/classifier-mobilenet_v1-100c-constrainted-by-knowledge-inject")
# mobilenet_v1.load_weights("./ckpt/classifier-mobilenet_v1-100c-constrainted-by-labelmask-union/classifier-mobilenet_v1-100c-constrainted-by-labelmask-union")

mobilenet_v2 = MobileNet_v2(feature_out=True,vector_out=True,dim_out=para.classNums)
mobilenet_v2.load_weights("./ckpt/classifier-mobilenet_v2-100c-40-retrain/classifier-mobilenet_v2-100c-40-retrain")
# mobilenet_v2.load_weights("./ckpt/classifier-mobilenet_v2-100c-constrainted-by-knowledge-inject/classifier-mobilenet_v2-100c-constrainted-by-knowledge-inject")
# mobilenet_v2.load_weights("./ckpt/classifier-mobilenet_v2-100c-constrainted-by-labelmask-union/classifier-mobilenet_v2-100c-constrainted-by-labelmask-union")

jfs = open('./idx2name.json','r')
name_dict = json.load(jfs)
jfs.close()

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
X = np.zeros(shape=(2970,1024))

process = tqdm.tqdm(enumerate(ds),total=ds_len_)
for idx,(img,lab_mask,lab) in process:
    # plt.rcParams['font.sans-serif']=['SimHei']  # 用来正常显示中文标签 
    img = img / 255.0 
    # img = preprocess_input(img)
    img = np.expand_dims(img,axis=0)

    # (pred,convouts,hiddenout) = vgg(img)
    # (pred,convouts,hiddenout) = inception_v3(img)
    # (pred,convouts,hiddenout) = res50(img)
    # (pred,convouts,hiddenout) = res101(img)
    # (pred,convouts,hiddenout) = res152(img)
    # (pred,convouts,hiddenout) = mobilenet_v1(img)
    (pred,convouts,hiddenout) = mobilenet_v2(img)
    # print(pred.shape)
    X[idx,:] = hiddenout.numpy()
    # print(X.shape)
process.close()
# plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=5)

tsne = TSNE(n_components=2, n_iter=1000,random_state = 8)
embedding_2d = tsne.fit_transform(X)
print(embedding_2d.shape)

process = tqdm.tqdm(enumerate(ds),total=ds_len_)
for idx,(img,lab_mask,lab) in process:
    # if (dl.lab_dict[np.argmax(lab)] < 92):
    #     plt.scatter(embedding_2d[idx,0], embedding_2d[idx,1],c='b',s=10)
    # elif (dl.lab_dict[np.argmax(lab)] < 217):
    #     plt.scatter(embedding_2d[idx,0], embedding_2d[idx,1],c='r',s=10)
    # else:
    #     plt.scatter(embedding_2d[idx,0], embedding_2d[idx,1],c='g',s=10)
    plt.scatter(embedding_2d[idx,0], embedding_2d[idx,1],c=color_codes[np.argmax(lab)] ,s=10 ,alpha=0.7)
    # plt.legend()
# plt.show()

# plt.title('ResNet152(Base) Feature Vector Visualization',fontsize=18)
# plt.title('ResNet152(KI) Feature Vector Visualization',fontsize=18)
# plt.title('ResNet152(KI & HC) Feature Vector Visualization',fontsize=18)

plt.xticks([])
plt.yticks([])

# plt.savefig("./distribute_opt_union_vgg16", dpi=600)
# plt.savefig("./distribute_opt_ki_vgg16", dpi=600)
# plt.savefig("./distribute_base_vgg16", dpi=600)

# plt.savefig("./distribute_opt_union_inceptionv3", dpi=600)
# plt.savefig("./distribute_opt_ki_inceptionv3", dpi=600)
# plt.savefig("./distribute_base_inceptionv3", dpi=600)

# plt.savefig("./distribute_opt_union_resnet50", dpi=600)
# plt.savefig("./distribute_opt_ki_resnet50", dpi=600)
# plt.savefig("./distribute_base_resnet50", dpi=600)

# plt.savefig("./distribute_opt_union_resnet101", dpi=600)
# plt.savefig("./distribute_opt_ki_resnet101", dpi=600)
# plt.savefig("./distribute_base_resnet101", dpi=600)

# plt.savefig("./distribute_opt_union_resnet152", dpi=600)
# plt.savefig("./distribute_opt_ki_resnet152", dpi=600)
# plt.savefig("./distribute_base_resnet152", dpi=600)

# plt.savefig("./distribute_opt_union_mobilenetv1", dpi=600)
# plt.savefig("./distribute_opt_ki_mobilenetv1", dpi=600)
# plt.savefig("./distribute_base_mobilenetv1", dpi=600)

# plt.savefig("./distribute_opt_union_mobilenetv2", dpi=600)
# plt.savefig("./distribute_opt_ki_mobilenetv2", dpi=600)
plt.savefig("./distribute_base_mobilenetv2", dpi=600)













