#!/usr/bin/env python
from cProfile import label
import numpy as np
from sklearn.manifold import TSNE
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
import torchvision.utils as vutils
import h5py
import sys, os
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('--infile',action='store',type=str,required=True,help='input data file')
parser.add_argument('--trained',action='store',type=str,required=True,help='trained model path')
parser.add_argument('--batch', action='store', type=int, default=128, help='Batch size')
parser.add_argument('--ldim', action='store', type=int, default=8, help='latent dim')
parser.add_argument('--nClusters',action='store',type=int,default=10,help='number of clusters')
parser.add_argument('-o', '--outdir', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--shuffle', action='store', type=bool, default=True, help='Shuffle batches for each epochs')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')

args =parser.parse_args()

latent = args.ldim
nClusters = args.nClusters
#nClusters = 13
batch = args.batch
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available else {}

import model as MyModel
import h5py
from torch.utils.data import DataLoader, Dataset, TensorDataset

data_list = ['../data/normal_dataset.h5','../data/Anomaly_dataset.h5']
#data_list = ['../data/normal_dataset.h5']
imageList=[]
labelList=[]
for file_path in data_list:
    print('Loading data from ', file_path)
    dataset = h5py.File(file_path,'r',libver='latest',swmr=True)
    FimageList=[]
    FlabelList=[]
    for gName,group in dataset.items():
        for dName,data in group.items():
            if dName == 'images':
                FimageList.append(data)
            elif dName == 'labels':
                FlabelList.append(data)

    if len(FimageList) >= 2:
        #print("More than 2 gropus in File")
        image_concat = []
        for i in range(0,len(FimageList)):
            image_concat.append(FimageList[i][:])
        imageList.append(np.concatenate(image_concat))
        label_concat = []
        for i in range(0,len(FlabelList)):
            label_concat.append(FlabelList[i][:])
        labelList.append(np.concatenate(label_concat))
    else:
        imageList.append(FimageList[0][:])
        labelList.append(FlabelList[0][:])
imageList = np.concatenate(imageList)
labelList = np.concatenate(labelList)
print('input image shape : ',imageList.shape)
print('input label shape : ',labelList.shape)
ds = TensorDataset(torch.tensor(imageList),torch.tensor(labelList))
length = [int(len(ds)*0.7),int(len(ds)*0.2)]
length.append(len(ds)-sum(length))

trnSet,valSet,tstSet=torch.utils.data.random_split(ds,length)

#train Loader
train_loader = DataLoader(trnSet, batch_size=batch, shuffle=True, **kwargs)
#val Loader
val_loader = DataLoader(valSet, batch_size=batch, shuffle=False, **kwargs)
#test Loader
test_loader = DataLoader(tstSet, batch_size=batch, shuffle=False, **kwargs)

net = MyModel.MyModel(latent_dim = latent,nClusters=nClusters)
model = net.cuda()

model.load_state_dict(torch.load(args.trained))

recon_e = []
kl_div = []
test_la = []
from tqdm import tqdm
model.eval()
print("Calculating Errors")
for i,(data,y_) in tqdm(enumerate(test_loader)):
	#if 'mlp' in args.model:
		#data = data.view(-1,784)
	data = data.cuda()
	data = Variable(data,volatile=True)

	recon , mu , logvar = model(data)

	for j in range(0,len(data)):
		kld = model.KLD(mu[j].unsqueeze(0),logvar[j].unsqueeze(0))
		re = model.RE(data[j],recon[j])
		re = re.detach().cpu().numpy()
		kld = kld.detach().cpu().numpy()
		#print(re)
		#print(kld)
		recon_e.append(re)
		kl_div.append(kld)
		test_la.append(y_[j])
	
import pandas as pd
df_label = pd.DataFrame(data=test_la,columns=['Label'])
df_RE = pd.DataFrame(data=recon_e,columns=['RE'])
df_KLD = pd.DataFrame(data=kl_div,columns=['KLD'])

df_err = pd.merge(df_RE,df_KLD,left_index=True,right_index=True)
df = pd.merge(df_label,df_err,left_index=True,right_index=True)

df = df.sort_values(by=['Label'],ascending=True)
df = df.reset_index(drop=True)
df.to_csv(args.outdir+'/TestSetErrorInfo.csv')
