import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import argparse 
import sys
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

import h5py
from torch.utils.data import DataLoader, Dataset, TensorDataset

data_list = ['../../data/normal_dataset.h5','../../data/Anomaly_dataset.h5']
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

import torch.optim as optim
import itertools
from tqdm import tqdm
from torch.autograd import Variable

sys.path.append('../model')
import model as MyModel
#import model.model_test as MyModel

net = MyModel.MyModel(latent_dim = latent,nClusters=nClusters)
model = net.cuda()

model.load_state_dict(torch.load(args.trained))

x_1=[]
y_1=[]

for x , y in iter(test_loader):
    x_1.append(x)
    y_1.append(y)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

x_t = torch.cat(x_1)
y_t = torch.cat(y_1)
recon,mu,logvar = model(x_t.cuda())
#z = model.reparameterize(mu,logvar).detach().cpu().numpy()

c_lst = [plt.cm.nipy_spectral(a) for a in np.linspace(0.0, 1.0, len(np.unique(y_t)))]

import matplotlib
import plotly.express as px
import plotly.io as po
import plotly.graph_objs as go

#RE
RE_A = []
for idx in range(len(x_t)):
    re_ = model.RE(x_t[idx:idx+1,:].cuda(),recon[idx:idx+1,:]).detach().cpu().numpy()
    RE_A.append(re_)

#q(c|x)
z = model.reparameterize(mu,logvar).unsqueeze(1)
det = 1e-9
mu_c = model.mu_c.unsqueeze(0)
logvar_c = model.logvar_c.unsqueeze(0)
pc = torch.softmax(model.pi_,dim=0).unsqueeze(0)

h = z - mu_c
h = h * h / (torch.exp(logvar_c)+det)
h = torch.sum(h,dim=2)

ln2pi = torch.log(torch.ones(model.nClusters)*2*np.pi).cuda()
h = ln2pi + h

ln_pzc = -0.5 * (torch.sum(logvar_c,dim=2) + h)
lambda_ = 50
tr_ln_pzc = (ln_pzc - torch.max(ln_pzc,dim=1,keepdim=True)[0]) / (torch.max(ln_pzc,dim=1,keepdim=True)[0] - torch.min(ln_pzc,dim=1,keepdim=True)[0])
tr_ln_pzc = lambda_ * tr_ln_pzc

pc_pzc = pc * torch.exp(tr_ln_pzc) + det
qcx = pc_pzc / torch.sum(pc_pzc,dim=1,keepdim=True)

#kl-div for each cluster
h2 = torch.exp(logvar.unsqueeze(1)) + pow((mu.unsqueeze(1)-mu_c),2) + det
h2 = h2 / (torch.exp(logvar_c)+det)
h2 = torch.sum((h2 + logvar_c),2)

kld_c = 0.5 * qcx * h2
kld_c -= (qcx * torch.log(pc / qcx))
kld_c -= 0.5 * torch.sum((1+logvar),dim=1).unsqueeze(1)

#Cosine Similarity & distance(= 1-similarity)
cos = nn.CosineSimilarity(dim=0,eps=1e-8)
z1 = z.squeeze(1)
CS = []
CD = []
for v in z1:
    cs_a=np.array([])
    cd_a=np.array([])
    for c in range(nClusters):
        cs = cos(v,model.mu_c[c])
        cd = 1 - cs
        cs_a = np.append(cs_a,cs.detach().cpu().numpy().item())
        cd_a = np.append(cd_a,cd.detach().cpu().numpy().item())
    CS.append(cs_a)
    CD.append(cd_a)

#Mahalanobis distance(MD) & Euclidean Distance(UD)
MD = []
UD = []
for x in z:
    md = np.array([])
    ud = np.array([])
    for i in range(0,nClusters):
        h = x - model.mu_c[i]
        h = pow(h,2)
        ud = np.append(ud,torch.sqrt(torch.sum(h)).detach().cpu().numpy().item())
        h = h / model.logvar_c[i]
        #h = torch.mean(h)
        h = torch.sqrt(torch.sum(h))
        md=np.append(md,h.detach().cpu().numpy().item())
    MD.append(md)
    UD.append(ud)

qcx = qcx.detach().cpu().numpy()
kld_c = kld_c.detach().cpu().numpy()

np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.2f}".format(x)})

n_components = 3
#tsne = TSNE(n_components=n_components)
#tsne = TSNE(n_components=n_components, metric='cosine',perplexity=50,square_distances=True)
tsne = TSNE(n_components=n_components, metric='cosine',perplexity=100,square_distances=True)
z = model.reparameterize(mu,logvar).detach().cpu().numpy()
tsneArr = tsne.fit_transform(z)
c_lst = [matplotlib.colors.colorConverter.to_rgb(plt.cm.nipy_spectral(a)) for a in np.linspace(0.0, 1.0, len(np.unique(y_t)))]
c_list=[]

for i in y_t:
    #print(i)
    c_list.append(c_lst[int(i.data)])

data = go.Scatter3d(
    x = tsneArr[:,0],
    y = tsneArr[:,1],
    z = tsneArr[:,2],
    text=['GT:#{}<br>q(c|x):{}<br>Cluster:{}<br>KL-D:{}<br>min KL-D Cluster:{}\
    <br>mahalanobis distance:{}<br>min mahalanobis distance Cluster:{} \
        <br>Euclidean distance:{}<br>min euclidean distance Cluster:{} \
            <br> Cosine distance:{}<br>min Cosine distance Cluster:{} \
                <br> Reconstruction Error:{}'.format(
        int(a.data),
        str(qcx[b]),int(np.argmax(qcx[b])),
        str(kld_c[b]),np.argmin(kld_c[b]),
        str(MD[b]),np.argmin(MD[b]),
        str(UD[b]),np.argmin(UD[b]),
        str(CD[b]),np.argmin(CD[b]),
        RE_A[b]) for (b,a) in enumerate(y_t)],
    mode = 'markers',
    name = str(int(i)),
    marker = dict(
        size = 2,
        color = c_list)
    
)

fig = go.Figure(data=data)

#fig = px.scatter_3d(tsneArr, x=0,y=1,z=2,color=c_list,labels=y_t)
#fig = px.scatter_3d(tsneArr, x=0,y=1,z=2,color=c_list,name=y_t)
#fig.update_traces(marker_size=2)
#fig.show()
po.write_html(fig, file=args.outdir+'/3dManifold.html')
#po.write_html(fig, file='./s3_full2.html')
print('done')
