import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as tr
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.autograd import Variable
import torchvision.utils as vutils

import argparse
import sys, os
import h5py
import numpy as np

import time
startT = time.time()

parser = argparse.ArgumentParser()
parser.add_argument('--infile',action='store',type=str,required=True,help='input data file')
parser.add_argument('--batch', action='store', type=int, default=128, help='Batch size')
parser.add_argument('--model', action='store', choices=('mlp','conv','test'), default='mlp', help='choice of model')
parser.add_argument('--ldim', action='store', type=int, default=10, help='latent dim')
parser.add_argument('--nClusters',action='store',type=int,default=10,help='number of clusters')
parser.add_argument('-o', '--outdir', action='store', type=str, required=True, help='Path to output directory')
parser.add_argument('--shuffle', action='store', type=bool, default=True, help='Shuffle batches for each epochs')
parser.add_argument('--device', action='store', type=int, default=0, help='device name')

args =parser.parse_args()
if torch.cuda.is_available() and args.device >= 0 : torch.cuda.set_device(args.device)
if not os.path.exists(args.outdir): os.makedirs(args.outdir)

ldim = args.ldim
nC = args.nClusters
batch_size = args.batch
kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available else {}

file_list = args.infile.split(',')

imageList=[]
labelList=[]
for file_path in file_list:
	print('Loading file'+file_path)
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
train_loader = DataLoader(trnSet, batch_size=args.batch, shuffle=args.shuffle, **kwargs)
#validation Loader
val_loader = DataLoader(valSet, batch_size=args.batch, shuffle=False, **kwargs)
#test Loader
test_loader = DataLoader(tstSet, batch_size=args.batch, shuffle=False, **kwargs)

sys.path.append('../model')
if args.model == 'mlp':
	import model as MyModel
elif args.model == 'conv':
	import model_conv as MyModel
elif args.model == 'test':
	import model_test as MyModel
net = MyModel.MyModel(nClusters=nC,latent_dim=ldim)
model = net.cuda()
#model = net
print(model)

# Initial gamma-training 
gamma_ = 0.0005
g_step = 100000

optimizer = optim.Adam(model.parameters(),lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.96)
step = 0
while (step+1) <= g_step:
	for batch_idx,(data,y) in enumerate(train_loader):
		data = Variable(data)
		x = data.cuda()
		#x = data
		n_d = data + torch.FloatTensor(np.random.normal(0,0.000000005,data.shape))
		n_x = n_d.cuda()
		#n_x = n_d

		optimizer.zero_grad()
		recon_batch, mu, logvar = model(n_x)
		#recon_batch, mu, logvar = model(x)
		#print(recon_batch)
		#recon_batch = torch.nan_to_num(recon_batch,nan=0.0)
		loss = model.loss_function(x,recon_batch,mu,logvar,beta=gamma_)
		re = model.RE(x,recon_batch)
		kld = model.KLD(mu,logvar,beta=gamma_)
		#print(x)
		#print(recon_batch)

		loss.backward()
		optimizer.step()
		step+=1

		if (step+1)%100==0:
			print(step+1,'Step Loss : ',loss.data)

		if (step+1)%1000==0:
			scheduler.step()
			print("lr: ", optimizer.param_groups[0]['lr'])

#mini-batch GMM initialization
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
gmm_step = 200
subset = np.random.randint(len(train_loader),size=gmm_step*batch_size)
train_subset_loader = DataLoader(torch.utils.data.Subset(trnSet,subset))
print('Model initialization with ',gmm_step,'*',batch_size,'samples')

Z=[]
Y=[]
with torch.no_grad():
	for i, (data,y) in enumerate(train_loader):
		data = data.view(-1,784).cuda()
		z1, z2 = model.encode(data.cuda())
		#z1, z2 = model.encode(data)
		#assert nn.functional.mse_loss(z1,z2)==0
		Z.append(z1)
		Y.append(y)

Z=torch.cat(Z,0).detach().cpu().numpy()
Y=torch.cat(Y,0).detach().numpy()

kmeans = KMeans(n_clusters=nC, random_state=0)
kmeans.fit(Z)

gmm = GaussianMixture(n_components=nC,covariance_type='diag',max_iter=int(1e+04),means_init = kmeans.cluster_centers_,random_state=100)
#gmm = GaussianMixture(n_components=self.nClusters,covariance_type='diag')

pre = gmm.fit_predict(Z)

model.pi_.data = torch.from_numpy(gmm.weights_).cuda().float()
model.mu_c.data = torch.from_numpy(gmm.means_).cuda().float()
model.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).cuda().float())
#model.pi_.data = torch.log(torch.from_numpy(gmm.weights_).float())
#model.mu_c.data = torch.from_numpy(gmm.means_).float()
#model.logvar_c.data = torch.log(torch.from_numpy(gmm.covariances_).float())
print(model.pi_)
print(torch.sum(model.pi_))
period = 10
beta_step = 9000
static_step = 1000

start = gamma_

B = np.ones(beta_step)
for bs in range(beta_step):
	B[int(bs)] = gamma_ + pow((bs/beta_step),3)

optimizer = optim.Adam(model.parameters(),lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.96)
best_ = 150
for period_ in range(period):
	step = 0
	model.train()
	while step < beta_step:
		L = 0
		beta_ = B[step]
		for batch_idx,(data,y) in enumerate(train_loader):
			data = Variable(data)
			x = data.cuda()
			#x = data

			n_d = data + torch.FloatTensor(np.random.normal(0,0.000000005,data.shape))
			n_x = n_d.cuda()
					
			optimizer.zero_grad()
			recon_batch, mu, logvar = model(n_x)
			loss = model.loss_function(x,recon_batch,mu,logvar,beta=beta_)
						
			loss.backward()
			optimizer.step()
			step +=1
			if step % 1000 == 0:
				print('Period ',period_,step,'beta-step, Loss : ',loss.data)
	step = 0
	while step < static_step:
		L = 0
		for batch_idx,(data,y) in enumerate(train_loader):
			data = Variable(data)
			x = data.cuda()
			#x = data

			n_d = data+torch.FloatTensor(np.random.normal(0,0.000000005,data.shape))
			n_x = n_d.cuda()
			#n_x = n_d
		
			optimizer.zero_grad()
			recon_batch, mu, logvar = model(n_x)
			loss = model.loss_function(x,recon_batch,mu,logvar,beta=1)
						
			loss.backward()
			optimizer.step()
			step +=1
			if step % 1000 == 0:
				print('Period ',period_,step,'static-step, Loss : ',loss.data)
				scheduler.step()
				print("lr: ", optimizer.param_groups[0]['lr'])

	model.eval()
	test_loss = 0
	for i, (data, _) in enumerate(test_loader):
		if torch.cuda.is_available():
			data = data.cuda()
			#data = data
		data = Variable(data)
		recon_batch, mu, logvar = model(data)
		test_loss += model.loss_function(data, recon_batch, mu, logvar,beta=1).data
		if i == 0:
			n = min(data.size(0), 16)
			comparison = torch.cat([data[:n],
								  recon_batch.view(batch_size, 1, 28, 28)[:n]])
	test_loss /= len(test_loader)#.dataset)
	print('====> Test set loss: {:.4f}'.format(test_loss))
	if test_loss.cpu()<best_:
		best_ = test_loss.cpu()
		torch.save(model.state_dict(),args.outdir+'/best_model.pth')
		print(args.outdir+'/best_model.pth Saved')
torch.save(model.state_dict(),args.outdir+'/model.pth')
print(args.outdir+'/model.pth Saved')

print("time:", time.time() - startT,'seconds')
