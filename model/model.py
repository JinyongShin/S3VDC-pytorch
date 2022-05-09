#!/usr/bin/env python
import torch
import torch.utils.data
import torch.nn as nn
from torch.autograd import Variable 
import numpy as np

class MyModel(nn.Module):
	def __init__(self,latent_dim,nClusters):
		super(MyModel,self).__init__()
		
		self.latent_dim = latent_dim
		self.nClusters = nClusters
		#self.pi_=nn.Parameter(torch.FloatTensor(self.nClusters,).fill_(1)/self.nClusters,requires_grad=True)
		self.pi_=nn.Parameter(torch.zeros(self.nClusters))
		self.mu_c = nn.Parameter(torch.randn(self.nClusters, self.latent_dim))
		self.logvar_c = nn.Parameter(torch.randn(self.nClusters, self.latent_dim))

		self.mu = nn.Linear(256,latent_dim)
		self.logvar = nn.Linear(256,latent_dim)

		self.fc4 = nn.Linear(latent_dim, 256)

		self.relu = nn.ReLU()
		self.sigmoid = nn.Sigmoid()

		self.encoder = nn.Sequential(
			nn.Linear(784,2000),
			nn.ReLU(),
			nn.Linear(2000,500),
			nn.ReLU(),
			nn.Linear(500,500),
			nn.ReLU(),
			nn.Linear(500,256)
		)

		self.decoder = nn.Sequential(
			nn.Linear(256,500),
			nn.ReLU(),
			nn.Linear(500,500),
			nn.ReLU(),
			nn.Linear(500,2000),
			nn.ReLU(),
			nn.Linear(2000,784),
			nn.Sigmoid(),
		)

	def encode(self, x):
		h= self.encoder(x)
		return self.mu(h), self.logvar(h)

	def decode(self, z):
		h3 = self.relu(self.fc4(z))
		return self.decoder(h3)

	def reparameterize(self,mu, logvar):
		"""Reparameterization trick.
		"""
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		z = mu + eps * std
		return z

	def forward(self, x):
		mu, logvar = self.encode(x.view(-1,784))
		z = self.reparameterize(mu,logvar)
		decoded = self.decode(z)
		return decoded, mu, logvar

	@property
	def weights(self):
		return torch.softmax(self.pi_,dim=0)

	def RE(self,x,recon_x):
		return torch.nn.functional.binary_cross_entropy(recon_x.view(-1,784),x.view(-1,784),reduction='sum')
	
		
	def KLD(self,mu,logvar,beta):
		det = 1e-9
		z = self.reparameterize(mu,logvar).unsqueeze(1)
		#print(z.shape)							#Batch_size * 1 * ldim
		mu_c = self.mu_c.unsqueeze(0)			#1 * nC * ldim
		logvar_c = self.logvar_c.unsqueeze(0)	#1 * nC * ldim
		pc = self.weights.unsqueeze(0)			#1 * nC
		
		h = z - mu_c							#Batch * nC * ldim
		h = h * h / (torch.exp(logvar_c)+det)  #Batch * nC * ldim
		h = torch.sum(h,dim=2) 					#Batch * nC
		
		ln2pi = torch.log(torch.ones(self.nClusters)*2*np.pi).cuda()
		#ln2pi = torch.log(torch.ones(self.nClusters)*2*np.pi)
		h = ln2pi + h

		ln_pzc = -0.5 * (torch.sum(logvar_c,dim=2) + h) #Batch * nC

		lambda_ = 50
		tr_ln_pzc = (ln_pzc - torch.max(ln_pzc,dim=1,keepdim=True)[0]) / (torch.max(ln_pzc,dim=1,keepdim=True)[0] - torch.min(ln_pzc,dim=1,keepdim=True)[0])
		tr_ln_pzc = lambda_ * tr_ln_pzc			#Batch * nC

		pc_pzc = pc * torch.exp(tr_ln_pzc)+det		#Batch * nC
		qcx = pc_pzc / torch.sum(pc_pzc,dim=1,keepdim=True) #Batch * nC
		
		#logvar (Batch * ldim)
		#mu     (Batch * ldim)
		h2 = torch.exp(logvar.unsqueeze(1)) + pow((mu.unsqueeze(1)-mu_c),2) + det #Batch * nC * ldim
		h2 = h2 / (torch.exp(logvar_c)+det) #Batch * nC * ldim
		h2 = torch.sum((h2 + logvar_c),2)	#Batch * nC

		#qcx * h2 (Batch * nC)
		loss = 0.5 * torch.sum((qcx * h2),dim=1) 			#Batch,
		#(qcx * torch.log(pc/qcx)) (Batch * nC)
		loss -= torch.sum(qcx * torch.log(pc/qcx),dim=1)	#Batch,
		loss -= 0.5 * torch.sum((1+logvar),dim=1)			#Batch,
		loss = torch.mean(loss)		
		return loss * beta

	def loss_function(self,x,recon_x,mu,log_var,beta):
		RE = self.RE(x,recon_x)/len(recon_x)
		KLD = self.KLD(mu,log_var,beta)

		loss = RE + KLD
		loss = loss #/ len(recon_x)
		return loss
