import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np


nc =1
ndf = 64
ngf = 64
class MyModel(nn.Module):
	"""Variational Deep Embedding(VaDE).
	Args:
		nClusters (int): Number of clusters.
		data_dim (int): Dimension of observed data.
		latent_dim (int): Dimension of latent space.
	"""
	def __init__(self, nClusters=13, latent_dim=10):
		super(MyModel, self).__init__()
		
		self.data_dim = 28*28
		self.latent_dim = latent_dim
		self.nClusters = nClusters
		
		#self.pi_ = Parameter(torch.zeros(self.nClusters))
		self.pi_ = Parameter(torch.ones(self.nClusters)/self.nClusters)
		#self.mu_c = Parameter(torch.zeros(self.nClusters, self.latent_dim))
		#self.logvar_c = Parameter(torch.zeros(self.nClusters, self.latent_dim))
		self.mu_c = Parameter(torch.randn(self.nClusters, self.latent_dim))
		self.logvar_c = Parameter(torch.randn(self.nClusters, self.latent_dim))

		self.encoder = nn.Sequential(
			# input is (nc) x 28 x 28
			nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf) x 14 x 14
			nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 2),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*2) x 7 x 7
			nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
			nn.BatchNorm2d(ndf * 4),
			nn.LeakyReLU(0.2, inplace=True),
			# state size. (ndf*4) x 4 x 4
			nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
			# nn.BatchNorm2d(1024),
			nn.LeakyReLU(0.2, inplace=True),
			# nn.Sigmoid()
		)

		self.encoder_fc = torch.nn.Linear(1024, 512)
		self.encoder_mu = torch.nn.Linear(512, self.latent_dim)
		self.encoder_logvar = torch.nn.Linear(512, self.latent_dim)		   


		self.decoder = nn.Sequential(
			# input is Z, going into a convolution
			nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False),
			nn.BatchNorm2d(ngf * 8),
			nn.ReLU(True),
			# state size. (ngf*8) x 4 x 4
			nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 4),
			nn.ReLU(True),
			# state size. (ngf*4) x 8 x 8
			nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
			nn.BatchNorm2d(ngf * 2),
			nn.ReLU(True),
			# state size. (ngf*2) x 16 x 16
			nn.ConvTranspose2d(ngf * 2,		nc, 4, 2, 1, bias=False),
			# nn.BatchNorm2d(ngf),
			# nn.ReLU(True),
			# state size. (ngf) x 32 x 32
			# nn.ConvTranspose2d(	 ngf,	   nc, 4, 2, 1, bias=False),
			# nn.Tanh()
			nn.Sigmoid()
			# state size. (nc) x 64 x 64
		)
 
		self.relu = nn.ReLU()
		self.decoder_in1 = nn.Linear(self.latent_dim,512)
		self.decoder_in2 = nn.Linear(512,1024)
	
	def encode(self, x):
		h = self.encoder(x)
		#h = self.encoder(x.view(-1,self.data_dim))
		h = self.encoder_fc(h.view(-1,1024))
		mu = self.encoder_mu(h)
		logvar = self.encoder_logvar(h)
		return mu, logvar

	def decode(self, z):
		h=self.relu(self.decoder_in1(z))
		h=self.decoder_in2(h)
		h=h.view(-1,1024,1,1)
		return self.decoder(h)

	def reparameterize(self,mu, logvar):
		"""Reparameterization trick.
		"""
		std = torch.exp(0.5 * logvar)
		eps = torch.randn_like(std)
		z = mu + eps * std
		return z
	
	def forward(self, x):
		#x = x.view(-1,self.data_dim)
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		recon_x = self.decode(z)
		return recon_x, mu, logvar

	#@property
	#def weights(self):
	#	return torch.softmax(self.pi_,dim=0)

	def RE(self,x,recon_x):
		return nn.functional.binary_cross_entropy(recon_x.view(-1,784),x.view(-1,784),reduction='sum')
	
	def KLD(self,mu,logvar,beta=1):
		z = self.reparameterize(mu,logvar).unsqueeze(1)
		#Calculate ln(p(z|c)) 
		G=[]
		for c in range(self.nClusters):
			ln_pzc = -0.5*(self.logvar_c[c:c+1,:]+np.log(2*np.pi)+(pow((z-self.mu_c[c:c+1,:]),2)/torch.exp(self.logvar_c[c:c+1,:])))
			G.append(ln_pzc)
		#print(torch.cat(G,1).shape)
		ln_pzc = torch.cat(G,1)
		ln_pzc = torch.sum(ln_pzc,2)
		#print(ln_pzc.shape)

		#re-scale each element in ln_pzc to the range of [-lambda,0] to avoid NaN Loss
		lambda_ = 50
		tr_ln_pzc = lambda_*((ln_pzc - torch.max(ln_pzc,1,keepdim=True)[0])/(torch.max(ln_pzc,1,keepdim=True)[0]-torch.min(ln_pzc,1,keepdim=True)[0]))
		
		# approximate q(c|x) from p(c|z)
		pc_pzc = torch.exp(torch.log(self.pi_.unsqueeze(0)) + tr_ln_pzc)
		qcx = pc_pzc / torch.sum(pc_pzc,dim=1,keepdim=True)

		# Loss
		h = logvar.exp().unsqueeze(1) + (mu.unsqueeze(1) - self.mu_c).pow(2)
		#print(h.shape) # batch_size * nClusters * ldim
		h = torch.sum(self.logvar_c + (h / self.logvar_c.exp()), dim=2)
		#print(h.shape) # batch_size * nClusters
		loss = 0.5*torch.mean(torch.sum(qcx*h,dim=1))
		loss -= torch.mean(torch.sum(qcx*(torch.log(self.pi_.unsqueeze(0))-torch.log(qcx)),dim=1))
		loss -= 0.5*torch.mean(torch.sum((1+logvar),dim=1))
		#loss = 0.5*torch.mean(torch.sum(qcx*torch.sum(self.logvar_c.unsqueeze(0)+ \
                #torch.exp(logvar.unsqueeze(1)-self.logvar_c.unsqueeze(0))+ \
                #(mu.unsqueeze(1)-self.mu_c.unsqueeze(0)).pow(2)/torch.exp(self.logvar_c.unsqueeze(0)),2),1))
		#loss -= torch.mean(torch.sum(qcx*torch.log(self.pi_.unsqueeze(0)/(qcx)),1))+0.5*torch.mean(torch.sum(1+logvar,1))
		return loss * beta

	def loss_function(self,x,recon_x,mu,logvar,beta=1):
		RE = self.RE(x,recon_x)
		KLD = self.KLD(mu,logvar,beta)

		loss = RE + KLD
		loss = loss / len(recon_x)
		return loss

