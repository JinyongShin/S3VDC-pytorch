#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','--indir',action='store',type=str,required=True,help='input dir')
args =parser.parse_args()

df = pd.read_csv(args.indir + '/TestSetErrorInfo.csv')

n_la=[]
n_re=[]
n_kld=[]

an_la=['A1','A2','A3']
an_re=[]
an_kld=[]

for i in range(0,len(np.unique(df['Label'].to_numpy()))):
	if i <= 9:
		n_la.append(str(i))
		n_re.append(df['RE'][df['Label']==i].to_numpy())
		n_kld.append(df['KLD'][df['Label']==i].to_numpy())

	else:
		an_re.append(df['RE'][df['Label']==i].to_numpy())
		an_kld.append(df['KLD'][df['Label']==i].to_numpy())

c_lst = [plt.cm.Set3(a) for a in np.linspace(0.0, 1.0, len(n_la))]

n_re = np.concatenate(n_re)
n_kld = np.concatenate(n_kld)
an_re = np.concatenate(an_re)
an_kld = np.concatenate(an_kld)

from tqdm import tqdm
_x = np.array([])
_y = np.array([])
_z = np.array([])
for _re in tqdm(range(0,600)):
	n_ab = 0
	n_norm = 0
	eff = 0
	for _kld in range(0,40):
		n_ab = 0
		n_norm = 0
		eff = 0
		_x = np.append(_x,_re)
		_y = np.append(_y,_kld)

		mask = (n_re > _re) & (n_kld>_kld)
		n_norm = np.count_nonzero(n_re[mask])
		mask = (an_re > _re) & (an_kld>_kld)
		n_ab = np.count_nonzero(an_re[mask])
		eff = n_ab / np.sqrt(n_norm+n_ab)
		_z = np.append(_z,eff)

_z[np.isnan(_z)] = 1

print(_x[np.argmax(_z)])
print(_y[np.argmax(_z)])
print(_z[np.argmax(_z)])

df = {
	'RE' : _x,
	'KLD' : _y,
	'Eff' : _z
}

import pandas as pd
data = pd.DataFrame(df)
data = data.pivot('RE','KLD','Eff')
#print(data)
print(np.count_nonzero(n_re[n_re>_x[np.argmax(_z)]]),len(n_re))
print(np.count_nonzero(an_re[an_re>_x[np.argmax(_z)]]),len(an_re))
print(np.count_nonzero(n_re[n_re>_x[np.argmax(_z)]])/len(n_re)*100)
print(np.count_nonzero(an_re[an_re>_x[np.argmax(_z)]])/len(an_re)*100)

import matplotlib.pyplot as plt
plt.pcolor(data.columns,data.index,data)
cl = plt.colorbar()
cl.set_label("Expected Value")
plt.title('Expected Value')
plt.ylabel("Reconstruction Error")
plt.xlabel("KL-Div")
#plt.savefig('Cut_optim.png')

print(np.count_nonzero(n_re[n_re>100]),len(n_re))
print(np.count_nonzero(an_re[an_re>100]),len(an_re))
print(np.count_nonzero(n_re[n_re>100])/len(n_re)*100)
print(np.count_nonzero(an_re[an_re>100])/len(an_re)*100)
