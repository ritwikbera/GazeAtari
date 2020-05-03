import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn import Parameter, Module

'''
h, w : input dimensions for image
d : feature vector size from ResNet backbone output. 
V : number of nodes
outfeatures : dimension of output features

W : anchor feature vectors (one for each node)
weight : graph convolution parameters
'''

class GCU(Module):
    def __init__(self, batch=1, h=16, w=16, d=3, V=8, outfeatures=1):
        super(GCU, self).__init__()
        
        
        self.ht = h
        self.wdth= w
        self.d = d
        self.no_of_vert = V
        self.outfeatures = outfeatures
        self.batch = batch
        self.W = Parameter(torch.Tensor(d,V))
        self.variance = Parameter(torch.Tensor(d,V))
        self.weight = Parameter(torch.Tensor(d, outfeatures))

        self.iden = torch.eye(self.d)
        self.iden = torch.cat((self.iden, self.iden))

        for i in range(int(np.log2(V))):
            self.iden = torch.cat((self.iden,self.iden), dim=1)
        
        self.count = 0
        

    def init_param(self,x):

        vari = x.clone()
        x.detach()

        c = np.reshape(vari.detach().numpy(), (self.ht*self.wdth, self.d))
        kmeans = KMeans(n_clusters=self.no_of_vert, random_state=0).fit(c)

        W = Parameter(torch.Tensor(np.array(kmeans.cluster_centers_)))

        lab = kmeans.labels_
        c_s = np.square(c)

        sig1 = np.zeros((self.d, self.no_of_vert))
        sig2 = np.zeros((self.d, self.no_of_vert))

        count = np.array([0 for i in range(self.no_of_vert)])

        for i in range(len(lab)):
            sig1[:,lab[i]] += np.transpose(c[i])
            sig2[:,lab[i]] += np.transpose(c_s[i])
            count[lab[i]] += 1

        sig2 = sig2/count
        sig1 = sig1/count

        sig1 = np.square(sig1)

        variance = Parameter(torch.Tensor((sig2 - sig1 + 1e-6)))

        return W, variance

    def GraphProject_optim(self, X):

        Adj = torch.Tensor(self.batch, self.no_of_vert, self.no_of_vert)
        Z = torch.Tensor(self.batch, self.d, self.no_of_vert)
        Q = torch.Tensor(self.batch, self.ht*self.wdth, self.no_of_vert)

        X = torch.reshape(X,(self.batch, self.ht*self.wdth, self.d))
        
        zero = torch.Tensor(X.shape).fill_(0)
        new = torch.cat((zero, X), dim=2)

        extend = torch.matmul(new, self.iden)

        W = torch.reshape(self.W, (self.d*self.no_of_vert,))
        variance = torch.reshape(self.variance, (self.d*self.no_of_vert,))

        q = extend - W[None, None, :]
        
        q1 = (q/variance[None, None, :])
        q = q1**2
        q = torch.reshape(q, (self.batch, self.ht*self.wdth, self.d, self.no_of_vert))
        q = torch.sum(q, dim=2)
        q = torch.reshape(q, (self.batch, self.ht*self.wdth, self.no_of_vert))
        Q = q
        
        Q -= torch.min(Q, 2)[0][:, :, None]
        Q = torch.exp(-Q*0.5)
        norm = torch.sum(Q, dim=2)
        Q = torch.div(Q, norm[:, :, None])

        z = torch.reshape(q1, (self.batch, self.d, self.ht*self.wdth, self.no_of_vert))
        z = torch.mul(z,Q)
        z = torch.sum(z, dim=2)
        z = torch.add(z, 10e-8)/torch.add(torch.sum(Q,dim=1), 10e-8)

        norm = torch.sum(z**2, dim=1)
        Z = torch.div(Z,norm)
        Adj = torch.matmul(torch.transpose(Z,1,2), Z)

        return (Q, Z, Adj)


    def GraphReproject(self, Z_o,Q):
        X_new = torch.matmul(Z_o,Q)
        return torch.reshape(X_new, (self.batch, self.outfeatures, self.ht, self.wdth))

    def forward(self, X):
        X = torch.reshape(X,(self.batch, self.ht, self.wdth, self.d)).float()
    
        Q, Z, Adj = self.GraphProject_optim(X)

        out = torch.matmul(torch.transpose(Z,1,2), self.weight)
        out = torch.matmul(Adj, out)
        Z_o = F.relu(out)

        out = self.GraphReproject(Q, Z_o)

        return out

if __name__=='__main__':
	gcu = GCU()
	img = torch.randn(16,16,3)
	out = gcu(img).permute(0,2,3,1)
	print(out.size())