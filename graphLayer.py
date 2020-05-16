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
        self.W.requires_grad = True
        self.W.retain_grad()
        self.variance = Parameter(torch.Tensor(d,V))
        self.variance.requires_grad = True
        self.variance.retain_grad()
        self.weight = Parameter(torch.ones(d, outfeatures))
        self.weight.requires_grad = True
        self.device = 'cpu'

        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.W)
        torch.nn.init.xavier_uniform_(self.variance)
        # print(self.weight)

        self.weight.register_hook(self.save_grad('weight_g'))
        self.W.register_hook(self.save_grad('W'))
        self.variance.register_hook(self.save_grad('variance'))

        self.iden = torch.eye(self.d)
        self.iden = torch.cat((self.iden, self.iden))

        for i in range(int(np.log2(V))):
            self.iden = torch.cat((self.iden,self.iden), dim=1)
        
        self.count = 0
        self.initialized = False
        self.grads = {}

    def save_grad(self, name):
        def hook(grad):
            self.grads[name] = grad
            assert not (grad == torch.zeros(grad.size())).all()
            # print(name, grad.size(), (grad == torch.zeros(grad.size())).all())
        return hook

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

        # print(W, variance)

        self.W.data = W.data
        self.variance.data = variance.data

    def GraphProject_optim(self, X):

        Adj = torch.Tensor(self.batch, self.no_of_vert, self.no_of_vert).to(self.device)
        Z = torch.Tensor(self.batch, self.d, self.no_of_vert).to(self.device)
        Q = torch.Tensor(self.batch, self.ht*self.wdth, self.no_of_vert).to(self.device)

        X = torch.reshape(X,(self.batch, self.ht*self.wdth, self.d))
        
        zero = torch.Tensor(X.shape).fill_(0).to(self.device)
        new = torch.cat((zero, X), dim=2)

        extend = torch.matmul(new, self.iden.to(self.device))

        W = torch.reshape(self.W, (self.d*self.no_of_vert,))
        variance = torch.reshape(self.variance, (self.d*self.no_of_vert,))

        # W.register_hook(self.save_grad('W'))
        # variance.register_hook(self.save_grad('variance'))

        q = extend - W[None, None, :]


        q1 = (q/variance[None, None, :])
        q = q1**2
        q = torch.reshape(q, (self.batch, self.ht*self.wdth, self.d, self.no_of_vert))
        q = torch.sum(q, dim=2)
        q = torch.reshape(q, (self.batch, self.ht*self.wdth, self.no_of_vert))
        Q = q

        # q.register_hook(self.save_grad('q'))
        
        Q = Q.refine_names('B','HW','V')
        Q -= torch.min(Q, 'V')[0].align_as(Q) # backprop bottleneck point

        # Q.register_hook(self.save_grad('Q'))

        Q = torch.exp(-Q*0.5)
        norm = torch.sum(Q, dim='V')
        Q = torch.div(Q, norm.align_as(Q))

        z = torch.reshape(q1, (self.batch, self.d, self.ht*self.wdth, self.no_of_vert))

        z = z.refine_names('B','d','HW', 'V')
        Q = Q.align_as(z)

        z = torch.mul(z,Q)

        z = torch.sum(z, dim='HW')
        z = torch.add(z, 10e-8)/torch.add(torch.sum(Q,dim='HW'), 10e-8)

        # z.register_hook(self.save_grad('z'))

        norm = torch.sum(z**2, dim='d')
        Z = torch.div(z,norm.align_as(z))
        Z = Z.rename(None)

        Adj = torch.matmul(torch.transpose(Z,1,2), Z)

        # Z.register_hook(self.save_grad('Z'))
        # Adj.register_hook(self.save_grad('Adj'))


        return (Q, Z, Adj)


    def GraphReproject(self, Z_o, Q):
        Q = torch.sum(Q, dim='d')
        assert Q.names == ('B','HW','V') and Z_o.names == ('B','V','d_out')

        X_new = torch.matmul(Q, Z_o)
        assert X_new.names == ('B','HW','d_out')

        return X_new.unflatten('HW', (('H',self.ht),('W',self.wdth))).rename(None)

    def forward(self, X):
        self.batch = X.size(0)
        self.device = 'cpu' if X.get_device() == -1 else 'cuda:{}'.format(X.get_device())

        if not self.initialized:
            self.init_param(X[0]) # one input from batch
            self.initialized = True

        X = torch.reshape(X,(self.batch, self.ht, self.wdth, self.d)).float()
    
        Q, Z, Adj = self.GraphProject_optim(X)

        out = torch.matmul(torch.transpose(Z,1,2), self.weight[None,:,:])
        out = torch.matmul(Adj, out)
        Z_o = F.relu(out)
        Z_o = Z_o.refine_names('B','V','d_out')

        # Z_o.register_hook(self.save_grad('Z_o'))

        out = self.GraphReproject(Z_o, Q).permute(0,3,1,2) # permute channel dimension

        return out

