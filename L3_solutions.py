import torch
def sol1(x): return x.permute((2,0,1))
def sol2(x): return x.type(torch.FloatTensor).mean(dim=2)
def sol3(x): return x.t().reshape(-1)
def sol4(x): return x.view(x.shape[0]//2,2,-1,2,3)[:,0,:,0,:]
def sol5(x): return x[:,None,:,None,...].expand(-1,2,-1,2,-1).reshape(x.shape[0]*2, x.shape[1]*2,-1)
def sol6(x,batch): return torch.abs(x-batch).view(-1,28*28).sum(dim=1).argmin()