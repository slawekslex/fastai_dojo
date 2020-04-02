def sol1(x): return x.permute((2,0,1))
def sol2(x): return x.type(torch.FloatTensor).mean(dim=2)
def sol3(x): return x.t().reshape(-1)