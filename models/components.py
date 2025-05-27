import torch, torch.nn as nn, torch.nn.functional as F

class GraphConvolution(nn.Module):
    def __init__(self, num_in, num_out):
        super().__init__(); self.w = nn.Parameter(torch.empty(num_in,num_out)); nn.init.xavier_normal_(self.w)
    def forward(self,x,adj):
        return adj @ x @ self.w

class EnhancedChannelAttention(nn.Module):
    def __init__(self,in_c,ratio=4):
        super().__init__()
        red=max(1,in_c//ratio)
        self.avg,self.max= nn.AdaptiveAvgPool1d(1), nn.AdaptiveMaxPool1d(1)
        self.mlp = nn.Sequential(nn.Linear(in_c,red), nn.LeakyReLU(0.2), nn.Linear(red,in_c))
        self.mix = nn.Sequential(nn.Conv1d(2,1,3,padding=1), nn.LeakyReLU(0.2))
    def forward(self,x):
        B,N,F = x.shape; xt = x.transpose(1,2)
        avg, mx = self.mlp(self.avg(xt).squeeze(-1)), self.mlp(self.max(xt).squeeze(-1))
        att = torch.sigmoid(self.mix(torch.stack([avg,mx],1)).squeeze(1))
        return (xt*att.unsqueeze(-1)).transpose(1,2)

class EnhancedBandAttention(nn.Module):
    def __init__(self,num_bands,hid=32):
        super().__init__()
        self.w = nn.Sequential(nn.Linear(num_bands,hid), nn.LeakyReLU(0.2), nn.Linear(hid,num_bands), nn.Softmax(1))
        self.g = nn.Sequential(nn.Linear(num_bands,hid), nn.LeakyReLU(0.2), nn.Linear(hid,num_bands), nn.Sigmoid())
    def forward(self,x):
        mean, mx = x.mean(1), x.max(1).values
        comb = 0.5*(mean+mx)
        att = self.w(comb*self.g(comb)).unsqueeze(1)
        return x*att

class DynamicGraphConvolution(nn.Module):
    def __init__(self,in_f,out_f,k):
        super().__init__(); self.k=k
        self.layers=nn.ModuleList([GraphConvolution(in_f,out_f) for _ in range(k)])
        self.alpha=nn.Parameter(torch.ones(k)/k)
    def forward(self,x,adjs):
        outs=[l(x,adjs[i]) for i,l in enumerate(self.layers)]
        w=F.softmax(self.alpha,0)
        return sum(w[i]*outs[i] for i in range(self.k))

class MultiScalePooling(nn.Module):
    def forward(self,x):
        return torch.cat([x.mean(1), x.sum(1)],1)

class ResidualBlock(nn.Module):
    def __init__(self,dim):
        super().__init__(); self.block=nn.Sequential(nn.Linear(dim,dim), nn.LeakyReLU(0.2), nn.BatchNorm1d(dim))
    def forward(self,y):
        return y + self.block(y)