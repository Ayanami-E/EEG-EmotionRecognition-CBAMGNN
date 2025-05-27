import torch, torch.nn as nn
from models.components import EnhancedChannelAttention, EnhancedBandAttention, DynamicGraphConvolution, MultiScalePooling, ResidualBlock
from utils.graph_utils import normalize_A, generate_cheby_adj

class ChannelBandAttentionModelM1(nn.Module):
    def __init__(self, num_nodes, num_bands, k_adj=2, num_out=88, nclass=3):
        super().__init__()
        self.chan = EnhancedChannelAttention(num_bands,2)
        self.band = EnhancedBandAttention(num_bands) if num_bands>1 else None
        self.gcn = DynamicGraphConvolution(num_bands,num_out,k_adj)
        self.A = nn.Parameter(torch.randn(num_nodes,num_nodes))
        self.pool= MultiScalePooling()
        dim=2*num_out
        self.feat = nn.Sequential(nn.Linear(dim,512), nn.LeakyReLU(0.2), nn.BatchNorm1d(512))
        self.res = nn.Sequential(ResidualBlock(512), ResidualBlock(512))
        self.cls = nn.Sequential(nn.Linear(512,256), nn.LeakyReLU(0.2), nn.BatchNorm1d(256), nn.Linear(256,nclass))
    def forward(self,x):
        x=self.chan(x)
        if self.band: x=self.band(x)
        adjs=generate_cheby_adj(normalize_A(self.A), self.gcn.k)
        x=self.gcn(x,adjs); x=self.pool(x)
        x=self.res(self.feat(x))
        return self.cls(x)