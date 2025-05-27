import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import GraphConvolution
from utils import generate_cheby_adj, normalize_A

# ================== 通道注意力 ==================
class EnhancedChannelAttention(nn.Module):
    """优化的通道注意力机制"""
    def __init__(self, in_channels, reduction_ratio=4):
        super(EnhancedChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        reduced_channels = max(1, in_channels // reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, reduced_channels),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(reduced_channels),
            nn.Dropout(0.2),  # 防止过拟合
            nn.Linear(reduced_channels, in_channels)
        )
        
        # 通道混合模块
        self.channel_mixer = nn.Sequential(
            nn.Conv1d(2, 1, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # x形状: [B, node, feat] 或 [B, sample, node, feat]
        input_shape = x.shape
        if len(input_shape) == 4:
            batch_size, num_samples, num_nodes, num_features = input_shape
            x = x.reshape(-1, num_nodes, num_features)  # [B*, node, feat]
        else:
            batch_size = x.size(0)

        # 转置: [B*, feat, node] 方便池化
        x_transposed = x.transpose(1, 2)
        
        # 池化操作
        avg_out = self.avg_pool(x_transposed).squeeze(-1)  # [B*, feat]
        max_out = self.max_pool(x_transposed).squeeze(-1)  # [B*, feat]
        
        # MLP处理
        avg_out = self.mlp(avg_out)
        max_out = self.mlp(max_out)
        
        # 通道信息融合
        mixed_features = torch.stack([avg_out, max_out], dim=1)  # [B*, 2, feat]
        channel_attention = self.sigmoid(self.channel_mixer(mixed_features).squeeze(1))  # [B*, feat]
        
        # 应用注意力
        result = (x_transposed * channel_attention.unsqueeze(-1)).transpose(1, 2)
        
        # 如果原始是4维，则还原
        if len(input_shape) == 4:
            result = result.reshape(batch_size, num_samples, num_nodes, num_features)
            
        return result

class EnhancedBandAttention(nn.Module):
    """优化的频带注意力机制"""
    def __init__(self, num_bands, hidden_dim=32):
        super(EnhancedBandAttention, self).__init__()
        self.band_weights = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_bands),
            nn.Softmax(dim=1)
        )
        self.band_interaction = nn.Sequential(
            nn.Linear(num_bands, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_bands),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, node, band] 或 [B, sample, node, band]
        input_shape = x.shape
        if len(input_shape) == 4:
            batch_size, num_samples, num_nodes, num_bands = input_shape
            x = x.reshape(-1, num_nodes, num_bands)
        else:
            batch_size = x.size(0)
        
        # 提取频带特征
        band_features = torch.mean(x, dim=1)  # [B*, band]
        max_band_features, _ = torch.max(x, dim=1)  # [B*, band]
        
        # 合并特征
        combined_features = (band_features + max_band_features) / 2
        
        # 频带间交互
        band_interaction = self.band_interaction(combined_features)
        
        # 注意力权重
        weights = self.band_weights(combined_features * band_interaction)  # [B*, band]
        weights = weights.unsqueeze(1)  # [B*, 1, band]
        
        # 应用权重
        result = x * weights
        
        if len(input_shape) == 4:
            result = result.reshape(batch_size, num_samples, num_nodes, num_bands)
        return result

class DynamicGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, k, activation=F.relu):
        super(DynamicGraphConvolution, self).__init__()
        self.k = k
        self.activation = activation
        
        self.gc_layers = nn.ModuleList()
        for i in range(k):
            self.gc_layers.append(GraphConvolution(in_features, out_features))
        
        self.fusion = nn.Parameter(torch.ones(k) / k)
        
    def forward(self, x, L):
        device = x.device
        adj = generate_cheby_adj(L, self.k, device)
        
        outputs = []
        for i in range(len(self.gc_layers)):
            outputs.append(self.gc_layers[i](x, adj[i]))
        
        weights = F.softmax(self.fusion, dim=0)
        result = 0
        for i in range(len(outputs)):
            result += weights[i] * outputs[i]
        
        if self.activation:
            result = self.activation(result)
        return result

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if len(x.shape) == 3:
            b, c, _ = x.size()
            y = self.avg_pool(x.transpose(1, 2)).view(b, c)
            y = self.fc(y).view(b, c, 1)
            return x * y.transpose(1, 2)
        else:
            b, c = x.size()
            y = self.fc(x)
            return x * y

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_features),
            nn.Dropout(0.2),
            nn.Linear(hidden_features, in_features),
            nn.BatchNorm1d(in_features)
        )
        self.se = SEBlock(in_features)
        self.relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out = residual + out
        out = self.relu(out)
        return out

# ================== 多尺度特征池化模块 ==================
class MultiScaleFeaturePooling(nn.Module):

    def __init__(self, pool_dims=["node","feature"], pool_types=["sum","avg"]):
        super(MultiScaleFeaturePooling, self).__init__()
        self.pool_dims = pool_dims   # ["node", "feature", "both"] 中的组合
        self.pool_types = pool_types # ["avg", "max", "sum"] 的组合

    def forward(self, x):
        """
        x: [B, N, F]
        返回: [B, ?] 拼接后的向量
        """
        # 如果 x 来自 [B, N, F]
        B, N, F = x.shape
        out_list = []

        for dim_str in self.pool_dims:
            if dim_str == "node":
                # 在 node维度做池化 => 维度1
                for pt in self.pool_types:
                    if pt == "avg":
                        pooled = x.mean(dim=1)  # [B, F]
                    elif pt == "sum":
                        pooled = x.sum(dim=1)   # [B, F]
                    elif pt == "max":
                        pooled, _ = x.max(dim=1) # [B, F]
                    else:
                        raise ValueError(f"unknown pool type: {pt}")
                    out_list.append(pooled)
            elif dim_str == "feature":
                # 在 feature维度做池化 => 维度2
                for pt in self.pool_types:
                    if pt == "avg":
                        pooled = x.mean(dim=2)  # [B, N]
                    elif pt == "sum":
                        pooled = x.sum(dim=2)   # [B, N]
                    elif pt == "max":
                        pooled, _ = x.max(dim=2) # [B, N]
                    else:
                        raise ValueError(f"unknown pool type: {pt}")
                    out_list.append(pooled)
            elif dim_str == "both":
                # 一次性在 N、F 上全局池化
                for pt in self.pool_types:
                    if pt == "avg":
                        pooled = x.mean(dim=(1,2))  # [B]
                        pooled = pooled.unsqueeze(1) # [B,1]
                    elif pt == "sum":
                        pooled = x.sum(dim=(1,2))   # [B]
                        pooled = pooled.unsqueeze(1)
                    else:
                        raise ValueError(f"unknown pool type: {pt}")
                    out_list.append(pooled)
            else:
                raise ValueError(f"unknown pool dim: {dim_str}")

        # 拼接
        final_out = torch.cat(out_list, dim=1) # [B, X]
        return final_out

class ChannelBandAttentionModelM1(nn.Module):
    """
    在原ChannelBandAttentionModel基础上添加 多尺度池化(MultiScaleFeaturePooling).
    """
    def __init__(self, xdim, k_adj=2, num_out=88, nclass=3):
        super(ChannelBandAttentionModelM1, self).__init__()
        self.K = k_adj
        
        # 通道注意力 => in_channels = xdim[2]
        self.channel_attention = EnhancedChannelAttention(
            in_channels=xdim[2],
            reduction_ratio=2
        )
        
        # 如果频带>1，则使用 band_attention
        if xdim[2] > 1:
            self.band_attention = EnhancedBandAttention(xdim[2], hidden_dim=64)
        else:
            self.band_attention = None

        self.BN1 = nn.BatchNorm1d(xdim[2])
        self.layer1 = DynamicGraphConvolution(xdim[2], num_out, k_adj)

        # 可学习邻接矩阵
        self.A = nn.Parameter(torch.FloatTensor(xdim[1], xdim[1]))
        nn.init.xavier_normal_(self.A)

        # === Multi-Scale Pooling (新增) ===
        self.multi_scale_pool = MultiScaleFeaturePooling(
            pool_dims=["node","feature"], 
            pool_types=["sum","avg"]
        )


        self.num_nodes = xdim[1]
        self.num_out = num_out
        # => 2F + 2N
        pool_out_dim = 2 * self.num_out + 2 * self.num_nodes

        self.feature_extractor = nn.Sequential(
            nn.Linear(pool_out_dim, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3)
        )

        # 残差块
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(512, 512) for _ in range(2)
        ])

        # 主分类器
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, nclass)
        )

        # 集成分类器 (可选)
        self.classifier_ensemble = nn.ModuleList([
            nn.Linear(pool_out_dim, nclass) for _ in range(2)
        ])
        self.ensemble_weights = nn.Parameter(torch.ones(2) / 2)

    def forward(self, x):
        # x: [B, node, band] 或 [B, sample, node, band]
        input_shape = x.shape
        if len(input_shape) == 4:
            b, s, n, f = input_shape
            x_reshaped = x.reshape(-1, n, f)
        else:
            x_reshaped = x

        # 通道注意力
        x_reshaped = self.channel_attention(x_reshaped)
        # 频带注意力
        if self.band_attention is not None:
            x_reshaped = self.band_attention(x_reshaped)

        # BN
        x_bn = self.BN1(x_reshaped.transpose(1, 2)).transpose(1, 2)

        # 图卷积
        L = normalize_A(self.A)
        gc_out = self.layer1(x_bn, L)  # [B*, N, num_out]

        # === 多尺度池化 ===
        pooled_out = self.multi_scale_pool(gc_out)  # [B*, 2F + 2N] (示例中)

        # 特征抽取
        features = self.feature_extractor(pooled_out)

        # 残差块
        for block in self.residual_blocks:
            features = block(features)

        # 主分类器
        main_output = self.classifier(features)

        # ========== Ensemble ========== 
        ensemble_outputs = []
        for cl in self.classifier_ensemble:
            ensemble_outputs.append(cl(pooled_out))

        weights = F.softmax(self.ensemble_weights, dim=0)
        ensemble_result = sum(w * out for w, out in zip(weights, ensemble_outputs))

        alpha = 0.7
        final_output = alpha * main_output + (1 - alpha) * ensemble_result
        return final_output
