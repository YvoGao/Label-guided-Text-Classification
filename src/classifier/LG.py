import torch
import torch.nn as nn


class SingleFusion(nn.Module):
    def __init__(self, feature_dim):
        super(SingleFusion, self).__init__()

    def forward(self, feature_1, feature_2):
        return feature_1


class MeanFusion(nn.Module):
    def __init__(self, feature_dim):
        super(MeanFusion, self).__init__()

    def forward(self, feature_1, feature_2):
        return (feature_1 + feature_2)/2


class FMeanFusion(nn.Module):
    def __init__(self, feature_dim, alpha):
        super(FMeanFusion, self).__init__()
        self.alpha = alpha

    def forward(self, feature_1, feature_2):
        return self.alpha * feature_1 + (1 - self.alpha) * feature_2


class ConnectFusion(nn.Module):
    def __init__(self, feature_dim):
        super(ConnectFusion, self).__init__()
        # self.mlp = nn.Linear(feature_dim * 2, feature_dim)
        self.mlp = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim))

    def forward(self, feature_1, feature_2):
        feature = torch.cat([feature_1, feature_2], dim=1)
        # import pdb
        # pdb.set_trace()
        return self.mlp(feature)


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion, self).__init__()
        # 定义注意力权重计算模型
        self.attention_weights_model = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Tanh(),
            nn.Flatten()
        )

    def forward(self, feature_1, feature_2):
        # 计算注意力权重
        attention_weights_1 = self.attention_weights_model(feature_1)
        attention_weights_2 = self.attention_weights_model(feature_2)
        attention_weights = torch.nn.functional.softmax(
            torch.cat([attention_weights_1, attention_weights_2], dim=1), dim=1)

        # 对两个特征进行加权平均
        fused_feature = attention_weights[:, 0:1] * \
            feature_1 + attention_weights[:, 1:2] * feature_2

        return fused_feature


class AttentionFusionTransformer(nn.Module):
    def __init__(self, input_size, num_heads=8, hidden_size=128):
        super(AttentionFusionTransformer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Linear layers for query, key, and value
        self.query_linear = nn.Linear(input_size, hidden_size)
        self.key_linear = nn.Linear(input_size, hidden_size)
        self.value_linear = nn.Linear(input_size, hidden_size)

        # Attention layer
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)

        # Feedforward layer
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, x1, x2):
        # import pdb
        # pdb.set_trace()
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        # x1 and x2 are the input feature vectors
        # query = self.query_linear(x1)  # [batch_size, seq_len, hidden_size]
        # key = self.key_linear(x2)      # [batch_size, seq_len, hidden_size]
        # value = self.value_linear(x2)  # [batch_size, seq_len, hidden_size]
        query = x1
        key = x2
        value = x2
        # Compute self-attention
        attention_output, _ = self.attention(query, key, value)

        # Add and normalize
        x = self.norm1(x1 + attention_output)

        # Feedforward pass
        feedforward_output = self.feedforward(x)

        # Add and normalize
        x = self.norm2(x + feedforward_output)
        x = x.squeeze(dim=1)
        return x


class AttentionFusion3(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion3, self).__init__()
        # 定义注意力权重计算模型
        self.attention_weights_model = nn.Sequential(
            nn.Linear(feature_dim, 1),
            nn.Tanh(),
            nn.Flatten()
        )

    def forward(self, A, B):
        similarity_matrix = torch.matmul(A, B.t())
        # 计算注意力权重
        # attention_weights = torch.softmax(similarity_matrix, dim=1)
        attention_weights = self.attention_weights_model(
            torch.diag(similarity_matrix))
        # 加权并对应位置的量
        merged_vector = A * attention_weights + B * (1 - attention_weights)
        # 对第二维求平均，得到（32, 128）的输出
        # merged_vector = torch.mean(merged_vector, dim=1)
        return merged_vector


class AutoFusion(nn.Module):
    def __init__(self, feature_dim):
        super(AutoFusion, self).__init__()
        # 定义注意力权重计算模型
        self.alpha = nn.Parameter(torch.tensor(1, dtype=torch.float))

    def forward(self, feature_1, feature_2):
        return self.alpha * feature_1 + (1 - self.alpha) * feature_2


class AttentionFusion4(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion4, self).__init__()
        # 定义注意力权重计算模型
        self.attention_weights_model = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, A, B):
        # similarity_matrix = torch.matmul(A, B.t())
        dot = torch.matmul(
            A.unsqueeze(0).unsqueeze(-2),
            B.unsqueeze(1).unsqueeze(-1)
        )
        dot = dot.squeeze(-1).squeeze(-1)

        scale = (torch.norm(A, dim=1).unsqueeze(0) *
                 torch.norm(B, dim=1).unsqueeze(1))

        scale = torch.max(scale,
                          torch.ones_like(scale) * 1e-8)

        similarity_matrix = 1 - dot/scale
        # 计算注意力权重
        # sig = torch.unsqueeze(torch.diag(similarity_matrix), dim=0)

        # attention_weights = self.attention_weights_model(sig)
        attention_weights = torch.unsqueeze(
            torch.diag(similarity_matrix), dim=0)
        # 加权并对应位置的量[32,1]
        # merged_vector = (attention_weights.t() * A +
        #                  (2-attention_weights.t()) * B)/2
        # import pdb
        # pdb.set_trace()
        merged_vector = (attention_weights.t() * A +
                         (2-attention_weights.t()) * B)/2
        # merged_vector = attention_weights.t() * A +  B
        # 对第二维求平均，得到（32, 128）的输出
        # merged_vector = torch.mean(merged_vector, dim=1)
        return merged_vector


class AttentionFusion5(nn.Module):
    def __init__(self, feature_dim):
        super(AttentionFusion5, self).__init__()
        # self.mlp = nn.Linear(feature_dim * 2, feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, feature_1, feature_2):
        feature = torch.cat([feature_1, feature_2], dim=1)
        weight = self.mlp(feature)
        fused_feature = weight * feature_1 + (1 - weight) * feature_2
        return fused_feature


class AttentionFusionTransformer2(nn.Module):
    def __init__(self, input_size, num_heads=8, hidden_size=128):
        super(AttentionFusionTransformer2, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Attention layer
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, feature_1, feature_2):
        feature_1 = torch.unsqueeze(feature_1, dim=1)
        feature_2 = torch.unsqueeze(feature_2, dim=1)
        attention_output, _ = self.attention(feature_1, feature_2, feature_2)
        weight1 = self.mlp(attention_output.squeeze(dim=1))
        attention_output2, _ = self.attention(feature_2, feature_1, feature_1)
        weight2 = self.mlp2(attention_output2.squeeze(dim=1))
        attention_weights = torch.nn.functional.softmax(torch.cat([weight1, weight2], dim=1),
                                                        dim=1)
        fuse_feature = attention_weights[:, 0:1] * \
            feature_1.squeeze(
                dim=1) + attention_weights[:, 1:2] * feature_2.squeeze(dim=1)

        # fuse_feature = weight1 * \
        #     feature_1.squeeze(dim=1) + weight2 * feature_2.squeeze(dim=1)
        return fuse_feature


class AttentionFusionTransformer3(nn.Module):
    def __init__(self, input_size, num_heads=8, hidden_size=128):
        super(AttentionFusionTransformer3, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size

        # Attention layer
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
            nn.Flatten()
        )

    def forward(self, feature_1, feature_2):
        feature_1 = torch.unsqueeze(feature_1, dim=1)
        feature_2 = torch.unsqueeze(feature_2, dim=1)
        attention_output, _ = self.attention(feature_1, feature_2, feature_2)
        weight1 = self.mlp(attention_output.squeeze(dim=1))
        fuse_feature = (weight1 *
                        feature_1.squeeze(dim=1) + (2-weight1) *
                        feature_2.squeeze(dim=1))/2
        return fuse_feature


class SG(nn.Module):
    def __init__(self, args):
        super(SG, self).__init__()
        if args.SG == 'att':
            self.fusion = AttentionFusion(args.embedding_dim)
        if args.SG == 'mean':
            self.fusion = MeanFusion(args.embedding_dim)
        if args.SG == 'connect':
            self.fusion = ConnectFusion(args.embedding_dim)
        if args.SG == 'single':
            self.fusion = SingleFusion(args.embedding_dim)
        if args.SG == 'fmean':
            self.fusion = FMeanFusion(args.embedding_dim, args.falpha)
        if args.SG == 'att2':
            self.fusion = AttentionFusionTransformer(
                args.embedding_dim, num_heads=4, hidden_size=args.embedding_dim)
        if args.SG == 'att3':
            self.fusion = AttentionFusion3(args.embedding_dim)
        if args.SG == 'auto':
            self.fusion = AutoFusion(args.embedding_dim)
        if args.SG == 'att4':
            self.fusion = AttentionFusion4(args.way*args.shot)
        if args.SG == 'att5':
            self.fusion = AttentionFusion5(args.embedding_dim)
        if args.SG == 'att6':
            self.fusion = AttentionFusionTransformer2(
                args.embedding_dim, num_heads=4, hidden_size=args.embedding_dim)
        if args.SG == 'att7':
            self.fusion = AttentionFusionTransformer3(
                args.embedding_dim, num_heads=4, hidden_size=args.embedding_dim)

    def forward(self, feature_1, feature_2):
        return self.fusion(feature_1, feature_2)
