import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ==========================================
# Shared Components
# ==========================================
class ResidualBlock(nn.Module):
    """Standard ResNet Block: Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN -> Add -> ReLU"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

# ==========================================
# Logos Network (Dynamic Symbolic Residual)
# ==========================================
class LogosNet(nn.Module):
    """
    Neurosymbolic Chess Network with Dynamic Confidence
    Backbone: ResNet
    Policy: Standard
    Value: Dual Head (Deep Value + Confidence Scalar)
    """
    def __init__(self, input_channels=12, filters=128, num_res_blocks=10, policy_output_size=4672):
        super(LogosNet, self).__init__()
        
        # Input Block
        self.conv_input = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(filters)
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(filters) for _ in range(num_res_blocks)
        ])
        
        # Policy Head (Convolutional)
        # Output: 73 planes per square (8x8x73) -> 4672 moves
        self.policy_conv = nn.Conv2d(filters, 73, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(73)
        # No Linear layer for policy! We flatten the conv output directly.
        
        # Value Backbone (Shared by V and K)
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc_hidden = nn.Linear(8 * 8, 256)
        
        # Dual Heads
        # 1. Deep Value Head (V_net) - outputs Logit
        self.val_head = nn.Linear(256, 1)
        
        # 2. Confidence Head (K_net) - outputs Logit
        self.k_head = nn.Linear(256, 1)
        
        # Zero Initialization for K-head
        # Result: k_logit starts at 0.0
        # Softplus(0) = ln(1 + e^0) = ln(2)
        # k = ln(2) / (2 * ln(2)) = 0.5
        nn.init.constant_(self.k_head.weight, 0.0)
        nn.init.constant_(self.k_head.bias, 0.0)
        
        # Denominator Constant: 2 * ln(2)
        self.k_scale = 2 * math.log(2)

    def forward(self, x, material_scalar):
        """
        Args:
            x: [B, C, 8, 8] Board features
            material_scalar: [B] or [B, 1] Material Imbalance
        Returns:
            policy: [B, 4672] Log probabilities
            value: [B, 1] Final evaluation (-1 to 1)
            k: [B, 1] The predicted material confidence scalar
        """
        # Backbone
        x = F.relu(self.bn_input(self.conv_input(x)))
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        # Flatten to [B, 4672] (8*8*73)
        # Note: PyTorch flattens Channel-first by default?
        # Conv2d output is [B, 73, 8, 8].
        # We want alignment with our move_to_index which is (src * 73 + plane).
        # Src is 0..63 (Rank*8 + File).
        # So we want [B, 8, 8, 73] then flattened? 
        # Or does [B, 73, 8, 8] flatten to something else?
        # x.view(B, -1) flattens [B, C, H, W] -> [B, C*H*W].
        # Index order: C changes slowest, then H, then W.
        # Index = c * (H*W) + h * W + w.
        # Our target index is: src * 73 + plane.
        # src = h*8 + w.
        # Target Index = (h*8 + w) * 73 + c.
        # This matches permuting to [B, 8, 8, 73] then flattening.
        
        p = p.permute(0, 2, 3, 1) # [B, 8, 8, 73]
        p = p.contiguous().view(p.size(0), -1) # [B, 8*8*73]
        
        policy = F.log_softmax(p, dim=1) # Log probabilities
        
        # Value Backbone
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc_hidden(v))
        
        # Dual Heads
        v_logit = self.val_head(v) # [B, 1]
        k_logit = self.k_head(v)   # [B, 1]
        
        # Calculate k (Confidence Scalar)
        # k = Softplus(k_logit) / (2 * ln2)
        k = F.softplus(k_logit) / self.k_scale
        
        # Ensure material_scalar matches shape [B, 1]
        if material_scalar.dim() == 1:
            material_scalar = material_scalar.unsqueeze(1)
            
        # Residual Recombination
        # V_final = Tanh( V_net + k * DeltaM )
        total_logit = v_logit + (k * material_scalar)
        value = torch.tanh(total_logit)
        
        return policy, value, k