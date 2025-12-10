import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Small/Minimal Architecture (Debugging)
# ==========================================
class SmallNet(nn.Module):
    """
    A simple two-layer neural network for debugging purposes.
    Not expected to play well, but useful for testing pipeline integration.
    """
    def __init__(self, input_channels=12, input_height=8, input_width=8, policy_output_size=4096):
        super(SmallNet, self).__init__()
        self.input_size = input_channels * input_height * input_width
        
        # Layer 1: Flatten -> Dense
        self.fc1 = nn.Linear(self.input_size, 128)
        
        # Layer 2: Heads
        self.policy_head = nn.Linear(128, policy_output_size)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        # Flatten input: [Batch, 12, 8, 8] -> [Batch, 768]
        x = x.view(-1, self.input_size)
        
        # Layer 1
        x = F.relu(self.fc1(x))
        
        # Heads
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        
        return policy, value


# ==========================================
# 2. Minimal Viable Model (MVM)
# ==========================================
class MinimalViableNet(nn.Module):
    """
    A 7-layer Convolutional Network with residual connections.
    Includes specific head structures:
    - Eval Head: 2-layer dense
    - Policy Head: Convolutional structure
    """
    def __init__(self, input_channels=12, filters=64, policy_output_size=4096):
        super(MinimalViableNet, self).__init__()
        
        # Initial Conv Layer (Layer 1)
        self.conv_input = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(filters)
        
        # 3 Residual Blocks (2 layers each = 6 layers) 
        # Total depth = 1 (input) + 6 (res) = 7 convolutional layers
        self.res_blocks = nn.ModuleList([
            ResidualBlock(filters) for _ in range(3)
        ])
        
        # -----------------------------------------------------
        # Policy Head (Convolutional -> Linear)
        # -----------------------------------------------------
        self.policy_conv = nn.Conv2d(filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        # Note: Purely convolutional policy heads are rare for chess (move=from*to).
        # We use a standard Conv projection -> Flatten -> Linear to map to 4096 moves.
        self.policy_fc = nn.Linear(32 * 8 * 8, policy_output_size)
        
        # -----------------------------------------------------
        # Value Head (Two-layer Dense)
        # -----------------------------------------------------
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        # Flattened size = 1 * 8 * 8 = 64
        self.value_fc1 = nn.Linear(64, 64)   # Dense Layer 1
        self.value_fc2 = nn.Linear(64, 1)    # Dense Layer 2

    def forward(self, x):
        # Initial Conv
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual Tower
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = F.log_softmax(self.policy_fc(p), dim=1)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))   # Layer 1
        value = torch.tanh(self.value_fc2(v)) # Layer 2
        
        return policy, value


# ==========================================
# 3. Large / AlphaZero Architecture
# ==========================================
class AlphaZeroNet(nn.Module):
    """
    The full AlphaZero architecture (Chess configuration).
    - Tower of 20 (AlphaGo Zero) or 40 (AlphaZero) Residual Blocks.
    - 256 Filters.
    """
    def __init__(self, input_channels=12, filters=256, num_res_blocks=20, policy_output_size=4096):
        super(AlphaZeroNet, self).__init__()
        
        # Input Block
        self.conv_input = nn.Conv2d(input_channels, filters, kernel_size=3, padding=1, bias=False)
        self.bn_input = nn.BatchNorm2d(filters)
        
        # Residual Tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(filters) for _ in range(num_res_blocks)
        ])
        
        # Policy Head
        self.policy_conv = nn.Conv2d(filters, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * 8 * 8, policy_output_size)
        
        # Value Head
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Input
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Tower
        for block in self.res_blocks:
            x = block(x)
            
        # Policy Head
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.view(p.size(0), -1)
        policy = F.log_softmax(self.policy_fc(p), dim=1)
        
        # Value Head
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.view(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        value = torch.tanh(self.value_fc2(v))
        
        return policy, value


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
