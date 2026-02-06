import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import struct
import glob
import sys
from model import LogosNet

# Configuration
INPUT_CHANNELS = 17
BOARD_SIZE = 8 * 8
POLICY_SIZE = 4672
SAMPLE_SIZE_FLOATS = (INPUT_CHANNELS * BOARD_SIZE) + 1 + 1 + POLICY_SIZE 

class ChessDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(os.path.join(data_dir, "*.bin"))
        self.samples = []
        
        print(f"Loading data from {len(self.files)} files in {data_dir}...")
        for file_path in self.files:
            self._load_file(file_path)
        print(f"Loaded {len(self.samples)} training samples.")

    def _load_file(self, path):
        file_size = os.path.getsize(path)
        bytes_per_sample = SAMPLE_SIZE_FLOATS * 4 
        num_samples = file_size // bytes_per_sample
        
        with open(path, 'rb') as f:
            data = np.fromfile(f, dtype=np.float32)
            
        if data.size == 0:
            return

        try:
            data = data.reshape(num_samples, SAMPLE_SIZE_FLOATS)
            board_end = INPUT_CHANNELS * BOARD_SIZE
            self.samples.extend(data) 
        except ValueError as e:
            print(f"Error loading {path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        flat_data = self.samples[idx]
        
        # 1. Board [17, 8, 8]
        board_end = INPUT_CHANNELS * BOARD_SIZE
        board_data = flat_data[:board_end]
        board_tensor = torch.from_numpy(board_data).view(INPUT_CHANNELS, 8, 8)
        
        # 2. Material Scalar [1]
        material_idx = board_end
        material_scalar = torch.tensor([flat_data[material_idx]])
        
        # 3. Value Target [1]
        value_idx = material_idx + 1
        value_target = torch.tensor([flat_data[value_idx]])
        
        # 4. Policy Target [4672]
        policy_start = value_idx + 1
        policy_target = torch.from_numpy(flat_data[policy_start:])
        
        return board_tensor, material_scalar, value_target, policy_target

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train LogosNet chess model")
    parser.add_argument('data_dir', type=str, nargs='?', default='data/training',
                        help='Directory containing .bin training data')
    parser.add_argument('output_path', type=str, nargs='?', default='python/models/latest.pt',
                        help='Output model path')
    parser.add_argument('resume_path', type=str, nargs='?', default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--optimizer', type=str, default='adam',
                        choices=['adam', 'adamw', 'muon'],
                        help='Optimizer to use (default: adam)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 0.001 for adam/adamw, 0.02 for muon)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size (default: 64)')
    return parser.parse_args()


def train():
    args = parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Set default LR based on optimizer
    if args.lr is None:
        lr = 0.02 if args.optimizer == 'muon' else 0.001
    else:
        lr = args.lr

    print(f"Using device: {DEVICE}")
    print(f"Data Dir: {args.data_dir}")
    print(f"Output Model: {args.output_path}")
    print(f"Optimizer: {args.optimizer} (lr={lr})")
    print(f"Epochs: {args.epochs}")

    # Data
    dataset = ChessDataset(args.data_dir)
    if len(dataset) == 0:
        print("No training data found.")
        return

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Model
    model = LogosNet().to(DEVICE)

    # Optimizer selection
    if args.optimizer == 'muon':
        from muon import Muon
        optimizer = Muon(model.parameters(), lr=lr, backend_lr=lr * 0.1)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    if args.resume_path and os.path.exists(args.resume_path):
        print(f"Resuming from {args.resume_path}")
        try:
            model.load_state_dict(torch.load(args.resume_path, map_location=DEVICE))
        except Exception as e:
            print(f"Warning: Failed to load resume checkpoint: {e}")

    # Training Loop
    model.train()
    for epoch in range(args.epochs):
        total_policy_loss = 0
        total_value_loss = 0
        total_k = 0

        for batch_idx, (boards, materials, values, policies) in enumerate(dataloader):
            boards = boards.to(DEVICE)
            materials = materials.to(DEVICE)
            values = values.to(DEVICE)
            policies = policies.to(DEVICE)

            # Forward: returns (policy, value, k)
            pred_policy, pred_value, k = model(boards, materials)

            # Loss
            policy_loss = F.kl_div(pred_policy, policies, reduction='batchmean')
            value_loss = F.mse_loss(pred_value, values)

            loss = policy_loss + value_loss

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            current_k = k.mean().item()
            total_k += current_k

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} Batch {batch_idx}: Loss={loss.item():.4f} P_Loss={policy_loss.item():.4f} V_Loss={value_loss.item():.4f} K={current_k:.4f}")

        n_batches = len(dataloader)
        avg_policy = total_policy_loss / n_batches
        avg_value = total_value_loss / n_batches
        avg_k = total_k / n_batches
        print(f"Epoch {epoch+1} Average: Policy={avg_policy:.4f} Value={avg_value:.4f} K={avg_k:.4f}")

        # Save checkpoint
        os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
        torch.save(model.state_dict(), args.output_path)
        print(f"Saved {args.output_path}")

if __name__ == "__main__":
    train()