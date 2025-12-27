import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import struct
import glob
from model import LogosNet

# ... (dataset logic) ...

def train():
    # Hyperparameters
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    EPOCHS = 10
    DATA_DIR = "data/training"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # Data
    dataset = ChessDataset(DATA_DIR)
    if len(dataset) == 0:
        print("No training data found. Run self-play first.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    model = LogosNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
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
                print(f"Epoch {epoch+1} Batch {batch_idx}: P_Loss={policy_loss.item():.4f} V_Loss={value_loss.item():.4f} K={current_k:.4f}")

        avg_k = total_k / len(dataloader)
        print(f"Epoch {epoch+1} Average: Policy={total_policy_loss/len(dataloader):.4f} Value={total_value_loss/len(dataloader):.4f} K={avg_k:.4f}")
        
        # Save checkpoint
        os.makedirs("python/models", exist_ok=True)
        torch.save(model.state_dict(), "python/models/latest.pt")
        print("Saved python/models/latest.pt")

if __name__ == "__main__":
    import torch.nn.functional as F
    train()
