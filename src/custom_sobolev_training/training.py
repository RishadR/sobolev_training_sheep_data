"""
Training a model using Sobolev Training techniques
"""
import torch
import torch.nn as nn
from torch.func import vmap, grad
from torch.utils.data import DataLoader
from typing import Dict, Callable, Optional, List
from tqdm import tqdm
import numpy as np


def sobolev_train(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    evaluator_fn: Callable[[nn.Module, DataLoader, torch.device], Dict[str, float]],
    num_epochs: int,
    alpha: float = 1.0,
    scheduler: Optional[Callable[[torch.optim.Optimizer, int, Dict[str, float]], torch.optim.Optimizer]] = None,
) -> Dict[str, List[float]]:
    """
    Train a model using Sobolev training methodology.
    
    Sobolev training matches both labels and gradients of labels w.r.t. features.
    The input features are split in half: first half contains actual features,
    second half contains gradient values.
    
    Args:
        model: PyTorch model to train
        device: Torch device (cuda/cpu)
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer for training
        loss_fn: Loss function for both label and gradient losses
        evaluator_fn: Function that takes (model, dataloader, device) and returns 
                     a dict of metric names to float values
        num_epochs: Number of epochs to train
        alpha: Hyperparameter to weight gradient loss (final_loss = label_loss + alpha * gradient_loss)
        scheduler: Optional learning rate scheduler
        
    Returns:
        Dictionary containing training history with keys:
            - 'train_loss': List of training losses per epoch
            - 'train_label_loss': List of label losses per epoch
            - 'train_gradient_loss': List of gradient losses per epoch
            - 'train_metrics': List of dicts containing training metrics per epoch
            - 'val_metrics': List of dicts containing validation metrics per epoch
    """
    
    # Initialize history storage
    history = {
        'train_loss': [],
        'train_label_loss': [],
        'train_gradient_loss': [],
        'train_metrics': [],
        'val_metrics': []
    }
    
    # Move model to device
    model.to(device)
    
    def get_prediction_unsqueezed(x):
        return model(x).squeeze()
    
    print(f"\n{'='*80}")
    print(f"Starting Sobolev Training for {num_epochs} epochs")
    print(f"Alpha (gradient loss weight): {alpha}")
    print(f"Device: {device}")
    print(f"{'='*80}\n")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_label_loss = 0.0
        epoch_gradient_loss = 0.0
        num_batches = 0
        
        # Progress bar for training
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        
        for batch_idx, (features, labels) in enumerate(train_pbar):
            # Move data to device
            features = features.to(device)
            labels = labels.to(device)
            
            # Split features into actual features and gradients
            # First half: actual features, Second half: gradients
            num_features = features.shape[1] // 2
            actual_features = features[:, :num_features]
            target_gradients = features[:, num_features:]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass with actual features
            predictions = model(actual_features)
            
            # Compute label loss
            label_loss = loss_fn(predictions, labels)
            
            # Compute gradients of predictions w.r.t. input features
            # We need to enable gradient computation for input features
            actual_features.requires_grad_(True)
            # Compute gradients for each output w.r.t. inputs
            predicted_gradients = vmap(grad(get_prediction_unsqueezed))(actual_features)
            
            # Compute gradient loss
            gradient_loss = loss_fn(predicted_gradients, target_gradients)
            
            # Combined loss
            total_loss = label_loss + alpha * gradient_loss
            
            # Backward pass
            total_loss.backward()
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate losses
            epoch_loss += total_loss.item()
            epoch_label_loss += label_loss.item()
            epoch_gradient_loss += gradient_loss.item()
            num_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'loss': f'{total_loss.item():.3E}',
                'label': f'{label_loss.item():.3E}',
                'grad': f'{gradient_loss.item():.3E}'
            })
        
        # Average losses for the epoch
        avg_train_loss = epoch_loss / num_batches
        avg_label_loss = epoch_label_loss / num_batches
        avg_gradient_loss = epoch_gradient_loss / num_batches
        
        # Store training losses
        history['train_loss'].append(avg_train_loss)
        history['train_label_loss'].append(avg_label_loss)
        history['train_gradient_loss'].append(avg_gradient_loss)
        
        # Evaluation phase
        print(f"\nEvaluating on training set...")
        train_metrics = evaluator_fn(model, train_loader, device)
        history['train_metrics'].append(train_metrics)
        
        print(f"Evaluating on validation set...")
        val_metrics = evaluator_fn(model, val_loader, device)
        history['val_metrics'].append(val_metrics)
        
        # Pretty print results
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{num_epochs} Summary:")
        print(f"{'-'*80}")
        print(f"Training Loss:    {avg_train_loss:.4E}")
        print(f"  - Label Loss:   {avg_label_loss:.4E}")
        print(f"  - Gradient Loss: {avg_gradient_loss:.4E}")
        print(f"\nTraining Metrics:")
        for metric_name, metric_value in train_metrics.items():
            print(f"  - {metric_name}: {metric_value:.4E}")
        print(f"\nValidation Metrics:")
        for metric_name, metric_value in val_metrics.items():
            print(f"  - {metric_name}: {metric_value:.4E}")
        print(f"{'='*80}\n")
        
        # Update scheduler if provided
        if scheduler is not None:
            optimizer = scheduler(optimizer, epoch, val_metrics)
    
    print(f"\n{'='*80}")
    print(f"Training Complete!")
    print(f"{'='*80}\n")
    
    return history

