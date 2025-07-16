"""
Contains functions for training and testing a PyTorch model.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import time
# from sheduler import warmup_lambda, EarlyStopping

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device,)-> Tuple[float, float]:
    
    """
    Trains a PyTorch model for a single epoch and records backpropagation time and memory usage.
    
    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").
    
    Returns:
    A tuple of training loss, training accuracy, backpropagation time, and memory usage metrics.
    """
    model.train()

    train_loss, train_acc = 0, 0

    for elems in dataloader:
        
       
        X, y = elems[0].to(device), elems[1].to(device)
        outputs = model(X)
            
        loss = loss_fn(outputs.float(), y)
        train_loss += loss.item()

        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

        preds = torch.argmax(outputs, dim=1)
        
        if y.dtype == torch.float32 or y.dtype == torch.float64:
            y = y.long()
            
        train_acc += (preds == y).sum().item() / len(y)

    # Average loss and accuracy over batches
    train_loss /= len(dataloader)
    train_acc /= len(dataloader)

    return train_loss, train_acc



def val_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device,
            ) -> Tuple[float, float]:
    """Validates a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a validation dataset.

    Args:
    model: A PyTorch model to be validated.
    dataloader: A DataLoader instance for the model to be validated on.
    loss_fn: A PyTorch loss function to calculate loss on the validation data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of validation loss and validation accuracy metrics.
    In the form (val_loss, val_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval() 

    # Setup validation loss and validation accuracy values
    val_loss, val_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for elems in dataloader:
        
            X, y = elems[0].to(device), elems[1].to(device)
            outputs = model(X)
                
            # 2. Calculate and accumulate loss
            loss = loss_fn(outputs, y)
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            val_pred_labels = outputs.argmax(dim=1)
            
            # Check if y is of type float and convert to long (integer)
            if y.dtype == torch.float32 or y.dtype == torch.float64:
                y = y.long()

            val_acc += ((val_pred_labels == y).sum().item() / len(val_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch 
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          val_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer, 
          loss_fn: torch.nn.Module, 
          device: torch.device, 
          epochs: int,
          model_save_path: str,
          use_scheduler: bool = False,  # New parameter to toggle scheduler usage
          scheduler=None,
          ) -> Dict[str, float]:
    """
    Trains and validates a model for a given number of epochs, recording backpropagation time and memory usage.
    
    Args:
    model: A PyTorch model to be trained and validated.
    train_dataloader: DataLoader for the training data.
    val_dataloader: DataLoader for the validation data.
    optimizer: PyTorch optimizer to optimize the model's weights.
    loss_fn: PyTorch loss function to minimize.
    device: Target device to run the training on.
    epochs: Number of epochs to train for.
    model_save_path: Path to save the best model.
    use_scheduler: If True, applies the scheduler step after each epoch.
    scheduler: The learning rate scheduler instance (optional).
    
    Returns:
    A dictionary with total and average metrics including backpropagation time and memory usage.
    """
    results = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    best_val_acc = 0.0

    model.to(device)

    for epoch in tqdm(range(epochs), colour="blue"):
        train_loss, train_acc = train_step(model=model,
                                            dataloader=train_dataloader,
                                            loss_fn=loss_fn,
                                            optimizer=optimizer,
                                            device=device,
                                        )
        
        val_loss, val_acc = val_step(model=model,
                                     dataloader=val_dataloader,
                                     loss_fn=loss_fn,
                                     device=device,
                                     )
    
        
        print(f"Epoch: {epoch+1} | "
              f"train_loss: {train_loss:.4f} | "
              f"train_acc: {train_acc:.4f} | "
              f"val_loss: {val_loss:.4f} | "
              f"val_acc: {val_acc:.4f} | "
            )
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        
        
        # If scheduler is used, step it
        if use_scheduler and scheduler is not None:
            scheduler.step()
            
        
        # Save the best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"✅ Best model saved with accuracy: {best_val_acc:.4f}")


    # Load the best model after training
    print("Loading the best model for further tasks...")
    model.load_state_dict(torch.load(model_save_path, weights_only=True))

    return results, model 


