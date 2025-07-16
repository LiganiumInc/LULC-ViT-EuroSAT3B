import random 
import numpy as np 
import matplotlib.pyplot as plt

import torch 

from collections import defaultdict
from torch.utils.data import Subset
from tqdm import tqdm
import random
import time
import json

from PIL import Image

def set_seeds(seed: int = 42):
    """Sets random seeds for all relevant libraries to ensure reproducibility.

    Args:
        seed (int, optional): Random seed to set. Defaults to 42.
    """
    # Set the seed for Python's random module
    random.seed(seed)
    
    # Set the seed for NumPy
    np.random.seed(seed)
    
    # Set the seed for PyTorch CPU operations
    torch.manual_seed(seed)
    
    # Set the seed for CUDA operations (both current device and all devices)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    
    # Ensure deterministic behavior in PyTorch (for operations that allow it)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def sample_subset_per_class(subset, percentage=0.3, seed=42):
    """
    Returns a subset containing a fixed percentage of samples from each class in the given Subset.

    Args:
        subset (torch.utils.data.Subset): The subset to sample from (e.g., train_data).
        percentage (float): The percentage of each class to keep.
        seed (int): Random seed for reproducibility.

    Returns:
        torch.utils.data.Subset: A new subset containing the sampled indices.
    """
    start_time = time.time()
    random.seed(seed)

    dataset = subset.dataset
    indices = subset.indices

    # Group indices by class
    class_to_indices = defaultdict(list)
    for i in indices:
        _, label = dataset[i]
        class_to_indices[label].append(i)

    # Sample per class with tqdm
    selected_indices = []
    print("Sampling per class:")
    for cls in tqdm(class_to_indices, total=len(class_to_indices)):
        cls_indices = class_to_indices[cls]
        k = int(len(cls_indices) * percentage)
        sampled = random.sample(cls_indices, k)
        selected_indices.extend(sampled)

    elapsed = time.time() - start_time
    print(f"✅ Sampling completed in {elapsed:.2f} seconds.")

    return Subset(dataset, selected_indices)




def visualize_batch(dataloader, class_names, mean, std, n=4):
    """
    Visualize a batch of images from a PyTorch DataLoader.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader to fetch the batch from.
        class_names (list): List of class names indexed by label.
        mean (list or tuple): Normalization mean used on images.
        std (list or tuple): Normalization std used on images.
        n (int): Grid size (n x n) for visualization.
    """
    inputs, labels = next(iter(dataloader))
    fig, axes = plt.subplots(n, n, figsize=(8, 8))

    for i in range(n):
        for j in range(n):
            idx = i * n + j
            if idx >= len(inputs):
                axes[i, j].axis('off')
                continue

            image = inputs[idx].numpy().transpose((1, 2, 0))
            image = np.clip(std * image + mean, 0, 1)

            title = class_names[labels[idx]]
            axes[i, j].imshow(image)
            axes[i, j].set_title(title, fontsize=8)
            axes[i, j].axis('off')

    plt.tight_layout()
    plt.show()

def get_optimizer(optimizer_name: str, model: torch.nn.Module, lr: float, **kwargs) -> torch.optim.Optimizer:
    """
    Returns a PyTorch optimizer based on the specified optimizer name.

    Args:
        optimizer_name (str): Name of the optimizer to use. Supported values are "sgd", "adam", "adamw", "rmsprop".
        model (torch.nn.Module): The model whose parameters will be optimized.
        lr (float): Learning rate for the optimizer.
        **kwargs: Additional keyword arguments to pass to the optimizer.

    Returns:
        torch.optim.Optimizer: Instantiated optimizer object.

    Raises:
        ValueError: If an unsupported optimizer name is provided.
    """
    optimizer_name = optimizer_name.lower()
    if optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, **kwargs)
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=lr, **kwargs)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    

    

def save_training_results(results_dict, file_path):
    """
    Save a dictionary of training results to a text file in a readable JSON format.
    
    Args:
        results_dict (dict): The dictionary containing training results.
        file_path (str): The path to the file where the results will be saved.
    """
    with open(file_path, 'w') as f:
        # Convert the dictionary to a JSON string with indentation for readability
        json.dump(results_dict, f, indent=4)

    print(f"Training results saved to {file_path}")
    
    
def plot_loss_acc_curves(results, save_path, title):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "val_loss": [...],
             "val_acc": [...]}
        save_path (str): path to save the plot.
        title (str): global title for the plot.
    """
    # Extract loss and accuracy values
    loss = results["train_loss"]
    val_loss = results["val_loss"]

    accuracy = results["train_acc"]
    val_accuracy = results["val_acc"]

    epochs = range(len(results["train_loss"]))

    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.grid()
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.grid()
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Add global title in bold
    plt.suptitle(f"{title}", fontweight="bold")

    # Save the plot to the specified path
    plt.savefig(save_path)




def predict_and_visualize(image_path, model, transform, class_names, device='cpu'):
    """
    Predicts the class of an image using a trained model and visualizes it.

    Args:
        image_path (str): Path to the input image (PIL-compatible).
        model (torch.nn.Module): Trained PyTorch model for classification.
        transform (callable): Transformation function to preprocess the image.
        class_names (list): List of class names indexed by label.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        str: Predicted class label.
    """
    # Load and transform image
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Set model to eval mode and predict
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, pred_idx = torch.max(output, 1)
        pred_label = class_names[pred_idx.item()]

    # Visualize
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image)
    ax.set_title(f"Predicted class: {pred_label}")
    ax.axis('off')
    plt.show()

    return pred_label


def print_train_time(start, end, device=None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"\nTime on {device}: {total_time:.3f} seconds")

    return round(total_time,3)


############### Part 2 functions ##############################################################################################################