import numpy as np
import rasterio as rio
from rasterio.mask import mask
from PIL import Image
import torch


def crop_tile(image_path, shape):
    """
    Crop a raster image using a polygon geometry and return the result as a PIL Image.

    Args:
        image_path (str): Path to the raster (.tif) file.
        shape (list): List containing a single shapely geometry (e.g., [tiles.iloc[i].geometry]).

    Returns:
        PIL.Image.Image: Cropped tile image (converted to HWC, uint8 format).
    """
    # Open raster file
    with rio.open(image_path) as src:
        # Crop raster with polygon geometry
        out_image, _ = mask(src, shape, crop=True)

        # Remove black borders (pixels = 0)
        _, x_nonzero, y_nonzero = np.nonzero(out_image)
        out_image = out_image[
            :, np.min(x_nonzero):np.max(x_nonzero),
            np.min(y_nonzero):np.max(y_nonzero)
        ]

    # Convert CHW (bands, height, width) -> HWC (height, width, bands)
    out_image = np.moveaxis(out_image, 0, -1)

    # Cast to uint8 for PIL compatibility
    pil_img = Image.fromarray(out_image.astype(np.uint8))

    return pil_img


def batch_predict(images, classes, model, transform, batch_size=64):
    """
    Run batched inference on a list of images using a trained PyTorch model.

    Args:
        images (list[PIL.Image.Image]): List of cropped images.
        classes (list[str]): List of class names, indexed by model predictions.
        model (torch.nn.Module): Trained PyTorch model for classification.
        transform (callable): Transformation function to preprocess each image.
        batch_size (int, optional): Number of images per forward pass. Default = 64.

    Returns:
        list[str]: Predicted class labels for each input image.
    """
    results = []

    # Ensure model is in evaluation mode
    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        # Iterate over mini-batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]

            # Apply preprocessing transform and stack into a single tensor
            tensors = [transform(img).unsqueeze(0) for img in batch]
            inputs = torch.cat(tensors, dim=0).to(device)

            # Forward pass
            outputs = model(inputs)

            # Get class predictions
            preds = outputs.argmax(dim=1).cpu().numpy()
            results.extend([classes[p] for p in preds])

    return results
