from torch.utils import data
from torchvision import transforms

input_size = 224
imagenet_mean, imagenet_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])

test_transform = transforms.Compose([
    transforms.Resize(input_size),
    transforms.CenterCrop(input_size),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std)
])



class EuroSAT(data.Dataset):
    """
    PyTorch Dataset wrapper for the EuroSAT dataset.

    This class wraps a preloaded EuroSAT dataset (e.g., a list of (image, label) tuples)
    and optionally applies transformations to the input images.

    Attributes:
        dataset (list or Dataset): The underlying dataset containing image-label pairs.
        transform (callable, optional): A function/transform to apply to the input images.

    Methods:
        __getitem__(index): Returns the transformed image and label at the specified index.
        __len__(): Returns the total number of samples in the dataset.
    """

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        # Apply image transformations
        if self.transform:
            x = self.transform(self.dataset[index][0])
        else:
            x = self.dataset[index][0]
        # Get class label
        y = self.dataset[index][1]
        return x, y

    def __len__(self):
        return len(self.dataset)


