from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

def get_tiny_imagenet_loaders(root="data/tiny-imagenet-200", batch_size=32):
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = ImageFolder(root=f"{root}/train", transform=transform)
    val_dataset = ImageFolder(root=f"{root}/val", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
