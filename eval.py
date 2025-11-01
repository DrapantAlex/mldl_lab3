import torch
import torch.nn as nn
from models.custom_net_tiny_imagenet.customnet import CustomNet
from data.tiny_imagenet.dataloader import get_tiny_imagenet_loaders  # funzione che prepara train_loader e val_loader
from utils.training_and_validation import validate

def main():
    model = CustomNet().cuda()
    model.eval()

    _,test_loader = get_tiny_imagenet_loaders()
    criterion = nn.CrossEntropyLoss()

    val_acc = validate(model, test_loader, criterion)

    print(f"Test Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()
