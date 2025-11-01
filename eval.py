import torch
import torch.nn as nn
from models.custom_net_tiny_imagenet import CustomNet
from data import get_tiny_imagenet_loaders  # funzione che prepara train_loader e val_loader
from utils.training_and_validation import validate

def main():
    model = CustomNet().cuda()
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    _,test_loader = get_tiny_imagenet_loaders()
    criterion = nn.CrossEntropyLoss()

    val_acc = validate(model, test_loader, criterion)

    print(f"Test Accuracy: {val_acc:.2f}%")

if __name__ == "__main__":
    main()
