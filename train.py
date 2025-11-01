import torch
import torch.nn as nn
from models.custom_net_tiny_imagenet.customnet import CustomNet  # importa la tua rete
from dataset.tiny_imagenet.dataloader import get_tiny_imagenet_loaders  # funzione che prepara train_loader e val_loader
from utils.training_and_validation import validate, train  # funzione di validazione


def main():
    train_loader, val_loader = get_tiny_imagenet_loaders()
    model = CustomNet().cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    num_epochs = 10
    best_acc = 0.0

    for epoch in range(1, num_epochs + 1):
        
        train(epoch, model, train_loader, criterion, optimizer)

        val_accuracy = validate(model, val_loader, criterion)
        best_acc = max(best_acc, val_accuracy)
        print(f"Epoch [{epoch}/{num_epochs}] - Val Acc: {val_accuracy:.2f}%")

    print(f"Best validation accuracy: {best_acc:.2f}%")
    torch.save(model.state_dict(), "best_model.pth")

if __name__ == "__main__":
    main()