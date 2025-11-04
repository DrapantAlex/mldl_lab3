import torch
import wandb
def validate(model, val_loader, criterion):
    model.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss /= len(val_loader)
    val_acc = 100. * correct / total
    print(f"Validation: Loss {val_loss:.4f}, Acc {val_acc:.2f}%")
    return val_acc

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total
    print(f"Epoch {epoch}: Loss {train_loss:.4f}, Acc {train_acc:.2f}%")
    wandb.log({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_acc": train_acc
    })