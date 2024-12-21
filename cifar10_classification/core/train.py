
def train_loop(model, train_loader, optimizer, loss_fn, device):
    """1エポックの学習を行う関数
    args:
        model(nn.Module): モデル
        train_loader(torch.utils.data.DataLoader): 
        optimizer(torch.optim): 
        loss_fn(torch.Module): 損失関数
        device(torch.device):
    returns:
        dict[str, Any]: 学習の損失や精度などの情報
    """
    model.train()
    epoch_loss = 0
    correct = 0
    all_count = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = loss_fn(output, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        all_count += labels.size(0)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    train_acc = correct / all_count
    epoch_loss = epoch_loss / len(train_loader)

    return {
        "train_loss": epoch_loss,
        "train_acc": train_acc
    }