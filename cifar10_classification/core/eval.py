import torch

@torch.no_grad()
def test_loop(model, valid_loader, loss_fn, device):
    """1エポックのテストを行う関数
    args:
        model(nn.Module): モデル
        valid_loader(torch.utils.data.DataLoader): 
        optimizer(torch.optim): 
        loss_fn(torch.Module): 損失関数
        device(torch.device):
    returns:
        dict[str, Any]: 学習の損失や精度などの情報
    """
    model.eval()
    epoch_loss = 0
    correct = 0
    all_count = 0

    for images, labels in valid_loader:
        images, labels = images.to(device), labels.to(device)

        output = model(images)
        loss = loss_fn(output, labels)

        epoch_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        all_count += labels.size(0)
        correct += pred.eq(labels.view_as(pred)).sum().item()

    test_acc = correct / all_count
    epoch_loss = epoch_loss / len(valid_loader)

    return {
        "test_loss": epoch_loss,
        "test_acc": test_acc
    }