import torch.functional as F
import torch
from typing import Tuple
from tqdm import tqdm


def train(model, device, train_loader, optimizer, scheduler, loss_fn) -> Tuple[float, float]:
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar):
        # get samples
        data, target = data.to(device), target.to(device)

        # Init
        optimizer.zero_grad()

        # Predict
        y_pred = model(data)

        # Calculate loss
        loss = loss_fn(y_pred, target)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Update pbar-tqdm

        # get the index of the max log-probability
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        accuracy = 100*correct/processed

        pbar.set_description(
            desc=f'Loss={loss.item()} Batch_id={batch_idx} Accuracy={accuracy:0.2f}')

    return accuracy, loss


def validate(model, device, val_loader, loss_fn) -> Tuple[float, float]:
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_fn(output, target).item()
            # test_loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        accuracy))

    return accuracy, test_loss


def train_model(model, device, train_loader,val_loader,optimizer, scheduler,criterion,EPOCHS=50,model_path='model.pth'):
    train_losses = []
    train_acc = []
    test_losses = []
    test_acc = []

    best_test_acc = 0
    model_path = 'session11_model.pth'

    lrs = []
    for epoch in range(5):
        print("EPOCH:", epoch+1)
        #train
        train_epoch_acc, train_epoch_loss = train(
            model, device, train_loader, optimizer, criterion)
        train_losses.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)

        #test
        test_epoch_acc, test_epoch_loss = validate()(
            model, device, val_loader, criterion)
        test_losses.append(test_epoch_loss)
        test_acc.append(test_epoch_acc)

        scheduler.step()

        # remember best accuracy and save the model
        is_best = test_epoch_acc > best_test_acc
        best_test_acc = max(test_epoch_acc, best_test_acc)

        if is_best:
            print('Saving Model for accuracy: ', test_epoch_acc)
            torch.save(model.state_dict(), model_path)

        lrs.append(scheduler.get_last_lr())
    
    return train_losses,train_acc, test_losses, test_acc



#def 