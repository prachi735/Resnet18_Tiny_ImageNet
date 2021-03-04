import random
import numpy as np
import torch
import matplotlib.pyplot as plt


def seed_everything(seed=1):
    ''' Seed for reproducability '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # is cuda available
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.manual_seed_all(seed)


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    return device


def get_misclassified_images(gbn_model, test_loader):
    test_images = []
    target_labels = []
    target_predictions = []
    for img, target in test_loader:
        prediction = torch.argmax(gbn_model(img), dim=1)
        test_images.append(img)
        target_labels.append(target)
        target_predictions.append(prediction)

    test_images = torch.cat(test_images)
    target_labels = torch.cat(target_labels)
    target_predictions = torch.cat(target_predictions)
    misclassified_index = target_labels.ne(target_predictions).numpy()
    test_images = test_images[misclassified_index]
    target_labels = target_labels[misclassified_index]
    target_predictions = target_predictions[misclassified_index]

    return test_images, target_labels, target_predictions


def plot_results(train_losses, train_acc, test_losses, test_acc):
    data = {'train_loss': train_losses,  'train_acc': train_acc,
            'test_loss': test_losses,  'test_acc': test_acc}
    _, axs = plt.subplots(1, 4, figsize=(30, 5))
    axs_pos = {'train_loss': (0),
               'train_acc': (1),
               'test_loss': (0),
               'test_acc': (1)}

    for i in data:
        ax = axs[axs_pos[i]]
        ax.plot(data[i])
        ax.set_title(i)


def show_misclassified_images(test_images, target_labels, target_predictions, nrow=5, ncol=5):
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=(15, 15))
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Misclassified Images in GBN Model')

    for ax, image, target, prediction in zip(axes.flatten(), test_images, target_labels, target_predictions):
        ax.imshow(np.uint8(torch.Tensor.cpu(image[0].permute(2, 1, 0))))
        ax.set(title='target:{t} prediction:{p}'.format(
            t=target.item(), p=prediction.item()))
        ax.axis('off')


def show_images_predictions(test_images, target_labels, target_predictions, classes, nrow=5, ncol=5, fig_size=(3, 3)):
    fig, axes = plt.subplots(nrows=nrow, ncols=ncol, figsize=fig_size)
    fig.subplots_adjust(hspace=0.5)
    fig.suptitle('Misclassified Images in Model')
    for ax, image, target, prediction in zip(axes.flatten(), test_images, target_labels, target_predictions):
        ax.imshow(np.uint8(torch.Tensor.cpu(image[0].permute(2, 1, 0))))
        ax.set(title='target:{t} prediction:{p}'.format(
            t=classes[target.item()], p=classes[prediction.item()]))
        ax.axis('off')


def denormalize(tensor, mean, std):
    if not tensor.ndimension() == 4:
        raise TypeError('tensor should be 4D')

    mean = torch.FloatTensor(mean).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)
    std = torch.FloatTensor(std).view(
        1, 3, 1, 1).expand_as(tensor).to(tensor.device)

    return tensor.mul(std).add(mean)
