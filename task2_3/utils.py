import os
import shutil
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import ImageOps
import wandb

def train(args, model, device, train_loader, optimizer, epoch,criterion):
    """
    args: training parameters
    model: CNN model
    device: cpu or gpu
    train_loader: train loader
    optimizer: Optimizer 
    epoch: current epoch
    criterion: loss function 
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    train_loss /= len(train_loader.dataset)

    accuracy = 100. * correct / len(train_loader.dataset)

    print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    train_loss, correct, len(train_loader.dataset),
    accuracy))

    if args.wandb:
        wandb.log({"Train Loss": train_loss, "Train Accuracy": accuracy,"Lr": optimizer.param_groups[0]['lr']}, step = epoch)
    
    return accuracy

def val_midas(args, model, device, test_loader,epoch,criterion):
    """
    args: validation parameters
    model: CNN model
    device: cpu or gpu
    test_loader: val loader
    epoch: current epoch
    criterion: loss function 
    """
    model.eval()
    test_loss = 0
    correct = 0
    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if args.wandb:
                example_images.append(wandb.Image(
                    data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)
    
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\n Val set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    if args.wandb:
        wandb.log({
            "Examples": example_images,
            "Val Accuracy": 100. * correct / len(test_loader.dataset),
            "Val Loss": test_loss}, step = epoch)

    return accuracy


def test(args, model, device, test_loader, epoch, criterion):
    """
    args: testing parameters
    model: CNN model
    device: cpu or gpu
    test_loader: test loader
    epoch: current epoch
    criterion: loss function 
    """
    model.eval()
    test_loss = 0
    correct = 0
    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            if args.wandb:
                example_images.append(wandb.Image(
                    data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)
    
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\n Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    if args.wandb:
        wandb.log({
            "Examples": example_images,
            "Test Accuracy": 100. * correct / len(test_loader.dataset),
            "Test Loss": test_loss}, step = epoch)

    return accuracy

def accuracy_per_class(args, model, device, testloader, epoch, classes):
    """
    args: accuracy per class parameters
    model: CNN model
    device: cpu or gpu
    test_loader: test loader
    epoch: current epoch
    classes: Classes in the test datatset
    """
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data, target in testloader:
            images, labels = data.to(device), target.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracies = []
    accuracy = 0
    for i in range(len(classes)):
        accuracy = 100 * class_correct[i] / class_total[i]
        print('Accuracy of %5s : %2d %%' % (
            classes[i], accuracy ))
        accuracies.append(accuracy)
    
    if args.wandb:    
        data = [[name, accuracy] for (name, accuracy) in zip(classes, accuracies)]
        table = wandb.Table(data=data, columns=["class names", "Accuracy"])
        wandb.log({"my_bar_chart_id" : wandb.plot.bar(table, "class names",
                   "Accuracy", title="Per Class Accuracy")})

def mnist_trainloader():
    """
    MNIST training dataset loader
    """

    mnist_ransforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
    
    dataset1 = datasets.MNIST('./MNIST',train=True,download=True,
                        transform=mnist_ransforms)
    return dataset1

def mnist_testloader():
    """
    MNIST Testing dataset loader
    """
    mnist_ransforms=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
    
    dataset2 = datasets.MNIST('./MNIST',train=False,download=True,
                        transform=mnist_ransforms)

    return dataset2
    
def midas_task1_split(path):
    """
    Task 1 splitted dataset loader
    path: path to the dataset
    """
    transform=transforms.Compose([
        transforms.CenterCrop((900,900)),
        ImageOps.invert,
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),                                     
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
        )
    
    dataset1 = datasets.ImageFolder(f'{path}/train',
                    transform=transform)
    dataset2 = datasets.ImageFolder(f'{path}/val',
                    transform=transform)
    
    return dataset1, dataset2

def midas_full_digits_dataset(path):
    """
    Task 2 subset of Task 1 dataset containing only digits loader
    path: path to the dataset
    """
    transform=transforms.Compose([
        transforms.CenterCrop((900,900)),
        ImageOps.invert,
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),                                     
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))]
        )
    
    dataset1 = datasets.ImageFolder(f'{path}/train',
                        transform=transform)

    return dataset1

def midas_mnist_trainloader(path):
    """
    Task 3  dataset loader
    path: path to the dataset
    """
    mnist_ransforms=transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
    
    dataset1 = datasets.ImageFolder(f'{path}/mnistTask',
                        transform=mnist_ransforms)
    return dataset1

def save_checkpoint(state, is_best, fdir):
    """
    Save checkpoint function
    state: current state
    is_best: If true best model will be save
    fdir: path for saving the checkpoint
    """
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'checkpoint_model_best.pth.tar'))

def load_ckpt(model, path):
    """
    Load checkpoint function
    model: model for loading the weights
    path: path to the checkpoint
    """
    print(f'=> loading checkpoint from {path}')
    checkpoint = torch.load(f"{path}/checkpoint_model_best.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    print(f'=> loaded checkpoint {path}')
