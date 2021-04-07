from __future__ import print_function
import argparse
import os
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from cosine_annearing_with_warmup import CosineAnnealingWarmupRestarts
from PIL import ImageOps
import wandb

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch,criterion):
    model.train()
    running_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
            
            if args.wandb:
                wandb.log({"epoch": epoch, "loss": loss})



def test(model, device, test_loader,args):
    model.eval()
    test_loss = 0
    correct = 0
    example_images = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            if args.wandb:
                example_images.append(wandb.Image(
                    data[0], caption="Pred: {} Truth: {}".format(pred[0].item(), target[0])))

    test_loss /= len(test_loader.dataset)
    
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        accuracy))
    
    if args.wandb:
        wandb.log({
            "Examples": example_images,
            "Test Accuracy": 100. * correct / len(test_loader.dataset),
            "Test Loss": test_loss})

    return accuracy

def accuracy_per_class(net, classes, device, testloader,args):
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data, target in testloader:
            images, labels = data.to(device), target.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    accuracies = []
    accuracy = 0
    # import pdb; pdb.set_trace()
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
    
def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'checkpoint_model_best.pth.tar'))
        

    


def main():
    best_accu = 0
    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='For wandb logging')
    parser.add_argument('--mnist', action='store_true', default=False,
                        help='For mnist dataset loader')
    parser.add_argument('--saved_ckpt', type=str, default="./checkpoints",
                        help='For loading checkpoint')
    parser.add_argument('--load_ckpt', action='store_true', default=False,
                        help='For loading checkpoint')
    parser.add_argument('--split_dataset', action='store_true', default=False,
                        help='For Training the model on midas task 1 split set')
    parser.add_argument('--path', type=str, default="./subtask1",
                        help='For Training the model on midas task 1 split set')
    
    
    args = parser.parse_args()
    
    if args.wandb:
        # wandb initalization
        print("==> wandb initalization of project")
        wandb.init(project="midas-task", reinit=True)
        wandb.config.update(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    
    
    os.makedirs(f'{args.saved_ckpt}', exist_ok = True)
    fdir = f'{args.saved_ckpt}/run_with_epochs_{args.epochs}_LR_{args.lr}'
    os.makedirs(fdir,exist_ok = True)

  
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
    torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

    device = torch.device("cuda" if use_cuda else "cpu")

    print("==> Loading dataset")
    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': 64}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    
    train_transforms_mnist=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])

    test_transforms_mnist=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))])
    
    if args.mnist:
        print("==> Loading MNIST dataset")
        dataset1 = datasets.MNIST('../MNIST',train=True,download=True,
                         transform=train_transforms_mnist)
        dataset2 = datasets.MNIST('../MNIST',train=False,download=True,
                           transform=test_transforms_mnist)
    else:
        transform=transforms.Compose([
        transforms.CenterCrop((900,900)),
        ImageOps.invert,
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28,28)),                                     
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))])
         
        
        if args.split_dataset:
            print("==> Loading splitted Midas dataset of subtask 1")
            dataset1 = datasets.ImageFolder(f'{args.path}/processed/train',
                            transform=transform)
            dataset2 = datasets.ImageFolder(f'{args.path}/processed/val',
                            transform=transform)
        else:
            print("==> Loading full Midas dataset of subtask 1")
            dataset1 = datasets.ImageFolder(f'{args.path}/train',
                       transform=transform)
            dataset2 = datasets.MNIST(root='../MNIST', train=False,
                                 transform=torchvision.transforms.ToTensor(),
                                 target_transform=torchvision.transforms.Compose([
                                  # or just torch.tensor
                                 lambda x:F.one_hot(torch.tensor(x),62)]),
                                 download=True)

    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    classes = dataset2.classes

    print("==> Building model...")
    model = Net().to(device)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=200,max_lr=1, gamma=args.gamma)
    
    
    if args.load_ckpt:
        print(f'=> loading checkpoint {args.saved_ckpt}/run_with_{args.epochs}_LR_{args.lr}')
        checkpoint = torch.load(f"{args.saved_ckpt}/run_with_{args.epochs}_LR_{args.lr}/checkpoint_model_best.pth.tar")
        # # # # # args.start_epoch = checkpoint['epoch']
        # # # # best_accu = checkpoint['best_accu']
        model.load_state_dict(checkpoint['state_dict'])
        # # # # # optimizer.load_state_dict(checkpoint['optimizer'])
        print(f'=> loaded checkpoint {args.saved_ckpt}/run_with_{args.epochs}_LR_{args.lr}')
    
        print(model)

    if args.mnist:
        num_ftrs = model.fc2.in_features
        # # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        model.fc2 = nn.Linear(num_ftrs, 10)
        model.to(device)
        print(model)

    if args.wandb:
        wandb.watch(model)
    
    for epoch in range(1, args.epochs + 1):
        print("==> Model training started")
        train(args, model, device, train_loader, optimizer, epoch,criterion)
        print("==> Evaluating model")
        accu = test(model, device, test_loader, args)
        print("==> Accuracy per class")
        accuracy_per_class(model, classes, device, test_loader,args)
        scheduler.step()
        print("Lr after scheduler = ",optimizer.param_groups[0]['lr'])

        is_best = accu > best_accu

        print("==> Saving model checkpoint")
        best_accu = max(accu,best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc': best_accu,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)
    
    print(f"Best accuracy on testing set = {best_accu}")


if __name__ == '__main__':
    main()