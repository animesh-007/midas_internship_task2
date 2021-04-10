from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import wandb
from utils import *
from cosine_annealing_with_warmup import CosineAnnealingWarmupRestarts
import warnings
warnings.filterwarnings("ignore")

torch.backends.cudnn.deterministic = True
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

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
        output = self.fc2(x)
        
        return output

def main():
    best_accu = 0

    
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--wandb', action='store_true', default=False,
                        help='For wandb logging')
    parser.add_argument('--train', action='store_false', default=True,
                        help='Start training')
    parser.add_argument('--val', action='store_false', default=True,
                        help='Start validation')
    parser.add_argument('--test', action='store_false', default=True,
                        help='Start testing')
    parser.add_argument('--ckpt', action='store_true', default=False,
                        help='For loading ckpt')
    parser.add_argument('--per_class', action='store_false', default=True,
                        help='Calulate accuracy per class')
    parser.add_argument('--saved_ckpt', type=str, default="./checkpoints", metavar='saved_ckpt',
                        help='Path for saving the checkpoint')
    parser.add_argument('--load_ckpt', type=str, default="./checkpoints", metavar='load_ckpt',
                        help='For loading checkpoint')
    parser.add_argument('--path', type=str, default="./data", metavar='path',
                        help='For Training the model on midas task 3')

    args = parser.parse_args()

    if args.wandb:
        # wandb initalization
        print("==> wandb initalization of project")
        wandb.init(project="midas-tasks-solutions", reinit=True)
        wandb.config.update(args)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(f'{args.saved_ckpt}', exist_ok = True)
    fdir = f'{args.saved_ckpt}/run_with_epochs_{args.epochs}_LR_{args.lr}'
    os.makedirs(fdir,exist_ok = True)

    print("==> Loading dataset")
    train_kwargs = {'batch_size': args.batch_size}
    val_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        

    
    print("==> Loading Midas dataset")
    midas_train = midas_mnist_trainloader(args.path) 


    print("==> Loading MNIST dataset")
    mnist_test = mnist_testloader()

    midas_train_loader = torch.utils.data.DataLoader(midas_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(mnist_test, **test_kwargs)

    print("==> Building model...")
    midas_model = Net().to(device)

    if args.ckpt:
        print("==> Loading model checkpoint")
        load_ckpt(midas_model, args)
        print(midas_model)
    

    optimizer = optim.Adadelta(midas_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=args.epochs,
                                          cycle_mult=1.0,
                                          max_lr=1.0,
                                          min_lr=args.lr,
                                          warmup_steps=5,
                                          gamma=1.0)    
    if args.wandb:
        wandb.watch(midas_model)


    print(f"==> Starting Learning Rate {args.lr}")
    for epoch in range(1, args.epochs + 1):
        print(f"==> Epoch {epoch}/{args.epochs + 1}")
        
        if args.train:
            print("==> Model training started")
            train(args, midas_model, device, midas_train_loader, optimizer, epoch,criterion)

        if args.test:
            print("==> Testing model on mnist")
            mnist_accu = test(args, midas_model, device, test_loader, epoch, criterion)

            print(f"==> Saving model checkpoint at {fdir}")
            is_best = mnist_accu > best_accu
            best_accu = max(mnist_accu,best_accu)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': midas_model.state_dict(),
                'best_accu': best_accu,
                'optimizer': optimizer.state_dict(),
            }, is_best, fdir)
        
        if args.per_class:
            print("==> Accuracy per class on mnist test")
            accuracy_per_class(args, midas_model, device, test_loader, epoch, mnist_test.classes)

        scheduler.step()
        print("Lr after scheduler = ",optimizer.param_groups[0]['lr'])
        
    print(f"Best accuracy on testing set = {best_accu}")


if __name__ == '__main__':
    main()
