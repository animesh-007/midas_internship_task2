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
from model_nologsoftmax.model_nologsoftmax import *
from cosine_annearing_with_warmup import CosineAnnealingWarmupRestarts
import warnings
warnings.filterwarnings("ignore")
# from model_logsoftmax.task1_model import Net3

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
        # output = F.log_softmax(x, dim=1)
        # return output
        return x



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
    midas_train, midas_val = midas_task1_split(args.path+"/processed")
    # midas_train = midas_mnist_trainloader(args.path) 
    # midas_train = midas_full_digits_dataset(args.path)

    print("==> Loading MNIST dataset")
    # mnist_train = mnist_trainloader()
    mnist_test = mnist_testloader()

    midas_train_loader = torch.utils.data.DataLoader(midas_train, **train_kwargs)
    # midas_train_loader = torch.utils.data.DataLoader(midas_train, **train_kwargs)
    midas_val_loader = torch.utils.data.DataLoader(midas_val, **val_kwargs)
    # train_loader = torch.utils.data.DataLoader(midas_train, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(mnist_test, **test_kwargs)

    print("==> Building model...")
    midas_model = Net().to(device)

    # print("==> Loading model checkpoint")
    # load_ckpt(midas_model, args)
    # midas_model.fc2.out_features = 10
    print(midas_model)
    mnist_model = Net().to(device)
    # print(mnist_model)

    optimizer = optim.Adadelta(midas_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    # criterion = nn.NLLLoss()
    
    # scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    scheduler = CosineAnnealingWarmupRestarts(optimizer,
                                          first_cycle_steps=args.epochs,
                                          cycle_mult=1.0,
                                          max_lr=1.0,
                                          min_lr=args.lr,
                                          warmup_steps=5,
                                          gamma=1.0)    
    if args.wandb:
        wandb.watch(midas_model)
        wandb.watch(mnist_model)


    print(f"==> Starting Learning Rate {args.lr}")
    for epoch in range(1, args.epochs + 1):
        print(f"==> Epoch {epoch}/{args.epochs + 1}")
        print("==> Model training started")

        train(args, midas_model, device, midas_train_loader, optimizer, epoch,criterion)
        
        print("==> Evaluating midas model on midas")
        midas_accu = val_midas(midas_model, device, midas_val_loader, args,criterion,epoch)

        print(f"==> Saving model checkpoint at {fdir}")
        is_best = midas_accu > best_accu
        best_accu = max(midas_accu,best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': midas_model.state_dict(),
            'best_accu': best_accu,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)
        


        print("==> Loading model checkpoint")
        load_ckpt(mnist_model, args)
        mnist_model.fc2.out_features = 10
        print(mnist_model)

        print("==> Testing model on mnist")
        mnist_accu = test(mnist_model, device, test_loader, args,criterion,epoch)

        # print(f"==> Saving model checkpoint at {fdir}")
        # is_best = midas_accu > best_accu
        # best_accu = max(midas_accu,best_accu)
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'state_dict': midas_model.state_dict(),
        #     'best_accu': best_accu,
        #     'optimizer': optimizer.state_dict(),
        # }, is_best, fdir)

        print("==> Accuracy per class on midas val")
        accuracy_per_class(midas_model,midas_val.classes, device, midas_val_loader,args,epoch)

        # print("==> Accuracy per class on mnist test")
        # accuracy_per_class(midas_model,mnist_test.classes, device, test_loader,args,epoch)

        scheduler.step()
        print("Lr after scheduler = ",optimizer.param_groups[0]['lr'])
        
        # if args.wandb:
        #     wandb.log({"Lr": optimizer.param_groups[0]['lr']}, step = epoch)

        
    
    print(f"Best accuracy on testing set = {best_accu}")


if __name__ == '__main__':
    main()