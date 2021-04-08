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

torch.backends.cudnn.deterministic = True
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)



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
    test_kwargs = {'batch_size': args.batch_size}
    
    if torch.cuda.is_available():
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    
    midas_train, midas_val = midas_task1_split(args.path+"/processed")
    mnist_test = mnist_testloader()

    midas_train_loader = torch.utils.data.DataLoader(midas_train, **train_kwargs)
    midas_val_loader = torch.utils.data.DataLoader(midas_val, **train_kwargs)
    mnist_test_loader = torch.utils.data.DataLoader(mnist_test, **test_kwargs)

    print("==> Building model...")
    midas_model = Net3().to(device)

    mnist_model = Net3().to(device)

    optimizer = optim.Adam(midas_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    if args.wandb:
        wandb.watch(midas_model)
        wandb.watch(mnist_model)


    
    for epoch in range(1, args.epochs + 1):
        print(f"==> Epoch {epoch}/{args.epochs + 1}")
        print("==> Model training started")

        train(args, midas_model, device, midas_train_loader, optimizer, epoch,criterion)
        
        print("==> Evaluating midas model on midas")
        midas_accu = val_midas(midas_model, device, midas_val_loader, args,criterion,epoch)
        

        print("==> Saving model checkpoint")
        is_best = midas_accu > best_accu
        best_accu = max(midas_accu,best_accu)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': midas_model.state_dict(),
            'best_accu': best_accu,
            'optimizer': optimizer.state_dict(),
        }, is_best, fdir)

        print("==> Loading model checkpoint")
        mnist_model = load_ckpt(mnist_model, args)
        mnist_model.fc2.out_channels = 10
        print(mnist_model)

        print("==> Testing model on mnist")
        mnist_accu = test(mnist_model, device, mnist_test_loader, args,criterion,epoch)

        print("==> Accuracy per class on midas val")
        accuracy_per_class(midas_model,midas_val.classes, device, midas_val_loader,args,epoch)

        scheduler.step()
        print("Lr after scheduler = ",optimizer.param_groups[0]['lr'])
        
        if args.wandb:
            wandb.log({"Lr": optimizer.param_groups[0]['lr']}, step = epoch)

        
    
    print(f"Best accuracy on testing set = {best_accu}")


if __name__ == '__main__':
    main()
