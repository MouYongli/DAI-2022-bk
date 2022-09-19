import sys
sys.path.append('..')
import os
import os.path as osp

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

import time
import copy
import argparse

def fix_all_seeds(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
fix_all_seeds(2021)

from datasets import get_dataloaders
from models import get_model

here = osp.dirname(osp.abspath(__file__))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch FedBN")
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--num-data-mnist', type=int, default=60000, help='number of data of MNIST (max 60000)')
    parser.add_argument('--num-data-svhn', type=int, default=73257, help='number of data of SVHN (max 73257)')
    parser.add_argument('--num-data-usps', type=int, default=7291, help='number of data of USPS')
    parser.add_argument('--num-data-synth', type=int, default=479400, help='number of data of Synthetic Digits')
    parser.add_argument('--num-data-mnistm', type=int, default=60000, help='number of data of MNIST-M')

    parser.add_argument('--model', type=str, default="digits", help='model types')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch-size', type = int, default= 32, help ='batch size')

    parser.add_argument('--num-rounds', type = int, default=10, help = 'number of rounds for communication') #100
    parser.add_argument('--num-epochs', type = int, default=1, help = 'number epochs of local training on client between communication')

    parser.add_argument('--dp', type = str, default='none', help='differential privacy type: none, gaussian, laplacian')
    args = parser.parse_args()

    exp_dir = osp.join(here, "../../Experiments", f"d_{args.num_data_mnist}_{args.num_data_svhn}_{args.num_data_usps}_{args.num_data_synth}_{args.num_data_mnistm}", "fedbn", f"r_{args.num_rounds}_e_{args.num_epochs}")
    if not osp.exists(exp_dir):
        os.makedirs(exp_dir)

    log_headers = [
        'round',
        'epoch',
        'dataset',
        'stage',
        'loss',
        'acc',
    ]
    if not osp.exists(osp.join(exp_dir, 'log.csv')):
        with open(osp.join(exp_dir, 'log.csv'), 'w') as f:
            f.write(','.join(log_headers) + '\n')

    if args.log:
        log_path = os.path.join(exp_dir, "log")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'fedavg.log'), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch size: {}\n'.format(args.batch_size))
        logfile.write('    rounds: {}\n'.format(args.num_rounds))
        logfile.write('    epochs: {}\n'.format(args.num_epochs))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    ## set up dataloaders, models and loss functino
    train_loaders, test_loaders = get_dataloaders(args)
    ### name of each client dataset
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
    num_clients = len(datasets)
    num_data_on_clients = [args.num_data_mnist, args.num_data_svhn, args.num_data_usps, args.num_data_synth, args.num_data_mnistm]
    client_alpha = [num_data / sum(num_data_on_clients) for num_data in num_data_on_clients]
    print('Client alpha:', client_alpha)
    server_model = get_model(args).to(device)
    client_model_set = [copy.deepcopy(server_model).to(device) for idx in range(num_clients)]

    loss_fun = nn.CrossEntropyLoss()
    ## start training
    for rd in range(args.num_rounds):
        print("============ Communication round {} ============".format(rd))
        if args.log: logfile.write("============ Communication round {} ============".format(rd))

        optimizers = [optim.SGD(params=client_model_set[idx].parameters(), lr=args.lr) for idx in range(num_clients)]
        for ep in range(args.num_epochs):
            print("============ Train epoch {} ============".format(ep))
            if args.log:
                logfile.write("============ Train epoch {} ============\n".format(ep))

            for client_idx in range(num_clients):
                ## training local model
                model, train_loader, optimizer = client_model_set[client_idx], train_loaders[client_idx], optimizers[client_idx]
                model.train()
                num_data = 0
                correct = 0
                loss_all = 0
                train_iter = iter(train_loader)
                for step in range(len(train_iter)):
                    optimizer.zero_grad()
                    x, y = next(train_iter)
                    num_data += y.size(0)
                    x = x.to(device).float()
                    y = y.to(device).long()
                    output = model(x)
                    loss = loss_fun(output, y)
                    loss.backward()
                    loss_all += loss.item()
                    optimizer.step()
                    pred = output.data.max(1)[1]
                    correct += pred.eq(y.view(-1)).sum().item()
                train_loss, train_acc = loss_all / len(train_iter), correct / num_data
                print(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], train_loss, train_acc))
                with open(osp.join(exp_dir, 'log.csv'), 'a') as f:
                    log = [rd, ep, datasets[client_idx], 'train', train_loss, train_acc]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')
                if args.log:
                    logfile.write(' {:<11s}| Train Loss: {:.4f} | Train Acc: {:.4f}\n'.format(datasets[client_idx], train_loss, train_acc))

        ## communication for model aggregation
        with torch.no_grad():
            # aggregate params
            for key in server_model.state_dict().keys():
                if 'bn' not in key:
                    temp = torch.zeros_like(server_model.state_dict()[key], dtype=torch.float32)
                    for client_idx in range(num_clients):
                        temp += client_alpha[client_idx] * client_model_set[client_idx].state_dict()[key]
                    server_model.state_dict()[key].data.copy_(temp)
                    for client_idx in range(num_clients):
                        client_model_set[client_idx].state_dict()[key].data.copy_(server_model.state_dict()[key])

        # start testing
        print("============ Test ============")
        if args.log: logfile.write("============ Test ============\n")
        for test_idx, test_loader in enumerate(test_loaders):
            model = client_model_set[test_idx]
            model.eval()
            test_loss = 0
            correct = 0
            targets = []
            for data, target in test_loader:
                data = data.to(device).float()
                target = target.to(device).long()
                targets.append(target.detach().cpu().numpy())
                output = model(data)
                test_loss += loss_fun(output, target).item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1)).sum().item()
            test_loss, test_acc = test_loss / len(test_loader), correct / len(test_loader.dataset)
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            with open(osp.join(exp_dir, 'log.csv'), 'a') as f:
                log = [rd, '', datasets[test_idx], 'test', test_loss, test_acc]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            if args.log:
                logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss,
                                                                                          test_acc))

    torch.save({
            'model_0': client_model_set[0].state_dict(),
            'model_1': client_model_set[1].state_dict(),
            'model_2': client_model_set[2].state_dict(),
            'model_3': client_model_set[3].state_dict(),
            'model_4': client_model_set[4].state_dict(),
            'server_model': server_model.state_dict(),
    }, osp.join(exp_dir, "fedbn.pth"))
    if args.log:
        logfile.flush()
        logfile.close()