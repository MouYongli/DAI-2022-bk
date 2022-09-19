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

global eps
eps = 1e-5

def get_csd_loss(client_model, mu, omega, round_num):
    loss_set = []
    for name, param in client_model.named_parameters():
        theta =client_model.state_dict()[name]
        loss_set.append((0.5 / round_num) * (omega[name] * ((theta - mu[name]) ** 2)).sum())
    return sum(loss_set)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch Fed OLA")
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

    parser.add_argument('--csd-importance', type=float, default=0)
    parser.add_argument('--clip', type=float, default=10)
    parser.add_argument('--dp', type = str, default='none', help='differential privacy type: none, gaussian, laplacian')
    args = parser.parse_args()

    exp_dir = osp.join(here, "../../Experiments", f"d_{args.num_data_mnist}_{args.num_data_svhn}_{args.num_data_usps}_{args.num_data_synth}_{args.num_data_mnistm}", "fola", f"r_{args.num_rounds}_e_{args.num_epochs}_csd_{args.csd_importance}")
    if not osp.exists(exp_dir):
        os.makedirs(exp_dir)

    log_headers = [
        'round',
        'epoch',
        'dataset',
        'stage',
        'loss',
        'csd',
        'acc',
    ]
    if not osp.exists(osp.join(exp_dir, 'log.csv')):
        with open(osp.join(exp_dir, 'log.csv'), 'w') as f:
            f.write(','.join(log_headers) + '\n')

    if args.log:
        log_path = os.path.join(exp_dir, "log")
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logfile = open(os.path.join(log_path,'fola.log'), 'a')
        logfile.write('==={}===\n'.format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
        logfile.write('===Setting===\n')
        logfile.write('    lr: {}\n'.format(args.lr))
        logfile.write('    batch size: {}\n'.format(args.batch_size))
        logfile.write('    rounds: {}\n'.format(args.num_rounds))
        logfile.write('    epochs: {}\n'.format(args.num_epochs))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    ## set up dataloaders, models and loss functinon
    ### name of each client dataset
    datasets = ['MNIST', 'SVHN', 'USPS', 'SynthDigits', 'MNIST-M']
    num_clients = len(datasets)
    train_loaders, test_loaders = get_dataloaders(args)

    num_data_on_clients = [args.num_data_mnist, args.num_data_svhn, args.num_data_usps, args.num_data_synth, args.num_data_mnistm]
    client_alpha = [num_data / sum(num_data_on_clients) for num_data in num_data_on_clients]
    print('Client alpha:', client_alpha)

    loss_fun = nn.CrossEntropyLoss()

    server_model = get_model(args).to(device)
    client_model_set = [get_model(args) for _ in range(num_clients)]
    server_omega = dict()
    client_omega_set = [dict() for _ in range(num_clients)]

    # Inital client weights
    init_state_dict = server_model.state_dict()
    for client_idx in range(num_clients):
        client_model_set[client_idx].load_state_dict(init_state_dict)
        for name, param in copy.deepcopy(client_model_set[client_idx]).named_parameters():
            client_omega_set[client_idx][name] = torch.zeros_like(param.data).to(device)
    for name, param in copy.deepcopy(server_model).named_parameters():
        server_omega[name] = torch.zeros_like(param.data).to(device)

    ## start training
    for rd in range(args.num_rounds):
        print("============ Communication round {} ============".format(rd))
        if args.log: logfile.write("============ Communication round {} ============".format(rd))
        ## training local model

        print("============ Train ============")
        if args.log: logfile.write("============ Train ============\n")
        for client_idx, train_loader in enumerate(train_loaders):
            client_model_set[client_idx] = client_model_set[client_idx].to(device)
            optimizer = optim.SGD(client_model_set[client_idx].parameters(), lr=args.lr)
            new_omega = dict()
            new_mu = dict()
            server_model_state_dict = server_model.state_dict()
            for name, param in client_model_set[client_idx].named_parameters():
                new_omega[name] = copy.deepcopy(server_omega[name])
                new_mu[name] = copy.deepcopy(server_model_state_dict[name])
            client_model_set[client_idx].train()
            for ep in range(args.num_epochs):
                num_data = 0
                correct = 0
                loss_all = 0
                log_ce_loss = 0
                log_csd_loss = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    num_data += target.size(0)
                    optimizer.zero_grad()
                    output = client_model_set[client_idx](data)
                    ce_loss = loss_fun(output, target)
                    csd_loss = get_csd_loss(client_model_set[client_idx], new_mu, new_omega, rd+1) if args.csd_importance > 0 else 0
                    ce_loss.backward(retain_graph=True)
                    for name, param in client_model_set[client_idx].named_parameters():
                        if param.grad is not None:
                            client_omega_set[client_idx][name] += (len(target) / num_data_on_clients[client_idx]) * param.grad.data.clone() ** 2
                    optimizer.zero_grad()
                    loss = ce_loss + args.csd_importance * csd_loss
                    loss.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_norm_(client_model_set[client_idx].parameters(), args.clip)
                    optimizer.step()
                    pred = output.data.max(1)[1]
                    correct += pred.eq(target.view(-1)).sum().item()
                    log_ce_loss += ce_loss.item()
                    log_csd_loss += csd_loss.item() if args.csd_importance > 0 else 0
                loss_all = log_ce_loss + log_csd_loss
                train_loss, train_ce_loss, train_csd_loss, train_acc = loss_all / len(train_loader), log_ce_loss / len(train_loader), log_csd_loss/ len(train_loader), correct / num_data
                print(' {:<11s}| Epoch {:02d} | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], ep, train_loss, train_acc))
                if args.log: logfile.write(' {:<11s}| Epoch {:02d} | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], ep, train_loss, train_acc))
                with open(osp.join(exp_dir, 'log.csv'), 'a') as f:
                    log = [rd, ep, datasets[client_idx], 'train', train_loss, train_csd_loss, train_acc]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')
            # client_model_set[client_idx] = client_model_set[client_idx].cpu()

        ## communication for model aggregation
        new_param = {}
        new_omega = {}
        with torch.no_grad():
            # aggregate params
            for name, param in server_model.named_parameters():
                new_param[name] = param.data.zero_()
                new_omega[name] = server_omega[name].data.zero_()
                for client_idx in range(num_clients):
                    new_param[name] += client_alpha[client_idx] * client_omega_set[client_idx][name] * client_model_set[client_idx].state_dict()[name].to(device)
                    new_omega[name] += client_alpha[client_idx] * client_omega_set[client_idx][name]
                new_param[name] /= (new_omega[name] + eps)

            for name, param in server_model.named_parameters():
                server_model.state_dict()[name].data.copy_(new_param[name])  # https://discuss.pytorch.org/t/how-can-i-modify-certain-layers-weight-and-bias/11638
                server_omega[name] = new_omega[name]
                for client_idx in range(num_clients):
                    client_model_set[client_idx].state_dict()[name].data.copy_(new_param[name].cpu())
                    client_omega_set[client_idx][name].data.copy_(new_omega[name])

        # start testing
        print("============ Test ============")
        if args.log: logfile.write("============ Test ============\n")
        for test_idx, test_loader in enumerate(test_loaders):
            client_model_set[test_idx] = client_model_set[test_idx].to(device)
            client_model_set[test_idx].eval()
            test_loss = 0
            correct = 0
            num_data = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                num_data += target.size(0)
                output = client_model_set[client_idx].eval()(data)
                test_loss += loss_fun(output, target).item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1)).sum().item()
            test_loss, test_acc = test_loss / len(test_loader), correct / num_data
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            with open(osp.join(exp_dir, 'log.csv'), 'a') as f:
                log = [rd, '', datasets[test_idx], 'test', test_loss, '', test_acc]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            if args.log:
                logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss,
                                                                                          test_acc))

    torch.save({
        'server_model': server_model.state_dict(),
    }, osp.join(exp_dir, "fola.pth"))
    if args.log:
        logfile.flush()
        logfile.close()