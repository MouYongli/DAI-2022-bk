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

global eps
eps = 1e-5

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
from models import bayesian_kl_loss

here = osp.dirname(osp.abspath(__file__))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch Fed VBI")
    parser.add_argument('--log', action='store_true', help ='whether to make a log')
    parser.add_argument('--num-data-mnist', type=int, default=729, help='number of data of MNIST (max 60000)')
    parser.add_argument('--num-data-svhn', type=int, default=729, help='number of data of SVHN (max 73257)')
    parser.add_argument('--num-data-usps', type=int, default=729, help='number of data of USPS')
    parser.add_argument('--num-data-synth', type=int, default=729, help='number of data of Synthetic Digits')
    parser.add_argument('--num-data-mnistm', type=int, default=729, help='number of data of MNIST-M')

    parser.add_argument('--model', type=str, default="bayes", help='model types')
    parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
    parser.add_argument('--batch-size', type = int, default= 32, help ='batch size')

    parser.add_argument('--num-rounds', type = int, default=10, help = 'number of rounds for communication') #100
    parser.add_argument('--num-epochs', type = int, default=1, help = 'number epochs of local training on client between communication')
    parser.add_argument('--importance-bayes', type=float, default=0.1)
    parser.add_argument('--dp', type = str, default='none', help='differential privacy type: none, gaussian, laplacian')
    args = parser.parse_args()

    exp_dir = osp.join(here, "../../Experiments", f"d_{args.num_data_mnist}_{args.num_data_svhn}_{args.num_data_usps}_{args.num_data_synth}_{args.num_data_mnistm}", "fedvbi", f"r_{args.num_rounds}_e_{args.num_epochs}")
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
        logfile = open(os.path.join(log_path,'fedvbi.log'), 'a')
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

    # Inital client weights
    init_state_dict = server_model.state_dict()
    for client_idx in range(num_clients):
        client_model_set[client_idx].load_state_dict(init_state_dict)

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
            client_model_set[client_idx].train()
            for ep in range(args.num_epochs):
                num_data = 0
                correct = 0
                loss_all = 0
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    num_data += target.size(0)
                    optimizer.zero_grad()
                    # clientmodel_set[client_idx].reset_parameters()
                    output = client_model_set[client_idx](data)
                    ce_loss = loss_fun(output, target)
                    kl_local_loss = bayesian_kl_loss(client_model_set[client_idx], server_model) if args.importance_bayes > 0 else 0.0
                    # 定义loss，包括ce loss和kl loss
                    loss = ce_loss + args.importance_bayes * kl_local_loss
                    loss.backward()
                    optimizer.step()
                    pred = output.data.max(1)[1]
                    correct += pred.eq(target.view(-1)).sum().item()
                    loss_all += loss.item()
                train_loss, train_acc = loss_all / num_data, correct / num_data
                print(' {:<11s}| Epoch {:02d} | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], ep, train_loss, train_acc))
                if args.log: logfile.write(' {:<11s}| Epoch {:02d} | Train Loss: {:.4f} | Train Acc: {:.4f}'.format(datasets[client_idx], ep, train_loss, train_acc))
                with open(osp.join(exp_dir, 'log.csv'), 'a') as f:
                    log = [rd, ep, datasets[client_idx], 'train', train_loss, train_acc]
                    log = map(str, log)
                    f.write(','.join(log) + '\n')

        ## communication for model aggregation
        new_mu = {}
        new_sigma = {}
        mu_set = [{} for _ in range(num_clients)]
        log_sigma_set = [{} for _ in range(num_clients)]
        module_set = set([name.split("_")[0] for name, param in server_model.named_parameters()])
        with torch.no_grad():
            # aggregate params
            server_state_dict = server_model.state_dict()
            for m in module_set:
                new_mu[m + "_mu"] = torch.zeros_like(server_state_dict[m + "_mu"])
                new_sigma[m + "_log_sigma"] = torch.zeros_like(server_state_dict[m + "_log_sigma"])

            for client_idx in range(num_clients):
                client_state_dict = client_model_set[client_idx].state_dict()
                for m in module_set:
                    mu_set[client_idx][m + "_mu"] = client_state_dict[m + "_mu"]
                    log_sigma_set[client_idx][m + "_log_sigma"] = client_state_dict[m + "_log_sigma"]

            for m in module_set:
                for client_idx in range(num_clients):
                    new_mu[m+"_mu"] += client_alpha[client_idx] * mu_set[client_idx][m+"_mu"] / (torch.exp_(log_sigma_set[client_idx][m+"_log_sigma"]) + eps)
                    new_sigma[m+"_log_sigma"] += client_alpha[client_idx] / (torch.exp_(log_sigma_set[client_idx][m+"_log_sigma"]) + eps)
                new_mu[m+"_mu"] *= (new_sigma[m+"_log_sigma"] + eps)

            for m in module_set:
                server_model.state_dict()[m+"_mu"].data.copy_(new_mu[m+"_mu"])
                server_model.state_dict()[m+"_log_sigma"].data.copy_(torch.log_(new_sigma[m+"_log_sigma"]))  # https://discuss.pytorch.org/t/how-can-i-modify-certain-layers-weight-and-bias/11638

            for name, param in server_model.named_parameters():
                for client_idx in range(num_clients):
                    client_model_set[client_idx].state_dict()[name].data.copy_(param.data)


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
                output = client_model_set[client_idx](data)
                loss = loss_fun(output, target)
                test_loss += loss.item()
                pred = output.data.max(1)[1]
                correct += pred.eq(target.view(-1)).sum().item()
            test_loss, test_acc = test_loss / num_data, correct / num_data
            print(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}'.format(datasets[test_idx], test_loss, test_acc))
            with open(osp.join(exp_dir, 'log.csv'), 'a') as f:
                log = [rd, '', datasets[test_idx], 'test', test_loss, test_acc]
                log = map(str, log)
                f.write(','.join(log) + '\n')
            if args.log:
                logfile.write(' {:<11s}| Test  Loss: {:.4f} | Test  Acc: {:.4f}\n'.format(datasets[test_idx], test_loss,
                                                                                          test_acc))

    torch.save({
        'server_model': server_model.state_dict(),
    }, osp.join(exp_dir, "fedvbi.pth"))
    if args.log:
        logfile.flush()
        logfile.close()