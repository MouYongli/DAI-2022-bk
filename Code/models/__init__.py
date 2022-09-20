import torch
from .models import DigitModel, AlexNet, DigitBayesModel, DigitVariationalModel
from .torchbnn.linear import BayesLinear
from .torchbnn.batchnorm import BayesBatchNorm2d, BayesBatchNorm1d
from .torchbnn.conv import BayesConv2d

def _kl_loss(mu_0, log_sigma_0, mu_1, log_sigma_1):
    """
    An method for calculating KL divergence between two Normal distribtuion.
    Arguments:
        mu_0 (Float) : mean of normal distribution.
        log_sigma_0 (Float): log(standard deviation of normal distribution).
        mu_1 (Float): mean of normal distribution.
        log_sigma_1 (Float): log(standard deviation of normal distribution).

    """
    kl = log_sigma_1 - log_sigma_0 + (torch.exp(log_sigma_0) ** 2 + (mu_0 - mu_1) ** 2) / (2 * torch.exp(log_sigma_1) ** 2) - 0.5
    return kl.sum()


def bayesian_kl_loss(model1, model2, reduction='mean', last_layer_only=False):
    """
    An method for calculating KL divergence of whole layers in the model.
    Arguments:
        model (nn.Module): a model to be calculated for KL-divergence.
        reduction (string, optional): Specifies the reduction to apply to the output:
            ``'mean'``: the sum of the output will be divided by the number of
            elements of the output.
            ``'sum'``: the output will be summed.
        last_layer_only (Bool): True for return only the last layer's KL divergence.

    """
    device = torch.device("cuda" if next(model1.parameters()).is_cuda else "cpu")
    kl = torch.Tensor([0]).to(device)
    kl_sum = torch.Tensor([0]).to(device)
    n = torch.Tensor([0]).to(device)

    for (m1, m2) in zip(model1.modules(), model2.modules()):
        if isinstance(m1, (BayesLinear, BayesConv2d)):
            kl = _kl_loss(m1.weight_mu, m1.weight_log_sigma, m2.weight_mu, m2.weight_log_sigma)
            kl_sum += kl
            n += len(m1.weight_mu.view(-1))

            if m1.bias:
                kl = _kl_loss(m1.bias_mu, m1.bias_log_sigma, m2.bias_mu, m2.bias_log_sigma,)
                kl_sum += kl
                n += len(m1.bias_mu.view(-1))

        if isinstance(m1, (BayesBatchNorm1d, BayesBatchNorm2d)):
            if m1.affine:
                kl = _kl_loss(m1.weight_mu, m1.weight_log_sigma, m2.weight_mu, m2.weight_log_sigma)
                kl_sum += kl
                n += len(m1.weight_mu.view(-1))

                kl = _kl_loss(m1.bias_mu, m1.bias_log_sigma, m2.bias_mu, m2.bias_log_sigma, )
                kl_sum += kl
                n += len(m1.bias_mu.view(-1))
    if last_layer_only or n == 0:
        return kl
    if reduction == 'mean':
        return kl_sum / n
    elif reduction == 'sum':
        return kl_sum
    else:
        raise ValueError(reduction + " is not valid")

def get_model(args):
    """
    :param args:
    :return: model
    """
    # assert args.model in ["digit", "alexnet"]
    if args.model == "bayes":
        print("Bayesian Model is used...")
        return DigitBayesModel()
    elif args.model == "variational":
        print("Variational Model is used...")
        return DigitVariationalModel()
    else:
        # return AlexNet()
        print("Digit Model is used...")
        return DigitModel()


