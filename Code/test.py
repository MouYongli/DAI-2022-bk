from models import bayesian_kl_loss
from models import DigitBayesModel

if __name__ == '__main__':
    m1, m2 = DigitBayesModel(), DigitBayesModel()
    bayesian_kl_loss(m1, m2)