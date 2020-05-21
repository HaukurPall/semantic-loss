import torch

# Product t-norm, co-norm and canonical negation


class Predicate(torch.nn.Module):
    def __init__(self, in_dim):
        super(Predicate, self).__init__()
        self.linear = torch.nn.Linear(in_dim, 1)

    def forward(self, tensor):
        return torch.sigmoid(self.linear(tensor))


def truthy(parameter: torch.nn.Parameter):
    return torch.sigmoid(parameter)


def and_(*xs):
    """
    Product t-norm: $T_{P}(x_1, \dots, x_n)=\prod_{i=1}^n x_i $
    """
    prod = torch.prod(torch.cat([x for x in xs], dim=0))
    return torch.min(prod, torch.ones_like(prod))


def or_(x, y):
    """
    Product s-norm: $S_{P}(x_1, \dots, x_n)=1-\prod_{i=1}^n (1-x_i )$
    """
    return x + y - (x * y)


def neg_(x):
    return torch.ones_like(x) - x


def clause_loss(clauses):
    """
    Compute the loss of given clauses (as a dict).
    ANDs the clauses together (this is total satisfied) and then negates the result (missing satisfaction)
    """
    return neg_(and_(*clauses.values()))
