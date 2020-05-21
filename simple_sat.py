#!/usr/bin/env python
import torch

from loss import and_, or_, neg_, clause_loss, truthy

zero = torch.tensor([0])
one = torch.tensor([1])
assert and_(zero, zero).item() == 0
assert and_(zero, one).item() == 0
assert and_(one, zero).item() == 0
assert and_(one, one).item() == 1

assert or_(zero, zero).item() == 0
assert or_(zero, one).item() == 1
assert or_(one, zero).item() == 1
assert or_(one, one).item() == 1

assert neg_(zero).item() == 1
assert neg_(one).item() == 0


def simple_optim():
    """
    Try to SAT:
    x -- and
    neg(x) or y -- and
    neg(y) or z

    should be all 1

    neg(x) or y is equivalent to x -> y.
    Thus the clauses are: x and x -> y and y -> z
    """
    print('Simple example')
    x = torch.nn.Parameter(torch.tensor([0.5]))
    y = torch.nn.Parameter(torch.tensor([0.5]))
    z = torch.nn.Parameter(torch.tensor([0.5]))

    def get_clause():
        return {
            'x': x,
            'neg(x)_or_y': or_(neg_(x), y),
            'neg(y)_or_z': or_(neg_(y), z)
        }

    assert clause_loss(get_clause()).item() != 1

    optimizer = torch.optim.SGD((x, y, z), lr=0.2)
    for i in range(10):
        optimizer.zero_grad()
        # We need to build the graph each time
        loss = clause_loss(get_clause())
        print(loss.item())
        loss.backward()
        optimizer.step()
        # We need to clamp the parameters to avoid them becoming higher than 1
        x.data = x.data.clamp(0, 1)
        y.data = y.data.clamp(0, 1)
        z.data = z.data.clamp(0, 1)
    print(x, y, z)


simple_optim()


def many_answers(optimizer, runs=100):
    """
    Try to SAT:
    neg(x) or y -- and
    neg(y) or z

    # can be (1, 1, 1) or (0, 1, 1) or (0, 0, 1) or (0, 0, 0) in classical
    """
    print('Many answers')
    x = torch.nn.Parameter(torch.tensor([0.5]))
    y = torch.nn.Parameter(torch.tensor([0.5]))
    z = torch.nn.Parameter(torch.tensor([0.5]))

    def get_clause():
        return {
            # truthy just adds a sigmoid, to avoid clamping but slows down training
            'neg(x)_or_y': or_(neg_(truthy(x)), truthy(y)),  # x -> y
            'neg(y)_or_z': or_(neg_(truthy(y)), truthy(z))  # y -> z
        }

    assert clause_loss(get_clause()).item() != 1

    if optimizer == 'sdg':
        optimizer = torch.optim.SGD((x, y, z), lr=1.0)
    elif optimizer == 'adam':
        optimizer = torch.optim.Adam((x, y, z), lr=1.0)
    else:
        raise ValueError('Read the code!')

    # More runs needed to sat
    for i in range(runs):
        optimizer.zero_grad()
        # We need to build the graph each time
        loss = clause_loss(get_clause())
        print(loss.item())
        loss.backward()
        optimizer.step()
    print(truthy(x), truthy(y), truthy(z))


many_answers('sdg', 20)
# Lets try using some momentum it helps vs sigmoid
many_answers('adam', 20)


# Now lets do this properly
class SatSolver(torch.nn.Module):
    def __init__(self, parameters, generate_clauses):
        super(SatSolver, self).__init__()
        for name, param in parameters.items():
            self.register_parameter(name=name, param=param)
        self.generate_clauses = generate_clauses

    def forward(self):
        # We need to generate the graph each time
        return and_(*self.generate_clauses())


def as_sat_solver():
    p = {
        'x': torch.nn.Parameter(torch.tensor([0.0])),
        'y': torch.nn.Parameter(torch.tensor([0.0])),
        'z': torch.nn.Parameter(torch.tensor([0.0]))
    }

    def generate_clauses():
        return [
            truthy(p['x']),
            or_(neg_(truthy(p['x'])), truthy(p['y'])),
            or_(neg_(truthy(p['y'])), truthy(p['z']))
        ]
    sat_solver = SatSolver(parameters=p, generate_clauses=generate_clauses)
    optimizer = torch.optim.Adam(sat_solver.parameters(), lr=1.0)
    for i in range(10):
        optimizer.zero_grad()
        sat_val = sat_solver()
        loss = neg_(sat_val)
        print(f"Sat={sat_val.item()}, loss={loss.item()}")
        loss.backward()
        optimizer.step()


as_sat_solver()
