from random import random
from math import exp
from itertools import izip

def add(x, y):
    if len(x) != len(y): raise  ValueError('it should be same dimension of two vectors')
    return map(lambda v: v[0] + v[1], izip(x, y))
    
def dot(x, y):
    if len(x) != len(y): raise  ValueError('it should be same dimension of two vectors')
    return sum(map(lambda v: v[0] * v[1], izip(x, y)))

def sig(x):
    return 1 / (1 + exp(-x))

def create_layer(n_inputs, n_outputs):
    # return [[2 * random() -1 for _ in xrange(n_inputs)] for _ in xrange(n_outputs)]
    return [[random() for _ in xrange(n_inputs)] for _ in xrange(n_outputs)]
    

def cost(t, y): 
    return 0.5 * (t - y) * (t - y)

def learn_xor():
    learning_rate = 0.01
    
    n_inputs = 2
    n_hiddens = 3
    n_outputs = 1
    
    l1 = create_layer(n_inputs,n_hiddens)
    l2 = create_layer(n_hiddens, n_outputs)
        
    def _calc(input):
        c1 = map(lambda v: sig(dot(v, input)), l1)
        c2 = map(lambda v: sig(dot(v, c1)), l2)
        return c1, c2, c2[0]
    
    def _train(xs, ys):
        assert(len(xs) == len(ys))
        n_trains = len(xs)
        while True:
            for t in xrange(n_trains):
                    o1, o2, pred_y = _calc(xs[t])
                    if learning_rate >= cost(ys[t], pred_y):
                            continue

                    # update weights of output layer 
                    for i in xrange(n_outputs):
                            for j in xrange(n_hiddens):
                                    l2[i][j] += learning_rate * o2[i] * (1 - o2[i]) * (ys[t] - o2[i]) * o1[j]
                    
                    # update weights of hidden layer
                    for i in xrange(n_hiddens):
                            for j in xrange(n_inputs):
                                    s = 0
                                    for k in xrange(n_outputs):
                                            s += l2[k][i] * o2[k] * (1 - o2[k]) * (ys[t] - o2[k]) * xs[t][j]
                                    l1[i][j] += learning_rate * o1[i] * (1 - o1[i]) * s
            ok = []
            for t in xrange(n_trains):
                    _, _, pred_y = _calc(xs[t])
                    ok.append(learning_rate >= cost(ys[t], pred_y))
            if all(ok):
                    break
                
    _train([[0, 0], [0, 1], [1, 0], [1, 1]], [0, 1, 1, 0])
    
    print l1
    print l2
learn_xor()
    
