import numpy as np

def movingaverage(values, window):
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma

def decay(x, d=0.9):
    return x * ( d ** np.arange(len(x))[::-1] )
