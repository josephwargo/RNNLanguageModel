import numpy as np

# functions to compute costs and gradients of costs
def MSE(y, y_pred):
    return np.mean(.5*((y-y_pred)**2))
def MSEgradient(y, y_pred):
    return -(y-y_pred)

# no gradient function because this will only be paired with softmax, and they have a joint gradient function
def crossEntropyLoss(yIndex, yPred):
    # determining maxes to normalize
    maxLogits = np.max(yPred) #, axis=1, keepdims=True)

    # using log sum exp trick
    normalizedExp = np.exp(yPred - maxLogits)
    normalizedProbs = np.sum(normalizedExp) #, axis=1, keepdims=True)
    logSumExp = maxLogits + np.log(normalizedProbs)

    # geting the log probability
    logProb = logSumExp - yPred[:, yIndex]

    return logProb

# functions to compute activation and gradient of activations
def sigmoid(x):
    return 1 / (1 + np.exp(-(x)))
def sigmoidGradient(y):
    return y*(1-y)

def relu(x):
    return np.maximum(0, x)
def reluGradient(y):
    return (y>0)*1

def tanH(x):
    numerator = np.exp(x) - np.exp(-x)
    denominator = np.exp(x) + np.exp(-x)
    return numerator/denominator
def tanHGradient(y):
    return 1 - y**2

# no gradient function because this will only be paired with Cross Entropy Loss,
# and they have a joint gradient function
def softmax(x):
    if len(x.shape)>1:
        normalization = np.max(x, axis=1, keepdims=True)
        numerator = np.exp(x - normalization)
        denominator = np.sum(numerator, axis=1, keepdims=True)
    else:
        normalization = np.max(x)
        numerator = np.exp(x - normalization)
        denominator = np.sum(numerator)
    return numerator / (denominator+10e-8)

# special gradient for softmax & cross entropy loss
def softmaxLocalError(wordIndex, logits):
    prob = softmax(logits)
    prob[:, wordIndex] -= 1.0
    return prob

# choosing activation
def activation(activationName, z):
    if activationName == 'relu':
        return relu(z)
    elif activationName =='sigmoid':
        return sigmoid(z)
    elif activationName == 'tanH':
        return tanH(z)
    elif activationName == 'softmax':
        return softmax(z)
    else:
        raise Exception('Unknown activation function')

# choosing gradient
def localError(activationName, hiddenState, dLdH):
    if activationName == 'relu':
        dHdZ = reluGradient(hiddenState)
        localError = dLdH * dHdZ
    elif activationName == 'sigmoid':
        dHdZ = sigmoidGradient(hiddenState)
        localError = dLdH * dHdZ
    elif activationName == 'tanH':
        dHdZ = tanHGradient(hiddenState)
        localError = dLdH * dHdZ
    # special case of softmax & cross entropy loss
    # TBD
    return localError

