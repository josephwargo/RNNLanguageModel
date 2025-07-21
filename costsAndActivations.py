import numpy as np

# functions to compute costs and gradients of costs
def MSE(y, y_pred):
    return .5*np.mean((y-y_pred)**2)
def MSEgradient(y, y_pred):
    return (y_pred-y) / y.size

# no gradient function because this will only be paired with softmax, and they have a joint gradient function
def crossEntropyLoss(yIndex, logits):
    # determining maxes to normalize
    maxLogits = logits.max(axis=1, keepdims=True)

    # using log sum exp trick
    normalizedExp = np.exp(logits - maxLogits)
    normalizedProbs = normalizedExp.sum(axis=1, keepdims=True)
    logSumExp = maxLogits + np.log(normalizedProbs+1e-12)

    # geting the log probability
    updatedLogits = logits[np.arange(logits.shape[0]), yIndex][:, None]
    logProb = logSumExp - updatedLogits

    return logProb.mean()
    # return logProb

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
def softmax(logits):
    if len(logits.shape)>1:
        normalization = np.max(logits, axis=1, keepdims=True)
        numerator = np.exp(logits - normalization)
        denominator = np.sum(numerator, axis=1, keepdims=True)
    else:
        normalization = np.max(logits)
        numerator = np.exp(logits - normalization)
        denominator = np.sum(numerator)
    return numerator / (denominator+1e-8)

# special gradient for softmax & cross entropy loss
def softmaxLocalError(wordIndex, logits):
    prob = softmax(logits)
    prob[np.arange(prob.shape[0]), wordIndex] -= 1.0
    return prob / logits.shape[0]
    # return prob

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