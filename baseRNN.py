import numpy as np
from neuronLayer import neuronLayer
import costsAndActivations as caa

# entire net
class neuralNet(object):
    def __init__(self, embeddings, outputActivation, hiddenLayerShapes, 
                 hiddenLayerActivations, lossFunction='MSE', learningRate=.001, epochs=1, batchSize=1,
                 adam=False, debug=False):
        # errors
        if len(hiddenLayerShapes)!=len(hiddenLayerActivations):
            raise Exception('Length of hiddenLayerShapes does not match length of hiddenLayerActivations')
        if ((lossFunction=='crossEntropyLoss') & (outputActivation!='softmax')) or ((lossFunction!='crossEntropyLoss') & (outputActivation=='softmax')):
            raise Exception('A cost function of Cross Entropy Loss and an output layer activation of Softmax must be paired with each other')
        if adam & (learningRate>.01):
            print('Warning: Learning rate may be too high for ADAM optimizer to function properly')
        # variables straight from initialization
        self.embeddings = embeddings
        self.embeddingsShape = embeddings.shape[0]
        self.batchSize = batchSize
        self.epochs = epochs
        self.debug = debug
        self.adam = adam
        self.learningRate = learningRate
        self.lossFunction = lossFunction
        self.activations = hiddenLayerActivations + [outputActivation]
        self.reverseActivations = self.activations.copy()
        self.reverseActivations.reverse()
        
        # initializing hidden layers and adding to dictionary of all layers
        hiddenLayer1 = neuronLayer(self.embeddingsShape, hiddenLayerShapes[0], adam)
        self.allLayers = {'hiddenLayer1': hiddenLayer1}
        if len(hiddenLayerShapes) > 1:
            for count, value in enumerate(hiddenLayerShapes):
                layerNum = count+2
                if count<len(hiddenLayerShapes)-1:
                    self.allLayers["hiddenLayer{}".format(layerNum)] = neuronLayer(value, hiddenLayerShapes[count+1], adam)
       
        # adding output layer to dictionary of all layers
        outputLayer = neuronLayer(hiddenLayerShapes[-1], self.embeddingsShape, adam)
        self.allLayers['outputLayer'] = outputLayer

        # to track error
        self.error = None

    # training methods
    def forwardPass(self, corpus):
        # cycling through each word
        for wordIndex in enumerate(corpus):
            input = self.embeddingsself.word2ind[corpus[wordIndex]]
            output = corpus[wordIndex+1]
            # cycling through each layer
            for count, layerName in enumerate(self.allLayers.keys()):
                layer = self.allLayers[layerName]
                if self.batchSize>1:
                    bias = np.tile(layer.b, (self.batchSize, 1))
                else:
                    bias = layer.b

                # calculating dot product + activation
                z = np.dot(input, layer.W) + bias
                if self.activations[count] == 'relu':
                    layer.N = caa.relu(z)
                elif self.activations[count]=='sigmoid':
                    layer.N = caa.sigmoid(z)
                elif self.activations[count]=='tanH':
                    layer.N = caa.tanH(z)
                elif (self.activations[count]=='softmax'):
                    layer.N = caa.softmax(z)
                else:
                    raise Exception('Unknown activation function')
                input = layer.N
            # storing final error
            if self.lossFunction == 'MSE':
                self.error = caa.MSE(output, layer.N)
            elif self.lossFunction == 'crossEntropyLoss':
                self.error = caa.crossEntropyLoss(output, layer.N)
            else:
                raise Exception('Unknown cost function')

    def backwardPass(self, input, output):
        """
        Gradient Notation:
        C = cost function
        H = activation
        Z = Wx + b
        W = weights
        B = bias
        X = input from previous layer
        """
        reverseKeys = list(self.allLayers.keys())
        reverseKeys.reverse()
        # lists to hold the update values for weights and biases
        weightUpdates = []
        biasUpdates = []
        # gradient of the cost function to start backpropogation
        if self.lossFunction == 'MSE':
            dCdH = caa.MSEgradient(output, self.allLayers['outputLayer'].N)
        # iterating through layers backwards for backpropogation
        for count, layerName in enumerate(reverseKeys):
            currLayer = self.allLayers[layerName]
            # activation WRT output node value
            if self.reverseActivations[count] == 'relu':
                dHdZ = caa.reluGradient(currLayer.N)
                localError = dCdH * dHdZ
            elif self.reverseActivations[count] == 'sigmoid':
                dHdZ = caa.sigmoidGradient(currLayer.N)
                localError = dCdH * dHdZ
            elif self.reverseActivations[count] == 'tanH':
                dHdZ = caa.tanHGradient(currLayer.N)
                localError = dCdH * dHdZ
            # special case of softmax & cross entropy loss
            elif self.reverseActivations[count] == 'softmax':
                localError = caa.dCdZ(output, currLayer.N)
            
            # weight+bias updates, and the dCdH for the next round of backpropogation
            if layerName != 'hiddenLayer1':
                prevLayer = self.allLayers[reverseKeys[count+1]]
                # TODO: see if this should be divided by batch size
                dCdW = np.dot(prevLayer.N.T, localError) / self.batchSize
                dCdB = np.sum(localError, axis=0, keepdims=True) / self.batchSize # cost function WRT input biases - value used to update bias
                if currLayer.adam:
                    dCdW, dCdB = currLayer.updateAdam(dCdW, dCdB)
                dCdH = np.dot(localError, currLayer.W.T)
            # weight and bias updates for when we hit the first hidden layer
            else:
                dCdW = np.dot(input.T, localError) / self.batchSize
                dCdB = np.sum(localError, axis=0, keepdims=True) / self.batchSize # cost function WRT input biases - value used to update bias
                if currLayer.adam:
                    dCdW, dCdB = currLayer.updateAdam(dCdW, dCdB)
            weightUpdates.append(dCdW)
            biasUpdates.append(dCdB)
        # updating weights and biases
        for count, layerName in enumerate(reverseKeys):
            layer = self.allLayers[layerName]
            layer.W += -self.learningRate*weightUpdates[count]
            layer.b += -self.learningRate*(biasUpdates[count].reshape(-1,))
    
    # training model by repeatedly running forward and backward passes
    def trainModel(self, input, output):
        numSamples = input.shape[0]
        for epoch in range(self.epochs):
            indices = np.random.permutation(numSamples)
            inputShuffled = input[indices]
            outputShuffled = output[indices]
            for start in range(0, numSamples, self.batchSize):
                end = min(start+self.batchSize, numSamples)
                self.forwardPass(inputShuffled[start:end], outputShuffled[start:end])
                self.backwardPass(inputShuffled[start:end], outputShuffled[start:end])

    # return predicted output for a given input
    def query(self, input):
        for count, layerName in enumerate(self.allLayers.keys()):
            layer = self.allLayers[layerName]
            if self.activations[count] == 'relu':
                input = caa.relu(np.dot(input, layer.W) + layer.b)
            elif self.activations[count] == 'sigmoid':
                input = caa.sigmoid(np.dot(input, layer.W) + layer.b)
            elif self.activations[count] == 'tanH':
                input = caa.tanH(np.dot(input, layer.W) + layer.b)
            elif self.activations[count] == 'softmax':
                input = caa.softmax(np.dot(input, layer.W) + layer.b)
        return input