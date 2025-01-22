import numpy as np
from neuronLayer import neuronLayer
import costsAndActivations as caa

# entire net
class neuralNet(object):
    def __init__(self, embeddings, corpus, word2ind, outputActivation, hiddenLayerShapes, 
                 hiddenLayerActivations, lossFunction='crossEntropyLoss', learningRate=.001, epochs=1,
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
        self.corpus = corpus
        self.word2ind = word2ind
        self.embeddingsShape = embeddings.shape[1]
        self.numEmbeddings = embeddings.shape[0]
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
        outputLayer = neuronLayer(hiddenLayerShapes[-1], self.numEmbeddings, adam)
        self.allLayers['outputLayer'] = outputLayer

        # to track loss
        self.loss = None

    # training methods
    def forwardPass(self, text):
        # cycling through each word
        localLoss = 0
        for wordIndex in range(len(text)-1):
            # selecting proper input embeddings
            inputWord = text[wordIndex]
            outputWord = text[wordIndex+1]
            inputEmbedding = self.embeddings[self.word2ind[inputWord]]
            # creating a onehot vector to use to calculate loss
            outputOneHot = np.zeros(self.numEmbeddings)
            outputOneHot[self.word2ind[outputWord]] = 1
            print(self.word2ind[outputWord])
            print(np.argmax(outputOneHot))
            # cycling through each layer
            for count, layerName in enumerate(self.allLayers.keys()):
                layer = self.allLayers[layerName]
                # calculating dot product + activation
                z = np.dot(inputEmbedding, layer.W) + layer.b
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
                inputEmbedding = layer.N
            # storing final error
            if self.lossFunction == 'crossEntropyLoss':
                print('Pair Loss: ' + str(localLoss))
                print()
                localLoss += caa.crossEntropyLoss(outputOneHot, layer.N)
            else:
                raise Exception('Unknown cost function')
        self.loss = localLoss / (len(text)-1)
        print(self.loss)

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
                dCdW = np.dot(prevLayer.N.T, localError)
                dCdB = np.sum(localError, axis=0, keepdims=True) # cost function WRT input biases - value used to update bias
                if currLayer.adam:
                    dCdW, dCdB = currLayer.updateAdam(dCdW, dCdB)
                dCdH = np.dot(localError, currLayer.W.T)
            # weight and bias updates for when we hit the first hidden layer
            else:
                dCdW = np.dot(input.T, localError)
                dCdB = np.sum(localError, axis=0, keepdims=True) # cost function WRT input biases - value used to update bias
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
    def trainModel(self, corpus):
        numSamples = input.shape[0]
        for epoch in range(self.epochs):
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