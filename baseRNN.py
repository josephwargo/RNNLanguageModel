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
        hiddenLayer1 = neuronLayer(self.embeddingsShape, hiddenLayerShapes[0], rnn=True, adam=adam)
        self.layers = {'hiddenLayer1': hiddenLayer1}
        if len(hiddenLayerShapes) > 1:
            for count, value in enumerate(hiddenLayerShapes):
                layerNum = count+2
                if count<len(hiddenLayerShapes)-1:
                    self.layers["hiddenLayer{}".format(layerNum)] = neuronLayer(value, hiddenLayerShapes[count+1], rnn=True, adam=adam)
       
        # adding output layer to dictionary of all layers
        outputLayer = neuronLayer(hiddenLayerShapes[-1], self.numEmbeddings, rnn=True, adam=adam)
        self.layers['outputLayer'] = outputLayer

        # to track loss
        self.loss = None
        self.losses = []

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
            
            # cycling through each layer
            for count, layerName in enumerate(self.layers.keys()):
                
                layer = self.layers[layerName]
                
                # calculating dot product + activation
                z = np.dot(inputEmbedding, layer.layerW) + np.dot(layer.N, layer.timeW) + layer.b
                layer.N = caa.activation(self.activations[count], z)
                inputEmbedding = layer.N
            
            # storing local loss
            if self.lossFunction == 'crossEntropyLoss':
                self.losses.append(caa.crossEntropyLoss(outputOneHot, layer.N))
            else:
                raise Exception('Unknown cost function')
        
        # calculating final loss (divided over all words in the text)
        self.loss = np.mean(self.losses)

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
        # reversing layers to iterate backwards through
        reverseKeys = list(self.layers.keys())
        reverseKeys.reverse()
        reverseLosses = list(self.losses)
        reverseLosses.reverse()
        # lists to hold the update values for weights and biases
        weightUpdates = []
        biasUpdates = []
        dCdH = 0

        # iterating through time backwards
        for loss in reverseLosses:
            

            # iterating through layers backwards
            for count, layerName in enumerate(reverseKeys):
                
                currLayer = self.layers[layerName]
                
                # activation WRT output node value
                localError = caa.localError(self.reverseActivations[count], currLayer, dCdH, output)
                
                # weight+bias updates, and the dCdH for the next round of backpropogation (if not first hidden layer)
                if layerName != 'hiddenLayer1':
                    prevLayer = self.layers[reverseKeys[count+1]]
                    dCdW = np.dot(prevLayer.N.T, localError)
                    dCdB = np.sum(localError, axis=0, keepdims=True) # cost function WRT input biases - value used to update bias
                    
                    dCdH = np.dot(localError, currLayer.layerW.T)
                
                # weight and bias updates for when we hit the first hidden layer
                else:
                    dCdW = np.dot(input.T, localError)
                    dCdB = np.sum(localError, axis=0, keepdims=True) # cost function WRT input biases - value used to update bias
                    
                weightUpdates.append(dCdW)
                biasUpdates.append(dCdB)
        
        # updating weights and biases
        for count, layerName in enumerate(reverseKeys):
            layer = self.layers[layerName]
            layer.layerW += -self.learningRate*weightUpdates[count]
            layer.b += -self.learningRate*(biasUpdates[count].reshape(-1,))
    
    # training model by repeatedly running forward and backward passes
    def trainModel(self, corpus):
        numSamples = input.shape[0]
        for epoch in range(self.epochs):
            self.forwardPass(inputShuffled[start:end], outputShuffled[start:end])
            self.backwardPass(inputShuffled[start:end], outputShuffled[start:end])

    # return predicted output for a given input
    def query(self, input):
        for count, layerName in enumerate(self.layers.keys()):
            layer = self.layers[layerName]
            if self.activations[count] == 'relu':
                input = caa.relu(np.dot(input, layer.layerW) + layer.b)
            elif self.activations[count] == 'sigmoid':
                input = caa.sigmoid(np.dot(input, layer.layerW) + layer.b)
            elif self.activations[count] == 'tanH':
                input = caa.tanH(np.dot(input, layer.layerW) + layer.b)
            elif self.activations[count] == 'softmax':
                input = caa.softmax(np.dot(input, layer.layerW) + layer.b)
        return input