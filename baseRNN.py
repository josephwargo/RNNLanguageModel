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
        # embeddings
        self.embeddings = embeddings
        self.corpus = corpus
        self.word2ind = word2ind
        self.embeddingsShape = embeddings.shape[1]
        self.numEmbeddings = embeddings.shape[0]
        
        # hyperparameters
        self.epochs = epochs
        self.debug = debug
        self.adam = adam
        self.learningRate = learningRate
        self.lossFunction = lossFunction
        self.activations = hiddenLayerActivations + [outputActivation]
        self.reverseActivations = self.activations.copy()
        self.reverseActivations.reverse()

        # loss
        self.loss = []
        self.lossGradients = []
        
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

    # training methods
    def forwardPass(self, text):
        ### Resetting losses and gradients in advance of forward pass ###
        # setting/resetting loss
        localLoss = [] # list to store the local loss to average at the end
        self.lossGradients = [] # list used to store loss gradients for backwards pass
                                # resetting list to empty so we only store gradients from this
                                # instance of the forward pass
        
        # setting/resetting layer gradients
        for layerName in self.layers.keys():
            layer = self.layers[layerName] # layer
            layer.N = np.zeros(layer.N.shape) # hidden state at time 0
            layer.NMemory = [] # list to store hidden states (for BPTT)

        ### Executing forward pass ###
        # cycling through each word (timestep)
        for wordIndex in range(len(text)-1):
            # selecting proper input embeddings
            inputWord = text[wordIndex]
            prevLayerHiddenState = self.embeddings[self.word2ind[inputWord]]

            # creating a onehot vector to use to calculate loss
            outputWord = text[wordIndex+1]
            outputOneHot = np.zeros(self.numEmbeddings)
            outputOneHot[self.word2ind[outputWord]] = 1
            
            # cycling through each layer
            for count, layerName in enumerate(self.layers.keys()):
                layer = self.layers[layerName]
                # for output layer
                if layerName == 'outputLayer':
                    z = np.dot(prevLayerHiddenState, layer.layerWeights) + layer.bias
                    logits = caa.activation(self.activations[count], z)
                    layer.thisLayerHiddenState = logits
                    layer.thisLayerHiddenStateMemory.append(logits)
                else:
                    # calculating dot product and activation
                    layerDotProduct = np.dot(prevLayerHiddenState, layer.layerWeights)
                    timeDotProduct = np.dot(layer.thisLayerHiddenState, layer.timeWeights)
                    z = layerDotProduct + timeDotProduct + layer.bias # z = Uh + Wx + b
                    
                    prevLayerHiddenState = caa.activation(self.activations[count], z) # activation - depending on the layer

                    # updating hidden state and hidden state memory
                    layer.thisLayerHiddenState = prevLayerHiddenState
                    layer.thisLayerHiddenStateMemory.append(prevLayerHiddenState)
                    # layer.zMemory.append(z) # only necessary for select loss functions - commented out for now
            
            # storing local loss and loss gradients
            if self.lossFunction == 'crossEntropyLoss':
                localLoss.append(caa.crossEntropyLoss(outputOneHot, logits))
                self.lossGradients.append(caa.softmaxLocalError(outputOneHot, logits))
            else:
                raise Exception('Unknown loss function')

        # calculating final loss (divided over all words in the text)
        self.loss.append(np.mean(localLoss))

    def backwardPass(self, text):
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
        reverseLossGradients = list(self.lossGradients)
        reverseLossGradients.reverse()
        # lists to hold the update values for weights and biases
        layerWeightUpdates = []
        timeWeightUpdates = []
        biasUpdates = []
        dCdH = 0

        # iterating through time backwards
        for reverseTimeStep in range(len(text)-1):
            timeStep = len(reverseLossGradients) - reverseTimeStep
            # getting proper local error
            localError = reverseLossGradients[timeStep]
            
            # selecting proper input embeddings
            inputWord = text[timeStep]
            inputEmbedding = self.embeddings[self.word2ind[inputWord]]

            # iterating through layers backwards
            for layerNum, layerName in enumerate(reverseKeys):
                currLayer = self.layers[layerName]
                
                # weight+bias updates, and the dCdH for the next round of backpropogation (if not first hidden layer)
                if layerName != 'hiddenLayer1':
                    prevLayer = self.layers[reverseKeys[layerNum+1]]

                    # determining hidden state to use to calculate gradient
                    prevLayerAtCurrTime = prevLayer.NMemory[timeStep]
                    currLayerAtPrevTime = currLayer.NMemory[timeStep-1]

                    # calculating weight and bias gradients
                    dCdLayerW = np.dot(prevLayerAtCurrTime.T, localError)
                    dCdTimeW = np.dot(currLayerAtPrevTime.T, localError)
                    dCdB = np.sum(localError, axis=0, keepdims=True) # cost function WRT input biases - value used to update bias
                    
                    # calculating loss gradient to pass back
                    # TODO: determine how the gradient is passed back - see Notion for notes
                    dCdLayerH = np.dot(localError, currLayer.layerW.T)
                    dCdTimeH = np.dot(localError, currLayer.timeW.T)
                
                # weight and bias updates for when we hit the first hidden layer
                else:
                    currLayerAtPrevTime = currLayer.NMemory[timeStep-1]
                    dCdLayerW = np.dot(inputEmbedding.T, localError)
                    dCdTimeW = np.dot(currLayerAtPrevTime.T, localError)
                    dCdB = np.sum(localError, axis=0, keepdims=True) # cost function WRT input biases - value used to update bias
                    
                layerWeightUpdates.append(dCdLayerW)
                timeWeightUpdates.append(dCdTimeW)
                biasUpdates.append(dCdB)

                # activation WRT output node value
                localError = caa.localError(self.reverseActivations[layerNum], currLayer, dCdH)
        
        # updating weights and biases
        # for count, layerName in enumerate(reverseKeys):
        #     layer = self.layers[layerName]
        #     layer.layerW += -self.learningRate*layerWeightUpdates[count]
        #     layer.timeW += -self.learningRate*timeWeightUpdates[count]
        #     layer.b += -self.learningRate*(biasUpdates[count].reshape(-1,))
    
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