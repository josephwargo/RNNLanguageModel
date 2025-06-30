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
        hiddenLayer1 = neuronLayer(self.embeddingsShape, hiddenLayerShapes[0],
                                   hiddenLayerActivations[0], rnn=True, adam=adam)
        self.layers = {'hiddenLayer1': hiddenLayer1}
        if len(hiddenLayerShapes) > 1:
            for count, inputShape in enumerate(hiddenLayerShapes):
                layerNum = count+2
                if count<len(hiddenLayerShapes)-1:
                    self.layers["hiddenLayer{}".format(layerNum)] = neuronLayer(
                        inputShape, hiddenLayerShapes[count+1], hiddenLayerActivations[count+1], rnn=True, adam=adam)
       
        # adding output layer to dictionary of all layers
        outputLayer = neuronLayer(
            hiddenLayerShapes[-1], self.numEmbeddings, outputActivation, rnn=True, adam=adam)
        self.layers['outputLayer'] = outputLayer

    # training methods
    def forwardPass(self, text):
        ### Resetting losses and gradients in advance of forward pass ###
        
        # setting/resetting loss
        localLoss = [] # list to store the local loss to average at the end
        self.lossGradients = [] # list used to store loss gradients for backwards pass
                                # resetting list to empty so we only store gradients from this
                                # instance of the forward pass
        
        # setting/resetting gradients & other stored variables
        for layerName in self.layers.keys():
            # forward pass
            layer = self.layers[layerName] # layer
            layer.thisLayerMostRecentOutput = np.zeros(layer.thisLayerMostRecentOutput.shape) # hidden state at time 0
            layer.prevLayerOutputMemory = [] # memory of hidden states from the previous layer in  this timestep
            layer.prevTimeStepOutputMemory = [] # memory of hidden states from this layer in the previous timestep
            layer.thisLayerOutputMemory = [] # memory of the output from this layer

            # backward pass
            layer.thisLayerTimeLocalError = np.zeros(shape=(layer.thisLayerTimeLocalError.shape))
            layer.timeWeightUpdates = []
            layer.layerWeightUpdates = []
            layer.biasUpdates = []

        ### Executing forward pass ###
        # cycling through each word (timestep)
        for wordIndex in range(len(text)-1):
            # selecting proper input embeddings
            inputWord = text[wordIndex]
            inputVocabIndex = self.word2ind[inputWord]
            prevLayerOutput = self.embeddings[inputVocabIndex]

            # selecting index for output word
            outputWord = text[wordIndex+1]
            outputVocabIndex = self.word2ind[outputWord]
            
            # cycling through each layer
            for count, layerName in enumerate(self.layers.keys()):
                layer = self.layers[layerName]
                # for output layer
                if layerName == 'outputLayer':
                    # adding previous layer hidden state to memory (for BPTT)
                    layer.prevLayerOutputMemory.append(prevLayerOutput)

                    z = np.dot(prevLayerOutput, layer.layerWeights) + layer.bias
                    logits = caa.activation(layer.activation, z)
                    
                    layer.thisLayerMostRecentOutput = prevLayerOutput

                    # updating memory for BPTT
                    
                    # layer.prevTimeStepOutputMemory.append(logits)
                # for non-output layers
                else:
                    # adding previous layer and timestep hidden states to memory (for BPTT)
                    layer.prevLayerOutputMemory.append(prevLayerOutput)
                    layer.prevTimeStepOutputMemory.append(layer.thisLayerMostRecentOutput)
                    
                    # calculating hidden state (dot products and activation)
                    layerDotProduct = np.dot(prevLayerOutput, layer.layerWeights)
                    timeDotProduct = np.dot(layer.thisLayerMostRecentOutput, layer.timeWeights)
                    z = layerDotProduct + timeDotProduct + layer.bias # z = Uh + Wx + b
                    hiddenState = caa.activation(layer.activation, z) # activation - depending on the layer

                    # updating hidden state and hidden state memory
                    layer.thisLayerMostRecentOutput = hiddenState
                    layer.thisLayerOutputMemory.append(hiddenState)

                    # updating so that this outputs hidden state feeds into 
                    prevLayerOutput = hiddenState
                    
            # storing local loss and loss gradients
            if self.lossFunction == 'crossEntropyLoss':
                loss = caa.crossEntropyLoss(outputVocabIndex, logits)
                localLoss.append(loss)
                self.lossGradients.append(caa.softmaxLocalError(outputVocabIndex, logits)) # dLdH
            else:
                raise Exception('Unknown loss function')

        # calculating final loss (divided over all words in the text)
        meanLoss = np.mean(localLoss)
        self.loss.append(meanLoss)

    def backwardPass(self, text):
        """
        Gradient Notation:
        L = loss function
        H = activation
        Z = Wx + b
        W = weights
        B = bias
        X = input from previous layer
        """
        # reversing layers to iterate backwards through
        reverseKeys = list(self.layers.keys())
        reverseKeys.reverse()
        # reverseLossGradients = list(self.lossGradients)
        # reverseLossGradients.reverse()
        # lists to hold the update values for weights and biases
        dLdH = 0

        # iterating through time backwards
        for reverseTimeStep in range(1, len(text)):
            timeStep = len(text) - reverseTimeStep - 1
            
            # getting proper local error that was stored during forward pass
            layerLocalError = self.lossGradients[timeStep]
            timeLocalError = 0

            # selecting proper input embeddings
            inputWord = text[timeStep]
            inputEmbedding = self.embeddings[self.word2ind[inputWord]]

            # selecting proper input embeddings
            inputWord = text[timeStep]
            inputVocabIndex = self.word2ind[inputWord]
            # inputWordEmbedding = self.embeddings[inputVocabIndex]

            # selecting index for output word
            # outputWord = text[timeStep+1]
            # outputVocabIndex = self.word2ind[outputWord]

            # iterating through layers backwards
            for layerNum, layerName in enumerate(reverseKeys):
                layerNum = len(reverseKeys) - 1 - layerNum
                currLayer = self.layers[layerName]
                
                if layerName == 'outputLayer':
                    # dLossdZ (dLdZ) = stored during forward pass
                    dLossdZ = layerLocalError

                    # dLossdOutputWeights (dLdW)
                    # dLdZ = stored during forward pass
                    # dZdW
                    prevLayerHiddenState = currLayer.prevLayerOutputMemory[timeStep]
                    # dLdW = dLdZ * dZdW
                    dLossdOutputWeights = np.outer(prevLayerHiddenState, dLossdZ)
                    
                    # dLossdOutputBias (dLdB)
                    # dLdZ = stored during forward pass
                    # dZdB = 1
                    # dLdB = dLdZ * dZdB
                    dLossdOutputBias = layerLocalError

                    # dLossdPreviousHiddenLayer (dLdH)
                    # dLdZ = stored during forward pass
                    # dZdH = layerWeights
                    # dLdH = dLdZ * dZdH
                    dLossdPrevLayerHiddenState = np.dot(currLayer.layerWeights, layerLocalError)
                    
                    # updating localError to pass back
                    layerLocalError = dLossdPrevLayerHiddenState

                    # adding gradients to list for weight updates
                    currLayer.layerWeightUpdates.append(dLossdOutputWeights)
                    currLayer.biasUpdates.append(dLossdOutputBias)
                
                else:
                    # dLossdZ (dLdZ)
                    # dLdH = passed back from previous layer
                    dLdH = layerLocalError+currLayer.thisLayerTimeLocalError
                    # hiddenState = most recent version of this hidden state
                    hiddenState = currLayer.thisLayerOutputMemory[timeStep]
                    # dLdZ
                    dLossdZ = caa.localError(currLayer.activation, hiddenState, dLdH)
                    
                    # dLossdTimeWeights (dLdWt)
                    # dLdZ = calculated above
                    # dZdWt
                    prevTimeStepHiddenState = currLayer.prevTimeStepOutputMemory[timeStep]
                    # dLdWt = dLdZ * dZdWt
                    dLossdTimeWeights = np.outer(prevTimeStepHiddenState, dLossdZ)
                    
                    # dLossdLayerWeights (dLdWl)
                    # dLdZ = calculated above
                    # dZdWl
                    prevLayerHiddenState = currLayer.prevLayerOutputMemory[timeStep]
                    # dLdWl = dLdZ * dZdWl
                    dLossdLayerWeights = np.outer(prevLayerHiddenState, dLossdZ)
                    
                    # dLossdOutputBias (dLdB)
                    # dLdZ = calculated above
                    # dZdb = 1
                    # dLdWl = dLdZ * dZdb
                    dLossdOutputBias = dLossdZ

                    # dLossdPreviousHiddenLayer (dLdH)
                    # dLdZ = calculated above
                    # dZdH = layerWeights
                    # dLdH = dLdZ * dZdH
                    layerLocalError = np.dot(currLayer.layerWeights, dLossdZ)
                    
                    # dLossdPreviousTimeStep (dLdH)
                    # dLdZ = calculated above
                    # dZdH = timeWeights
                    # dLdH = dLdZ * dZdH
                    currLayer.thisLayerTimeLocalError = np.dot(currLayer.timeWeights, dLossdZ)

                    # adding gradients to list for weight updates
                    currLayer.layerWeightUpdates.append(dLossdLayerWeights)
                    currLayer.timeWeightUpdates.append(dLossdTimeWeights)
                    currLayer.biasUpdates.append(dLossdOutputBias)
        
        # updating weights and biases
        for count, layerName in enumerate(reverseKeys):
            currLayer = self.layers[layerName]

            # layer weight calculation
            layerWeightUpdate = np.stack(currLayer.layerWeightUpdates).sum(axis=0)
            # print(currLayer.layerWeightUpdates[1])

            # time weight calculation
            if len(currLayer.timeWeightUpdates) > 0:
                timeWeightUpdate = np.stack(currLayer.timeWeightUpdates).sum(axis=0)

            
            # bias calculation
            biasUpdate = np.stack(currLayer.biasUpdates).sum(axis=0)
            
            # updates
            currLayer.layerWeights += -self.learningRate*layerWeightUpdate
            if len(currLayer.timeWeightUpdates) > 0:
                currLayer.timeWeights += -self.learningRate*timeWeightUpdate
            currLayer.bias += -self.learningRate*biasUpdate
    
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
            if layer.activation == 'relu':
                input = caa.relu(np.dot(input, layer.layerW) + layer.b)
            elif layer.activation == 'sigmoid':
                input = caa.sigmoid(np.dot(input, layer.layerW) + layer.b)
            elif layer.activation == 'tanH':
                input = caa.tanH(np.dot(input, layer.layerW) + layer.b)
            elif layer.activation == 'softmax':
                input = caa.softmax(np.dot(input, layer.layerW) + layer.b)
        return input